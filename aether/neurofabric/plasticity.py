"""SynapseForge structural plasticity and lesion adaptation engine.

Enables causal ablation of message/scalar routes and activity-dependent
re-wiring with atomic buffer reconciliation and graph mode updates.
"""

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from aether.neurofabric.topology import Topology
from aether.neurowave.graph_modes import graph_modes


@dataclass(frozen=True)
class PlasticityReport:
    original_version: str
    updated_version: str
    scalar_edges_pruned: int
    scalar_edges_added: int
    region_links_pruned: int
    region_links_added: int
    new_eigenvalues: list[float]


def lesion_region_links(topology: Topology, severed_pairs: list,
                        version_tag: str = 'lesion') -> Topology:
    """Sever specific directed inter-region communication links."""
    severed_set = {tuple(pair) for pair in severed_pairs}
    r_sources = topology.region_sources.tolist()
    r_targets = topology.region_targets.tolist()

    kept_sources = []
    kept_targets = []
    for s, t in zip(r_sources, r_targets):
        if (s, t) not in severed_set:
            kept_sources.append(s)
            kept_targets.append(t)

    new_version = f'{topology.version}-{version_tag}'
    return Topology(
        scalar_sources=topology.scalar_sources,
        scalar_targets=topology.scalar_targets,
        region_sources=np.array(kept_sources, dtype=np.int64),
        region_targets=np.array(kept_targets, dtype=np.int64),
        neuron_regions=topology.neuron_regions,
        version=new_version,
    )


def reconcile_model_topology(model, new_topology: Topology) -> None:
    """Atomically reconcile a TissueModel with a newly modified Topology.

    Recomputes sparse degrees, pooling matrix, carrier eigenvalues/eigenvectors,
    and attention mask without disrupting model parameters or neuron states.
    """
    c = model.config
    # 1. Update basic graph topology buffers
    for name in ('scalar_sources', 'scalar_targets', 'region_sources',
                 'region_targets', 'neuron_regions'):
        model._buffers[name] = torch.tensor(getattr(new_topology, name).copy())

    # 2. Recompute scalar degree
    scalar_deg = np.bincount(new_topology.scalar_targets, minlength=c.neuron_count)
    model._buffers['scalar_degree'] = torch.tensor(np.maximum(scalar_deg, 1), dtype=torch.float32)

    # 3. Recompute region degree
    region_deg = np.bincount(new_topology.region_targets, minlength=c.region_count)
    model._buffers['region_degree'] = torch.tensor(np.maximum(region_deg, 1), dtype=torch.float32)

    # 4. Recompute pooling
    pool = torch.zeros(c.region_count, c.neuron_count)
    neuron_regions = torch.tensor(new_topology.neuron_regions.copy(), dtype=torch.int64)
    pool[neuron_regions, torch.arange(c.neuron_count)] = 1.0
    pool /= pool.sum(1, keepdim=True)
    model._buffers['pooling'] = pool

    # 5. Recompute carrier eigenvalues and eigenvectors
    eigenvalues, carriers = graph_modes(new_topology.region_adjacency(), c.mode_count)
    model._buffers['carrier_eigenvalues'] = torch.tensor(eigenvalues, dtype=torch.float32)
    model._buffers['carriers'] = torch.tensor(carriers, dtype=torch.float32)

    # 6. Recompute attention mask
    mask = torch.zeros(c.region_count, c.region_count, dtype=torch.bool)
    if len(new_topology.region_targets) > 0:
        targets = torch.tensor(new_topology.region_targets.copy(), dtype=torch.int64)
        sources = torch.tensor(new_topology.region_sources.copy(), dtype=torch.int64)
        mask[targets, sources] = True
    model._buffers['attention_mask'] = mask

    # 7. Reconcile scalar edge weights if scalar topology edge count changed
    new_scalar_count = len(new_topology.scalar_sources)
    if new_scalar_count != model.edge_weights.numel():
        old_pairs = list(zip(model.topology.scalar_sources.tolist(), model.topology.scalar_targets.tolist()))
        new_pairs = list(zip(new_topology.scalar_sources.tolist(), new_topology.scalar_targets.tolist()))
        weight_map = {pair: w.item() for pair, w in zip(old_pairs, model.edge_weights)}

        new_weights = []
        for pair in new_pairs:
            new_weights.append(weight_map.get(pair, 0.4))

        model.edge_weights = nn.Parameter(torch.tensor(new_weights, dtype=torch.float32))

    # Update authoritative topology snapshot and config link counts
    model.topology = new_topology
    from dataclasses import replace
    model.config = replace(
        model.config,
        region_message_edges=len(new_topology.region_sources),
        scalar_edges=len(new_topology.scalar_sources),
    )


class HebbianPlasticityEngine:
    """Activity-dependent structural rewiring engine with degree conservation."""

    def __init__(self, pruning_percentile: float = 10.0, rewiring_budget: int = 4):
        self.pruning_percentile = float(pruning_percentile)
        self.rewiring_budget = int(rewiring_budget)

    def adapt_region_links(self, model, regional_activity_history: np.ndarray) -> PlasticityReport:
        """Prune inactive regional links and grow new links between co-active regions.

        regional_activity_history shape: (T, region_count)
        """
        top = model.topology
        r_count = model.config.region_count
        T = len(regional_activity_history)
        if T < 2:
            raise ValueError('activity history requires at least 2 ticks to compute correlation')

        # Correlation between regional activity series
        act = regional_activity_history - regional_activity_history.mean(axis=0, keepdims=True)
        std = act.std(axis=0, keepdims=True) + 1e-8
        norm_act = act / std
        corr = (norm_act.T @ norm_act) / float(T)

        r_sources = top.region_sources.tolist()
        r_targets = top.region_targets.tolist()
        existing_edges = list(zip(r_sources, r_targets))
        existing_set = set(existing_edges)

        # Measure link utilization by correlation
        link_corrs = [corr[s, t] for s, t in existing_edges]
        if not link_corrs:
            return PlasticityReport(top.version, top.version, 0, 0, 0, 0,
                                    model.carrier_eigenvalues.tolist())

        threshold = np.percentile(link_corrs, self.pruning_percentile)
        prune_candidates = [edge for edge, c_val in zip(existing_edges, link_corrs) if c_val <= threshold]
        pruned = prune_candidates[:self.rewiring_budget]
        surviving_edges = [edge for edge in existing_edges if edge not in set(pruned)]

        # Find candidate pairs to sprout with highest correlation
        candidates = []
        for s in range(r_count):
            for t in range(r_count):
                if s != t and (s, t) not in existing_set:
                    candidates.append(((s, t), corr[s, t]))

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        to_add = [pair for pair, _ in candidates[:len(pruned)]]

        final_edges = surviving_edges + to_add
        final_edges.sort(key=lambda pair: (pair[1], pair[0]))

        new_sources = np.array([s for s, _ in final_edges], dtype=np.int64)
        new_targets = np.array([t for _, t in final_edges], dtype=np.int64)

        new_version = f'{top.version}-hebbian-{len(pruned)}'
        new_top = Topology(
            scalar_sources=top.scalar_sources,
            scalar_targets=top.scalar_targets,
            region_sources=new_sources,
            region_targets=new_targets,
            neuron_regions=top.neuron_regions,
            version=new_version,
        )

        reconcile_model_topology(model, new_top)

        return PlasticityReport(
            original_version=top.version,
            updated_version=new_version,
            scalar_edges_pruned=0,
            scalar_edges_added=0,
            region_links_pruned=len(pruned),
            region_links_added=len(to_add),
            new_eigenvalues=model.carrier_eigenvalues.tolist(),
        )
