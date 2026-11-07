"""SynapseForge structural plasticity and lesion adaptation engine.

Enables causal ablation of message/scalar routes and activity-dependent
re-wiring with atomic buffer reconciliation and graph mode updates.
"""

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
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

    All validation and derived tensors are prepared before committing changes.
    Neuron and region counts are fixed; membership and edges may change. Scalar
    weights follow their directed endpoints, with new edges initialized to 0.4.
    Parameter identities survive unchanged shapes. Changing scalar edge count
    replaces ``edge_weights``; recreate the optimizer afterwards. Reordering or
    replacing scalar endpoints also requires resetting/remapping optimizer
    moments and pending gradients, which this model-only operation does not own.
    """
    c = model.config
    if not isinstance(new_topology, Topology):
        raise ValueError('new_topology must be a validated Topology')
    if (len(new_topology.neuron_regions) != c.neuron_count
            or int(new_topology.neuron_regions.max()) + 1 != c.region_count):
        raise ValueError('topology must preserve neuron and region counts')
    new_config = replace(c, scalar_edges=len(new_topology.scalar_sources),
                         region_message_edges=len(new_topology.region_sources))
    # Use the same canonical FP32 derivations as construction/checkpoint loading,
    # then restore each buffer's existing dtype and device.
    prepared = {name: torch.tensor(getattr(new_topology, name).copy())
                for name in ('scalar_sources', 'scalar_targets', 'region_sources',
                             'region_targets', 'neuron_regions')}
    scalar_deg = np.bincount(new_topology.scalar_targets, minlength=c.neuron_count)
    prepared['scalar_degree'] = torch.tensor(np.maximum(scalar_deg, 1), dtype=torch.float32)
    region_deg = np.bincount(new_topology.region_targets, minlength=c.region_count)
    prepared['region_degree'] = torch.tensor(np.maximum(region_deg, 1), dtype=torch.float32)
    pool = torch.zeros(c.region_count, c.neuron_count)
    neuron_regions = prepared['neuron_regions']
    pool[neuron_regions, torch.arange(c.neuron_count)] = 1.0
    pool /= pool.sum(1, keepdim=True)
    prepared['pooling'] = pool
    eigenvalues, carriers = graph_modes(new_topology.region_adjacency(), c.mode_count)
    prepared['carrier_eigenvalues'] = torch.tensor(eigenvalues, dtype=torch.float32)
    prepared['carriers'] = torch.tensor(carriers, dtype=torch.float32)
    mask = torch.zeros(c.region_count, c.region_count, dtype=torch.bool)
    mask[prepared['region_targets'], prepared['region_sources']] = True
    prepared['attention_mask'] = mask
    prepared['input_profile'] = (torch.linspace(0.6, 1.4, c.neuron_count)
                                 * (neuron_regions == 0)).view(1, -1, 1)
    prepared['readout_neurons'] = neuron_regions == c.region_count - 1
    prepared = {name: value.to(device=model._buffers[name].device,
                               dtype=model._buffers[name].dtype)
                for name, value in prepared.items()}

    old_parameter = model.edge_weights
    old_pairs = list(zip(model.topology.scalar_sources, model.topology.scalar_targets))
    old_indexes = {pair: index for index, pair in enumerate(old_pairs)}
    new_pairs = list(zip(new_topology.scalar_sources, new_topology.scalar_targets))
    new_weights = old_parameter.detach().new_full((len(new_pairs),), 0.4)
    for index, pair in enumerate(new_pairs):
        if pair in old_indexes:
            new_weights[index] = old_parameter.detach()[old_indexes[pair]]
    replacement = (old_parameter if new_weights.shape == old_parameter.shape else
                   nn.Parameter(new_weights, requires_grad=old_parameter.requires_grad))
    originals = dict(model._buffers)
    old_topology, old_weights = model.topology, old_parameter.detach().clone()
    try:
        if replacement is old_parameter:
            with torch.no_grad():
                old_parameter.copy_(new_weights)
        model.edge_weights = replacement
        model._buffers.update(prepared)
        model.config, model.topology = new_config, new_topology
    except Exception:
        model._buffers.clear()
        model._buffers.update(originals)
        model.edge_weights = old_parameter
        with torch.no_grad():
            old_parameter.copy_(old_weights)
        model.config, model.topology = c, old_topology
        raise


class HebbianPlasticityEngine:
    """Exploratory correlation heuristic preserving the total regional edge budget.

    Per-region degrees and task performance are not conserved or guaranteed.
    """

    def __init__(self, pruning_percentile: float = 10.0, rewiring_budget: int = 4):
        if (isinstance(pruning_percentile, bool) or not isinstance(pruning_percentile, Real)
                or not math.isfinite(pruning_percentile) or not 0 <= pruning_percentile <= 100):
            raise ValueError('pruning_percentile must be finite and in [0, 100]')
        if (isinstance(rewiring_budget, bool) or not isinstance(rewiring_budget, Integral)
                or rewiring_budget < 0):
            raise ValueError('rewiring_budget must be a nonnegative integer')
        self.pruning_percentile = float(pruning_percentile)
        self.rewiring_budget = int(rewiring_budget)

    def adapt_region_links(self, model, regional_activity_history: np.ndarray) -> PlasticityReport:
        """Prune inactive regional links and grow new links between co-active regions.

        regional_activity_history shape: (T, region_count)
        """
        top = model.topology
        r_count = model.config.region_count
        activity = np.asarray(regional_activity_history)
        if (activity.ndim != 2 or activity.shape[1] != r_count or len(activity) < 2
                or activity.dtype.kind not in 'iuf' or not np.isfinite(activity).all()):
            raise ValueError('activity must be finite real values with shape (at least 2 ticks, region_count)')
        # Scale before centering to avoid overflow on finite, large observations.
        activity = activity.astype(np.float64)
        scale = np.max(np.abs(activity), axis=0, keepdims=True)
        act = np.divide(activity, scale, out=np.zeros_like(activity), where=scale > 0)
        act -= act.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(act, axis=0, keepdims=True)
        normalized = np.divide(act, norms, out=np.zeros_like(act), where=norms > 0)
        corr = normalized.T @ normalized

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
        prune_candidates = sorted(
            [edge for edge, c_val in zip(existing_edges, link_corrs) if c_val <= threshold],
            key=lambda edge: (corr[edge], edge))

        # Find candidate pairs to sprout with highest correlation
        candidates = []
        for s in range(r_count):
            for t in range(r_count):
                if s != t and (s, t) not in existing_set:
                    candidates.append(((s, t), corr[s, t]))

        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        edit_count = min(self.rewiring_budget, len(prune_candidates), len(candidates))
        if not edit_count:
            return PlasticityReport(top.version, top.version, 0, 0, 0, 0,
                                    model.carrier_eigenvalues.tolist())
        pruned = prune_candidates[:edit_count]
        removed = set(pruned)
        surviving_edges = [edge for edge in existing_edges if edge not in removed]
        to_add = [pair for pair, _ in candidates[:edit_count]]

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
