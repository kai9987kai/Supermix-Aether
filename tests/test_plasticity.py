import numpy as np
import pytest
import torch
from dataclasses import replace

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel
from aether.neurofabric.topology import Topology
from aether.neurofabric.plasticity import (
    PlasticityReport,
    lesion_region_links,
    reconcile_model_topology,
    HebbianPlasticityEngine,
)


def small_model():
    c = ModelConfig(
        neuron_count=16,
        scalar_edges=32,
        region_count=4,
        region_message_edges=8,
        state_dimension=4,
        mode_count=2,
        complex_dimensions=2,
    )
    return TissueModel(c, 'wave', seed=42)


def test_lesion_region_links():
    model = small_model()
    initial_top = model.topology
    initial_edges = len(initial_top.region_sources)

    # Pick first link to sever
    sever_pair = (int(initial_top.region_sources[0]), int(initial_top.region_targets[0]))
    lesioned_top = lesion_region_links(initial_top, [sever_pair])

    assert len(lesioned_top.region_sources) == initial_edges - 1
    assert sever_pair not in zip(lesioned_top.region_sources.tolist(), lesioned_top.region_targets.tolist())
    assert lesioned_top.version != initial_top.version

    # Reconcile model with lesioned topology
    reconcile_model_topology(model, lesioned_top)
    assert model.topology == lesioned_top
    assert model.region_sources.shape[0] == initial_edges - 1
    assert model.carriers.shape == (4, 2)

    # Step the model to ensure forward pass works post-lesion
    obs = torch.zeros(1, 12)
    state = model.initial_state(1)
    logits, next_state = model.step(obs, state)
    assert logits.shape == (1, 7)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(next_state).all()


def test_hebbian_structural_plasticity():
    model = small_model()
    initial_version = model.topology.version
    engine = HebbianPlasticityEngine(pruning_percentile=50.0, rewiring_budget=2)

    # Generate synthetic correlated activity across regions (T=20 ticks, R=4 regions)
    rng = np.random.default_rng(1234)
    activity = rng.standard_normal((20, 4))
    # Make regions 0 and 3 strongly correlated
    activity[:, 3] = activity[:, 0] * 0.9 + rng.standard_normal(20) * 0.1

    report = engine.adapt_region_links(model, activity)
    assert isinstance(report, PlasticityReport)
    assert report.original_version == initial_version
    assert report.updated_version != initial_version
    assert report.region_links_pruned > 0
    assert report.region_links_added == report.region_links_pruned
    assert len(report.new_eigenvalues) == 2

    # Step the adapted model to verify execution stability
    obs = torch.zeros(1, 12)
    state = model.initial_state(1)
    logits, next_state = model.step(obs, state)
    assert logits.shape == (1, 7)
    assert torch.isfinite(logits).all()


def test_rejected_topology_preserves_the_entire_model():
    model = small_model()
    digest, config, topology = model.state_digest(), model.config, model.topology
    identities = {name: id(value) for name, value in model.named_parameters()}
    incompatible = Topology.build(replace(config, neuron_count=20), seed=9)
    with pytest.raises(ValueError, match='neuron|region'):
        reconcile_model_topology(model, incompatible)
    assert model.state_digest() == digest
    assert model.config == config
    assert model.topology == topology
    assert {name: id(value) for name, value in model.named_parameters()} == identities


def test_same_size_edge_reordering_keeps_weights_with_their_endpoints():
    model = small_model()
    original_parameter = model.edge_weights
    with torch.no_grad():
        model.edge_weights.copy_(torch.arange(32, dtype=model.edge_weights.dtype))
    top = model.topology
    reordered = Topology(top.scalar_sources[::-1], top.scalar_targets[::-1],
                         top.region_sources, top.region_targets, top.neuron_regions, 'reordered')
    reconcile_model_topology(model, reordered)
    assert model.edge_weights is original_parameter
    assert torch.equal(model.edge_weights, torch.arange(31, -1, -1, dtype=torch.float32))


def test_membership_change_rebuilds_sensory_and_readout_buffers():
    model = small_model().double()
    top = model.topology
    swapped = Topology(top.scalar_sources, top.scalar_targets, top.region_sources,
                       top.region_targets, 3 - top.neuron_regions, 'swapped')
    reconcile_model_topology(model, swapped)
    assert torch.equal(model.readout_neurons, model.neuron_regions == 3)
    assert torch.equal(model.input_profile.flatten() != 0, model.neuron_regions == 0)
    assert model.pooling.dtype == torch.float64
    assert model.carriers.dtype == torch.float64
    logits, _ = model.step(torch.zeros(1, 12, dtype=torch.float64), model.initial_state(1))
    assert torch.isfinite(logits).all()
    # Checkpoint validation must agree with every recomputed derived buffer.
    restored = TissueModel(model.config, 'wave', seed=42).double()
    restored.load_state_dict(model.state_dict())
    assert restored.state_digest() == model.state_digest()


def test_complete_graph_rewiring_is_a_budget_preserving_noop():
    original = small_model()
    model = TissueModel(replace(original.config, region_message_edges=12), 'wave', seed=42)
    before = model.state_digest()
    report = HebbianPlasticityEngine(50, 2).adapt_region_links(
        model, np.random.default_rng(1).normal(size=(10, 4)))
    assert report.region_links_pruned == report.region_links_added == 0
    assert model.config.region_message_edges == 12
    assert model.state_digest() == before
    assert report.original_version == report.updated_version


@pytest.mark.parametrize('percentile,budget', [(-1, 2), (101, 2), (np.nan, 2),
                                               (50, -1), (50, 1.5), (50, True)])
def test_invalid_plasticity_settings_are_rejected(percentile, budget):
    with pytest.raises(ValueError):
        HebbianPlasticityEngine(percentile, budget)


@pytest.mark.parametrize('activity', [np.ones((3, 3)), np.ones((3, 4, 1)),
                                      np.full((3, 4), np.nan), np.full((3, 4), np.inf),
                                      np.ones((3, 4), dtype=complex)])
def test_invalid_activity_preserves_model(activity):
    model = small_model()
    before = model.state_digest()
    with pytest.raises(ValueError):
        HebbianPlasticityEngine().adapt_region_links(model, activity)
    assert model.state_digest() == before
