import numpy as np
import pytest
import torch

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel
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
