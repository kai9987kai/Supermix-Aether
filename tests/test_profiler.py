import pytest
import torch
from dataclasses import replace

from aether.config import ModelConfig
from aether.kernel.runtime import TissueModel
from aether.kernel.profiler import (
    ComputeProfile,
    count_step_flops,
    profile_model,
    verify_resource_comparability,
)


def small_config():
    return ModelConfig(
        neuron_count=16,
        scalar_edges=32,
        region_count=4,
        region_message_edges=8,
        state_dimension=4,
        mode_count=2,
        complex_dimensions=2,
    )


def test_count_step_flops_monotonic():
    c = small_config()
    m_scalar = TissueModel(c, 'scalar', seed=1)
    m_vector = TissueModel(c, 'vector', seed=1)
    m_attention = TissueModel(c, 'attention', seed=1)
    m_wave = TissueModel(c, 'wave', seed=1)

    f_scalar = count_step_flops(m_scalar)
    f_vector = count_step_flops(m_vector)
    f_attention = count_step_flops(m_attention)
    f_wave = count_step_flops(m_wave)

    assert f_scalar > 0
    # Communication protocols have additional message encoder/decoder/routing FLOPs
    assert f_vector > f_scalar
    assert f_attention > f_vector
    assert f_wave > f_scalar


def test_profile_model_metrics():
    c = small_config()
    m_wave = TissueModel(c, 'wave', seed=42)
    profile = profile_model(m_wave, warmup_steps=2, benchmark_steps=5)

    assert isinstance(profile, ComputeProfile)
    assert profile.protocol == 'wave'
    assert profile.parameter_count > 0
    assert profile.parameter_bytes > 0
    assert profile.buffer_bytes > 0
    assert profile.optimizer_state_bytes == 2 * profile.parameter_count * 4
    assert profile.flops_per_step > 0
    assert profile.latency_mean_us > 0.0
    assert profile.latency_median_us > 0.0
    assert profile.latency_p95_us >= profile.latency_median_us

    # Test dictionary export
    p_dict = profile.to_dict()
    assert 'flops_per_step' in p_dict
    assert 'latency_mean_us' in p_dict


def test_verify_resource_comparability():
    c = small_config()
    wave_prof = profile_model(TissueModel(c, 'wave', seed=1), warmup_steps=1, benchmark_steps=3)
    scalar_prof = profile_model(TissueModel(c, 'scalar', seed=1), warmup_steps=1, benchmark_steps=3)
    vector_prof = profile_model(TissueModel(c, 'vector', seed=1), warmup_steps=1, benchmark_steps=3)

    baselines = {
        'scalar': scalar_prof,
        'vector': vector_prof,
    }

    # Under relaxed tolerance, should be verified comparable
    result = verify_resource_comparability(wave_prof, baselines, max_flop_ratio=5.0, max_latency_ratio=5.0)
    assert result['comparable'] is True
    assert result['verdict'] == 'verified_comparable'
    assert 'scalar' in result['flop_ratios']

    # Under artificially strict tolerance (e.g. 1.05x), FLOP ratio of wave vs scalar exceeds 1.05
    strict_res = verify_resource_comparability(wave_prof, baselines, max_flop_ratio=1.05, max_latency_ratio=1.05)
    assert strict_res['comparable'] is False
    assert strict_res['verdict'] == 'disproportionate_compute'
    assert len(strict_res['reasons']) > 0


def test_runtime_resource_metrics_integration():
    c = small_config()
    model = TissueModel(c, 'wave', seed=1)
    metrics = model.resource_metrics()
    assert 'step_flops' in metrics
    assert metrics['step_flops'] > 0

    p = model.profile(warmup_steps=1, benchmark_steps=2)
    assert p.protocol == 'wave'


@pytest.mark.parametrize('protocol,expected', [('scalar', 189), ('vector', 431),
                                               ('attention', 495)])
def test_flop_estimate_counts_executed_dense_pooling_and_pooled_readout(protocol, expected):
    # Four neurons, two per region, state width two: the decoder runs once
    # after readout pooling; both auxiliary pooling calls are dense matmuls.
    config = ModelConfig(neuron_count=4, scalar_edges=4, region_count=2,
                         region_message_edges=2, state_dimension=2, mode_count=1,
                         complex_dimensions=1)
    model = TissueModel(config, protocol, seed=1)
    assert count_step_flops(model) == expected
    assert count_step_flops(model, batch_size=3) == 3 * expected - 2 * (4 + (2 if protocol != 'scalar' else 0))


@pytest.mark.parametrize('kwargs', [{'benchmark_steps': 0}, {'benchmark_steps': True},
                                    {'warmup_steps': -1}, {'batch_size': 0}])
def test_invalid_profile_workloads_are_rejected_without_changing_mode(kwargs):
    model = TissueModel(small_config(), 'wave', seed=1)
    model.train()
    with pytest.raises(ValueError):
        profile_model(model, **kwargs)
    assert model.training


def test_profile_labels_estimates_and_workload():
    model = TissueModel(small_config(), 'wave', seed=1).double()
    profile = profile_model(model, warmup_steps=0, benchmark_steps=2, batch_size=2)
    details = profile.to_dict()
    assert details['workload']['batch_size'] == 2
    assert details['workload']['dtype'] == 'torch.float64'
    assert details['workload']['model_config'] == model.config.to_dict()
    assert details['activation_memory_method'] == 'analytical_estimate_not_measured_peak'
    assert details['flops_method'] == 'analytical_estimate_v2'
    assert profile.optimizer_state_bytes == 2 * profile.parameter_bytes


def test_comparability_rejects_missing_invalid_and_mismatched_workloads():
    candidate = profile_model(TissueModel(small_config(), 'wave', seed=1),
                              warmup_steps=0, benchmark_steps=2)
    baseline = replace(candidate, protocol='vector')
    assert not verify_resource_comparability(candidate, {})['comparable']
    assert not verify_resource_comparability(candidate, {'wave': candidate})['comparable']
    for invalid in (replace(candidate, latency_mean_us=float('nan')),
                    replace(candidate, latency_p95_us=float('inf')),
                    replace(candidate, benchmark_samples=0),
                    replace(candidate, flops_per_step=-1)):
        assert not verify_resource_comparability(invalid, {'vector': baseline})['comparable']
    mismatched = replace(baseline, workload={**baseline.workload, 'batch_size': 2})
    assert not verify_resource_comparability(candidate, {'vector': mismatched})['comparable']
    missing_metadata = replace(baseline, workload={})
    assert not verify_resource_comparability(candidate, {'vector': missing_metadata})['comparable']
