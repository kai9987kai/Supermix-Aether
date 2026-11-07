"""Rigorous analytical and benchmarked computational profiler for neural models.

Measures active FLOPs, CPU step latency, activation memory, and optimizer state
to establish verified empirical resource comparability across protocols.
"""

from dataclasses import asdict, dataclass, field
import math
import time
import torch
import numpy as np


@dataclass
class ComputeProfile:
    protocol: str
    parameter_count: int
    parameter_bytes: int
    buffer_bytes: int
    optimizer_state_bytes: int
    flops_per_step: int
    peak_activation_bytes: int
    latency_mean_us: float
    latency_median_us: float
    latency_p95_us: float
    benchmark_samples: int
    workload: dict = field(default_factory=dict)
    activation_memory_method: str = 'analytical_estimate_not_measured_peak'
    flops_method: str = 'analytical_estimate_v2'

    def to_dict(self) -> dict:
        return asdict(self)


def count_step_flops(model, batch_size: int = 1) -> int:
    """Calculate theoretical floating-point operations for one model.step() call."""
    c = model.config
    b = batch_size

    # Unbatched structural overhead (edge weight & degree normalizer derivations)
    unbatched = c.scalar_edges + (c.region_message_edges if model.protocol != 'scalar' else 0)

    # 1. Input encoding: (B, input_dim) -> (B, state_dim)
    flops = 2 * c.input_dimension * c.state_dimension * b
    # Scaling by input profile
    flops += c.neuron_count * c.state_dimension * b

    # 2. Local sparse graph propagation:
    # Sparse COO mm with shape (N, N) [E nonzeros] x (N, B * state_dim)
    flops += 2 * c.scalar_edges * (b * c.state_dimension)

    # 3. Dense local transformation and self-recurrence:
    # local_transform: (state_dim, state_dim) on each neuron
    flops += 2 * c.neuron_count * c.state_dimension * c.state_dimension * b
    # self_gain + bias + tanh
    flops += 4 * c.neuron_count * c.state_dimension * b

    # 4. Auxiliary regional communication (if not scalar-only):
    if model.protocol != 'scalar':
        width = c.mode_count * c.complex_dimensions * 2

        # Dense regional pooling: (R, N) x (N, D) done twice (sender and receiver)
        pooling_flops = 2 * (2 * c.region_count * c.neuron_count * c.state_dimension * b)

        # Message encoder: Linear(state_dim, width) on R regions
        encoder_flops = 2 * c.region_count * c.state_dimension * width * b

        # Message decoder: Linear(width, state_dim) on R regions
        decoder_flops = 2 * c.region_count * width * c.state_dimension * b

        # Residual injection & clipping
        injection_flops = 3 * c.neuron_count * c.state_dimension * b

        # Base auxiliary communication scaled across forward & delivery channel
        aux_base = pooling_flops + encoder_flops + decoder_flops + injection_flops
        flops += 2 * aux_base

        if model.protocol == 'scalar_matched':
            # Bottleneck projection and expand
            flops += 2 * c.region_count * c.state_dimension * b + width * b
        elif model.protocol == 'attention':
            # Routing queries & keys: Linear(state_dim, state_dim)
            flops += 4 * c.region_count * c.state_dimension * c.state_dimension * b
            # Attention dot-product logits & projection: 4 * E_edges x D_state
            flops += 4 * c.region_message_edges * c.state_dimension * b
            # Softmax and value weighting
            flops += 4 * c.region_message_edges * width * b
        elif model.protocol in ('phase', 'wave'):
            # Phase encoder: Linear(state_dim, complex_dimensions)
            flops += 2 * c.region_count * c.state_dimension * c.complex_dimensions * b
            # Complex multiplication / rotation per edge
            flops += 6 * c.region_message_edges * c.complex_dimensions * b

        if model.protocol in ('modes', 'wave'):
            # Spatial carrier projection using eigenvectors: (R, M) x (M, width)
            flops += 2 * c.region_count * c.mode_count * width * b

    # 5. Output readout:
    # Readout pooling runs once, followed by decoder Linear
    readout_count = int(model.readout_neurons.sum().item())
    readout_pool_flops = readout_count * c.action_count * b
    decoder_flops = (2 * c.state_dimension + 1) * c.action_count * b
    flops += readout_pool_flops + decoder_flops

    return int(flops + unbatched)


def profile_model(model, warmup_steps: int = 5, benchmark_steps: int = 25,
                  batch_size: int = 1) -> ComputeProfile:
    """Benchmark CPU inference latency, memory allocations, and compute FLOPs."""
    if isinstance(benchmark_steps, bool) or not isinstance(benchmark_steps, int) or benchmark_steps < 1:
        raise ValueError('benchmark_steps must be a positive integer')
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError('warmup_steps must be a non-negative integer')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')

    was_training = model.training
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    optimizer_state_bytes = 2 * param_bytes

    flops = count_step_flops(model, batch_size=batch_size)

    # Estimate peak activation memory during a step
    # State tensors: recurrent state + intermediate projections
    c = model.config
    state_bytes = batch_size * c.neuron_count * c.state_dimension * 4
    interm_bytes = batch_size * c.region_count * (c.mode_count * c.complex_dimensions * 2) * 2
    peak_activation_bytes = state_bytes * 3 + interm_bytes * 2

    # Match observation dtype with model parameters
    param_dtype = next(model.parameters()).dtype if list(model.parameters()) else torch.float32
    obs = torch.zeros(batch_size, c.input_dimension, dtype=param_dtype)
    state = model.initial_state(batch_size)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            _, state = model.step(obs, state)

    # Benchmark CPU latency
    latencies_us = []
    with torch.no_grad():
        for _ in range(benchmark_steps):
            t0 = time.perf_counter_ns()
            _, state = model.step(obs, state)
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)

    model.train(was_training)

    workload = {
        'batch_size': batch_size,
        'dtype': str(param_dtype),
        'model_config': c.to_dict(),
    }

    return ComputeProfile(
        protocol=model.protocol,
        parameter_count=param_count,
        parameter_bytes=param_bytes,
        buffer_bytes=buffer_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        flops_per_step=flops,
        peak_activation_bytes=peak_activation_bytes,
        latency_mean_us=round(float(np.mean(latencies_us)), 2),
        latency_median_us=round(float(np.median(latencies_us)), 2),
        latency_p95_us=round(float(np.percentile(latencies_us, 95)), 2),
        benchmark_samples=benchmark_steps,
        workload=workload,
        activation_memory_method='analytical_estimate_not_measured_peak',
        flops_method='analytical_estimate_v2',
    )


def verify_resource_comparability(candidate_profile: ComputeProfile,
                                  baseline_profiles: dict[str, ComputeProfile],
                                  max_flop_ratio: float = 2.0,
                                  max_latency_ratio: float = 2.5) -> dict:
    """Certify whether candidate compute is comparable to baseline protocols."""
    reasons = []
    flop_ratios = {}
    latency_ratios = {}
    memory_ratios = {}

    def _is_valid_profile(p: ComputeProfile) -> bool:
        if not isinstance(p, ComputeProfile):
            return False
        if not math.isfinite(p.latency_mean_us) or p.latency_mean_us <= 0:
            return False
        if not math.isfinite(p.latency_p95_us) or p.latency_p95_us <= 0:
            return False
        if p.benchmark_samples <= 0:
            return False
        if p.flops_per_step < 0:
            return False
        if not p.workload or not isinstance(p.workload, dict):
            return False
        if 'batch_size' not in p.workload or 'dtype' not in p.workload or 'model_config' not in p.workload:
            return False
        return True

    if not _is_valid_profile(candidate_profile):
        reasons.append('Candidate profile is invalid or contains nonfinite/missing metrics')

    other_baselines = {name: base for name, base in baseline_profiles.items()
                       if name != candidate_profile.protocol}
    if not other_baselines:
        reasons.append('No baseline profiles provided for comparison')

    for name, base in other_baselines.items():
        if not _is_valid_profile(base):
            reasons.append(f'Baseline profile {name} is invalid or contains nonfinite/missing metrics')
            continue

        if candidate_profile.workload and base.workload:
            if candidate_profile.workload.get('batch_size') != base.workload.get('batch_size'):
                reasons.append(f'Workload batch_size mismatch vs {name}: {candidate_profile.workload.get("batch_size")} != {base.workload.get("batch_size")}')
            if candidate_profile.workload.get('dtype') != base.workload.get('dtype'):
                reasons.append(f'Workload dtype mismatch vs {name}: {candidate_profile.workload.get("dtype")} != {base.workload.get("dtype")}')

        f_ratio = candidate_profile.flops_per_step / max(1, base.flops_per_step)
        flop_ratios[name] = round(f_ratio, 3)
        if f_ratio > max_flop_ratio:
            reasons.append(f'FLOP ratio vs {name} exceeds limit: {f_ratio:.2f}x > {max_flop_ratio}x')

        l_ratio = candidate_profile.latency_mean_us / max(0.1, base.latency_mean_us)
        latency_ratios[name] = round(l_ratio, 3)
        if l_ratio > max_latency_ratio:
            reasons.append(f'Latency ratio vs {name} exceeds limit: {l_ratio:.2f}x > {max_latency_ratio}x')

        m_ratio = (candidate_profile.parameter_bytes + candidate_profile.buffer_bytes) / max(
            1, base.parameter_bytes + base.buffer_bytes)
        memory_ratios[name] = round(m_ratio, 3)

    is_comparable = len(reasons) == 0
    return {
        'comparable': is_comparable,
        'candidate_protocol': candidate_profile.protocol,
        'flop_ratios': flop_ratios,
        'latency_ratios': latency_ratios,
        'memory_ratios': memory_ratios,
        'reasons': reasons,
        'verdict': 'verified_comparable' if is_comparable else 'disproportionate_compute',
    }
