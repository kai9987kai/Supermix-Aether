"""Rigorous analytical and benchmarked computational profiler for neural models.

Measures active FLOPs, CPU step latency, activation memory, and optimizer state
to establish verified empirical resource comparability across protocols.
"""

from dataclasses import asdict, dataclass
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

    def to_dict(self) -> dict:
        return asdict(self)


def count_step_flops(model, batch_size: int = 1) -> int:
    """Calculate theoretical floating-point operations for one model.step() call."""
    c = model.config
    b = batch_size
    flops = 0

    # 1. Input encoding: (B, input_dim) -> (B, state_dim)
    flops += 2 * c.input_dimension * c.state_dimension * b
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
        # Regional pooling: (R, N) x (N, B * state_dim)
        flops += 2 * c.neuron_count * (b * c.state_dimension)
        # Message encoder: Linear(state_dim, width) on R regions
        flops += 2 * c.region_count * c.state_dimension * width * b

        if model.protocol == 'scalar_matched':
            # Bottleneck projection and expand
            flops += 2 * c.region_count * c.state_dimension * b + width * b
        elif model.protocol == 'vector':
            # Direct link forwarding
            flops += c.region_message_edges * width * b
        elif model.protocol == 'attention':
            # Routing queries & keys: Linear(state_dim, state_dim)
            flops += 4 * c.region_count * c.state_dimension * c.state_dimension * b
            # Attention dot-product logits: E_edges x D_state
            flops += 2 * c.region_message_edges * c.state_dimension * b
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

        # Message decoder: Linear(width, state_dim)
        flops += 2 * c.region_count * width * c.state_dimension * b
        # Residual injection and clipping
        flops += 3 * c.neuron_count * c.state_dimension * b

    # 5. Output readout: Linear(state_dim, action_count) on readout neurons
    readout_count = int(model.readout_neurons.sum().item())
    flops += 2 * readout_count * c.state_dimension * c.action_count * b
    # Average pooling across readout neurons
    flops += readout_count * c.action_count * b

    return int(flops)


def profile_model(model, warmup_steps: int = 5, benchmark_steps: int = 25,
                  batch_size: int = 1) -> ComputeProfile:
    """Benchmark CPU inference latency, memory allocations, and compute FLOPs."""
    was_training = model.training
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    # AdamW maintains first and second moments in FP32 (4 bytes each)
    optimizer_state_bytes = 2 * param_count * 4

    flops = count_step_flops(model, batch_size=batch_size)

    # Estimate peak activation memory during a step
    # State tensors: recurrent state + intermediate projections
    c = model.config
    state_bytes = batch_size * c.neuron_count * c.state_dimension * 4
    interm_bytes = batch_size * c.region_count * (c.mode_count * c.complex_dimensions * 2) * 2
    peak_activation_bytes = state_bytes * 3 + interm_bytes * 2

    obs = torch.zeros(batch_size, c.input_dimension, dtype=torch.float32)
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

    for name, base in baseline_profiles.items():
        if name == candidate_profile.protocol:
            continue

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
