"""Paired percentile bootstrap over independent training seeds, never ticks."""

import numbers

import numpy as np


def _samples(values, name):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional numeric sequence") from exc
    if array.ndim != 1 or not array.size or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty one-dimensional real numeric sequence")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite seed outcomes")
    return array


def paired_summary(candidate, baseline, seed=0):
    """Return n, mean paired difference, and a 95% percentile bootstrap CI.

    Each input entry must be one aggregate outcome from the same independent
    training seed in both arms. The caller owns seed identity/order: raw arrays
    cannot establish independence. Resampling uses 20,000 paired seed draws.
    Small-n intervals are descriptive; this function does not authorize claims.
    """
    candidate = _samples(candidate, "candidate")
    baseline = _samples(baseline, "baseline")
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline must have equal seed counts")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with np.errstate(over="ignore", invalid="ignore"):
        differences = candidate - baseline
    if not np.isfinite(differences).all():
        raise ValueError("paired differences overflowed")
    n = int(differences.size)
    # Scale first so finite, very large outcomes cannot overflow a mean/sum.
    scale = max(float(np.abs(differences).max()), 1.0)
    normalized = differences / scale
    rng = np.random.default_rng(int(seed))
    draws = np.empty(20_000, dtype=np.float64)
    chunk_size = max(1, min(1000, 1_000_000 // n))
    for start in range(0, len(draws), chunk_size):
        count = min(chunk_size, len(draws) - start)
        indices = rng.integers(0, n, size=(count, n))
        draws[start:start + count] = normalized[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975]) * scale
    return {"n": n, "mean": float(normalized.mean() * scale),
            "ci95": [float(lower), float(upper)]}
