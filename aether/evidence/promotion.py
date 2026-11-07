"""Fail-closed research promotion gates; these are not real-world guarantees."""

import math
import numbers
from collections.abc import Mapping

from .statistics import paired_summary


def promotion_decision(comparisons, *, retention_candidate, retention_baseline,
                       retention_margin, replay_valid, integrity_valid,
                       resources_comparable, min_confirmatory_seeds=10, seed=0):
    """Evaluate every predeclared control and preserve explicit failure reasons.

    comparisons maps control names to (candidate, control) seed outcomes. All
    comparisons and retention outcomes must use the same ordered training seeds.
    The caller establishes this identity and supplies independently verified
    replay/integrity and measured-resource comparability as literal booleans.
    At least ten independent training seeds are required; a study may require
    more (e.g. min_confirmatory_seeds=30). Retention's lower paired 95% bound
    must be >= -retention_margin, and every control's lower bound must be > 0.
    """
    if (isinstance(min_confirmatory_seeds, bool)
            or not isinstance(min_confirmatory_seeds, numbers.Integral)
            or min_confirmatory_seeds < 10):
        raise ValueError("min_confirmatory_seeds must be an integer >= 10")
    if (isinstance(retention_margin, bool)
            or not isinstance(retention_margin, numbers.Real)
            or not math.isfinite(retention_margin) or retention_margin < 0):
        raise ValueError("retention_margin must be finite and nonnegative")
    if not isinstance(comparisons, Mapping):
        raise ValueError("comparisons must map predeclared control names to paired outcomes")
    retention = paired_summary(retention_candidate, retention_baseline, seed)
    reasons = []
    summaries = {}
    if not comparisons:
        reasons.append("no predeclared controls supplied")
    for name, outcomes in comparisons.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("control names must be nonempty strings")
        if not isinstance(outcomes, (list, tuple)) or len(outcomes) != 2:
            raise ValueError(f"{name}: expected candidate/control paired sequences")
        summary = paired_summary(outcomes[0], outcomes[1], seed)
        if summary["n"] != retention["n"]:
            raise ValueError("all comparisons and retention require equal ordered seed counts")
        summaries[name] = summary
        if summary["ci95"][0] <= 0:
            reasons.append(f"{name}: lower paired confidence bound is not positive")
    insufficient = retention["n"] < min_confirmatory_seeds
    if insufficient:
        reasons.append(f"insufficient independent training seeds: {retention['n']} < {min_confirmatory_seeds}")
    if retention["ci95"][0] < -float(retention_margin):
        reasons.append("retention: lower paired confidence bound exceeds allowed degradation")
    for name, value in (("replay_valid", replay_valid), ("integrity_valid", integrity_valid),
                        ("resources_comparable", resources_comparable)):
        if value is not True:
            reasons.append(f"{name}: not established as true")
    promoted = not reasons
    return {"promoted": promoted,
            "status": "promoted" if promoted else "inconclusive" if insufficient else "rejected",
            "reasons": reasons, "comparisons": summaries, "retention": retention,
            "retention_margin": float(retention_margin),
            "min_confirmatory_seeds": int(min_confirmatory_seeds),
            "inference_unit": "independent_training_seed"}
