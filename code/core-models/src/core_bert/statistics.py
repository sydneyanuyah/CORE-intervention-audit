"""Graph-level paired inference required by EXPERIMENT_REGISTRY_v2."""

from __future__ import annotations

import math
import random
from statistics import mean, stdev


def paired_intervals(a: list[float], b: list[float], seed: int = 20260904, draws: int = 10000) -> dict[str, object]:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("paired graph arrays must have equal length >= 2")
    differences = [right - left for left, right in zip(a, b)]
    rng = random.Random(seed); n = len(differences)
    bootstrap = sorted(mean(rng.choices(differences, k=n)) for _ in range(draws))
    # Student-t critical values for common preregistered graph counts; conservative 1.96 fallback.
    t95 = {4: 2.776, 5: 2.571, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 20: 2.086, 24: 2.064}.get(n - 1, 1.96)
    center = mean(differences); margin = t95 * stdev(differences) / math.sqrt(n)
    return {"n_graphs": n, "mean_difference": center, "bootstrap_95": [bootstrap[int(.025 * draws)], bootstrap[int(.975 * draws) - 1]], "student_t_95": [center - margin, center + margin]}


def mcnemar_counts(baseline_correct: list[bool], method_correct: list[bool]) -> dict[str, int]:
    if len(baseline_correct) != len(method_correct): raise ValueError("item arrays must be paired")
    return {"baseline_only": sum(a and not b for a, b in zip(baseline_correct, method_correct)), "method_only": sum(b and not a for a, b in zip(baseline_correct, method_correct))}
