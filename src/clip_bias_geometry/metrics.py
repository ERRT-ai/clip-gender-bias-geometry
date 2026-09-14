"""Bias metrics and bootstrap utilities."""

from __future__ import annotations

import numpy as np


def signed_gender_gap(male_scores: np.ndarray, female_scores: np.ndarray) -> float:
    """Mean male similarity minus mean female similarity."""
    return float(np.mean(male_scores) - np.mean(female_scores))


def absolute_gender_gap(male_scores: np.ndarray, female_scores: np.ndarray) -> float:
    """Absolute value of the signed group similarity gap."""
    return abs(signed_gender_gap(male_scores, female_scores))


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 5329,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return mean and percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    low = 100 * (alpha / 2)
    high = 100 * (1 - alpha / 2)
    return (
        float(values.mean()),
        float(np.percentile(boot, low)),
        float(np.percentile(boot, high)),
    )
