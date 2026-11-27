"""Seeded percentile bootstrap over queries.

Resampling queries, not documents, respects that one query's score is the unit of observation.
A fixed ``seed`` reproduces the interval exactly; the paired form resamples the same queries for
both systems, so a difference is judged against the variation between queries, not between runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Interval:
    mean: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def _interval(values: np.ndarray, seed: int, resamples: int, alpha: float) -> Interval:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if resamples < 1 or not 0 < alpha < 1:
        raise ValueError("resamples must be positive and alpha between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(values.mean()), float(low), float(high))


def bootstrap_mean(
    values: Sequence[float], *, seed: int, resamples: int = 10_000, alpha: float = 0.05
) -> Interval:
    return _interval(np.asarray(values, dtype=np.float64), seed, resamples, alpha)


def paired_bootstrap(
    first: Sequence[float],
    second: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
    alpha: float = 0.05,
) -> Interval:
    """Interval of ``mean(first - second)`` over the same queries."""
    if len(first) != len(second):
        raise ValueError("both systems need scores for the same queries")
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    return _interval(delta, seed, resamples, alpha)
