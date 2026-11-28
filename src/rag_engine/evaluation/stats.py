"""Small, exact statistics for comparing runs and checking a judge.

Everything here is a pure function with a known answer: McNemar's exact test and the sign test for
paired outcomes, a bootstrap that resamples clusters (questions that share a gold document are not
independent), Holm's correction for several comparisons, Cohen's kappa for agreement, and the two
warnings the evaluation design asks for (a saturated set cannot show an improvement; fewer than 30
questions is anecdotal).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np

from rag_engine.evaluation.bootstrap import Interval

MIN_QUESTIONS = 30
SATURATED_MEAN = 0.98


def _two_sided_binomial(successes: int, trials: int) -> float:
    """Exact two-sided p for ``successes`` out of ``trials`` fair coin flips."""
    if trials == 0:
        return 1.0
    tail = min(successes, trials - successes)
    lower = sum(comb(trials, i) for i in range(tail + 1))
    return float(min(1.0, 2.0 * lower / 2**trials))


def mcnemar_exact(first_only: int, second_only: int) -> float:
    """Exact McNemar p-value from the two discordant counts."""
    if first_only < 0 or second_only < 0:
        raise ValueError("counts cannot be negative")
    return _two_sided_binomial(first_only, first_only + second_only)


@dataclass(frozen=True, slots=True)
class PairedBinary:
    first_only: int
    second_only: int
    p_value: float

    @property
    def first_better(self) -> bool:
        return self.first_only > self.second_only


def paired_binary(first: Sequence[bool], second: Sequence[bool]) -> PairedBinary:
    if len(first) != len(second):
        raise ValueError("both systems need an outcome for the same questions")
    first_only = sum(a and not b for a, b in zip(first, second, strict=True))
    second_only = sum(b and not a for a, b in zip(first, second, strict=True))
    return PairedBinary(first_only, second_only, mcnemar_exact(first_only, second_only))


def sign_test(differences: Sequence[float]) -> float:
    """Exact two-sided sign test; zero differences are ties and are ignored."""
    positive = sum(d > 0 for d in differences)
    negative = sum(d < 0 for d in differences)
    return _two_sided_binomial(positive, positive + negative)


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    resamples: int = 2000,
    alpha: float = 0.05,
) -> Interval:
    """Percentile interval of the mean, resampling whole clusters (for example gold documents)."""
    if len(values) != len(clusters):
        raise ValueError("every value needs a cluster")
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    members: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters, strict=True):
        members.setdefault(cluster, []).append(value)
    sums = np.array([sum(v) for v in members.values()], dtype=np.float64)
    sizes = np.array([len(v) for v in members.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(members), size=(resamples, len(members)))
    means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(np.mean(values)), float(low), float(high))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def percent_agreement(first: Sequence[str], second: Sequence[str]) -> float:
    _check_raters(first, second)
    return sum(a == b for a, b in zip(first, second, strict=True)) / len(first)


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float:
    """Chance-corrected agreement of two raters; 1.0 when both always give the same label."""
    _check_raters(first, second)
    n = len(first)
    observed = percent_agreement(first, second)
    one, two = Counter(first), Counter(second)
    expected = sum(one[label] * two[label] for label in one.keys() | two.keys()) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _check_raters(first: Sequence[str], second: Sequence[str]) -> None:
    if len(first) != len(second):
        raise ValueError("both raters need a label for the same items")
    if not first:
        raise ValueError("agreement needs at least one item")


def saturation(mean: float, upper: float) -> bool:
    """A metric this close to its ceiling cannot show an improvement."""
    return mean >= SATURATED_MEAN and upper >= 1.0


def sample_label(n: int) -> str:
    return f"anecdotal (N < {MIN_QUESTIONS})" if n < MIN_QUESTIONS else f"N = {n}"
