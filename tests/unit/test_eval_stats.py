"""Statistics with known answers: exact tests, clustering, multiple comparisons, agreement."""

from __future__ import annotations

import math

import pytest

from rag_engine.evaluation.stats import (
    cluster_bootstrap,
    cohen_kappa,
    holm_adjust,
    mcnemar_exact,
    paired_binary,
    percent_agreement,
    sample_label,
    saturation,
    sign_test,
)

pytestmark = pytest.mark.unit


def test_mcnemar_exact_matches_hand_computed_binomial_tails() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0  # a perfectly balanced disagreement
    # b=0, c=6: two-sided exact p = 2 * (1/2)^6 = 0.03125
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
    # b=1, c=7: n=8, p = 2 * (C(8,0)+C(8,1)) / 2^8 = 2 * 9 / 256
    assert mcnemar_exact(1, 7) == pytest.approx(18 / 256)
    assert mcnemar_exact(7, 1) == mcnemar_exact(1, 7)


def test_paired_binary_counts_only_the_discordant_pairs() -> None:
    first = [True, True, False, False, True, True, True, True]
    second = [True, False, False, True, False, False, False, False]
    result = paired_binary(first, second)
    assert (result.first_only, result.second_only) == (5, 1)
    assert result.p_value == pytest.approx(mcnemar_exact(5, 1))
    assert result.first_better and not paired_binary(second, first).first_better
    with pytest.raises(ValueError, match="same"):
        paired_binary([True], [True, False])


def test_sign_test_ignores_ties_and_is_two_sided() -> None:
    assert sign_test([0.0, 0.0]) == 1.0
    assert sign_test([0.2, 0.1, 0.3, 0.0, 0.5, 0.4, 0.2]) == pytest.approx(0.03125)
    assert sign_test([0.2, -0.2]) == 1.0


def test_cluster_bootstrap_is_wider_than_the_naive_interval_when_clusters_correlate() -> None:
    values = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    clusters = ["a", "a", "a", "a", "b", "b", "b", "b"]
    clustered = cluster_bootstrap(values, clusters, seed=1, resamples=2000)
    independent = cluster_bootstrap(values, list("abcdefgh"), seed=1, resamples=2000)
    assert clustered.mean == independent.mean == 0.5
    assert clustered.high - clustered.low > independent.high - independent.low
    assert cluster_bootstrap(values, clusters, seed=1, resamples=2000) == clustered  # seeded
    with pytest.raises(ValueError, match="cluster"):
        cluster_bootstrap([1.0], ["a", "b"], seed=1)


def test_holm_adjustment_is_monotone_and_capped() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.6, 0.9]) == [1.0, 1.0]
    assert holm_adjust([]) == []


def test_kappa_is_one_for_identical_raters_and_zero_for_chance_agreement() -> None:
    labels = ["yes", "no", "yes", "no", "yes", "yes"]
    assert cohen_kappa(labels, labels) == 1.0
    assert percent_agreement(labels, labels) == 1.0
    # observed 0.5, expected 0.5 -> 0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    # hand computed: po = 8/10, pe = 0.5*0.6 + 0.5*0.4 = 0.5 -> kappa = 0.6
    one = ["y"] * 5 + ["n"] * 5
    two = ["y"] * 4 + ["n"] + ["y"] + ["n"] * 4
    assert percent_agreement(one, two) == 0.8
    assert cohen_kappa(one, two) == pytest.approx(0.6)


def test_kappa_handles_a_single_label_and_bad_input() -> None:
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    assert not math.isnan(cohen_kappa(["a", "a"], ["a", "b"]))
    with pytest.raises(ValueError, match="at least one"):
        cohen_kappa([], [])
    with pytest.raises(ValueError, match="same items"):
        cohen_kappa(["a"], ["a", "b"])


def test_saturation_needs_a_high_mean_and_an_interval_reaching_one() -> None:
    assert saturation(1.0, 1.0)
    assert saturation(0.98, 1.0)
    assert not saturation(0.97, 1.0)
    assert not saturation(0.99, 0.995)


def test_small_samples_are_labelled_anecdotal() -> None:
    assert sample_label(29) == "anecdotal (N < 30)"
    assert sample_label(30) == "N = 30"
