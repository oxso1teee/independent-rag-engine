"""Pure rules: the mass-delete guard table and the retry delay with jitter."""

from __future__ import annotations

import pytest

from rag_engine.domain import ErrorCode, RagError
from rag_engine.ingestion.jobs import RetryPolicy
from rag_engine.ingestion.lifecycle import check_mass_delete, mass_delete_rule


@pytest.mark.parametrize(
    ("vanished", "tracked", "seen", "rule"),
    [
        (0, 50, 50, None),  # nothing vanished
        (1, 50, 49, None),
        (10, 50, 40, None),  # 10 files and 20 %: both limits not exceeded
        (11, 50, 39, "threshold"),
        (10, 40, 30, None),  # max(10, 8) = 10
        (11, 40, 29, "threshold"),
        (20, 100, 80, None),  # 20 %
        (21, 100, 79, "threshold"),
        (3, 3, 5, "all_known_gone"),  # everything known vanished, new files exist
        (1, 1, 0, "empty_scan"),  # a single document and an empty scan
        (5, 12, 0, "empty_scan"),  # empty scan wins over the other rules
        (7, 7, 0, "empty_scan"),
    ],
)
def test_rule_table(vanished: int, tracked: int, seen: int, rule: str | None) -> None:
    assert mass_delete_rule(vanished=vanished, tracked=tracked, entries_seen=seen) == rule
    if rule is None:
        check_mass_delete(vanished=vanished, tracked=tracked, entries_seen=seen)
    else:
        with pytest.raises(RagError) as info:
            check_mass_delete(vanished=vanished, tracked=tracked, entries_seen=seen)
        assert info.value.code is ErrorCode.E_MASS_DELETE_GUARD
        assert info.value.context["rule"] == rule
        # explicit confirmation always lets it through
        check_mass_delete(vanished=vanished, tracked=tracked, entries_seen=seen, accept=True)


def test_backoff_is_exponential_capped_and_jitter_only_shortens() -> None:
    policy = RetryPolicy(base_delay_s=30, factor=2, max_delay_s=100, jitter=0.2)
    no_jitter = [policy.delay_for(n).total_seconds() for n in (1, 2, 3, 4)]
    assert no_jitter == [30, 60, 100, 100]
    worst = [policy.delay_for(n, lambda: 0.999999).total_seconds() for n in (1, 2, 3, 4)]
    assert worst == pytest.approx([24, 48, 80, 80], rel=1e-3)  # shortened by at most 20 %
    middle = policy.delay_for(2, lambda: 0.5).total_seconds()
    assert middle == pytest.approx(54)
    assert all(w <= n for w, n in zip(worst, no_jitter, strict=True))


def test_jitter_zero_disables_it_and_invalid_policies_are_rejected() -> None:
    assert RetryPolicy(jitter=0).delay_for(1, lambda: 0.9).total_seconds() == 30
    for bad in ({"jitter": -0.1}, {"jitter": 1.5}, {"max_attempts": 0}, {"lease_s": 0}):
        with pytest.raises(RagError):
            RetryPolicy(**bad)
