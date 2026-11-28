# 0020. The sufficiency gate trusts the strongest signal alone

Status: Accepted (2026-11-28). Provisional: labels are unreviewed. The quality target is **not met**.

## Context

The gate refused only when both the dense cosine and the reranker logit were below their
thresholds ("either one is enough"). A refusal therefore needed agreement of the weaker signal too,
and the gate could never refuse more unanswerable questions than the weaker of the two.

An independent set (`gate-heldout@v2`: 24 documents, 136 questions, 40 of them unanswerable, built
after the earlier sets had been seen) was written for this decision. Before it was opened, the
rule was chosen on the calibration data only (hard dev and mini dev, 96 questions, 21
unanswerable), by five-fold cross-validation over single and combined signals. The held-out set was
then read once through the counted guard.

## Evidence

| Rule (fixed before the held-out read) | Correct refusals | False refusals |
|---|---|---|
| reranker logit alone, calibration CV | 95.2 % (20/21) | 6.7 % |
| reranker logit alone, **held-out** | 72.5 % (29/40) | 4.2 % (4/96) |
| best dense cosine, same recipe, held-out | 32.5 % (13/40) | 6.3 % (6/96) |

The fitted combinations with other features did not beat the reranker logit alone on calibration
and overfit with seven features.

## Decision

The gate uses the first signal that has both a score and a threshold, in this order: reranker
logit, dense cosine, lexical coverage. A weaker signal no longer rescues a poor stronger one.
Thresholds, the calibration record and the "never block what cannot be judged" rule are unchanged.

## Consequences

- The held-out correct-refusal rate (72.5 %) is below the target of 90 %. The criterion stays open;
  it is not weakened. The set is spent for tuning: another read of it would no longer be held-out.
- The reranker is off by default and needs about 2.4 GiB at peak, so most installations use the
  dense signal, which refuses about a third of the unanswerable questions in this set. Dense alone
  is not a reliable gate for near-miss questions.
- No command writes a calibration for a user's index (see `docs/AUDIT.md`, F-1), so the gate is
  open until one is added.
- Next steps that could change the picture: reviewed labels, a larger and untouched held-out set,
  and a calibration made on the user's own corpus.
