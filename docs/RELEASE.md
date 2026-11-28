# Release checklist

No release has been cut. This page lists what is ready, what a release needs, and the decisions
that belong to the owner. Nothing here tags, publishes or licenses anything.

## Ready

| Item | Evidence |
|---|---|
| Quality gates | ruff, format, mypy (strict), import contracts, tests on Python 3.12 and 3.13 in CI |
| Reproducible install | `uv.lock` checked in CI; the Docker image installs from hashes |
| Docker image | `Dockerfile`, `scripts/docker_smoke.sh`, a CI job; about 80 MB |
| Package | `scripts/package_smoke.sh` in CI |
| Backup and restore | `rag backup`, `rag verify`, `rag restore`; the drill is a test; runbook in [OPERATIONS.md](OPERATIONS.md) |
| Dependency audit | `pip-audit` and a licence check in CI; no forbidden licences ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)) |
| SBOM | `sbom.cdx.json` (CycloneDX 1.5), checked against the lock file in CI |
| Secret scan | gitleaks in CI |
| API contract | `docs/openapi.json`, compared in the tests |
| Documentation | README, architecture, API, CLI, operations, evaluation, testing, ADRs, security policy, provenance |

## Open before a release can claim quality

These are not packaging problems; they are the project's open criteria and none is closed:

1. **Sufficiency gate:** 72.5 % correct refusals at 4.2 % false refusals on an independent held-out,
   unreviewed set with the reranker, 32.5 % with dense alone, against targets of 90 % and 10 %
   ([EVALUATION.md](EVALUATION.md), [AUDIT.md](AUDIT.md)). A user can now calibrate the gate on their
   own index (`rag calibrate`, [CALIBRATION.md](CALIBRATION.md)); until they do it is open and says so.
2. **No real language model has run through the pipeline**, so answer latency, faithfulness and
   hallucination rate are unmeasured; the memory gate refused the 4B model on the reference machine
   ([LLM_VERIFICATION.md](LLM_VERIFICATION.md) says how to run it elsewhere).
3. **No evaluation label has been reviewed by a person**; the criterion of 100 reviewed questions is
   unmet and no judge is calibrated.
4. **The one-hour soak** covers search only; see [TESTING.md](TESTING.md) for its result. The harder
   variant (`--workers`, `--churn-every`) exists but has not been run: it needs 2.5 GiB of free
   memory and the reference machine had 1.9–2.4 GiB while the desktop was in use.
5. **The database is about 5.6× the source text without vectors** against a target of about 3×
   ([AUDIT.md](AUDIT.md), F-2). The owner chose to ship first and migrate afterwards
   ([ADR 0022](adr/0022-slimmer-storage.md)); the target stays as written and the release notes must
   say the database is larger than it.

## Decisions for the owner

- **Licence.** MIT (`LICENSE`, `pyproject.toml` and the README agree). Dependency licences are
  listed in `THIRD_PARTY_NOTICES.md`.
- **Version and tag.** The package says `0.0.1`. Whether the first release is `v0.1.0` and whether
  it is called a pre-release is the owner's call.
- **Publication.** Making the repository public, publishing the image or the package, and the
  repository's description and topics.
- **Review of the datasets** (`docs/DATASET_REVIEW.md`), which is what would make the evaluation
  numbers more than exploratory.

### Version: a recommendation, not a choice

- **Licence.** Decided by the owner (MIT). The model weights have their own licences and are not
  part of the repository.
- **Version.** Because the quality criteria are open (gate, real model, reviewed labels), `0.1.0`
  described as a pre-release fits better than `1.0.0`.

## Steps once those are decided

1. Merge everything; `main` green.
2. Set the version in `src/rag_engine/__init__.py`, move `CHANGELOG.md`'s Unreleased section under
   the version and date.
3. Regenerate the notices (`scripts/generate_notices.py`); the `LICENSE` file is already in place.
4. Run the Docker, package and soak checks on the exact commit.
5. Tag, create the GitHub release with the wheel, the SBOM and the notices attached.
6. Publish the image or the package only if the owner chose to.
