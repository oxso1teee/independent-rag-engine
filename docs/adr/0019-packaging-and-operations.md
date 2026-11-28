# 0019. Packaging and operations: small, pinned, unprivileged, verifiable

Status: Accepted (2026-11-28).

## Context

The engine is a local service that holds a user's documents. It has to be installable
reproducibly, run without privileges, be backed up and restored with confidence, and say what it is
made of, without a hosted component, a daemon manager or telemetry.

## Decisions

1. **One image, small by omission.** The Docker image holds the API and no embedding runtime (about
   80 MB); the ONNX runtime is a build argument (about 250 MB more). The base image is pinned by
   digest, dependencies are installed from the hashes in `uv.lock`, the process runs as an
   unprivileged user, and the image refuses to start without a token because it listens on `0.0.0.0`
   inside the container. A CI job builds it and checks those properties.
2. **The core stays installable without the API.** FastAPI and uvicorn are an extra (and part of the
   dev group so CI tests them); the container imports them lazily, and `rag serve` without the extra
   says so instead of crashing. A packaging smoke test installs the wheel into an empty environment.
3. **A backup is only as good as its restore.** `rag backup` verifies the copy; `rag verify` checks a
   backup without opening it for writing; `rag restore` verifies first, refuses to replace without
   `--replace`, and keeps the displaced database aside. The drill (backup, verify, restore, doctor)
   is a test, and the runbook says to do it before it is needed.
4. **Observability is local and optional.** Metrics are Prometheus text, off by default, behind the
   token, labelled by route template only. Logs rotate by size into a file readable by its owner.
   Nothing is exported anywhere.
5. **Say what is inside.** Third-party notices and a CycloneDX SBOM are generated from the lock file
   and checked in CI; licences are read from package metadata and marked unknown rather than guessed.
6. **Soak what can be soaked.** The one-hour soak runs real searches against a real server with the
   real embedding model, behind a memory gate. It does not cover answers: no language model has run
   on the reference machine.
7. **A release is a decision, not a build.** No tag, release, licence or publication is made here;
   [RELEASE.md](../RELEASE.md) lists what is ready, what is open and what the owner decides.

## Consequences

An operator can run, back up, restore and audit the service with documented commands, and a change
that breaks the image, the package, the notices or the restore drill fails CI. The quality criteria
that are still open (gate, real-model measurement, faithfulness, reviewed labels) are not closed by
any of this, and RELEASE.md says so.
