# Provenance

Where the parts of this repository come from, so that nothing is mistaken for something it is not.

## Code and prompts

All code and the system prompt (`answer.v1`) were written for this project, in this repository's
history. No code or prompt text was copied from another project. Design ideas are standard in the
field (reciprocal rank fusion, cross-encoder reranking, hexagonal architecture) and are credited in
the ADRs where they matter. The requirements in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) were written for this project.

## Evaluation data

- `eval/corpus/mini_v1` and `eval/datasets/retrieval_mini_v1.json`: invented documents and 44
  questions drafted with the help of an AI language model from them; **all unreviewed by a person**.
- `eval/corpus/hard_v1` and `eval/datasets/retrieval_hard_v1.json`: 26 invented documents and 141
  questions generated from one table by `eval/scripts/build_hard_dataset.py`; **all unreviewed**.
- `eval/datasets/embedding_probe_v1.json`: a small probe set used in the embedding bake-off.
- The NoMIRACL judged-pool bake-off used a public benchmark that is downloaded on demand and not
  stored in the repository (`data/` is ignored); its licence is the benchmark's own.

A record's `provenance` and `review` fields say who made it and whether a person accepted it.
Metrics count only accepted records; everything computed on the data above is exploratory.

## Models

No model weights are in the repository or the Docker image. The code can use, when provisioned by
the user: `intfloat/multilingual-e5-small` (MIT), `BAAI/bge-reranker-v2-m3` quantised
(Apache-2.0), and models served by Ollama or an OpenAI-compatible server, each under its own licence.

## Dependencies

Listed with their licences in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[sbom.cdx.json](sbom.cdx.json), both generated from `uv.lock`
(`uv run python scripts/generate_notices.py`).

## Licence of this project

Not chosen. Until the owner chooses one, no licence is granted.
