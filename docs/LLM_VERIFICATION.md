# Verifying a real language model

Nothing in the test suite or in the published metrics comes from a real chat model: the pipeline
has been exercised with scripted and stub models only. This page says what is needed to measure a
real one and how, so the run is reproducible on another machine.

## State on the reference machine (2026-11-28)

8 GiB RAM, 4 cores, no GPU; about 3.2 GiB available and no free swap with the usual desktop load.
`qwen3.5:4b` (3.1 GiB on disk) is installed in the local Ollama. The memory gate of
`eval/scripts/llm_smoke.py` needs the model size plus 1 GiB headroom and 0.5 GiB free swap, and
refused the run (`need 4.1 GiB, available 3.23 GiB, swap free 0`). The gate is deliberately not
weakened, and no process was stopped to make room. The 9B model is not run on this machine.

## Requirements

| item | requirement |
|---|---|
| memory | available RAM ≥ model size + 1 GiB, free swap ≥ 0.5 GiB (gate); in practice 8 GiB machine with nothing else heavy running, or 16 GiB |
| model | `qwen3.5:4b` through Ollama (`ollama pull qwen3.5:4b`), or any OpenAI-compatible endpoint served by the owner |
| embeddings | the E5 ONNX model directory used by the other evaluations |
| time | about 50 questions × 20–60 s on CPU: under an hour |

## Steps

```bash
# 1. the gate only: loads nothing, exits with a refusal report if memory is short
uv run python eval/scripts/llm_smoke.py --model qwen3.5:4b --context-window 4096

# 2. the opt-in integration tests against the real model
RAG_TEST_OLLAMA_CHAT_MODEL=qwen3.5:4b uv run pytest tests/external -m external

# 3. the measured end-to-end run over the mini dataset
uv run python eval/scripts/generation_eval.py --embed-dir <e5 dir> --llm ollama \
    --model qwen3.5:4b --include-unreviewed --output eval_runs/generation_qwen35_4b.json
```

The report's `quality` block holds answer correctness and completeness (match against the reference
facts), faithfulness, hallucination rate and citation precision/recall (claims checked by the
lexical judge), the answered/refused matrix, latency p50/p95/max and token counts. Record peak RSS
of the Ollama process separately (`/usr/bin/time -v` or `ps`), since it runs outside Python.

## Reading the result

- The lexical judge is a screen, not an entailment check, and no judge is calibrated against
  people. Faithfulness and hallucination figures are therefore indicative until a judge has been
  compared with a human sample (`calibrate_judge`: agreement ≥ 90 %, κ ≥ 0.8). The judge model
  should not be the model that wrote the answers.
- Labels are unreviewed (`DATASET_REVIEW.md`); report results as exploratory until they are.
- A real run is a separate result from the stub run; the stub only tests the wiring.

## End-to-end through a hosted OpenAI-compatible provider

`eval/scripts/llm_e2e.py` runs the whole pipeline against a real chat provider (Groq or any
OpenAI-compatible endpoint) and checks ingestion, retrieval, generation with citation validation,
refusal on insufficient evidence, streaming, cancellation, and three provider failures (a wrong key,
an unknown model, an unreachable server). It was tested against a scripted fake endpoint
(`tests/e2e/test_llm_e2e_script.py`); it **has not been run against a real provider**, because no key
was configured for this project on the reference machine (2026-11-28) and none was used.

```bash
export GROQ_API_KEY=...        # supplied by the owner; read from the environment only
uv run python eval/scripts/llm_e2e.py --approved \
    --embed-dir data/models/multilingual-e5-small/<revision> \
    --base-url https://api.groq.com/openai/v1 --model <model> --api-key-env GROQ_API_KEY \
    --output eval_runs/llm_e2e_groq.json
```

Rules the script enforces: it refuses to start without `--approved` (the owner agreed to use this
provider and its cost), refuses any corpus outside `eval/corpus` (public synthetic text only, nothing
private is ever sent), never prints or stores the key and asserts that the key is in none of its
output, and sends about 30 requests of at most 256 new tokens (the report gives the token totals).

**What a hosted provider does not show.** It checks the OpenAI-compatible adapter and the pipeline.
It says nothing about a local Ollama server (the native `/api/chat` adapter, `keep_alive`, the
memory behaviour of a resident model, the resource gate): the 4B model on this machine is still
unmeasured and needs the steps above. Hosted-model answers also say nothing about how a 4B model
will cite or refuse.

## If this machine is too small

1. Run the same three commands on another machine with the repository and the model files copied.
2. Or use an OpenAI-compatible endpoint the owner already pays for or hosts: the same script with
   `--llm openai_compat --base-url <url> --api-key-env <VARIABLE> --model <name>` (the key stays in
   the environment; the resource gate does not apply because nothing is loaded locally). The
   project does not sign up for or call paid services by itself; this needs the owner's agreement
   and an API key supplied through the environment.
