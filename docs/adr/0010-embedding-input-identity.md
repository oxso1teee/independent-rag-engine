# 0010. Embedding input and model identity

Status: Accepted (2026-11-25).

## Context

`Chunk.text_hash` identifies the normalized body. The embedding input includes its heading path.
Two chunks with the same body under different headings can therefore produce different vectors.
A cache keyed only by `(text_hash, model_id)` would return a vector for the wrong input.

## Decision

1. Keep `text_hash = SHA-256(UTF-8(chunk.text))` for body deduplication and integrity checks.
2. Construct document embedding input in one function: `"\n".join(heading_path) + "\n" + text`
   when headings exist, otherwise `text`. Apply the model's document prefix in the same function.
   Compute `embedding_input_hash` as SHA-256 of the exact UTF-8 bytes submitted to the provider.
   Store this hash on each indexed chunk and key cached vectors by
   `(embedding_input_hash, model_id)`. Query and document prefixes are part of `EmbeddingModelId`.
3. A model ID includes provider, name, immutable revision or digest, dimension, normalization
   policy and prefix hash. A change in any component creates a different ID. If a provider cannot
   supply a stable revision, configuration must declare one; it cannot silently reuse an ID.
4. The service rejects zero, non-finite, wrong-shape and wrong-dimension vectors, and input that
   exceeds the provider token limit. It normalizes valid vectors to L2 norm 1 before fp16 storage,
   then checks the decoded stored norm within `1e-3`. Failed or partial batches never publish a
   document version.
5. Batches are bounded by both item count and aggregate input tokens. The `low` profile permits
   one embedding call at a time and at most 16 items per batch. Timeouts and retryable failures
   use bounded retries with jitter; permanent failures do not retry.

## Consequences

The cache safely reuses vectors across versions only for identical model inputs. Changing heading
context may require a new vector even when the body stays unchanged. The NFR-05 test must use a
document whose unchanged chunks also retain their heading context. A model change is staged and
activated only after every visible indexed chunk has a compatible vector.
