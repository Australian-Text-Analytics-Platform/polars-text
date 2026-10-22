# Changelog

## 0.7.0

The release line for Wordflow 0.7.x (polars-based); 0.8 is the Rust-core line
for Wordflow 0.8.

- Remote embeddings ask for base64 vectors (little-endian float32): about a
  quarter of the size of JSON numbers and nothing to parse, so each batch
  comes back faster. Replies in numbers are still accepted, and a server that
  refuses base64 is asked for numbers for the rest of the run. Vectors that
  aren't whole float32 values fail the run.

## 0.6.9

- Remote embeddings: `embedding` and `topic_modeling` take
  `embedding_endpoint`, `embedding_api_key_env` or `embedding_api_key_file`,
  `embedding_remote_model`, `embedding_user`, `embedding_batch_size`,
  `embedding_concurrency` and `embedding_timeout_s`, to embed on an
  OpenAI-compatible `/v1/embeddings` server (vLLM, LiteLLM) instead of the
  local model. Keys stay out of plans; a key file is read before every request
  so SSO tokens can be refreshed; retries with backoff on connection errors,
  429 and 5xx; no fallback to the local model; a separate cache scope. Without
  the options nothing changes. Cargo feature `remote-embedding`, in `full`.
- HTTPS for remote embedding trusts the platform's certificates (ureq
  `platform-verifier`).

## 0.6.8

- macOS: the two curated MiniLM models embed on Core ML through static-shape
  length buckets (32, 64, 128, 256 tokens); length-sorted batches everywhere;
  CPU fallback with a recorded reason; embedding pipeline v3.
- Topic segmentation runs documents in parallel.
- New `segment_token_counts` expression.
