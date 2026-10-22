# Changelog

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
