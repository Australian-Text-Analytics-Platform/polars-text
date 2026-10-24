# Changelog

## 0.7.2

- Topic modelling gives the same topics for the same text, settings and seed
  on every run and with any number of threads (Wordflow issue 437). Before,
  runs with several threads on more than 8,000 segments differed (for example
  308, 305 and 313 topics for the same input). The cause was in pacmap 0.2.6,
  now patched in `vendor/pacmap` (see `POLARS_TEXT_CHANGES.md` there):
  - its approximate-neighbour index was filled from several threads, so the
    neighbours found depended on thread timing. It is now built as 4 fixed
    shards, each filled in row order, and every shard is searched;
  - its gradient sums were combined in whatever order threads finished. They
    are now added in chunk order.
  Results change once, compared with 0.7.1. The arranging step is about 1.5 s
  slower on 81,000 segments (6.0 s instead of 4.5 s with 8 threads).

## 0.7.1

- Topic modelling, last step (topic words and the topic map), Wordflow issue 431:
  - The tree that merges natural topics builds in about n^2 instead of n^3
    steps: each node keeps its nearest partner instead of a full scan of
    every pair before each merge. The merges, their order and the
    tie-breaking are unchanged. 1,197 topics: 18 s before, 38 ms now on an
    M5 Pro.
  - Counting each topic's words runs on the same processors as
    segmentation instead of one (200,000 segments: 2.6 s before, 0.6 s now).
    Counts are summed, so topic words are unchanged.

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
