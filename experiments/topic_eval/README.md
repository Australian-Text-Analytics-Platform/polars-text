# Topic-modeling evaluation harness (Phase 2 — manual, not in CI)

This directory holds **hand-run** experiments that validate the pure-Rust
topic-modeling pipeline in `polars-text` before it is wired into the backend.

It is intentionally **outside the automated test suite**: the embedding →
PaCMAP → HDBSCAN stages are non-deterministic, so we judge quality by inspecting
results on *labeled* corpora instead of asserting on output in CI (plan
feedback #5). Passing this harness is the **integration gate** for Phase 3.

## What it measures

For a chosen labeled dataset it:

1. samples documents and runs `polars_text._internal.run_topic_modeling`;
2. scores the per-document **dominant topic** against ground-truth labels with
   **Adjusted Rand Index**, NMI, and homogeneity/completeness/V-measure;
3. prints each discovered topic's c-TF-IDF keywords and dominant true label
   (cluster purity);
4. reports **long-text diagnostics** — chunks per document and the share of
   documents with a genuinely multi-topic distribution (the long-text value
   prop that the old single-topic path could not express);
5. optionally fits reference **Python BERTopic** on the same docs for a direct
   ARI comparison (`--bertopic`).

## Datasets (clean, parquet, offline-cacheable)

All carry ground-truth topic labels (`corpora.py`):

| key        | HF id                     | lang | topics | character of the test                |
| ---------- | ------------------------- | ---- | ------ | ------------------------------------ |
| `20ng`     | `SetFit/20_newsgroups`    | EN   | 20     | canonical BERTopic benchmark         |
| `bbc`      | `SetFit/bbc-news`         | EN   | 5      | long, clean EN articles              |
| `thucnews` | `oyxy2019/THUCNewsText`   | ZH   | 10     | long **Chinese** articles (lindera)  |

The first run downloads and caches via the `datasets` library; afterwards set
`HF_HUB_OFFLINE=1` to run fully offline.

## Prerequisites

Build the extension so `polars_text` is importable in the project env:

```bash
cd polars-text
uv run --no-project --with maturin maturin develop --release
```

## Running

```bash
cd polars-text

# EN long articles, 400 docs, balanced across classes
uv run --with datasets --with scikit-learn \
    python experiments/topic_eval/eval.py --dataset bbc --limit 400 --balanced

# Canonical 20 Newsgroups benchmark
uv run --with datasets --with scikit-learn \
    python experiments/topic_eval/eval.py --dataset 20ng --limit 2000 --balanced

# Long Chinese news (exercises the CJK/lindera path)
uv run --with datasets --with scikit-learn \
    python experiments/topic_eval/eval.py --dataset thucnews --limit 1000 --balanced

# Head-to-head against Python BERTopic (slow, needs the ML stack present)
uv run --with datasets --with scikit-learn --with bertopic --with sentence-transformers \
    python experiments/topic_eval/eval.py --dataset bbc --limit 400 --bertopic
```

Useful knobs: `--topic-size-mode {min,target,exact}` with `--topic-size-value N`
(post-cluster topic-count control — `min` sets the HDBSCAN floor, `target`/`exact`
over-segment then merge micro-topics down to ~N / exactly N),
`--min-cluster-size` (HDBSCAN granularity), `--max-tokens` / `--overlap`
(chunking), `--reduce-dims` (PaCMAP target), `--seed`, `--embedder <hf repo id>`.

```bash
# Ask for exactly 5 topics on BBC (matches its 5 true categories)
uv run --with datasets --with scikit-learn \
    python experiments/topic_eval/eval.py --dataset bbc --limit 400 --balanced \
    --topic-size-mode exact --topic-size-value 5
```

## Reading the output

- **ARI / NMI near BERTopic** on the same docs → the Rust results are
  trustworthy, not just plausible-looking.
- **Per-topic purity** high → clusters map onto real categories.
- **multi-topic docs %** non-trivial on `bbc`/`thucnews` → long documents are
  being represented as genuine topic *mixtures*, which is the whole point of the
  chunk-based redesign.
