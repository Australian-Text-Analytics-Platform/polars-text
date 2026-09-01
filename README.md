# polars-text

Polars 1.44.1 expression plugins for fast, practical text analysis. The
`pl.col("text").text.*` namespace is the sole expression façade; whole-Series
token-frequency and topic-projection utilities remain top-level functions.

## Quick start

```python
import polars as pl
import polars_text

df = pl.DataFrame({
    "text": [
        "Alice said \"Hello world\".",
        "Hello again, world!",
    ]
})

out = df.with_columns([
    pl.col("text").text.clean_text().alias("clean"),
    pl.col("text").text.word_count().alias("word_count"),
    pl.col("text").text.char_count().alias("char_count"),
    pl.col("text").text.sentence_count().alias("sentence_count"),
    pl.col("text").text.tokenize(
        model="native:plain_words_en",
        lowercase=True,
        remove_punctuation=True,
    ).alias("tokens"),
])
```

## Expressions and namespace

All expression operations are available through the `text` namespace.

### Tokenization

- `pl.col("text").text.tokenize(model="native:plain_words_en", lowercase=True, remove_punctuation=True, cache=None)`
- `pl.col("text").text.embedding(model=None, cache=None, batch_size=None)`
- `pl.col("text").text.clean_text()`
- `pl.col("text").text.word_count()`
- `pl.col("text").text.char_count()`
- `pl.col("text").text.sentence_count()`
- `pl.col("text").text.concordance(query, left_tokens=5, right_tokens=5, regex=False, case_sensitive=False, ignore_punctuation=False)`

### Namespace usage

```python
df = pl.DataFrame({"text": ["Hello world, hello again."]})

out = df.select([
    pl.col("text").text.clean_text().alias("clean"),
    pl.col("text").text.word_count().alias("word_count"),
    pl.col("text").text.tokenize(model="native:plain_words_en").alias("tokens"),
])
```

`tokenize` returns a list of structs with `token`, `start`, and `end`
character offsets. Pass an explicit `native:`, `huggingface:`, or `lindera:`
model ID. Pass `cache=Path("tokens.duckdb")` to persist tokenization results in
a DuckDB cache and reuse them by content hash; leave `cache=None` to compute
directly through the Rust plugin.

A configured cache path is dedicated, disposable `polars-text` storage. Schema
or model-pipeline changes replace the complete DuckDB file; do not put
unrelated user tables in it.

Pass `ignore_punctuation=True` to `concordance` to exclude punctuation and
symbol-only tokens from context counts and L1/R1. The returned contexts retain
the original punctuation and whitespace between lexical tokens and the match;
literal and regular-expression matching are unchanged.

### Embeddings

`embedding` accepts a string expression or a list-of-string expression. String
input returns `List(Float32)` per row; list input returns nested
`List(List(Float32))` per row.

```python
df = pl.DataFrame({"text": ["A short document."], "segments": [["first", "second"]]})

out = df.select([
    pl.col("text").text.embedding(cache="embeddings.duckdb").alias("embedding"),
    pl.col("segments").text.embedding(cache="embeddings.duckdb").alias("segment_embeddings"),
])
```

The Rust plugin downloads and loads Hugging Face ONNX sentence-transformer
repositories automatically through `hf-hub`. Repositories without ONNX files are
not supported. Passing `cache=Path("embeddings.duckdb")` persists vectors in a
separate DuckDB cache keyed by the immutable model snapshot, ONNX artifact,
pooling and normalization graph, canonical maximum length, execution provider,
pipeline version, and text hash. The default model is
`sentence-transformers/all-MiniLM-L6-v2`, whose declared maximum is 256 tokens.

## Concordance

Get left/right context windows around a search term. Output is a list of
structs that you can `explode` and `unnest` for tabular use.

```python
df = pl.DataFrame({"text": ["Hello world, hello again."]})

concordance = (
    pl.col("text")
    .text.concordance("hello", left_tokens=1, right_tokens=1)
    .list.explode()
    .struct.unnest()
)

out = df.select(concordance)
```

## Topic modelling

`topic_modeling` consumes a complete document column and returns one scalar run
result. The result keeps document outcomes separate from complete topic
metadata:

```text
{
  documents: [{doc_index, dominant_topic, topic_coverage}],
  topics: [{id, representative_words, x, y}],
  n_segments,
  projection_context
}
```

Automatic, Line, and Sentence modes differ only when constructing Topic
Segments. All modes then share embedding, clustering, c-TF-IDF, and document
rollup. Clustering treats every segment as one observation. Rollup weights each
non-overlapping source span by its owned Unicode-character length.

The token budget includes model-added special tokens. Oversized semantic units
are split without overlap or discarded tail text.
Corpora with too little density evidence return no Topics and a null projection
context. Use `project_topics` and `project_topic_basis` with a non-null context
for supported post-fit projections down to one Topic.

## Token frequencies and stats

Compute corpus token counts and compare corpora with standard statistics.

```python
series_0 = pl.Series("text", ["hello world", "hello again"])
series_1 = pl.Series("text", ["goodbye world"])

freqs_0 = pt.token_frequencies(series_0, model="native:plain_words_en")
freqs_1 = pt.token_frequencies(series_1, model="native:plain_words_en")

stats = pt.token_frequency_stats(freqs_0, freqs_1)
```

## Output schemas

**Tokenization** (list of structs):

- `token`
- `start`
- `end`

**Concordance** (list of structs):

- `left_context`, `matched_text`, `right_context`
- `start_idx`, `end_idx`
- `l1`, `r1` (first token on left/right for quick filtering)

## Models and downloads

Some features download tokenizer assets on first use and run on CPU:

- Hugging Face tokenizers: for example `huggingface:bert-base-uncased`
  (`tokenizer.json` via `hf-hub`)
- Lindera dictionaries: `lindera:cc-cedict`, `lindera:jieba`,
  `lindera:ja-ipadic`, `lindera:ja-ipadic-neologd`, `lindera:ja-unidic`,
  and `lindera:ko-dic` from official Lindera release zips

The initial call may take longer while models download and cache.

`TOKENIZER_MODELS` is the immutable catalogue of
`TokenizerModel(model_id, label, languages)` records.

Embedding features download ONNX artifacts on first use. Some ONNX repositories
store tensor data in sidecar files such as `onnx/model.onnx_data`; those files
are fetched automatically when present. ONNX Runtime uses CoreML on macOS,
DirectML on Windows, XNNPACK on Linux, and CPU fallback on every platform.

## Development

Build the extension locally with maturin and then import as `polars_text`.
See the repository-level
[development](../docs/runbooks/polars-text-development.md) and
[release](../docs/runbooks/polars-text-release.md) runbooks for complete
procedures.

```bash
make build
make test
```

For faster Rust iteration, use feature-scoped targets such as
`make check-tokenization`, `make build-tokenization`, or `make build-topic`.
Leave `JOBS` unset for Cargo's default parallelism, or pass `JOBS=<n>` to cap it.
