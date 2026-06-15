# Rust Plugins

## PyO3 Module

`src/lib.rs` defines the `_internal` Python module. It exports direct PyO3
functions for token frequencies, plan-path helpers, and tokenizer registry
helpers.

The module uses `PolarsAllocator` as the global allocator so memory allocated
inside Rust is compatible with Polars' expectations.

## Polars Expression Plugins

`src/expressions.rs` defines functions annotated with `#[polars_expr]`.
These are called by Polars through the plugin registration in Python.

Implemented expression plugins include:

- `clean_text`,
- `word_count`,
- `char_count`,
- `sentence_count`,
- `tokenize`,
- `concordance`,
- `embedding`,
- `topic_modeling`.

Each plugin declares its output schema through an output-type function. This is
important because Polars needs the schema while planning lazy queries.

## Simple Text Functions

`clean_text` lowercases text, replaces ASCII punctuation and digits with
spaces, collapses whitespace, and trims.

`word_count` keeps normal whitespace splitting for whitespace-delimited text.
For pure CJK runs without whitespace, it counts CJK characters as words so
Chinese/Japanese/Korean corpora do not all return one word per document.

`char_count` counts Unicode scalar values with `.chars().count()`.

`sentence_count` splits on ASCII terminators plus CJK full-width punctuation,
Arabic sentence marks, and Devanagari danda marks.

## Tokenization Output

`tokenize` returns `List[Struct[token: String, start: Int64, end: Int64]]`. It
tokenizes every row, accumulates flat token/start/end vectors, builds one inner
struct series, then slices that shared struct into per-row list entries. This
avoids per-row struct allocation overhead.

If the caller passes `cache=...`, tokenization uses the shared Rust DuckDB cache
flow in `cache.rs` with a token-specific table codec. The cache stores only
unique misses, uses the tokenizer model plus lowercase/punctuation parameters as
the namespace, computes misses outside the DB lock, and expands results back to
the original row order.

## Concordance Output

`concordance` returns a list of structs with:

- `left_context`,
- `matched_text`,
- `right_context`,
- `start_idx`,
- `end_idx`,
- `l1`,
- `r1`.

The Rust implementation finds regex matches, converts byte offsets to character
offsets in a batch, tokenizes left/right contexts with a lightweight BERT
pre-tokenizer, and builds one list entry per input row.

## Embedding Output

`embedding` accepts `String` and `List(String)` input. String input returns a
list of `Float32` values per row. List input returns a nested list so each input
string in the row gets its own embedding vector.

The plugin loads Hugging Face ONNX sentence-transformer repositories through the
Rust embedder registry. Model files and ONNX external-data sidecars are fetched
through `hf-hub` defaults. If the caller passes `cache=...`, vectors are stored
in a DuckDB file separate from the tokenization cache. Embedding uses the same
shared Rust cache flow as tokenization, with a table codec for binary `f32`
vectors keyed by model, revision, provider, and text hash.

## Topic Modeling Output

`topic_modeling` consumes the full input column and emits one struct per input
document. The pipeline chunks documents by paragraph, then sentence, then token
length; embeds chunks with ONNX Runtime; reduces with PaCMAP; clusters with
HDBSCAN; and labels topics with c-TF-IDF.
