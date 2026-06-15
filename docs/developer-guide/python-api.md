# Python API

## Public Exports

`polars_text/__init__.py` exports:

- `clean_text`,
- `word_count`,
- `char_count`,
- `sentence_count`,
- `concordance`,
- `embedding`,
- `topic_modeling`,
- `token_frequencies`,
- `token_frequency_stats`,
- tokenizer model registry helpers.

Importing the package also imports `namespace.py`, which registers the `.text`
expression namespace with Polars.

## Expression Functions

`functions.py` wraps `polars.plugins.register_plugin_function()`. Each wrapper
passes `PLUGIN_PATH`, the Rust function name, input expression, keyword
arguments, and the appropriate elementwise flag. `PLUGIN_PATH` points to the
exact imported `_internal` extension file, not just the package directory, so
Polars does not accidentally load a stale ABI sibling library in editable
development environments.

Tokenization is available through the `.text` expression namespace. Callers must
pass an explicit tokenizer model ID:

```python
import polars as pl
import polars_text

pl.col("text").text.tokenize(model="native:plain_words_en")
```

`tokenize()` returns a list of structs with `token`, `start`, and `end`
character offsets. Passing `cache=Path(...)` uses a DuckDB-backed cache at that
path; `cache=None` computes directly through the same Rust plugin. The Python
wrapper only registers the expression; Rust owns cache lookup, locking, miss
deduplication, and persistence. Wordflow uses the cached form only after
resolving a per-user cache path in the backend.

`embedding()` is available as `polars_text.embedding(...)` and
`.text.embedding(...)`. It accepts string input and list-of-string input:

```python
pl.col("text").text.embedding(cache="embeddings.duckdb")
pl.col("chunks").text.embedding(cache="embeddings.duckdb")
```

String input returns `List(Float32)` per row. List input returns nested
`List(List(Float32))` per row. The Rust side owns Hugging Face downloads and
only supports repositories with ONNX artifacts. It also fetches ONNX external
data sidecars such as `onnx/model.onnx_data` when present.

`topic_modeling()` is a whole-column expression, not an elementwise expression.
It chunks documents, embeds chunks with the same ONNX Runtime embedder, reduces
with PaCMAP, clusters with HDBSCAN, and returns one struct per source document.
Passing `cache=Path(...)` points the Rust embedding stage at a separate
`embeddings.duckdb` cache.

## Namespace API

`namespace.py` registers `TextNamespace` with:

```python
@pl.api.register_expr_namespace("text")
```

The namespace is intentionally thin. It forwards to functions in
`functions.py`, so tokenization, embedding, and topic-modeling behavior each
have a single Python wrapper implementation.

## Token Frequencies

`token_frequencies(series, model=...)` accepts a Polars `Series`, normalizes
`None` values to empty strings, and calls the Rust `token_frequencies` helper.
Callers must pass an explicit predefined tokenizer ID or Hugging Face model ID;
the model routes through the same registry as `.text.tokenize(model=...)`.

`token_frequency_stats(corpus_0, corpus_1)` builds a Polars DataFrame with
frequencies, expected counts, log likelihood, BIC-style Bayes factor,
effect-size estimate, significance markers, percentages, relative risk,
log ratio, and odds ratio. It is Python/Polars code because it is tabular
post-processing over already-counted dictionaries.

## Model Helpers

`models.py` exposes inventory, not recommendation policy:

- `PREDEFINED_MODELS` maps predefined model IDs to supported language codes.
  It includes `native:plain_words_en`, `huggingface:bert-base-uncased`,
  `lindera:cc-cedict`, `lindera:jieba`, `lindera:ja-ipadic`,
  `lindera:ja-ipadic-neologd`, `lindera:ja-unidic`, and `lindera:ko-dic`.
- `LINDERA_MODELS_BY_LANGUAGE` groups the Lindera dictionary-backed IDs by
  supported language: `zh`, `ja`, and `ko`.

`prefetch_model()` and `list_loaded_models()` call the Rust registry wrappers.

## Plan Source Utilities

Serialized LazyFrame plan source-path listing and rewriting now lives in the
separate `polars-source-utils` package. `docworkspace` imports that package for
workspace rebasing so `polars-text` stays focused on tokenizer and text
expression builds.
