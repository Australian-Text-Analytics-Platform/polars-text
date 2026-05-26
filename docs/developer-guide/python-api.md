# Python API

## Public Exports

`polars_text/__init__.py` exports:

- `tokenize`,
- `clean_text`,
- `word_count`,
- `char_count`,
- `sentence_count`,
- `concordance`,
- `token_frequencies`,
- `token_frequency_stats`,
- tokenizer model registry helpers.

Importing the package also imports `namespace.py`, which registers the `.text`
expression namespace with Polars.

## Expression Functions

`functions.py` wraps `polars.plugins.register_plugin_function()`. Each wrapper
passes `PLUGIN_PATH`, the Rust function name, input expression, keyword
arguments, and `is_elementwise=True`.

The expression API is available in two equivalent forms:

```python
import polars as pl
import polars_text as pt

pt.tokenize(pl.col("text"))
pl.col("text").text.tokenize()
```

`tokenize()` returns a list of structs with `token`, `start`, and `end`
character offsets. Passing `cache=Path(...)` uses a DuckDB-backed cache at that
path; `cache=None` computes directly through the Rust plugin. Wordflow uses the
cached form only after resolving a per-user cache path in the backend.

## Namespace API

`namespace.py` registers `TextNamespace` with:

```python
@pl.api.register_expr_namespace("text")
```

The namespace is intentionally thin. It forwards to functions in
`functions.py`, so behavior stays identical between module-level calls and
`.text.*` calls.

## Token Frequencies

`token_frequencies(series)` accepts a Polars `Series`, normalizes `None` to an
empty string, and calls the Rust `token_frequencies` helper.

`token_frequency_stats(corpus_0, corpus_1)` builds a Polars DataFrame with
frequencies, expected counts, log likelihood, BIC-style Bayes factor,
effect-size estimate, significance markers, percentages, relative risk,
log ratio, and odds ratio. It is Python/Polars code because it is tabular
post-processing over already-counted dictionaries.

## Model Helpers

`models.py` contains curated tokenizer ids:

- English: `bert-base-uncased`,
- Chinese: `jieba`,
- Japanese: `lindera-ja-ipadic`,
- Korean: `lindera-ko-dic`,
- multilingual: `xlm-roberta-base`,
- fallback: `bert-base-multilingual-cased`.

`recommended_tokenizer_for()` falls back to the multilingual model for unknown
language codes. `prefetch_model()` and `list_loaded_models()` call the Rust
registry wrappers.

## Plan Source Utilities

Serialized LazyFrame plan source-path listing and rewriting now lives in the
separate `polars-source-utils` package. `docworkspace` imports that package for
workspace rebasing so `polars-text` stays focused on tokenizer and text
expression builds.
