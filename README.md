# polars-text

Polars expression plugins for fast, practical text analysis. Use them as
expressions or via the `pl.col("text").text.*` namespace, plus a few
Series-based utilities for token frequency stats.

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
    pl.col("text").text.tokenize(lowercase=True, remove_punct=True).alias("tokens"),
])
```

## Expressions and namespace

Tokenization is available through the `text` namespace on expressions.

### Tokenization

- `pl.col("text").text.tokenize(lowercase=True, remove_punct=True, model=None, cache=None)`
- `clean_text(expr)`
- `word_count(expr)`
- `char_count(expr)`
- `sentence_count(expr)`
- `concordance(expr, search_word, num_left_tokens=5, num_right_tokens=5, regex=False, case_sensitive=False)`

### Namespace usage

```python
df = pl.DataFrame({"text": ["Hello world, hello again."]})

out = df.select([
    pl.col("text").text.clean_text().alias("clean"),
    pl.col("text").text.word_count().alias("word_count"),
    pl.col("text").text.tokenize().alias("tokens"),
])
```

`tokenize` returns a list of structs with `token`, `start`, and `end`
character offsets. Pass `cache=Path("tokens.duckdb")` to persist tokenization
results in a DuckDB cache and reuse them by content hash; leave `cache=None`
to compute directly through the Rust plugin.

## Concordance

Get left/right context windows around a search term. Output is a list of
structs that you can `explode` and `unnest` for tabular use.

```python
df = pl.DataFrame({"text": ["Hello world, hello again."]})

concordance = (
    pl.col("text")
    .text.concordance("hello", num_left_tokens=1, num_right_tokens=1)
    .list.explode()
    .struct.unnest()
)

out = df.select(concordance)
```

## Token frequencies and stats

Compute corpus token counts and compare corpora with standard statistics.

```python
series_0 = pl.Series("text", ["hello world", "hello again"])
series_1 = pl.Series("text", ["goodbye world"])

freqs_0 = pt.token_frequencies(series_0)
freqs_1 = pt.token_frequencies(series_1)

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

Some features download Hugging Face models on first use (via `hf-hub`) and run
on CPU:

- Tokenization: `bert-base-uncased` (`tokenizer.json`)

The initial call may take longer while models download and cache.

## Development

Build the extension locally with maturin and then import as `polars_text`.

For release and publishing procedures, see `PUBLISH.md`.

```bash
make build
make test
```
