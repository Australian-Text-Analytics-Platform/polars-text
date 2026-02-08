# polars-text

Polars expression plugins for fast, practical text analysis. Use them as
expressions or via the `pl.col("text").text.*` namespace, plus a few
Series-based utilities for token frequency stats and topic modeling.

## Quick start

```python
import polars as pl
import polars_text as pt

df = pl.DataFrame({
    "text": [
        "Alice said \"Hello world\".",
        "Hello again, world!",
    ]
})

out = df.with_columns([
    pt.clean_text(pl.col("text")).alias("clean"),
    pt.word_count(pl.col("text")).alias("word_count"),
    pt.char_count(pl.col("text")).alias("char_count"),
    pt.sentence_count(pl.col("text")).alias("sentence_count"),
    pt.tokenize(pl.col("text"), lowercase=True, remove_punct=True).alias("tokens"),
])
```

## Expressions and namespace

All expression functions are available both as module functions and through
the `text` namespace on expressions.

### Expression functions

- `tokenize(expr, lowercase=True, remove_punct=True)`
- `clean_text(expr)`
- `word_count(expr)`
- `char_count(expr)`
- `sentence_count(expr)`
- `concordance(expr, search_word, num_left_tokens=5, num_right_tokens=5, regex=False, case_sensitive=False)`
- `quotation(expr)`

### Namespace usage

```python
df = pl.DataFrame({"text": ["Hello world, hello again."]})

out = df.select([
    pl.col("text").text.clean_text().alias("clean"),
    pl.col("text").text.word_count().alias("word_count"),
    pl.col("text").text.tokenize().alias("tokens"),
])
```

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

## Quotation extraction

Extract quoted speech along with speaker, verb, and offsets. Output is a list
of structs you can `explode` and `unnest`.

```python
df = pl.DataFrame({"text": ["Alice said \"Hello world\"."]})

quotes = (
    pl.col("text")
    .text.quotation()
    .list.explode()
    .struct.unnest()
)

out = df.select(quotes)
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

## Topic modeling

Cluster documents and return topic labels plus per-document topic assignments.

```python
series = pl.Series("text", [
    "Policy changes were announced today.",
    "Elections are coming soon.",
    "The football match was thrilling.",
])

topics, doc_topics = pt.topic_modeling(series, min_points=2, max_terms=3)
```

`topics` is a dict of `topic_id -> label` and `doc_topics` is a Series of lists
of structs with `{topic_id, weight}`.

## Output schemas

**Concordance** (list of structs):

- `left_context`, `matched_text`, `right_context`
- `start_idx`, `end_idx`
- `l1`, `r1` (first token on left/right for quick filtering)

**Quotation** (list of structs):

- `speaker`, `speaker_start_idx`, `speaker_end_idx`
- `quote`, `quote_start_idx`, `quote_end_idx`
- `verb`, `verb_start_idx`, `verb_end_idx`
- `quote_type`, `quote_token_count`, `is_floating_quote`

**Topic modeling** (Series of list structs):

- `topic_id` (int), `weight` (float)

## Models and downloads

Some features download Hugging Face models on first use (via `hf-hub`) and run
on CPU:

- Tokenization: `bert-base-uncased` (`tokenizer.json`)
- Topic modeling embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Quotation POS tagging: `vblagoje/bert-english-uncased-finetuned-pos`

The initial call may take longer while models download and cache.

## Development

Build the extension locally with maturin and then import as `polars_text`.

```bash
make build
make test
```
