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
- `concordance`.

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
