# polars-text Architecture

`polars-text` is the compiled text-processing layer used by Wordflow. It is a
Python package with a Rust extension module built by maturin and PyO3.

## Package Role

The package provides:

- Polars expression plugins for text cleaning, counts, tokenization, offsets,
  and concordance;
- a `.text` namespace on Polars expressions;
- Series-based token-frequency helpers;
- tokenizer model recommendation and prefetch helpers.

The backend uses these functions directly. Serialized LazyFrame plan source-path
inspection and rewriting lives in the sibling `polars-source-utils` package and
is used by `docworkspace`.

## Main Boundaries

Python files under `polars_text/` expose a typed API and register plugins.
Rust files under `src/` implement the heavy work:

- `lib.rs`: PyO3 module and exported Python-callable functions.
- `expressions.rs`: Polars expression plugin functions and schemas.
- `tokenizer.rs`: tokenizer registry and backend dispatch.
- `lindera_dict.rs`: on-demand dictionary download and cache.
- `concordance.rs`: regex matching and context-window construction.
- `offsets.rs`: byte-to-character offset conversion.
- `token_frequencies.rs`: native token-frequency counting.

## Design Principle

The package keeps Python as the ergonomic API layer and Rust as the execution
layer. Plugin functions should expose Polars expressions so the caller can keep
work lazy. Operations that cannot be expression plugins, such as token
frequency dictionaries, are exposed as direct PyO3 functions.
