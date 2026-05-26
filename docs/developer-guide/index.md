# polars-text Developer Guide

`polars-text` is a Rust/PyO3 package that extends Polars with text-processing
expression plugins.

- [Architecture](architecture.md): big-picture package role.
- [Python API](python-api.md): functions, namespace registration, token
  statistics, and model helpers.
- [Rust plugins](rust-plugins.md): PyO3 module, Polars expression plugins,
  text functions, and output schemas.
- [Tokenizers](tokenizers-and-plan-paths.md): tokenizer registry,
  model/dictionary loading, and offsets.
- [Testing and release](testing-and-release.md): validation, maturin, and
  release process.
