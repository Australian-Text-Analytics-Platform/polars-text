# polars-text Developer Guide

`polars-text` is a Rust/PyO3 package that extends Polars with text-processing
expression plugins and low-level serialized plan path helpers.

- [Architecture](architecture.md): big-picture package role.
- [Python API](python-api.md): functions, namespace registration, token
  statistics, model helpers, and plan-path wrappers.
- [Rust plugins](rust-plugins.md): PyO3 module, Polars expression plugins,
  text functions, and output schemas.
- [Tokenizers and plan paths](tokenizers-and-plan-paths.md): tokenizer
  registry, model/dictionary loading, offsets, and `.plbin` source rewriting.
- [Testing and release](testing-and-release.md): validation, maturin, and
  release process.
