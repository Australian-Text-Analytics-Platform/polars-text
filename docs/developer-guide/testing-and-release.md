# Testing And Release

## Local Development

From `polars-text/`:

```bash
make build
make test
```

`make build` runs `maturin develop --release`, installing the compiled extension
into the active environment. `make test` runs `pytest -q`.

For compile-time profiling on a high-core local workstation:

```bash
make timings-full
make timings-lean
```

Both targets write Cargo's HTML timing report under `target/cargo-timings/`.
`timings-full` builds the default full feature set. `timings-lean` builds with
`--no-default-features` and is intended for fast iteration on the basic
expression boundary.

## Rust Feature Sets

Default builds enable `full`, so PyPI wheels and normal source installs keep the
same behavior. The feature split is only for developer build-time control:

- `cache`: DuckDB-backed text/vector caches.
- `tokenization`: Hugging Face/Lindera tokenization, concordance, tokenizer
  prefetch, and token-frequency helpers.
- `embedding`: ONNX Runtime embeddings and embedding cache support.
- `topic-modeling`: the full native topic-modeling pipeline; this includes
  `embedding`, `tokenization`, PaCMAP, HDBSCAN, and the static MKL backend used
  for Windows wheels.

When a developer builds with `--no-default-features`, only the basic
`clean_text`, `word_count`, `char_count`, and `sentence_count` expression
plugins are available. Python wrappers for gated functions remain importable but
raise a `RuntimeError` explaining which feature is missing before registering a
Polars plugin symbol.

## Test Coverage Shape

The suite covers:

- expression and namespace registration,
- tokenization and offset schemas,
- Chinese Jieba segmentation,
- Lindera integration and dictionary routing,
- model registry helpers,
- token-frequency statistics,
- concordance output,
- golden multilingual baselines,
- `.plbin` source path listing and rewriting.

Plan-path tests are especially important because they protect the workspace
load/save path in the backend and `docworkspace`.

## Packaging

The package is built with maturin. `pyproject.toml` defines Python metadata and
the extension module path; `Cargo.toml` defines the Rust crate version and
Polars/PyO3 dependencies. Keep both versions aligned for releases. Release
wheel jobs build the default `full` feature set, use Cargo and sccache caching,
and upload `target/cargo-timings/cargo-timing.html` when Cargo produces it.

Keep Python runtime dependencies limited to packages imported by the Python
wrapper layer. DuckDB cache support is implemented by the Rust `duckdb` crate
behind the Cargo `cache` feature, so the Python `duckdb` package is not a
runtime or test dependency. Cache tests inspect Rust-created cache files through
the private `_internal.debug_token_cache_snapshot` helper instead.

## Release Process

`PUBLISH.md` describes the trusted-publishing flow. CI builds platform wheels
and one source distribution, then publishes only for explicit `v*` tags or
manual TestPyPI dry runs.

Before tagging, run local validation and make sure the Python package version,
Rust crate version, and release tag agree.
