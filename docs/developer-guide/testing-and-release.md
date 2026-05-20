# Testing And Release

## Local Development

From `polars-text/`:

```bash
make build
make test
```

`make build` runs `maturin develop --release`, installing the compiled extension
into the active environment. `make test` runs `pytest -q`.

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
Polars/PyO3 dependencies. Keep both versions aligned for releases.

## Release Process

`PUBLISH.md` describes the trusted-publishing flow. CI builds platform wheels
and one source distribution, then publishes only for explicit `v*` tags or
manual TestPyPI dry runs.

Before tagging, run local validation and make sure the Python package version,
Rust crate version, and release tag agree.
