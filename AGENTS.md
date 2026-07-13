# polars-text operating guide

These rules extend the root `AGENTS.md` for the Rust/PyO3 Polars plugin
package. Read [the package architecture](../docs/architecture/packages/polars-text.md)
and [API reference](../docs/reference/polars-text-api.md) before changing a
public expression or namespace contract.

## Boundaries

- Rust expression plugins own vectorized computation; Python exposes typed
  wrappers and the `Expr.text` namespace.
- Keep the extension ABI and Python wrapper signatures synchronized.
- Serialized Polars-plan source inspection and rewriting belongs exclusively in
  `polars-source-utils`; do not add that responsibility here.
- Preserve lazy expression behavior. Do not collect a frame inside an
  expression wrapper.
- Treat tokenizer/model downloads and caches as explicit I/O boundaries and
  keep feature-specific dependencies behind their Cargo features.

## Development

Prefer feature-scoped Make targets during Rust iteration, then run the complete
package gates before completion:

```sh
make check-tokenization
make build-tokenization
make build-topic
make build
uvx ty check
uv run pytest -q
```

Use only targets relevant to the edit during development. Follow the
[development runbook](../docs/runbooks/polars-text-development.md) and
[release runbook](../docs/runbooks/polars-text-release.md) rather than copying
publishing commands into package documentation.
