"""Surgically remove targeted FFI-plugin expressions from a LazyFrame plan.

Polars' query optimiser does NOT dead-code-eliminate FFI plugin calls
(plugins may have side effects, and our :func:`tokenize_with_cache_lookup`
absolutely does — it writes a fresh delta-cache parquet whenever any row
misses the in-memory map). That means the natural-looking sequence
``lf.drop(col).with_columns(new_expr.alias(col))`` does **not** replace
the old plugin expression: the old `HStack` node stays alive and still
fires on every ``.collect()``, writing cache parquets to whatever
directory the *original* ``user_id`` kwarg pointed at.

:func:`scrub_plugin_expressions` is the proper fix. It walks the
serialized DSL plan and removes any ``Expr::Alias(<inner>, <name>)`` from
``HStack`` / ``Select`` where ``<name>`` matches a requested alias AND
``<inner>`` is a ``FunctionExpr::FfiPlugin`` whose ``symbol`` matches
the requested plugin symbol. After scrubbing the caller is free to
re-add a fresh expression under the same alias and trust that only the
new one will execute.

Use it at workspace-load time to align a cross-user-imported plan with
the importer's identity, and at the "delete derived column" path to
*actually* delete the underlying expression rather than just hiding its
output from the schema.
"""

from __future__ import annotations

import io
from typing import Iterable

import polars as pl

from ._internal import scrub_plugin_expressions as _scrub_plugin_expressions


def scrub_plugin_expressions(
    lf: pl.LazyFrame,
    *,
    aliases: Iterable[str],
    symbol: str,
) -> tuple[pl.LazyFrame, int]:
    """Remove every ``Alias(<name>, FfiPlugin{symbol=<symbol>})`` from
    ``lf``'s DSL plan, where ``<name>`` is in ``aliases``.

    Returns the rewritten ``LazyFrame`` and the number of expressions
    removed. When nothing matches, the original ``LazyFrame`` is
    returned unchanged (no expensive deserialize round-trip).
    """
    alias_list = list(aliases)
    if not alias_list:
        return lf, 0
    plan_bytes = lf.serialize(format="binary")
    new_bytes, removed = _scrub_plugin_expressions(plan_bytes, alias_list, symbol)
    if removed == 0:
        return lf, 0
    return pl.LazyFrame.deserialize(io.BytesIO(new_bytes), format="binary"), removed


__all__ = ["scrub_plugin_expressions"]
