from __future__ import annotations

import functools
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, cast

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from .utils import PLUGIN_PATH

CachePath = str | os.PathLike[str]

_TOKEN_STRUCT_DTYPE: pl.DataType = pl.Struct(
    [
        pl.Field("token", pl.String),
        pl.Field("start", pl.Int64),
        pl.Field("end", pl.Int64),
    ]
)
_TOKENS_DTYPE: pl.DataType = pl.List(_TOKEN_STRUCT_DTYPE)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS token_cache (
    model VARCHAR NOT NULL,
    params_hash VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL,
    tokens VARCHAR[] NOT NULL,
    start_offsets BIGINT[] NOT NULL,
    end_offsets BIGINT[] NOT NULL,
    PRIMARY KEY (model, params_hash, content_hash)
)
"""

_DB_LOCK = threading.Lock()


def _tokenize_kwargs(
    *, lowercase: bool, remove_punct: bool, model: str | None
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "lowercase": lowercase,
        "remove_punct": remove_punct,
    }
    if model is not None:
        kwargs["model_id"] = model
    return kwargs


def uncached_tokenize_expr(
    expr: IntoExpr,
    *,
    lowercase: bool = True,
    remove_punct: bool = True,
    model: str | None = None,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="tokenize",
        args=expr,
        kwargs=_tokenize_kwargs(
            lowercase=lowercase, remove_punct=remove_punct, model=model
        ),
        is_elementwise=True,
    )


def _params_hash(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _hash_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _connect(cache: CachePath):
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "Using pl.col(...).text.tokenize(..., cache=...) requires the duckdb package."
        ) from exc

    path = Path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(_SCHEMA_SQL)
    return conn


def _fetch_cached(
    conn: Any,
    *,
    model: str,
    params_hash: str,
    hashes: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not hashes:
        return {}
    requested = list(dict.fromkeys(hashes))
    rows = conn.execute(
        """
        SELECT content_hash, tokens, start_offsets, end_offsets
        FROM token_cache
        WHERE model = ?
          AND params_hash = ?
          AND content_hash IN (SELECT unnest(?))
        """,
        [model, params_hash, requested],
    ).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for content_hash, toks, starts, ends in rows:
        toks_list = list(toks or [])
        starts_list = list(starts or [])
        ends_list = list(ends or [])
        n = min(len(toks_list), len(starts_list), len(ends_list))
        out[str(content_hash)] = [
            {
                "token": str(toks_list[i]),
                "start": int(starts_list[i]),
                "end": int(ends_list[i]),
            }
            for i in range(n)
        ]
    return out


def _persist_new(
    conn: Any,
    *,
    model: str,
    params_hash: str,
    new_entries: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    if not new_entries:
        return
    records: list[tuple[str, str, str, list[str], list[int], list[int]]] = []
    for content_hash, tokens in new_entries:
        toks: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        for entry in tokens or []:
            if entry is None:
                continue
            toks.append(str(entry.get("token") or ""))
            starts.append(int(entry.get("start") or 0))
            ends.append(int(entry.get("end") or 0))
        records.append((model, params_hash, content_hash, toks, starts, ends))
    conn.executemany(
        """
        INSERT OR IGNORE INTO token_cache
        (model, params_hash, content_hash, tokens, start_offsets, end_offsets)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        records,
    )


def _tokenize_misses(
    texts: list[str],
    *,
    model: str | None,
    lowercase: bool,
    remove_punct: bool,
) -> list[list[dict[str, Any]]]:
    if not texts:
        return []
    miss_df = cast(
        pl.DataFrame,
        pl.DataFrame({"__src__": texts})
        .lazy()
        .select(
            uncached_tokenize_expr(
                pl.col("__src__"),
                model=model,
                lowercase=lowercase,
                remove_punct=remove_punct,
            ).alias("__tokens__")
        )
        .collect(),
    )
    return miss_df["__tokens__"].to_list()


def _tokenize_chunk(
    s: pl.Series,
    *,
    cache: CachePath,
    model: str,
    params_hash: str,
    lowercase: bool,
    remove_punct: bool,
) -> pl.Series:
    values = s.to_list()
    if not values:
        return pl.Series(name=s.name, values=[], dtype=_TOKENS_DTYPE)
    hashes = [_hash_text(v) for v in values]
    texts = ["" if v is None else str(v) for v in values]

    with _DB_LOCK:
        conn = _connect(cache)
        try:
            cached = _fetch_cached(
                conn, model=model, params_hash=params_hash, hashes=hashes
            )
            unique_misses: dict[str, str] = {}
            for content_hash, text in zip(hashes, texts):
                if content_hash not in cached and content_hash not in unique_misses:
                    unique_misses[content_hash] = text
            if unique_misses:
                miss_hashes = list(unique_misses.keys())
                miss_texts = list(unique_misses.values())
                computed = _tokenize_misses(
                    miss_texts,
                    model=model,
                    lowercase=lowercase,
                    remove_punct=remove_punct,
                )
                new_entries = list(zip(miss_hashes, computed))
                _persist_new(
                    conn,
                    model=model,
                    params_hash=params_hash,
                    new_entries=new_entries,
                )
                for content_hash, tokens in new_entries:
                    cached[content_hash] = tokens or []
        finally:
            conn.close()

    out = [cached.get(content_hash, []) for content_hash in hashes]
    return pl.Series(name=s.name, values=out, dtype=_TOKENS_DTYPE)


def cached_tokenize_expr(
    source_expr: pl.Expr,
    *,
    cache: CachePath,
    model: str | None,
    lowercase: bool = True,
    remove_punct: bool = True,
) -> pl.Expr:
    model_key = model or "bert-base-uncased"
    params = {"lowercase": lowercase, "remove_punct": remove_punct}
    params_hash = _params_hash(params)
    fn = functools.partial(
        _tokenize_chunk,
        cache=cache,
        model=model_key,
        params_hash=params_hash,
        lowercase=lowercase,
        remove_punct=remove_punct,
    )
    return source_expr.cast(pl.Utf8, strict=False).map_batches(
        fn,
        return_dtype=_TOKENS_DTYPE,
        is_elementwise=True,
    )


__all__ = [
    "CachePath",
    "cached_tokenize_expr",
    "uncached_tokenize_expr",
]
