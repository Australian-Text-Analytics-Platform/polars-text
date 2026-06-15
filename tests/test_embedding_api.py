from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import polars_text
from polars_text import functions
from polars_text.utils import PLUGIN_PATH


def test_embedding_is_exported_from_package() -> None:
    assert polars_text.embedding is functions.embedding


def test_plugin_path_points_to_imported_extension() -> None:
    assert PLUGIN_PATH.is_file()
    assert PLUGIN_PATH.name.startswith("_internal")
    assert PLUGIN_PATH.suffix == ".so"


def test_embedding_registers_plugin_kwargs(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_register_plugin_function(**kwargs: Any) -> pl.Expr:
        calls.append(kwargs)
        return pl.lit([0.0])

    monkeypatch.setattr(
        functions, "register_plugin_function", fake_register_plugin_function
    )

    cache_path = tmp_path / "embeddings.duckdb"

    expr = functions.embedding(
        "text",
        embedder_model="onnx-community/all-MiniLM-L6-v2-ONNX",
        cache=cache_path,
        batch_size=16,
    )

    assert isinstance(expr, pl.Expr)
    assert len(calls) == 1
    call = calls[0]
    assert call["function_name"] == "embedding"
    assert call["is_elementwise"] is True
    assert call["kwargs"] == {
        "embedder_model": "onnx-community/all-MiniLM-L6-v2-ONNX",
        "cache": str(cache_path),
        "batch_size": 16,
    }


def test_embedding_namespace_delegates_to_function(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_embedding(expr: pl.Expr, **kwargs: Any) -> pl.Expr:
        calls.append({"expr": expr, **kwargs})
        return pl.lit([0.0])

    monkeypatch.setattr(functions, "embedding", fake_embedding)

    cache_path = tmp_path / "embeddings.duckdb"
    text_namespace = getattr(pl.col("text"), "text")
    expr = text_namespace.embedding(
        embedder_model="onnx-community/all-MiniLM-L6-v2-ONNX",
        cache=cache_path,
        batch_size=32,
    )

    assert isinstance(expr, pl.Expr)
    assert len(calls) == 1
    assert calls[0]["embedder_model"] == "onnx-community/all-MiniLM-L6-v2-ONNX"
    assert calls[0]["cache"] == cache_path
    assert calls[0]["batch_size"] == 32


def test_topic_modeling_registers_embedding_cache_kwarg(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_register_plugin_function(**kwargs: Any) -> pl.Expr:
        calls.append(kwargs)
        return pl.lit([0])

    monkeypatch.setattr(
        functions, "register_plugin_function", fake_register_plugin_function
    )

    cache_path = tmp_path / "embeddings.duckdb"
    expr = functions.topic_modeling("text", cache=cache_path)

    assert isinstance(expr, pl.Expr)
    assert len(calls) == 1
    call = calls[0]
    assert call["function_name"] == "topic_modeling"
    assert call["kwargs"]["cache"] == str(cache_path)


def test_topic_modeling_namespace_delegates_cache(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_topic_modeling(expr: pl.Expr, **kwargs: Any) -> pl.Expr:
        calls.append({"expr": expr, **kwargs})
        return pl.lit([0])

    monkeypatch.setattr(functions, "topic_modeling", fake_topic_modeling)

    cache_path = tmp_path / "embeddings.duckdb"
    text_namespace = getattr(pl.col("text"), "text")
    expr = text_namespace.topic_modeling(cache=cache_path)

    assert isinstance(expr, pl.Expr)
    assert len(calls) == 1
    assert calls[0]["cache"] == cache_path
