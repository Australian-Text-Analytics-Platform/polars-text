from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

import polars_text.namespace as namespace_module


def test_embedding_registers_validated_plugin_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(namespace_module, "compiled_features", lambda: {"embedding"})
    monkeypatch.setattr(
        namespace_module,
        "register_plugin_function",
        lambda **kwargs: calls.append(kwargs) or pl.lit([0.0]),
    )

    cache_path = tmp_path / "embeddings.duckdb"
    expr = cast(Any, pl.col("text")).text.embedding(
        model=" onnx-community/all-MiniLM-L6-v2-ONNX ",
        cache=cache_path,
        batch_size=16,
    )

    assert isinstance(expr, pl.Expr)
    assert calls[0]["kwargs"] == {
        "embedder_model": "onnx-community/all-MiniLM-L6-v2-ONNX",
        "cache": str(cache_path),
        "batch_size": 16,
    }
    with pytest.raises(ValueError, match="batch_size"):
        cast(Any, pl.col("text")).text.embedding(batch_size=0)


def test_topic_modeling_registers_only_the_supported_fit_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        namespace_module, "compiled_features", lambda: {"topic-modeling"}
    )
    monkeypatch.setattr(
        namespace_module,
        "register_plugin_function",
        lambda **kwargs: calls.append(kwargs) or pl.lit([0]),
    )

    cast(Any, pl.col("text")).text.topic_modeling(
        segmentation="line", max_tokens=80, seed=7, min_topic_size=2
    )

    assert calls[0]["kwargs"] == {
        "embedder_model": None,
        "cache": None,
        "segmentation_method": "line",
        "max_tokens": 80,
        "seed": 7,
        "min_cluster_size": 2,
        "vectorizer_model": None,
        "lowercase": True,
    }
    with pytest.raises(ValueError, match="automatic.*line.*sentence"):
        cast(Any, pl.col("text")).text.topic_modeling(segmentation="paragraph")
