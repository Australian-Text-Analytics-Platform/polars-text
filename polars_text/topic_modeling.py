from __future__ import annotations

from typing import Any

import polars as pl

from ._internal import topic_modeling as _topic_modeling


def topic_modeling(
    series: pl.Series,
    *,
    min_points: int = 5,
    eps: float | None = None,
    max_terms: int = 5,
    seed: int = 42,
) -> tuple[dict[int, str], pl.Series]:
    if not isinstance(series, pl.Series):
        raise TypeError("topic_modeling expects a Polars Series")
    texts = [value if value is not None else "" for value in series.to_list()]
    topics, doc_topics = _topic_modeling(
        texts, min_points=min_points, eps=eps, max_terms=max_terms, seed=seed
    )
    doc_structs: list[list[dict[str, float | int]]] = [
        [{"topic_id": topic_id, "weight": weight} for topic_id, weight in row]
        for row in doc_topics
    ]
    doc_series = pl.Series(
        series.name,
        doc_structs,
        dtype=pl.List(
            pl.Struct([
                pl.Field("topic_id", pl.Int64),
                pl.Field("weight", pl.Float32),
            ])
        ),
    )
    return topics, doc_series


__all__ = ["topic_modeling"]
