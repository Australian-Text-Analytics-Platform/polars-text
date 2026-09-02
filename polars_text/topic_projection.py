from __future__ import annotations

import json
from typing import Any

from ._internal import project_topic_modeling_basis as _project_topic_basis
from ._internal import project_topic_modeling_context as _project_topics
from .namespace import _positive, _require_feature


def project_topics(projection_context: bytes, topic_count: int) -> dict[str, Any]:
    _require_feature("topic-modeling", "project_topics")
    _positive(topic_count, "topic_count")
    return json.loads(_project_topics(projection_context, topic_count))


def project_topic_basis(
    projection_context: bytes,
    topic_count: int,
    corpus_sizes: list[int],
) -> dict[str, Any]:
    _require_feature("topic-modeling", "project_topic_basis")
    _positive(topic_count, "topic_count")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in corpus_sizes
    ):
        raise ValueError("corpus_sizes must contain only non-negative integers")
    return json.loads(
        _project_topic_basis(projection_context, topic_count, corpus_sizes)
    )


__all__ = ["project_topic_basis", "project_topics"]
