from __future__ import annotations

import json
from typing import Any

from ._internal import project_topic_modeling_basis as _project_topic_basis
from ._internal import project_topic_modeling_context as _project_topics
from ._internal import project_topic_modeling_segments as _project_topic_segments
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


def project_topic_segments(
    projection_context: bytes,
    topic_count: int,
) -> list[tuple[int, int, int, int]]:
    """Map every Topic Segment to its Topic at ``topic_count``.

    Returns ``(document_index, start_char, end_char, topic_id)`` per segment, in
    segment order. Spans are Unicode-character offsets into the run's input
    documents, and ``topic_id`` uses the same merge cut as ``project_topics``
    (``-1`` for outliers). Raises ``ValueError`` for projection contexts from
    runs before segment spans were stored.
    """
    _require_feature("topic-modeling", "project_topic_segments")
    _positive(topic_count, "topic_count")
    return [
        (int(document), int(start), int(end), int(topic))
        for document, start, end, topic in json.loads(
            _project_topic_segments(projection_context, topic_count)
        )
    ]


__all__ = ["project_topic_basis", "project_topic_segments", "project_topics"]
