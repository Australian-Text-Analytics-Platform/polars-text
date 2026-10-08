from __future__ import annotations

import json
from typing import Any

from ._internal import project_topic_modeling_basis as _project_topic_basis
from ._internal import project_topic_modeling_context as _project_topics
from ._internal import project_topic_modeling_segments as _project_topic_segments
from ._internal import topic_modeling_segment_similarities as _topic_segment_similarities
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


def topic_segment_similarities(
    projection_context: bytes,
    topic_count: int,
    topic_id: int,
) -> list[tuple[int, int, int, int, float | None]]:
    """Every segment of one Topic with its similarity to the Topic's centre.

    Returns ``(segment_index, document_index, start_char, end_char,
    similarity)`` in segment order, for ``topic_id`` at ``topic_count`` (the
    same merge cut as ``project_topics``). The similarity is the cosine
    between the segment's compact embedding and the Topic's centre (the mean
    of its segments), for ranking how typical a segment is; it is ``None`` for
    runs before compact embeddings were stored. Raises ``ValueError`` for
    ``topic_id`` -1 (No topic) or contexts without segment spans.
    """
    _require_feature("topic-modeling", "topic_segment_similarities")
    _positive(topic_count, "topic_count")
    return [
        (
            int(segment),
            int(document),
            int(start),
            int(end),
            None if similarity is None else float(similarity),
        )
        for segment, document, start, end, similarity in json.loads(
            _topic_segment_similarities(projection_context, topic_count, topic_id)
        )
    ]


__all__ = [
    "project_topic_basis",
    "project_topic_segments",
    "project_topics",
    "topic_segment_similarities",
]
