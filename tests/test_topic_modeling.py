import os

import polars as pl
import polars_text as pt
import pytest


@pytest.mark.slow
def test_topic_modeling_returns_topics_and_series() -> None:
    if os.environ.get("POLARS_TEXT_SKIP_TOPIC_MODELING") == "1":
        pytest.skip("Skipping topic modeling test by env request")

    series = pl.Series(
        "text",
        [
            "Climate policy and emissions are rising.",
            "Markets reacted to inflation and rates.",
            "Renewable energy targets are expanding.",
            "Stocks fell after the central bank announcement.",
        ],
    )

    topics, doc_topics = pt.topic_modeling(series, min_points=2, eps=None, max_terms=3)

    assert isinstance(topics, dict)
    assert isinstance(doc_topics, pl.Series)
    assert doc_topics.len() == series.len()

    values = doc_topics.to_list()
    assert isinstance(values, list)
    assert len(values) == series.len()

    for entry in values:
        assert isinstance(entry, list)
        for item in entry:
            assert set(item.keys()) == {"topic_id", "weight"}
            assert isinstance(item["topic_id"], int)
            assert isinstance(item["weight"], float)

    topic_ids = {item["topic_id"] for entry in values for item in entry}
    assert topic_ids.issubset(set(topics.keys()))
