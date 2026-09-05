"""Native quotation contracts; real-model tests use an explicitly provisioned asset."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

import polars_text  # noqa: F401

MODEL_ENV = "WORDFLOW_TEST_UDPIPE_MODEL"


@pytest.fixture
def model():
    path = os.environ.get(MODEL_ENV)
    if not path:
        pytest.skip(f"Set {MODEL_ENV} to the pinned English EWT model")
    assert Path(path).is_file()
    return path


def extract(texts, model):
    return (
        pl.DataFrame({"text": texts}, schema={"text": pl.String})
        .select(
            cast(Any, pl.col("text"))
            .text.quotation(model_path=model)
            .alias("quotation")
        )["quotation"]
        .to_list()
    )


def test_quotation_schema_is_lazy_and_empty_inputs_need_no_model():
    expr = cast(Any, pl.col("text")).text.quotation(model_path="/missing/model")
    frame = pl.DataFrame({"text": [None, "", "  "]}, schema={"text": pl.String})
    assert (
        frame.lazy().select(expr).collect_schema()["text"]
        == frame.select(expr).schema["text"]
    )
    assert frame.select(expr)["text"].to_list() == [[], [], []]
    assert frame.head(0).select(expr).height == 0


def test_missing_and_corrupt_models_raise_compute_errors(tmp_path):
    with pytest.raises(pl.exceptions.ComputeError, match="model I/O"):
        extract(["some nonempty text"], str(tmp_path / "missing"))
    corrupt = tmp_path / "bad.udpipe"
    corrupt.write_bytes(b"not a model")
    with pytest.raises(pl.exceptions.ComputeError, match="Cannot load UDPipe"):
        extract(["some nonempty text"], str(corrupt))


def test_nonstring_and_empty_model_arguments_are_rejected():
    with pytest.raises(ValueError, match="model_path"):
        cast(Any, pl.col("text")).text.quotation(model_path="")
    with pytest.raises(pl.exceptions.ComputeError, match="String input"):
        pl.DataFrame({"text": [1]}).lazy().select(
            cast(Any, pl.col("text")).text.quotation(model_path="/missing")
        ).collect_schema()


_CORPUS = json.loads(
    (Path(__file__).parent / "fixtures/quotation_spacy_baseline.json").read_text()
)["cases"]
# These are category/speaker expectations for the supported Rust behavior, not
# exact spaCy output snapshots. Source spans are verified independently below.
_EXPECTATIONS = [
    ("speaker-before", [("Alice", "SVQCQ", False)]),
    ("speaker-after", [("Alice", "QCQVS", False)]),
    ("indirect-multiword-speaker", [("Company director Alice Smith", "SVC", False)]),
    ("according-to-before", [("Alice", "AccordingTo", False)]),
    ("according-to-after", [("Alice", "AccordingTo", False)]),
    ("floating", [("Alice", "QCQSV", False), ("Alice", "QCQ", True)]),
    (
        "floating-multiple-sentences",
        [("Alice", "QCQSV", False), ("Alice", "QCQ", True)],
    ),
    ("heuristic-sign", [(None, "Heuristic", False)]),
    ("heuristic-unattributed", [(None, "Heuristic", False)]),
    ("accents-curly-quotes", [("José", "SVQCQ", False)]),
    ("emoji", [("Alice", "SVQCQ", False)]),
    ("contraction", [("Alice", "SVQCQ", False)]),
    ("newline", [("Alice", "SVC", False), ("Bob", "SVC", False)]),
    ("repeated-spaces", [("Alice", "SVQCQ", False)]),
    ("no-quotation", []),
    ("empty", []),
]


@pytest.mark.model
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(case["text"], expected, id=name)
        for case, (name, expected) in zip(_CORPUS, _EXPECTATIONS, strict=True)
    ],
)
def test_real_quotation_categories_and_original_spans(model, text, expected):
    quotes = extract([text], model)[0]
    assert [
        (q["speaker"], q["quote_type"], q["is_floating_quote"]) for q in quotes
    ] == expected
    for index, quote in enumerate(quotes):
        assert quote["quote_row_idx"] == index
        assert quote["quote_token_count"] > 0
        for label in ("quote", "speaker", "verb"):
            start, end = quote[f"{label}_start_idx"], quote[f"{label}_end_idx"]
            if quote[label] is None:
                assert start is None and end is None
            else:
                assert 0 <= start < end <= len(text)
                assert text[start:end] == quote[label]


@pytest.mark.model
def test_duplicates_chunks_and_thread_confined_models(model):
    text = 'Alice said, "The project will finish tomorrow morning."'
    frame = pl.concat(
        [pl.DataFrame({"text": [text, None]}), pl.DataFrame({"text": [text, ""]})],
        rechunk=False,
    )
    result = frame.select(cast(Any, pl.col("text")).text.quotation(model_path=model))[
        "text"
    ].to_list()
    assert result[0] == result[2]
    assert result[1] == result[3] == []
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: extract([text], model), range(8)))
    assert all(output == [result[0]] for output in outputs)


@pytest.mark.model
def test_unicode_repeated_words_and_embedded_null_do_not_truncate(model):
    texts = [
        "🙂 José said, “The café will remain open until tomorrow.”",
        'Alice said, "We cannot finish the project before tomorrow"',
        'Noise\x00 Alice said, "The project will finish tomorrow morning."',
        'Alice said, "The project will finish tomorrow morning." Alice said, "The project will finish tomorrow morning."',
    ]
    for text, quotes in zip(texts, extract(texts, model), strict=True):
        assert quotes
        for quote in quotes:
            assert (
                text[quote["quote_start_idx"] : quote["quote_end_idx"]]
                == quote["quote"]
            )
    assert len(extract([texts[-1]], model)[0]) == 2
