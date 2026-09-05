import math

import polars as pl
import pytest

import polars_text as pt


def test_token_frequencies_counts_repeated_words() -> None:
    series = pl.Series("text", ["Hello world", "Hello there"])
    freqs = pt.token_frequencies(series, model="native:plain_words_en")
    assert freqs["hello"] == 2
    assert freqs["world"] == 1
    assert freqs["there"] == 1


def test_token_frequencies_accepts_plain_words_en_model() -> None:
    series = pl.Series("text", ["Hello, [UNK] ##sta Queensland"])
    freqs = pt.token_frequencies(series, model="native:plain_words_en")
    assert freqs == {"hello": 1, "sta": 1, "queensland": 1}


def test_token_frequencies_does_not_materialize_a_python_list(monkeypatch) -> None:
    def fail_to_list(self):
        raise AssertionError("token_frequencies must not call Series.to_list")

    monkeypatch.setattr(pl.Series, "to_list", fail_to_list)
    series = pl.concat(
        [pl.Series("text", ["Hello", None]), pl.Series("text", ["world", "hello"])],
        rechunk=False,
    )
    assert series.n_chunks() == 2
    assert pt.token_frequencies(series, model="native:plain_words_en") == {
        "hello": 2,
        "world": 1,
    }


def test_token_frequency_stats_aligns_union_and_expected_counts() -> None:
    freqs_0 = {"hello": 2, "world": 1}
    freqs_1 = {"hello": 1, "there": 2}
    stats = pt.token_frequency_stats(freqs_0, freqs_1)
    assert stats.select(
        "token",
        "freq_corpus_0",
        "freq_corpus_1",
        "expected_0",
        "expected_1",
        "corpus_0_total",
        "corpus_1_total",
    ).sort("token").rows() == [
        ("hello", 2, 1, 1.5, 1.5, 3, 3),
        ("there", 0, 2, 1.0, 1.0, 3, 3),
        ("world", 1, 0, 0.5, 0.5, 3, 3),
    ]


def test_token_frequency_stats_uses_ucrel_relative_measures() -> None:
    stats = pt.token_frequency_stats(
        {"shared": 2, "target_only": 1},
        {"shared": 1, "reference_only": 2},
    ).sort("token")

    rows = {row["token"]: row for row in stats.to_dicts()}
    assert rows["shared"]["percent_diff"] == pytest.approx(100.0)
    assert rows["shared"]["log_ratio"] == pytest.approx(1.0)
    assert rows["reference_only"]["percent_diff"] == pytest.approx(-100.0)
    assert rows["reference_only"]["log_ratio"] == pytest.approx(-2.0)
    assert rows["target_only"]["log_ratio"] == pytest.approx(1.0)
    assert math.isinf(rows["target_only"]["relative_risk"])


def test_token_frequency_stats_rejects_non_integer_counts() -> None:
    with pytest.raises(TypeError, match="nonnegative integer"):
        pt.token_frequency_stats({"bad": 1.5}, {"ok": 1})  # ty: ignore[invalid-argument-type]


def test_token_frequency_stats_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        pt.token_frequency_stats({"bad": -1}, {"ok": 1})


def test_token_frequency_stats_rejects_counts_outside_int64() -> None:
    with pytest.raises(ValueError, match="Int64"):
        pt.token_frequency_stats({"bad": 2**63}, {"ok": 1})


def test_token_frequency_stats_rejects_one_empty_corpus() -> None:
    with pytest.raises(ValueError, match="positive total"):
        pt.token_frequency_stats({"present": 1}, {})


def test_token_frequency_stats_empty_schema_is_typed() -> None:
    stats = pt.token_frequency_stats({}, {})
    assert stats.schema == {
        "token": pl.String,
        "freq_corpus_0": pl.Int64,
        "freq_corpus_1": pl.Int64,
        "expected_0": pl.Float64,
        "expected_1": pl.Float64,
        "corpus_0_total": pl.Int64,
        "corpus_1_total": pl.Int64,
        "log_likelihood_llv": pl.Float64,
        "bayes_factor_bic": pl.Float64,
        "effect_size_ell": pl.Float64,
        "significance": pl.String,
        "percent_corpus_0": pl.Float64,
        "percent_corpus_1": pl.Float64,
        "percent_diff": pl.Float64,
        "relative_risk": pl.Float64,
        "log_ratio": pl.Float64,
        "odds_ratio": pl.Float64,
    }
