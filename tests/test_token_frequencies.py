import math

import polars as pl
import polars_text as pt


def test_token_frequencies_returns_dict() -> None:
    series = pl.Series("text", ["Hello world", "Hello there"])
    freqs = pt.token_frequencies(series)
    assert isinstance(freqs, dict)
    assert freqs["hello"] == 2
    assert freqs["world"] == 1
    assert freqs["there"] == 1


def test_token_frequency_stats_columns() -> None:
    freqs_0 = {"hello": 2, "world": 1}
    freqs_1 = {"hello": 1, "there": 2}
    stats = pt.token_frequency_stats(freqs_0, freqs_1)
    assert set(stats.columns) == {
        "token",
        "freq_corpus_0",
        "freq_corpus_1",
        "expected_0",
        "expected_1",
        "corpus_0_total",
        "corpus_1_total",
        "log_likelihood_llv",
        "bayes_factor_bic",
        "effect_size_ell",
        "significance",
        "percent_corpus_0",
        "percent_corpus_1",
        "percent_diff",
        "relative_risk",
        "log_ratio",
        "odds_ratio",
    }
    assert stats.height == 3


def test_percent_diff_matches_gabrielatos_marchi_formula() -> None:
    # Reference corpus (corpus_0): 1000 tokens, target word appears 5 times.
    # Studied corpus (corpus_1):  1000 tokens, target word appears 10 times.
    # NF_ref = 0.005, NF_studied = 0.01.
    # %DIFF per Gabrielatos & Marchi: ((0.01 - 0.005) / 0.005) * 100 = +100.0
    freqs_ref = {"alpha": 5, **{f"f{i}": 1 for i in range(995)}}
    freqs_studied = {"alpha": 10, **{f"f{i}": 1 for i in range(990)}}
    stats = pt.token_frequency_stats(freqs_ref, freqs_studied)
    alpha = stats.filter(pl.col("token") == "alpha").to_dicts()[0]
    assert math.isclose(alpha["percent_diff"], 100.0, rel_tol=1e-9)


def test_log_ratio_is_log2_with_studied_corpus_in_numerator() -> None:
    # Same setup as above: NF_studied / NF_ref = 2 → log_2(2) = 1.0.
    freqs_ref = {"alpha": 5, **{f"f{i}": 1 for i in range(995)}}
    freqs_studied = {"alpha": 10, **{f"f{i}": 1 for i in range(990)}}
    stats = pt.token_frequency_stats(freqs_ref, freqs_studied)
    alpha = stats.filter(pl.col("token") == "alpha").to_dicts()[0]
    assert math.isclose(alpha["log_ratio"], 1.0, rel_tol=1e-9)


def test_percent_diff_reference_absent_emits_positive_infinity() -> None:
    # Token only in the studied corpus → relative growth is unbounded;
    # the API layer converts this to the "+Inf" string the UI renders.
    freqs_ref = {"only_ref": 3}
    freqs_studied = {"only_ref": 3, "studied_only": 4}
    stats = pt.token_frequency_stats(freqs_ref, freqs_studied)
    only_studied = stats.filter(pl.col("token") == "studied_only").to_dicts()[0]
    assert math.isinf(only_studied["percent_diff"])
    assert only_studied["percent_diff"] > 0
