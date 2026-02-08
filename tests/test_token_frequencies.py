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
