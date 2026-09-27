from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import polars as pl

from .namespace import _require_feature


def token_frequencies(series: pl.Series, model: str) -> dict[str, int]:
    _require_feature("tokenization", "token_frequencies")
    from ._internal import token_frequencies as _token_frequencies
    if not isinstance(series, pl.Series):
        raise TypeError("token_frequencies expects a Polars Series")
    if not model.strip():
        raise ValueError("token_frequencies requires an explicit tokenizer model ID")
    return _token_frequencies(series, model.strip())


def token_frequency_stats(
    corpus_0: Mapping[str, int],
    corpus_1: Mapping[str, int],
) -> pl.DataFrame:
    def validated_counts(corpus: Mapping[str, int], name: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for token, count in corpus.items():
            if not isinstance(token, str):
                raise TypeError(f"{name} token keys must be strings")
            if isinstance(count, bool) or not isinstance(count, Integral):
                raise TypeError(f"{name}[{token!r}] must be a nonnegative integer")
            value = int(count)
            if value < 0:
                raise ValueError(f"{name}[{token!r}] must be nonnegative")
            if value > 2**63 - 1:
                raise ValueError(f"{name}[{token!r}] exceeds the Int64 output range")
            counts[token] = value
        return counts

    counts_0 = validated_counts(corpus_0, "corpus_0")
    counts_1 = validated_counts(corpus_1, "corpus_1")
    from ._internal import frequency_stats
    rows = frequency_stats(counts_0, counts_1)
    return pl.DataFrame(rows, schema={
        "token": pl.String,
        "freq_corpus_0": pl.Int64, "freq_corpus_1": pl.Int64,
        "expected_0": pl.Float64, "expected_1": pl.Float64,
        "corpus_0_total": pl.Int64, "corpus_1_total": pl.Int64,
        "log_likelihood_llv": pl.Float64, "bayes_factor_bic": pl.Float64,
        "effect_size_ell": pl.Float64, "significance": pl.String,
        "percent_corpus_0": pl.Float64, "percent_corpus_1": pl.Float64,
        "percent_diff": pl.Float64, "relative_risk": pl.Float64,
        "log_ratio": pl.Float64, "odds_ratio": pl.Float64,
    })


__all__ = ["token_frequencies", "token_frequency_stats"]
