from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import polars as pl

from ._internal import token_frequencies as _token_frequencies
from .namespace import _require_feature


def token_frequencies(series: pl.Series, model: str) -> dict[str, int]:
    _require_feature("tokenization", "token_frequencies")
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
    all_tokens = sorted(
        token
        for token in set(counts_0) | set(counts_1)
        if counts_0.get(token, 0) > 0 or counts_1.get(token, 0) > 0
    )
    data = [
        {
            "token": token,
            "freq_corpus_0": counts_0.get(token, 0),
            "freq_corpus_1": counts_1.get(token, 0),
        }
        for token in all_tokens
    ]

    if not data:
        return pl.DataFrame(
            schema={
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
        )

    df = pl.DataFrame(data)

    df = df.with_columns(
        [
            (pl.col("freq_corpus_0") + pl.col("freq_corpus_1")).alias("total_freq"),
            pl.col("freq_corpus_0").sum().alias("corpus_0_total"),
            pl.col("freq_corpus_1").sum().alias("corpus_1_total"),
        ]
    )

    grand_total = df.select(
        pl.col("corpus_0_total").first() + pl.col("corpus_1_total").first()
    ).item()
    corpus_totals = df.select(
        pl.col("corpus_0_total").first(), pl.col("corpus_1_total").first()
    ).row(0)
    if 0 in corpus_totals:
        raise ValueError("both corpora must have a positive total frequency")

    df = df.with_columns(
        [
            (pl.col("total_freq") * pl.col("corpus_0_total") / grand_total).alias(
                "expected_0"
            ),
            (pl.col("total_freq") * pl.col("corpus_1_total") / grand_total).alias(
                "expected_1"
            ),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("freq_corpus_0") > 0)
            .then(
                pl.col("freq_corpus_0")
                * (
                    pl.col("freq_corpus_0")
                    / pl.max_horizontal("expected_0", pl.lit(1e-10))
                ).log()
            )
            .otherwise(0.0)
            .alias("ll_0"),
            pl.when(pl.col("freq_corpus_1") > 0)
            .then(
                pl.col("freq_corpus_1")
                * (
                    pl.col("freq_corpus_1")
                    / pl.max_horizontal("expected_1", pl.lit(1e-10))
                ).log()
            )
            .otherwise(0.0)
            .alias("ll_1"),
        ]
    )

    df = df.with_columns(
        [
            (2 * (pl.col("ll_0") + pl.col("ll_1"))).alias("log_likelihood_llv"),
        ]
    )

    dof = 1
    df = df.with_columns(
        [
            (pl.col("log_likelihood_llv") - (dof * pl.lit(grand_total).log())).alias(
                "bayes_factor_bic"
            ),
        ]
    )

    df = df.with_columns(
        [pl.min_horizontal("expected_0", "expected_1").alias("min_expected")]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("min_expected") > 0)
            .then(
                pl.col("log_likelihood_llv")
                / (grand_total * pl.max_horizontal("min_expected", pl.lit(1e-10)).log())
            )
            .otherwise(0.0)
            .alias("effect_size_ell"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("log_likelihood_llv") >= 15.13)
            .then(pl.lit("****"))
            .when(pl.col("log_likelihood_llv") >= 10.83)
            .then(pl.lit("***"))
            .when(pl.col("log_likelihood_llv") >= 6.63)
            .then(pl.lit("**"))
            .when(pl.col("log_likelihood_llv") >= 3.84)
            .then(pl.lit("*"))
            .otherwise(pl.lit(""))
            .alias("significance"),
        ]
    )

    result = df.select(
        [
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
        ]
    )

    result = result.with_columns(
        [
            (pl.col("freq_corpus_0") / pl.col("corpus_0_total") * 100).alias(
                "percent_corpus_0"
            ),
            (pl.col("freq_corpus_1") / pl.col("corpus_1_total") * 100).alias(
                "percent_corpus_1"
            ),
            (
                (
                    (pl.col("freq_corpus_0") / pl.col("corpus_0_total"))
                    - (pl.col("freq_corpus_1") / pl.col("corpus_1_total"))
                )
                * 100
                / pl.when(pl.col("freq_corpus_1") == 0)
                .then(pl.lit(1e-18))
                .otherwise(pl.col("freq_corpus_1") / pl.col("corpus_1_total"))
            ).alias("percent_diff"),
            (
                (pl.col("freq_corpus_0") / pl.col("corpus_0_total"))
                / (pl.col("freq_corpus_1") / pl.col("corpus_1_total"))
            ).alias("relative_risk"),
            (
                (
                    pl.when(pl.col("freq_corpus_0") == 0)
                    .then(0.5)
                    .otherwise(pl.col("freq_corpus_0"))
                    / pl.col("corpus_0_total")
                ) / (
                    pl.when(pl.col("freq_corpus_1") == 0)
                    .then(0.5)
                    .otherwise(pl.col("freq_corpus_1"))
                    / pl.col("corpus_1_total")
                )
            )
            .log(2.0)
            .alias("log_ratio"),
            (
                (
                    pl.col("freq_corpus_0")
                    / (pl.col("corpus_0_total") - pl.col("freq_corpus_0"))
                ) / (
                    pl.col("freq_corpus_1")
                    / (pl.col("corpus_1_total") - pl.col("freq_corpus_1"))
                )
            ).alias("odds_ratio"),
        ]
    )

    return result


__all__ = ["token_frequencies", "token_frequency_stats"]
