from __future__ import annotations

from typing import Mapping

import polars as pl

from ._internal import token_frequencies as _token_frequencies


def token_frequencies(series: pl.Series) -> dict[str, int]:
    if not isinstance(series, pl.Series):
        raise TypeError("token_frequencies expects a Polars Series")
    texts = [value if value is not None else "" for value in series.to_list()]
    return _token_frequencies(texts)


def token_frequency_stats(
    corpus_0: Mapping[str, int],
    corpus_1: Mapping[str, int],
) -> pl.DataFrame:
    all_tokens = sorted(set(corpus_0) | set(corpus_1))
    data = []
    for token in all_tokens:
        data.append({
            "token": token,
            "freq_corpus_0": int(corpus_0.get(token, 0)),
            "freq_corpus_1": int(corpus_1.get(token, 0)),
        })

    if not data:
        return pl.DataFrame({
            "token": [],
            "freq_corpus_0": [],
            "freq_corpus_1": [],
            "expected_0": [],
            "expected_1": [],
            "corpus_0_total": [],
            "corpus_1_total": [],
            "log_likelihood_llv": [],
            "bayes_factor_bic": [],
            "effect_size_ell": [],
            "significance": [],
            "percent_corpus_0": [],
            "percent_corpus_1": [],
            "percent_diff": [],
            "relative_risk": [],
            "log_ratio": [],
            "odds_ratio": [],
        })

    df = pl.DataFrame(data)

    df = df.with_columns([
        (pl.col("freq_corpus_0") + pl.col("freq_corpus_1")).alias("total_freq"),
        pl.col("freq_corpus_0").sum().alias("corpus_0_total"),
        pl.col("freq_corpus_1").sum().alias("corpus_1_total"),
    ])

    grand_total = df.select(
        pl.col("corpus_0_total").first() + pl.col("corpus_1_total").first()
    ).item()

    df = df.with_columns([
        (pl.col("total_freq") * pl.col("corpus_0_total") / grand_total).alias(
            "expected_0"
        ),
        (pl.col("total_freq") * pl.col("corpus_1_total") / grand_total).alias(
            "expected_1"
        ),
    ])

    df = df.with_columns([
        pl
        .when(pl.col("freq_corpus_0") > 0)
        .then(
            pl.col("freq_corpus_0")
            * (
                pl.col("freq_corpus_0") / pl.max_horizontal("expected_0", pl.lit(1e-10))
            ).log()
        )
        .otherwise(0.0)
        .alias("ll_0"),
        pl
        .when(pl.col("freq_corpus_1") > 0)
        .then(
            pl.col("freq_corpus_1")
            * (
                pl.col("freq_corpus_1") / pl.max_horizontal("expected_1", pl.lit(1e-10))
            ).log()
        )
        .otherwise(0.0)
        .alias("ll_1"),
    ])

    df = df.with_columns([
        (2 * (pl.col("ll_0") + pl.col("ll_1"))).alias("log_likelihood_llv"),
    ])

    dof = 1
    df = df.with_columns([
        (pl.col("log_likelihood_llv") - (dof * pl.lit(grand_total).log())).alias(
            "bayes_factor_bic"
        ),
    ])

    df = df.with_columns([
        pl.min_horizontal("expected_0", "expected_1").alias("min_expected")
    ])

    df = df.with_columns([
        pl
        .when(pl.col("min_expected") > 0)
        .then(
            pl.col("log_likelihood_llv")
            / (grand_total * pl.max_horizontal("min_expected", pl.lit(1e-10)).log())
        )
        .otherwise(0.0)
        .alias("effect_size_ell"),
    ])

    df = df.with_columns([
        pl
        .when(pl.col("log_likelihood_llv") >= 15.13)
        .then(pl.lit("****"))
        .when(pl.col("log_likelihood_llv") >= 10.83)
        .then(pl.lit("***"))
        .when(pl.col("log_likelihood_llv") >= 6.63)
        .then(pl.lit("**"))
        .when(pl.col("log_likelihood_llv") >= 3.84)
        .then(pl.lit("*"))
        .otherwise(pl.lit(""))
        .alias("significance"),
    ])

    result = df.select([
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
    ])

    # Normalised frequencies — used as the building blocks for %DIFF /
    # relative risk / log ratio below. Corpus 0 is conventionally the
    # reference and corpus 1 the studied corpus (the radio-selected
    # block in the Wordflow UI), so the direction-sensitive ratios put
    # corpus 1 in the numerator: positive %DIFF / LogRatio = "more
    # frequent in the studied corpus".
    nf_0 = pl.col("freq_corpus_0") / pl.col("corpus_0_total")
    nf_1 = pl.col("freq_corpus_1") / pl.col("corpus_1_total")
    result = result.with_columns([
        (nf_0 * 100).alias("percent_corpus_0"),
        (nf_1 * 100).alias("percent_corpus_1"),
        # %DIFF per Gabrielatos & Marchi (2012):
        #   ((NF_studied - NF_reference) / NF_reference) * 100
        # When the token is absent from the reference corpus the
        # quantity is theoretically infinite; emit `+Inf` (the API
        # layer's _safe_float converts it to the "+Inf" string the
        # frontend already renders as "+∞").
        pl
        .when(pl.col("freq_corpus_0") == 0)
        .then(
            pl
            .when(pl.col("freq_corpus_1") > 0)
            .then(pl.lit(float("inf")))
            .otherwise(None)
        )
        .otherwise(((nf_1 - nf_0) / nf_0) * 100)
        .alias("percent_diff"),
        pl
        .when(pl.col("freq_corpus_1") > 0)
        .then(nf_0 / nf_1)
        .otherwise(None)
        .alias("relative_risk"),
        # LogRatio per Hardie (CASS blog, 2014): binary log of the
        # relative risk with the studied corpus on top, so positive
        # values point at the studied block (matches the doc + the
        # significance interpretation in the keyness table).
        pl
        .when((pl.col("freq_corpus_0") > 0) & (pl.col("freq_corpus_1") > 0))
        .then((nf_1 / nf_0).log(2.0))
        .otherwise(None)
        .alias("log_ratio"),
        pl
        .when(
            (pl.col("freq_corpus_0") > 0)
            & (pl.col("freq_corpus_1") > 0)
            & (pl.col("corpus_1_total") > pl.col("freq_corpus_1"))
            & (pl.col("corpus_0_total") > pl.col("freq_corpus_0"))
        )
        .then(
            (
                pl.col("freq_corpus_0")
                * (pl.col("corpus_1_total") - pl.col("freq_corpus_1"))
            )
            / (
                pl.col("freq_corpus_1")
                * (pl.col("corpus_0_total") - pl.col("freq_corpus_0"))
            )
        )
        .otherwise(None)
        .alias("odds_ratio"),
    ])

    return result


__all__ = ["token_frequencies", "token_frequency_stats"]
