"""Phase 0.2 golden snapshots for the EN literary fixture.

Re-runs `polars_text.token_frequencies` and `text.concordance` against
`tests/fixtures/multilingual/literary/en.csv` and compares to committed
golden CSVs. Any drift (refactor change, polars dep update, fixture
edit) makes the test fail loudly.

Fixtures live in the parent ``ldaca_web_app`` repo, NOT this submodule,
so the test auto-skips when polars-text is checked out standalone (CI
on this repo) — it only runs when polars-text is mounted as a submodule
of ldaca_web_app.

To regenerate the goldens after an intentional change:

    cd polars-text && uv run python tests/test_golden_en_baseline.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

import polars_text as pt

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_EN = REPO_ROOT / "tests" / "fixtures" / "multilingual" / "literary" / "en.csv"
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "multilingual" / "golden"

# Skip the whole module when run standalone (no parent ldaca_web_app fixtures).
pytestmark = pytest.mark.skipif(
    not FIXTURE_EN.is_file(),
    reason=(
        "EN literary fixture lives in the parent ldaca_web_app repo; "
        "test only runs when polars-text is mounted as a submodule."
    ),
)

CONCORDANCE_KEYWORD = "time"
TOP_K = 50
CONTEXT_TOKENS = 5


def _compute_token_frequency_top_k() -> pl.DataFrame:
    df = pl.read_csv(FIXTURE_EN)
    freq = pt.token_frequencies(df["text"])
    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K]
    return pl.DataFrame(
        {"token": [t for t, _ in top], "frequency": [f for _, f in top]}
    )


def _compute_concordance() -> pl.DataFrame:
    df = pl.read_csv(FIXTURE_EN)
    concordance_expr = cast(Any, pl.col("text")).text.concordance(
        CONCORDANCE_KEYWORD,
        num_left_tokens=CONTEXT_TOKENS,
        num_right_tokens=CONTEXT_TOKENS,
    )
    return cast(
        pl.DataFrame,
        df.lazy()
        .with_columns(concordance_expr.alias("kwic"))
        .filter(pl.col("kwic").list.len() > 0)
        .explode("kwic")
        .unnest("kwic")
        .drop("text")
        .collect(),
    )


def regenerate() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    freq_df = _compute_token_frequency_top_k()
    freq_path = GOLDEN_DIR / "token_frequency_literary_en_top50.csv"
    freq_df.write_csv(freq_path)
    print(f"wrote {freq_path.relative_to(REPO_ROOT)}  rows={len(freq_df)}")

    kwic_df = _compute_concordance()
    kwic_path = GOLDEN_DIR / f"concordance_literary_en_{CONCORDANCE_KEYWORD}.csv"
    kwic_df.write_csv(kwic_path)
    print(f"wrote {kwic_path.relative_to(REPO_ROOT)}  rows={len(kwic_df)}")


def test_token_frequency_baseline_en_literary() -> None:
    actual_csv = _compute_token_frequency_top_k().write_csv()
    golden_path = GOLDEN_DIR / "token_frequency_literary_en_top50.csv"
    golden_csv = golden_path.read_text()
    assert actual_csv == golden_csv, (
        "Token-frequency baseline drift on literary/en.csv. "
        f"If the change is intentional, regenerate with "
        f"`uv run python {Path(__file__).relative_to(REPO_ROOT)}`."
    )


def test_concordance_baseline_en_literary() -> None:
    actual_csv = _compute_concordance().write_csv()
    golden_path = (
        GOLDEN_DIR / f"concordance_literary_en_{CONCORDANCE_KEYWORD}.csv"
    )
    golden_csv = golden_path.read_text()
    assert actual_csv == golden_csv, (
        f"Concordance baseline drift on literary/en.csv (keyword={CONCORDANCE_KEYWORD!r}). "
        f"If the change is intentional, regenerate with "
        f"`uv run python {Path(__file__).relative_to(REPO_ROOT)}`."
    )


if __name__ == "__main__":
    regenerate()
