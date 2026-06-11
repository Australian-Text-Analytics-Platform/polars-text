"""Labeled dataset registry for the Phase 2 topic-modeling evaluation harness.

Why this exists: the Rust pipeline's clustering/embedding output is
non-deterministic, so it is deliberately kept out of CI (plan feedback #5).
Instead we validate it by hand against *labeled* public corpora and score the
per-document dominant topic against ground truth (Adjusted Rand Index etc.).

Each entry points at a clean, parquet-backed Hugging Face dataset that the
``datasets`` library can download directly (no arbitrary-code loader) and cache
locally, so repeat runs work fully offline (``HF_HUB_OFFLINE=1``).

Used by ``eval.py`` (the harness CLI) to load a corpus, map its text/label
columns, and pick the right CJK-aware c-TF-IDF vectorizer + stopwords.

Dataset choices (all carry ground-truth topic labels):
  - ``20ng``      : the canonical BERTopic benchmark (20 EN categories).
  - ``bbc``       : long, clean EN news articles (5 categories).
  - ``thucnews``  : long *Chinese* news articles (10 categories) -> exercises
                    the long-CJK path and the lindera vectorizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    """One labeled corpus plus how to feed it to ``run_topic_modeling``.

    Fields:
      hf_id            : Hugging Face dataset id (parquet, no loader script).
      split            : split to read (we only need one labeled split).
      text_field       : column holding the document text.
      label_field      : column holding the integer ground-truth topic.
      language         : informational ("en"/"zh").
      vectorizer_model : tokenizer id for c-TF-IDF term counting. CJK corpora
                         need a lindera dictionary; EN uses the native splitter.
      stopwords_lang   : which built-in stopword list ``eval.py`` should apply
                         (``None`` = no stopword filtering).
    """

    hf_id: str
    split: str
    text_field: str
    label_field: str
    language: str
    vectorizer_model: str
    stopwords_lang: str | None = None
    # Free-form notes surfaced in the harness banner.
    notes: str = ""
    label_names: dict[int, str] = field(default_factory=dict)


# native: / lindera: ids mirror polars_text.tokenizer model-id constants.
_PLAIN_WORDS_EN = "native:plain_words_en"
_LINDERA_ZH = "lindera:cc-cedict"


REGISTRY: dict[str, DatasetSpec] = {
    "20ng": DatasetSpec(
        hf_id="SetFit/20_newsgroups",
        split="train",
        text_field="text",
        label_field="label",
        language="en",
        vectorizer_model=_PLAIN_WORDS_EN,
        stopwords_lang="en",
        notes="20 Newsgroups - canonical BERTopic benchmark, 20 EN topics.",
    ),
    "bbc": DatasetSpec(
        hf_id="SetFit/bbc-news",
        split="train",
        text_field="text",
        label_field="label",
        language="en",
        vectorizer_model=_PLAIN_WORDS_EN,
        stopwords_lang="en",
        notes="BBC News - long clean EN articles, 5 topics.",
    ),
    "thucnews": DatasetSpec(
        hf_id="oyxy2019/THUCNewsText",
        split="train",
        text_field="text",
        label_field="label",
        language="zh",
        vectorizer_model=_LINDERA_ZH,
        stopwords_lang=None,
        notes="THUCNews - long Chinese news articles, 10 topics (lindera).",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``name`` or raise a helpful error.

    Called by ``eval.py`` right after argument parsing.
    """
    try:
        return REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(REGISTRY))
        raise SystemExit(f"unknown dataset {name!r}; choose one of: {valid}") from exc
