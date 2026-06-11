"""Phase 2 manual evaluation harness for the Rust topic-modeling pipeline.

This is **not** part of the automated test suite. The embedding + PaCMAP +
HDBSCAN stages are non-deterministic, so per the plan (feedback #5) we validate
quality by hand on labeled corpora rather than asserting on output in CI.

What it does:
  1. Downloads a labeled dataset (see ``corpora.py``) and samples ``--limit``
     documents (seeded shuffle, optionally class-balanced).
  2. Runs the ``pl.col(...).text.topic_modeling`` Polars expression on them
     (the same entry point the backend uses) and reconstructs the per-topic /
     per-document result dict from its per-row struct output.
  3. Scores the per-document *dominant* topic against ground-truth labels with
     Adjusted Rand Index, NMI, and homogeneity/completeness/V-measure.
  4. Prints discovered topics with their c-TF-IDF keywords and the dominant
     true label per topic (cluster purity).
  5. Reports long-text diagnostics: chunks/doc and the share of documents whose
     topic *distribution* is genuinely multi-topic (the long-text value prop).

Run it (after ``maturin develop --release`` so ``polars_text`` is importable):

    cd polars-text
    HF_HUB_OFFLINE=0 uv run --with datasets --with scikit-learn \
        python experiments/topic_eval/eval.py --dataset bbc --limit 400

After the first download, re-run fully offline with ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from typing import Any, cast

# corpora.py lives next to this file; append (not insert) so it never shadows
# the Hugging Face ``datasets`` library on sys.path.
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent))
from corpora import get_spec  # noqa: E402  (local module)


def _load_documents(spec, limit: int, balanced: bool, seed: int):
    """Download + sample ``limit`` docs from ``spec``; return (texts, labels).

    Uses a seeded shuffle for reproducibility. With ``balanced`` we take an
    equal number per class so a dominant category can't skew the metrics.
    Called once from :func:`main` before the pipeline runs.
    """
    from datasets import load_dataset  # heavy import kept local

    ds = load_dataset(spec.hf_id, split=spec.split)
    ds = ds.shuffle(seed=seed)

    texts: list[str] = []
    labels: list[int] = []
    if balanced:
        per_class: dict[int, int] = defaultdict(int)
        # Aim for an even split across the classes actually present.
        n_classes = len(set(ds[spec.label_field]))
        cap = max(1, limit // max(1, n_classes))
        for row in ds:
            lbl = int(row[spec.label_field])
            if per_class[lbl] >= cap:
                continue
            txt = (row[spec.text_field] or "").strip()
            if not txt:
                continue
            texts.append(txt)
            labels.append(lbl)
            per_class[lbl] += 1
            if len(texts) >= limit:
                break
    else:
        for row in ds:
            txt = (row[spec.text_field] or "").strip()
            if not txt:
                continue
            texts.append(txt)
            labels.append(int(row[spec.label_field]))
            if len(texts) >= limit:
                break
    return texts, labels


def _stopwords_for(lang: str | None) -> list[str]:
    """Return a basic stopword list for ``lang`` (currently EN only).

    Pulled from scikit-learn so the harness has no extra dependency. CJK
    stopwording is left to the lindera vectorizer, so ``None``/``zh`` -> [].
    """
    if lang == "en":
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return sorted(ENGLISH_STOP_WORDS)
    return []


def _run_rust(texts, spec, args):
    """Run the Rust ``.text.topic_modeling`` expression; return (result, secs).

    The expression emits one struct per document (1:1 with the input rows). We
    reconstruct the result dict the rest of this harness consumes -- ``documents``
    (per-doc dominant topic + distribution) and ``topics`` (per-topic keywords,
    soft size, PaCMAP coords) -- with a plain ``group_by('dominant_topic')``,
    exactly the way the backend worker does. A single corpus is enough for
    evaluation; the app's two-corpus split only affects per-corpus sizes.

    ``cast(Any, ...)`` is needed because the dynamically-registered ``.text``
    namespace is invisible to the static type checker (same pattern the backend
    uses).
    """
    import polars as pl
    import polars_text  # noqa: F401  (importing registers the `.text` namespace)

    frame = pl.DataFrame({"__doc__": texts})
    expr = cast(Any, pl.col("__doc__")).text.topic_modeling(
        embedder_model=args.embedder,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        reduce_dims=args.reduce_dims,
        seed=args.seed,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        top_k=args.top_k,
        vectorizer_model=spec.vectorizer_model,
        lowercase=True,
        stopwords=_stopwords_for(spec.stopwords_lang) or None,
    )
    t0 = time.perf_counter()
    res = frame.select(expr.alias("__topic__")).unnest("__topic__")
    elapsed = time.perf_counter() - t0
    return _reconstruct_result(res), elapsed


def _reconstruct_result(res):
    """Rebuild the harness result dict from the per-row expression output.

    Mirrors the backend reconstruction: every row carries its dominant topic +
    full distribution, plus the keywords/coords replicated under its dominant
    topic and the global ``n_topics``/``n_chunks``. We roll the rows up by
    dominant topic into ``topics`` and compute a soft size = the summed document
    proportions assigned to each topic. ``chunk_count`` is estimated from the
    global chunk total (the per-row struct deliberately omits per-topic chunk
    counts); it is only an annotation on the bubble plot, never a metric.
    """
    rows = res.to_dicts()
    if not rows:
        return {"documents": [], "topics": [], "n_topics": 0, "n_chunks": 0}

    n_chunks = int(rows[0]["n_chunks"])
    n_topics = int(rows[0]["n_topics"])

    documents = []
    soft_size: dict[int, float] = defaultdict(float)
    # First row seen for each dominant topic supplies its keywords + coords.
    meta: dict[int, tuple[list[str], float, float]] = {}
    for i, row in enumerate(rows):
        dist = [
            (int(d["topic_id"]), float(d["proportion"]))
            for d in (row["topic_distribution"] or [])
        ]
        documents.append(
            {
                "doc_index": i,
                "dominant_topic": int(row["dominant_topic"]),
                "topic_distribution": dist,
            }
        )
        for tid, prop in dist:
            soft_size[tid] += prop
        dom = int(row["dominant_topic"])
        if dom not in meta:
            meta[dom] = (
                list(row["representative_words"] or []),
                float(row["x"]),
                float(row["y"]),
            )

    total_soft = sum(soft_size.values()) or 1.0
    topics = []
    for tid in sorted(soft_size):
        words, x, y = meta.get(tid, ([], 0.0, 0.0))
        size = soft_size[tid]
        topics.append(
            {
                "id": tid,
                "representative_words": words,
                "x": x,
                "y": y,
                "total_size": size,
                "chunk_count": round(n_chunks * size / total_soft),
            }
        )

    return {
        "documents": documents,
        "topics": topics,
        "n_topics": n_topics,
        "n_chunks": n_chunks,
    }


def _distribution_entropy(distribution) -> float:
    """Shannon entropy (nats) of a document's topic distribution.

    > 0 means the document spreads across multiple topics -- the long-text
    behaviour the single-topic BERTopic path could not represent. Excludes the
    outlier topic (-1) so noise chunks don't count as a "second topic".
    """
    probs = [p for tid, p in distribution if tid != -1 and p > 0.0]
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    return -sum((p / total) * math.log(p / total) for p in probs)


def _score(pred_labels, true_labels) -> dict[str, float]:
    """Clustering metrics comparing predicted vs ground-truth labels.

    Outlier topic ``-1`` is kept as its own predicted cluster so penalising
    excessive noise is reflected in the scores.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
        v_measure_score,
    )

    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "NMI": normalized_mutual_info_score(true_labels, pred_labels),
        "homogeneity": homogeneity_score(true_labels, pred_labels),
        "completeness": completeness_score(true_labels, pred_labels),
        "v_measure": v_measure_score(true_labels, pred_labels),
    }


def _print_topics(result, true_labels):
    """Print each discovered topic: keywords + dominant true label (purity).

    Helps eyeball whether clusters are coherent and map onto real categories.
    """
    # Map doc_index -> true label to compute per-topic label composition.
    dominant = {d["doc_index"]: d["dominant_topic"] for d in result["documents"]}
    members: dict[int, list[int]] = defaultdict(list)
    for doc_index, topic in dominant.items():
        members[topic].append(true_labels[doc_index])

    print(f"\nDiscovered {result['n_topics']} topic(s) "
          f"from {result['n_chunks']} chunks:")
    for topic in sorted(result["topics"], key=lambda t: -t["total_size"]):
        tid = topic["id"]
        words = ", ".join(topic["representative_words"][:8])
        labels_here = members.get(tid, [])
        purity_str = ""
        if labels_here:
            top_label, count = Counter(labels_here).most_common(1)[0]
            purity = count / len(labels_here)
            purity_str = (f"  dom_label={top_label} purity={purity:.2f} "
                          f"n={len(labels_here)}")
        tag = " (outliers)" if tid == -1 else ""
        print(f"  topic {tid:>3}{tag}: soft_size={topic['total_size']:.1f} "
              f"chunks={topic['chunk_count']}{purity_str}")
        print(f"            words: {words}")


def _render_plot(result, path, title="topic modeling"):
    """Render a topic bubble chart (PaCMAP 2D centroids) to ``path`` as PNG.

    Each bubble is one discovered topic positioned at its ``x``/``y`` PaCMAP
    coordinate, sized by soft document size (``total_size``, scaled by chunk
    count for legibility), and annotated with its top c-TF-IDF keywords. The
    outlier topic (-1) is drawn small and grey. This mirrors the app's bubble
    chart so the user can eyeball cluster separation and labels at a glance.
    Called by ``main()`` when ``--plot`` is given.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt

    topics = [t for t in result["topics"] if t["id"] != -1]
    outliers = [t for t in result["topics"] if t["id"] == -1]
    if not topics:
        print("[plot] no non-outlier topics to draw; skipping")
        return

    sizes = [max(1.0, t["total_size"]) for t in topics]
    smax = max(sizes)
    # Area-proportional bubbles, clamped to a readable range.
    areas = [400 + 6000 * (s / smax) for s in sizes]
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(13, 9))
    for i, t in enumerate(topics):
        ax.scatter(t["x"], t["y"], s=areas[i], color=cmap(i % 20),
                   alpha=0.6, edgecolors="black", linewidths=0.6, zorder=2)
        words = " / ".join(t["representative_words"][:3])
        ax.annotate(f"#{t['id']} {words}\n(size {t['total_size']:.0f}, "
                    f"{t['chunk_count']} chunks)",
                    (t["x"], t["y"]), fontsize=8, ha="center", va="center",
                    zorder=3)
    for t in outliers:
        ax.scatter(t["x"], t["y"], s=120, color="lightgrey",
                   alpha=0.5, edgecolors="grey", zorder=1)
        ax.annotate(f"outliers ({t['chunk_count']} chunks)", (t["x"], t["y"]),
                    fontsize=7, ha="center", va="center", color="grey")

    ax.set_title(f"{title}\n{len(topics)} topics, {result['n_chunks']} chunks "
                 f"(bubble area = soft document size)", fontsize=12)
    ax.set_xlabel("PaCMAP-1")
    ax.set_ylabel("PaCMAP-2")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bbc",
                        help="dataset key from datasets.REGISTRY (default: bbc)")
    parser.add_argument("--limit", type=int, default=400,
                        help="max documents to sample (default: 400)")
    parser.add_argument("--balanced", action="store_true",
                        help="sample an equal number of docs per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedder", default=None,
                        help="override embedder repo id (default: pipeline default)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--reduce-dims", type=int, default=5)
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="HDBSCAN minimum cluster size; the topic count is "
                             "whatever emerges (the only native topic-count control)")
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dump-json", default=None,
                        help="write the raw Rust result JSON to this path")
    parser.add_argument("--plot", default=None,
                        help="render a topic bubble-chart PNG to this path (needs matplotlib)")
    args = parser.parse_args(argv)

    spec = get_spec(args.dataset)
    print(f"== Topic-modeling evaluation ==")
    print(f"dataset : {args.dataset}  ({spec.hf_id})")
    print(f"note    : {spec.notes}")
    print(f"lang    : {spec.language}   vectorizer: {spec.vectorizer_model}")

    texts, true_labels = _load_documents(spec, args.limit, args.balanced, args.seed)
    n_true = len(set(true_labels))
    char_lens = [len(t) for t in texts]
    avg_len = sum(char_lens) / max(1, len(char_lens))
    print(f"loaded  : {len(texts)} docs, {n_true} true classes, "
          f"avg {avg_len:.0f} chars (max {max(char_lens)})")

    result, elapsed = _run_rust(texts, spec, args)

    # Long-text diagnostics: how much chunking + distribution actually happened.
    chunks_per_doc = result["n_chunks"] / max(1, len(texts))
    entropies = [_distribution_entropy(d["topic_distribution"]) for d in result["documents"]]
    multi_topic = sum(1 for e in entropies if e > 1e-6)
    print(f"\nRust pipeline: {elapsed:.1f}s, {result['n_chunks']} chunks "
          f"({chunks_per_doc:.2f}/doc), {result['n_topics']} topics")
    print(f"multi-topic docs (distribution entropy>0): "
          f"{multi_topic}/{len(texts)} ({100*multi_topic/max(1,len(texts)):.0f}%)")

    pred = [d["dominant_topic"] for d in sorted(result["documents"], key=lambda d: d["doc_index"])]
    ordered_true = [true_labels[d["doc_index"]] for d in sorted(result["documents"], key=lambda d: d["doc_index"])]
    metrics = _score(pred, ordered_true)
    print("\nRust vs ground truth:")
    for k, v in metrics.items():
        print(f"  {k:<13}: {v:.3f}")

    _print_topics(result, true_labels)

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"\n[dump] wrote raw result JSON -> {args.dump_json}")

    if args.plot:
        _render_plot(result, args.plot, title=f"{args.dataset} ({spec.hf_id})")
        print(f"[plot] wrote bubble chart -> {args.plot}")

    print("\nDone.")


if __name__ == "__main__":
    main()
