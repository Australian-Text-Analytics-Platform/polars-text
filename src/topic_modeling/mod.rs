//! Rust topic-modeling pipeline — an offline, long-text replacement for the
//! Python BERTopic path.
//!
//! Pipeline (one uniform path for short and long text alike):
//!   1. `chunking`  — split each document into token-budgeted semantic chunks
//!      (a short document is simply one chunk).
//!   2. `embedding` — candle multilingual sentence embeddings per chunk.
//!   3. `reduce`    — PaCMAP dimensionality reduction for clusterability.
//!   4. `cluster`   — HDBSCAN groups chunks into topics (with `-1` outliers).
//!   5. `ctfidf`    — c-TF-IDF keyword labels per topic.
//!   6. `rollup`    — aggregate chunk topics into a per-document distribution
//!      plus a dominant topic.
//!   7. `coords`    — 2D topic-centroid coordinates for the bubble chart.
//!
//! `run` chains these stages; `run_topic_modeling` (in `lib.rs`) is the PyO3
//! entry the backend worker calls. There is no length branching — the only
//! special case is a *numeric guard* for corpora too small for PaCMAP to fit,
//! which collapse to a single trivial topic (NOT a PCA fallback, NOT a
//! short-text path).
//!
//! Determinism note: the per-stage deterministic logic is unit-tested; `run`
//! itself depends on downloaded model weights and PaCMAP's seeded-but-not-
//! bit-exact reduction, so it is validated by the manual harness (Phase 2),
//! not CI.

pub mod chunking;
pub mod cluster;
pub mod coords;
pub mod ctfidf;
pub mod embedding;
pub mod plugin;
pub mod reduce;
pub mod rollup;

use std::collections::HashSet;

use anyhow::Result;
use serde::Serialize;

use crate::tokenizer::PLAIN_WORDS_EN_MODEL_ID;
use chunking::ChunkingConfig;
use cluster::ClusterConfig;
use ctfidf::CtfidfConfig;
use reduce::{ReduceConfig, MIN_POINTS_FOR_REDUCTION};

/// Number of dimensions for the visualization-only reduction feeding the bubble
/// chart. Always 2 (x, y).
const COORD_DIMS: usize = 2;

/// All knobs for one topic-modeling run. The backend maps its public options
/// (`min_topic_size`, `representative_words_count`, `random_seed`, sampling,
/// CJK vectorizer choice) onto these fields.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// HF repo id of the candle embedder; `None` uses the default multilingual
    /// model validated in Phase 0.
    pub embedder_repo_id: Option<String>,
    pub chunking: ChunkingConfig,
    /// PaCMAP clustering-space dimensionality (≈5–15).
    pub reduce_dims: usize,
    /// Seed shared by both PaCMAP passes for reproducibility.
    pub seed: u64,
    pub cluster: ClusterConfig,
    pub ctfidf: CtfidfConfig,
    /// Tokenizer model id used to segment topic text for c-TF-IDF (e.g.
    /// `lindera:jieba` for Chinese). `None` falls back to English plain words.
    pub vectorizer_model_id: Option<String>,
    pub lowercase: bool,
    pub stopwords: HashSet<String>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            embedder_repo_id: None,
            chunking: ChunkingConfig::default(),
            reduce_dims: ReduceConfig::default().output_dims,
            seed: ReduceConfig::default().seed,
            cluster: ClusterConfig::default(),
            ctfidf: CtfidfConfig::default(),
            vectorizer_model_id: None,
            lowercase: true,
            stopwords: HashSet::new(),
        }
    }
}

/// One topic for the bubble chart and topic table.
#[derive(Debug, Clone, Serialize)]
pub struct TopicInfo {
    pub id: i32,
    pub representative_words: Vec<String>,
    pub representative_scores: Vec<f32>,
    /// Per-corpus soft size (summed document proportions).
    pub size: Vec<f32>,
    pub total_size: f32,
    /// Raw chunk count assigned to this topic (the hard count, for `meta`).
    pub chunk_count: usize,
    pub x: f32,
    pub y: f32,
}

/// One document's topic outcome: the full distribution and its dominant topic.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentResult {
    pub doc_index: usize,
    pub corpus_index: usize,
    pub dominant_topic: i32,
    /// `(topic_id, proportion)` pairs summing to 1 over the document's chunks.
    pub topic_distribution: Vec<(i32, f32)>,
}

/// Full pipeline output handed back to Python.
#[derive(Debug, Clone, Serialize)]
pub struct TopicModelingResult {
    pub topics: Vec<TopicInfo>,
    pub documents: Vec<DocumentResult>,
    pub n_chunks: usize,
    pub n_topics: usize,
}

/// Run the full pipeline on `documents`, with `corpus_indices[d]` naming each
/// document's corpus (0-based; use all-zeros for a single corpus).
///
/// Flow:
///  1. Load the embedder and chunk every document with its sizing tokenizer.
///  2. If there are enough chunks for PaCMAP, embed → reduce(5D) → HDBSCAN, and
///     separately reduce(2D) for coordinates. Too few chunks collapse to one
///     trivial topic (numeric guard); zero chunks yield no topics.
///  3. Concatenate each topic's chunk text, then c-TF-IDF for keywords.
///  4. Roll chunk labels up to per-document distributions and per-corpus soft
///     sizes, and assemble the topic/document payload.
pub fn run(
    documents: &[String],
    corpus_indices: &[usize],
    cfg: &RunConfig,
) -> Result<TopicModelingResult> {
    if documents.len() != corpus_indices.len() {
        anyhow::bail!(
            "documents ({}) and corpus_indices ({}) length mismatch",
            documents.len(),
            corpus_indices.len()
        );
    }

    let embedder = embedding::ensure_embedder(cfg.embedder_repo_id.as_deref())?;
    let chunks = chunking::chunk_documents(documents, embedder.sizing_tokenizer(), &cfg.chunking)?;
    let n_chunks = chunks.len();

    // Stage 2-4 produce: a topic label per chunk, the topic count, and a 2D
    // coordinate per topic. The guard branches differ only in how labels/coords
    // are obtained — everything downstream is identical (no length branching).
    let (labels, n_topics, coords): (Vec<i32>, usize, Vec<(f32, f32)>) =
        if n_chunks == 0 {
            (Vec::new(), 0, Vec::new())
        } else if n_chunks < MIN_POINTS_FOR_REDUCTION {
            // Too few chunks for PaCMAP to fit a neighbor graph: one topic.
            (vec![0; n_chunks], 1, vec![(0.0, 0.0)])
        } else {
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = embedder.encode(&texts)?;

        let reduced = reduce::reduce(
            &embeddings,
            &ReduceConfig {
                output_dims: cfg.reduce_dims,
                seed: cfg.seed,
            },
        )?;

        // The topic count is whatever HDBSCAN yields for the configured
        // `min_cluster_size` (the only native topic-count control); there is no
        // post-fit merge step.
        let clustered = cluster::cluster(&reduced, &cfg.cluster)?;
        let labels = clustered.labels;
        let n_topics = clustered.n_topics;

        let two_d = reduce::reduce(
            &embeddings,
            &ReduceConfig {
                output_dims: COORD_DIMS,
                seed: cfg.seed,
            },
        )?;
        let coords = coords::topic_coords_2d(&two_d, &labels, n_topics);
        (labels, n_topics, coords)
    };

    // c-TF-IDF: one "document" per topic = its chunks concatenated.
    let mut topic_texts = vec![String::new(); n_topics];
    let mut chunk_counts = vec![0usize; n_topics];
    for (chunk, &label) in chunks.iter().zip(&labels) {
        if label >= 0 && (label as usize) < n_topics {
            let t = label as usize;
            topic_texts[t].push_str(&chunk.text);
            topic_texts[t].push(' ');
            chunk_counts[t] += 1;
        }
    }
    let vectorizer = cfg
        .vectorizer_model_id
        .as_deref()
        .unwrap_or(PLAIN_WORDS_EN_MODEL_ID);
    let term_counts =
        ctfidf::count_topic_terms(&topic_texts, Some(vectorizer), cfg.lowercase, &cfg.stopwords)?;
    let keywords = ctfidf::ctfidf_scores(&term_counts, &cfg.ctfidf);

    // Roll chunks up to documents and per-corpus soft sizes.
    let chunk_doc_index: Vec<usize> = chunks.iter().map(|c| c.doc_index).collect();
    let doc_topics = rollup::rollup(documents.len(), &chunk_doc_index, &labels);
    let n_corpora = corpus_indices.iter().copied().max().map_or(0, |m| m + 1);
    let sizes = rollup::corpus_topic_sizes(&doc_topics, corpus_indices, n_corpora, n_topics);

    let topics = (0..n_topics)
        .map(|t| {
            let words = keywords
                .get(t)
                .map(|kw| kw.iter().map(|(w, _)| w.clone()).collect())
                .unwrap_or_default();
            let scores = keywords
                .get(t)
                .map(|kw| kw.iter().map(|(_, s)| *s).collect())
                .unwrap_or_default();
            let size: Vec<f32> = (0..n_corpora).map(|c| sizes[c][t]).collect();
            let total_size = size.iter().sum();
            let (x, y) = coords.get(t).copied().unwrap_or((0.0, 0.0));
            TopicInfo {
                id: t as i32,
                representative_words: words,
                representative_scores: scores,
                size,
                total_size,
                chunk_count: chunk_counts[t],
                x,
                y,
            }
        })
        .collect();

    let document_results = doc_topics
        .into_iter()
        .enumerate()
        .map(|(i, dt)| DocumentResult {
            doc_index: i,
            corpus_index: corpus_indices[i],
            dominant_topic: dt.dominant_topic,
            topic_distribution: dt
                .topic_distribution
                .into_iter()
                .map(|p| (p.topic_id, p.proportion))
                .collect(),
        })
        .collect();

    Ok(TopicModelingResult {
        topics,
        documents: document_results,
        n_chunks,
        n_topics,
    })
}
