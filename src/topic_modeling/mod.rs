//! Rust topic-modeling pipeline — an offline, long-text replacement for the
//! Python BERTopic path.
//!
//! Pipeline (one uniform path for short and long text alike):
//!   1. `chunking`  — split each document into token-budgeted semantic chunks
//!      (a short document is simply one chunk).
//!   2. `embedding` — ONNX Runtime sentence embeddings per chunk.
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

#[cfg(feature = "topic-modeling")]
pub mod chunking;
#[cfg(feature = "topic-modeling")]
pub mod cluster;
#[cfg(feature = "topic-modeling")]
pub mod coords;
#[cfg(feature = "topic-modeling")]
pub mod ctfidf;
#[cfg(feature = "embedding")]
pub mod embedding;
#[cfg(feature = "embedding")]
pub mod embedding_cache;
#[cfg(feature = "topic-modeling")]
pub mod plugin;
#[cfg(feature = "topic-modeling")]
pub mod reduce;
#[cfg(feature = "topic-modeling")]
pub mod rollup;

#[cfg(feature = "topic-modeling")]
use std::{collections::HashSet, path::Path, time::Instant};

#[cfg(feature = "topic-modeling")]
use anyhow::Result;
#[cfg(feature = "topic-modeling")]
use serde::Serialize;

#[cfg(feature = "topic-modeling")]
use crate::tokenizer::PLAIN_WORDS_EN_MODEL_ID;
#[cfg(feature = "topic-modeling")]
use chunking::ChunkingConfig;
#[cfg(feature = "topic-modeling")]
use cluster::ClusterConfig;
#[cfg(feature = "topic-modeling")]
use ctfidf::CtfidfConfig;
#[cfg(feature = "topic-modeling")]
use embedding_cache::{get_or_insert_embeddings, CacheScope};
#[cfg(feature = "topic-modeling")]
use reduce::{ReduceConfig, MIN_POINTS_FOR_REDUCTION};

/// Number of dimensions for the visualization-only reduction feeding the bubble
/// chart. Always 2 (x, y).
#[cfg(feature = "topic-modeling")]
const COORD_DIMS: usize = 2;

/// ORT inference batch size for topic-modeling chunks. This mirrors the public
/// `.text.embedding(batch_size=None)` default so topic modeling is bounded even
/// when a corpus yields thousands of chunks.
#[cfg(feature = "topic-modeling")]
const TOPIC_EMBEDDING_BATCH_SIZE: usize = 32;

/// All knobs for one topic-modeling run. The backend maps its public options
/// (`min_topic_size`, `representative_words_count`, `random_seed`, sampling,
/// CJK vectorizer choice) onto these fields.
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// HF repo id of the ONNX embedder; `None` uses the default ONNX model.
    pub embedder_repo_id: Option<String>,
    /// Optional path to the per-user DuckDB embedding cache (`embeddings.duckdb`).
    pub embedding_cache_path: Option<String>,
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

#[cfg(feature = "topic-modeling")]
impl Default for RunConfig {
    fn default() -> Self {
        Self {
            embedder_repo_id: None,
            embedding_cache_path: None,
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
#[cfg(feature = "topic-modeling")]
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
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone, Serialize)]
pub struct DocumentResult {
    pub doc_index: usize,
    pub corpus_index: usize,
    pub dominant_topic: i32,
    /// `(topic_id, proportion)` pairs summing to 1 over the document's chunks.
    pub topic_distribution: Vec<(i32, f32)>,
}

/// One measured native topic-modeling stage, in milliseconds.
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone, Serialize)]
pub struct StageTiming {
    pub stage: String,
    pub elapsed_ms: f64,
}

/// Full pipeline output handed back to Python.
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone, Serialize)]
pub struct TopicModelingResult {
    pub topics: Vec<TopicInfo>,
    pub documents: Vec<DocumentResult>,
    pub n_chunks: usize,
    pub n_topics: usize,
    pub stage_timings_ms: Vec<StageTiming>,
}

#[cfg(feature = "topic-modeling")]
fn record_stage_timing(
    stage_timings_ms: &mut Vec<StageTiming>,
    stage: &'static str,
    started_at: Instant,
) {
    stage_timings_ms.push(StageTiming {
        stage: stage.to_string(),
        elapsed_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    });
}

#[cfg(feature = "topic-modeling")]
fn encode_topic_embedding_batches(
    texts: &[String],
    mut encode_batch: impl FnMut(&[String]) -> Result<Vec<Vec<f32>>>,
) -> Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::with_capacity(texts.len());
    for batch in texts.chunks(TOPIC_EMBEDDING_BATCH_SIZE) {
        let encoded = encode_batch(batch)?;
        if encoded.len() != batch.len() {
            anyhow::bail!(
                "topic embedding batch encoder returned {} vectors for {} texts",
                encoded.len(),
                batch.len()
            );
        }
        vectors.extend(encoded);
    }
    Ok(vectors)
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
#[cfg(feature = "topic-modeling")]
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

    let total_started_at = Instant::now();
    let mut stage_timings_ms = Vec::new();

    let stage_started_at = Instant::now();
    let embedder = embedding::ensure_embedder(cfg.embedder_repo_id.as_deref())?;
    record_stage_timing(&mut stage_timings_ms, "embedder_load", stage_started_at);

    let stage_started_at = Instant::now();
    let chunks = chunking::chunk_documents(documents, embedder.sizing_tokenizer(), &cfg.chunking)?;
    record_stage_timing(&mut stage_timings_ms, "chunking", stage_started_at);
    let n_chunks = chunks.len();

    // Materialize embeddings for every non-empty chunk set before the tiny-
    // corpus guard so the DuckDB embedding cache observes all text pieces. The
    // guard below skips only PaCMAP/HDBSCAN when there are too few points.
    let embeddings: Vec<Vec<f32>> = if n_chunks == 0 {
        Vec::new()
    } else {
        let stage_started_at = Instant::now();
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = if let Some(cache_path) = cfg.embedding_cache_path.as_deref() {
            get_or_insert_embeddings(
                Path::new(cache_path),
                CacheScope {
                    model_id: embedder.model_id(),
                    revision: embedder.model_revision(),
                    provider_id: embedder.provider_id(),
                },
                &texts,
                |misses| encode_topic_embedding_batches(misses, |batch| embedder.encode(batch)),
            )?
        } else {
            encode_topic_embedding_batches(&texts, |batch| embedder.encode(batch))?
        };
        record_stage_timing(&mut stage_timings_ms, "embedding", stage_started_at);
        embeddings
    };

    // Stages 3-4 produce: a topic label per chunk, the topic count, and a 2D
    // coordinate per topic. The guard branches differ only in how labels/coords
    // are obtained — everything downstream is identical (no length branching).
    let (labels, n_topics, coords): (Vec<i32>, usize, Vec<(f32, f32)>) = if n_chunks == 0 {
        (Vec::new(), 0, Vec::new())
    } else if n_chunks < MIN_POINTS_FOR_REDUCTION {
        // Too few chunks for PaCMAP to fit a neighbor graph: one topic.
        (vec![0; n_chunks], 1, vec![(0.0, 0.0)])
    } else {
        let stage_started_at = Instant::now();
        let reduced = reduce::reduce(
            &embeddings,
            &ReduceConfig {
                output_dims: cfg.reduce_dims,
                seed: cfg.seed,
            },
        )?;
        record_stage_timing(&mut stage_timings_ms, "reduce_clustering", stage_started_at);

        // The topic count is whatever HDBSCAN yields for the configured
        // `min_cluster_size` (the only native topic-count control); there is no
        // post-fit merge step.
        let stage_started_at = Instant::now();
        let clustered = cluster::cluster(&reduced, &cfg.cluster)?;
        let labels = clustered.labels;
        let n_topics = clustered.n_topics;
        record_stage_timing(&mut stage_timings_ms, "hdbscan", stage_started_at);

        let stage_started_at = Instant::now();
        let two_d = reduce::reduce(
            &embeddings,
            &ReduceConfig {
                output_dims: COORD_DIMS,
                seed: cfg.seed,
            },
        )?;
        record_stage_timing(
            &mut stage_timings_ms,
            "reduce_coordinates",
            stage_started_at,
        );

        let stage_started_at = Instant::now();
        let coords = coords::topic_coords_2d(&two_d, &labels, n_topics);
        record_stage_timing(&mut stage_timings_ms, "topic_coordinates", stage_started_at);
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
    let stage_started_at = Instant::now();
    let term_counts = ctfidf::count_topic_terms(
        &topic_texts,
        Some(vectorizer),
        cfg.lowercase,
        &cfg.stopwords,
    )?;
    record_stage_timing(
        &mut stage_timings_ms,
        "ctfidf_count_terms",
        stage_started_at,
    );

    let stage_started_at = Instant::now();
    let keywords = ctfidf::ctfidf_scores(&term_counts, &cfg.ctfidf);
    record_stage_timing(&mut stage_timings_ms, "ctfidf_scores", stage_started_at);

    // Roll chunks up to documents and per-corpus soft sizes.
    let stage_started_at = Instant::now();
    let chunk_doc_index: Vec<usize> = chunks.iter().map(|c| c.doc_index).collect();
    let doc_topics = rollup::rollup(documents.len(), &chunk_doc_index, &labels);
    let n_corpora = corpus_indices.iter().copied().max().map_or(0, |m| m + 1);
    let sizes = rollup::corpus_topic_sizes(&doc_topics, corpus_indices, n_corpora, n_topics);
    record_stage_timing(&mut stage_timings_ms, "rollup", stage_started_at);

    let stage_started_at = Instant::now();
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
    record_stage_timing(&mut stage_timings_ms, "assemble_topics", stage_started_at);
    record_stage_timing(&mut stage_timings_ms, "total", total_started_at);

    Ok(TopicModelingResult {
        topics,
        documents: document_results,
        n_chunks,
        n_topics,
        stage_timings_ms,
    })
}

#[cfg(all(test, feature = "topic-modeling"))]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn topic_embedding_batches_are_bounded_and_ordered() -> Result<()> {
        let texts = (0..70)
            .map(|index| format!("text-{index}"))
            .collect::<Vec<_>>();
        let mut batch_lengths = Vec::new();

        let vectors = encode_topic_embedding_batches(&texts, |batch| {
            batch_lengths.push(batch.len());
            batch
                .iter()
                .map(|text| {
                    let value = text
                        .strip_prefix("text-")
                        .context("test text prefix")?
                        .parse::<f32>()
                        .context("test text index parse")?;
                    Ok(vec![value])
                })
                .collect()
        })?;

        assert_eq!(batch_lengths, vec![32, 32, 6]);
        assert_eq!(vectors.first(), Some(&vec![0.0]));
        assert_eq!(vectors.last(), Some(&vec![69.0]));
        Ok(())
    }
}
