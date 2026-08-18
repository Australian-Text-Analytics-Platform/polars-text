//! Rust topic-modeling pipeline — an offline, long-text replacement for the
//! Python BERTopic path.
//!
//! Pipeline (one uniform path for short and long text alike):
//!   1. `chunking`  — split each document into token-budgeted Topic Segments
//!      (a short document is simply one segment).
//!   2. `embedding` — ONNX Runtime sentence embeddings per segment.
//!   3. `reduce`    — PaCMAP dimensionality reduction for clusterability.
//!   4. `cluster`   — HDBSCAN groups segments into topics (with `-1` outliers).
//!   5. `ctfidf`    — c-TF-IDF keyword labels per topic.
//!   6. `rollup`    — aggregate segment topics into a per-document distribution
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
pub mod projection;
#[cfg(feature = "topic-modeling")]
pub mod reduce;
#[cfg(feature = "topic-modeling")]
pub mod rollup;

#[cfg(feature = "topic-modeling")]
use std::{path::Path, time::Instant};

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
use ctfidf::RepresentativeWord;
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
/// (`random_seed`, sampling, CJK vectorizer choice) onto these fields and fixes
/// the natural HDBSCAN leaf size internally.
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
    /// Tokenizer model id used to segment topic text for c-TF-IDF (e.g.
    /// `lindera:jieba` for Chinese). `None` falls back to English plain words.
    pub vectorizer_model_id: Option<String>,
    pub lowercase: bool,
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
            vectorizer_model_id: None,
            lowercase: true,
        }
    }
}

/// One topic for the bubble chart and topic table.
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone, Serialize)]
pub struct TopicInfo {
    pub id: i32,
    pub representative_words: Vec<RepresentativeWord>,
    pub x: f32,
    pub y: f32,
}

/// One document's topic outcome: the full distribution and its dominant topic.
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone, Serialize)]
pub struct DocumentResult {
    pub doc_index: usize,
    pub dominant_topic: i32,
    /// `(topic_id, proportion)` pairs summing to 1 over the retained character
    /// length of the document's Topic Segments.
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
    pub truncated_segment_count: usize,
    pub stage_timings_ms: Vec<StageTiming>,
    #[serde(skip)]
    pub clustering_context: Vec<u8>,
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

/// Run the full pipeline on `documents`.
///
/// Flow:
///  1. Load the embedder and chunk every document with its sizing tokenizer.
///  2. If there are enough chunks for PaCMAP, embed → reduce(5D) → HDBSCAN, and
///     separately reduce(2D) for coordinates. Too few chunks collapse to one
///     trivial topic (numeric guard); zero chunks yield no topics.
///  3. Concatenate each topic's chunk text, then c-TF-IDF for keywords.
///  4. Roll segment labels up to length-weighted per-document distributions,
///     then assemble the topic/document payload.
#[cfg(feature = "topic-modeling")]
pub fn run(documents: &[String], cfg: &RunConfig) -> Result<TopicModelingResult> {
    let total_started_at = Instant::now();
    let mut stage_timings_ms = Vec::new();

    let stage_started_at = Instant::now();
    let embedder = embedding::ensure_embedder(cfg.embedder_repo_id.as_deref())?;
    record_stage_timing(&mut stage_timings_ms, "embedder_load", stage_started_at);

    let stage_started_at = Instant::now();
    let chunking_result =
        chunking::chunk_documents(documents, embedder.sizing_tokenizer(), &cfg.chunking)?;
    let chunks = chunking_result.chunks;
    let truncated_segment_count = chunking_result.truncated_count;
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
    let (labels, n_topics, reduced_5d, reduced_2d): (
        Vec<i32>,
        usize,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
    ) = if n_chunks == 0 {
        (Vec::new(), 0, Vec::new(), Vec::new())
    } else if n_chunks < MIN_POINTS_FOR_REDUCTION {
        // Too few chunks for PaCMAP to fit a neighbor graph: one topic.
        (
            vec![0; n_chunks],
            1,
            vec![vec![0.0; cfg.reduce_dims]; n_chunks],
            vec![vec![0.0; COORD_DIMS]; n_chunks],
        )
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

        // HDBSCAN establishes the natural maximum-resolution leaves. The
        // projection context built below can merge those real topics later.
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
        let _coords = coords::topic_coords_2d(&two_d, &labels, n_topics);
        record_stage_timing(&mut stage_timings_ms, "topic_coordinates", stage_started_at);
        (labels, n_topics, reduced, two_d)
    };

    // c-TF-IDF: one "document" per topic = its chunks concatenated.
    let mut topic_texts = vec![String::new(); n_topics];
    for (chunk, &label) in chunks.iter().zip(&labels) {
        if label >= 0 && (label as usize) < n_topics {
            let t = label as usize;
            topic_texts[t].push_str(&chunk.text);
            topic_texts[t].push(' ');
        }
    }
    let vectorizer = cfg
        .vectorizer_model_id
        .as_deref()
        .unwrap_or(PLAIN_WORDS_EN_MODEL_ID);
    let stage_started_at = Instant::now();
    let term_counts = ctfidf::count_topic_terms(&topic_texts, Some(vectorizer), cfg.lowercase)?;
    record_stage_timing(
        &mut stage_timings_ms,
        "ctfidf_count_terms",
        stage_started_at,
    );

    let chunk_doc_index: Vec<usize> = chunks.iter().map(|c| c.doc_index).collect();
    let chunk_weights: Vec<usize> = chunks.iter().map(|c| c.text.chars().count()).collect();
    let stage_started_at = Instant::now();
    let context = projection::prepare_context(
        documents.len(),
        &labels,
        &reduced_5d,
        &reduced_2d,
        &chunk_doc_index,
        &chunk_weights,
        &term_counts,
        n_chunks,
        truncated_segment_count,
    )?;
    let mut result = projection::project(&context, n_topics)?;
    record_stage_timing(&mut stage_timings_ms, "ctfidf_scores", stage_started_at);
    record_stage_timing(&mut stage_timings_ms, "total", total_started_at);
    result.stage_timings_ms = stage_timings_ms;
    result.clustering_context = projection::serialize_context(&context)?;
    Ok(result)
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
