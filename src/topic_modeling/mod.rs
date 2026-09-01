//! Native topic-modeling pipeline.
//!
//! One uniform path handles short and long text:
//! 1. `segmentation` creates non-overlapping, token-budgeted source spans.
//! 2. `embedding` produces sentence embeddings for those spans.
//! 3. `reduce` maps embeddings to an adaptive clustering space with PaCMAP.
//! 4. `cluster` assigns HDBSCAN Topics and explicit `-1` outliers.
//! 5. `ctfidf` labels each real Topic from its assigned source spans.
//! 6. `rollup` reports per-document source-character coverage.
//!
//! Corpora without enough density evidence return an explicit no-topic result;
//! the pipeline never fabricates a cluster to keep the result non-empty.

#[cfg(feature = "topic-modeling")]
pub mod cluster;
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
pub mod segmentation;

#[cfg(feature = "topic-modeling")]
use std::path::Path;

#[cfg(feature = "topic-modeling")]
use anyhow::{Context, Result};
#[cfg(feature = "topic-modeling")]
use serde::Serialize;

#[cfg(feature = "topic-modeling")]
use crate::tokenizer::PLAIN_WORDS_EN_MODEL_ID;
#[cfg(feature = "topic-modeling")]
use cluster::ClusterConfig;
#[cfg(feature = "topic-modeling")]
use ctfidf::RepresentativeWord;
#[cfg(feature = "topic-modeling")]
use embedding_cache::{get_or_insert_embeddings, CacheScope};
#[cfg(feature = "topic-modeling")]
use reduce::ReduceConfig;
#[cfg(feature = "topic-modeling")]
use segmentation::SegmentationConfig;

/// ORT inference batch size for Topic Segments.
#[cfg(feature = "topic-modeling")]
const TOPIC_EMBEDDING_BATCH_SIZE: usize = 32;

/// Supported controls for one topic-modeling run.
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub embedder_repo_id: Option<String>,
    pub embedding_cache_path: Option<String>,
    pub segmentation: SegmentationConfig,
    pub seed: u64,
    pub cluster: ClusterConfig,
    pub vectorizer_model_id: Option<String>,
    pub lowercase: bool,
}

#[cfg(feature = "topic-modeling")]
impl Default for RunConfig {
    fn default() -> Self {
        Self {
            embedder_repo_id: None,
            embedding_cache_path: None,
            segmentation: SegmentationConfig::default(),
            seed: ReduceConfig::default().seed,
            cluster: ClusterConfig::default(),
            vectorizer_model_id: None,
            lowercase: true,
        }
    }
}

/// One Topic for the bubble chart and Topic table.
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone, Serialize)]
pub struct TopicInfo {
    pub id: i32,
    pub representative_words: Vec<RepresentativeWord>,
    pub x: f32,
    pub y: f32,
}

/// One document's source-character Topic coverage and dominant Topic.
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone, Serialize)]
pub struct DocumentResult {
    pub doc_index: usize,
    pub dominant_topic: i32,
    /// `(topic_id, coverage)` pairs summing to one over owned Topic Segment
    /// characters. HDBSCAN outlier `-1` remains explicit.
    pub topic_coverage: Vec<(i32, f32)>,
}

/// Full pipeline output handed back to Polars.
#[cfg(feature = "topic-modeling")]
#[derive(Debug, Clone, Serialize)]
pub struct TopicModelingResult {
    pub topics: Vec<TopicInfo>,
    pub documents: Vec<DocumentResult>,
    pub n_segments: usize,
    #[serde(skip)]
    pub projection_context: Option<Vec<u8>>,
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
#[cfg(feature = "topic-modeling")]
pub fn run(documents: &[&str], cfg: &RunConfig) -> Result<TopicModelingResult> {
    let embedder = embedding::ensure_embedder(cfg.embedder_repo_id.as_deref())?;
    if cfg.segmentation.max_tokens > embedder.max_length() {
        anyhow::bail!(
            "segmentation max_tokens {} exceeds model {} maximum length {}",
            cfg.segmentation.max_tokens,
            embedder.model_id(),
            embedder.max_length()
        );
    }

    let segments = segmentation::segment_documents(
        documents,
        &embedder.sizing_tokenizer(),
        &cfg.segmentation,
    )?
    .segments;
    let segment_doc_indices = segments
        .iter()
        .map(|segment| segment.doc_index)
        .collect::<Vec<_>>();
    let segment_weights = segments
        .iter()
        .map(|segment| segment.owned_character_count)
        .collect::<Vec<_>>();

    // PaCMAP needs at least three points and HDBSCAN needs at least one full
    // minimum cluster. Anything smaller has no defensible density-based Topic.
    let minimum_evidence = cfg.cluster.min_cluster_size.max(3);
    if segments.len() < minimum_evidence {
        return Ok(no_topic_result(
            documents.len(),
            &segment_doc_indices,
            &segment_weights,
        ));
    }

    // Cache and ONNX APIs own strings, so source spans are materialized exactly
    // once at that boundary and reused for c-TF-IDF.
    let texts = segments
        .iter()
        .map(|segment| {
            segment
                .text(documents[segment.doc_index])
                .map(str::to_owned)
        })
        .collect::<Result<Vec<_>>>()?;
    let embeddings: Vec<std::sync::Arc<Vec<f32>>> =
        if let Some(cache_path) = cfg.embedding_cache_path.as_deref() {
            get_or_insert_embeddings(
                Path::new(cache_path),
                CacheScope {
                    model_id: embedder.model_id(),
                    fingerprint: embedder.cache_fingerprint(),
                    provider_id: embedder.provider_id(),
                },
                &texts,
                |misses| encode_topic_embedding_batches(misses, |batch| embedder.encode(batch)),
            )?
        } else {
            encode_topic_embedding_batches(&texts, |batch| embedder.encode(batch))?
                .into_iter()
                .map(std::sync::Arc::new)
                .collect()
        };

    let embedding_width = embeddings
        .first()
        .map(|embedding| embedding.len())
        .context("topic embedder returned no vectors")?;
    let reduce_dims = 5usize.min(embedding_width).min(segments.len());
    if reduce_dims < 2 {
        anyhow::bail!("topic embeddings do not have enough usable dimensions");
    }
    let reduced = reduce::reduce(
        &embeddings,
        &ReduceConfig {
            output_dims: reduce_dims,
            seed: cfg.seed,
        },
    )?;
    let clustered = cluster::cluster(&reduced, &cfg.cluster)?;
    if clustered.n_topics == 0 {
        return Ok(no_topic_result(
            documents.len(),
            &segment_doc_indices,
            &segment_weights,
        ));
    }

    let vectorizer = cfg
        .vectorizer_model_id
        .as_deref()
        .unwrap_or(PLAIN_WORDS_EN_MODEL_ID);
    let term_counts = ctfidf::count_topic_terms(
        clustered.n_topics,
        clustered
            .labels
            .iter()
            .copied()
            .zip(texts.iter().map(String::as_str)),
        Some(vectorizer),
        cfg.lowercase,
    )?;
    let embedding_points = embeddings
        .iter()
        .map(|embedding| embedding.as_slice())
        .collect::<Vec<_>>();
    let context = projection::prepare_context(projection::ProjectionInput {
        document_count: documents.len(),
        labels: &clustered.labels,
        embedding_points: &embedding_points,
        document_indices: &segment_doc_indices,
        owned_character_weights: &segment_weights,
        per_leaf_term_counts: &term_counts,
        seed: cfg.seed,
    })?;
    let mut result = projection::project(&context, clustered.n_topics)?;
    result.projection_context = Some(projection::serialize_context(&context)?);
    Ok(result)
}

#[cfg(feature = "topic-modeling")]
fn no_topic_result(
    document_count: usize,
    segment_doc_indices: &[usize],
    segment_weights: &[usize],
) -> TopicModelingResult {
    let labels = vec![cluster::OUTLIER_LABEL; segment_doc_indices.len()];
    let documents = rollup::rollup(
        document_count,
        segment_doc_indices,
        &labels,
        segment_weights,
    )
    .into_iter()
    .enumerate()
    .map(|(doc_index, topics)| DocumentResult {
        doc_index,
        dominant_topic: topics.dominant_topic,
        topic_coverage: topics
            .topic_coverage
            .into_iter()
            .map(|entry| (entry.topic_id, entry.coverage))
            .collect(),
    })
    .collect();
    TopicModelingResult {
        topics: Vec::new(),
        documents,
        n_segments: segment_doc_indices.len(),
        projection_context: None,
    }
}

#[cfg(all(test, feature = "topic-modeling"))]
mod tests {
    use super::*;

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

    #[test]
    fn insufficient_evidence_is_explicitly_all_outlier() {
        let result = no_topic_result(2, &[0], &[4]);
        assert!(result.topics.is_empty());
        assert!(result.projection_context.is_none());
        assert_eq!(result.documents[0].dominant_topic, cluster::OUTLIER_LABEL);
        assert_eq!(
            result.documents[0].topic_coverage,
            vec![(cluster::OUTLIER_LABEL, 1.0)]
        );
        assert!(result.documents[1].topic_coverage.is_empty());
    }
}
