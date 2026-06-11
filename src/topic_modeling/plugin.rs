//! Polars plugin expression wrapping the topic-modeling pipeline so it can be
//! used like every other `.text` function, e.g.
//!
//! ```python
//! lf.with_columns(
//!     pl.col("document").text.topic_modeling(seed=42).alias("topic")
//! )
//! ```
//!
//! Design (keeps the backend thin):
//! - The expression runs on the **whole** document column (it is *not*
//!   elementwise — clustering needs every document at once) and returns a
//!   **per-row struct** so the result lines up 1:1 with the input rows and can
//!   be appended with `with_columns`.
//! - Topic-level metadata that cannot be recovered by a `group_by`
//!   (`representative_words`, the bubble-chart `x`/`y` centroid, and the global
//!   `raw_n_topics`/`n_topics` counts) is **replicated onto each row** under its
//!   dominant topic. The backend then derives the bubble chart and per-corpus
//!   sizes with plain Polars `group_by` over `dominant_topic` (+ its own
//!   `corpus_index` column), so no orchestration logic leaks into Python.
//! - Corpus pooling is a backend concern: concatenate the 1-2 node columns into
//!   one column, run this expression once, then split the per-row output by a
//!   backend-side `corpus_index` column. The pipeline itself always clusters the
//!   pooled column as a single corpus.
//!
//! Used by: `polars_text.functions.topic_modeling` /
//! `polars_text.namespace.TextNamespace.topic_modeling`, which the backend
//! topic-modeling worker calls.

use std::collections::HashMap;

use polars::chunked_array::builder::{AnonymousOwnedListBuilder, ListBuilderTrait};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::{run, RunConfig};

/// Keyword arguments mirroring the pipeline knobs. Defaults match
/// `RunConfig::default` where applicable; the Python wrapper always sends every
/// field.
#[derive(Deserialize)]
struct TopicModelingKwargs {
    embedder_model: Option<String>,
    max_tokens: usize,
    overlap: usize,
    reduce_dims: usize,
    seed: u64,
    min_cluster_size: usize,
    min_samples: Option<usize>,
    top_k: usize,
    vectorizer_model: Option<String>,
    lowercase: bool,
    stopwords: Option<Vec<String>>,
}

/// Inner dtype of the per-document `topic_distribution` list elements.
fn distribution_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("topic_id".into(), DataType::Int32),
        Field::new("proportion".into(), DataType::Float32),
    ])
}

/// Output dtype of the expression: one struct per input row.
fn topic_modeling_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let dtype = DataType::Struct(vec![
        Field::new("dominant_topic".into(), DataType::Int32),
        Field::new(
            "topic_distribution".into(),
            DataType::List(Box::new(distribution_struct_type())),
        ),
        Field::new(
            "representative_words".into(),
            DataType::List(Box::new(DataType::String)),
        ),
        Field::new("x".into(), DataType::Float32),
        Field::new("y".into(), DataType::Float32),
        Field::new("n_topics".into(), DataType::UInt32),
        Field::new("n_chunks".into(), DataType::UInt32),
    ]);
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

#[polars_expr(output_type_func=topic_modeling_output)]
pub fn topic_modeling(inputs: &[Series], kwargs: TopicModelingKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let documents: Vec<String> = ca
        .into_iter()
        .map(|opt| opt.unwrap_or("").to_string())
        .collect();
    let n_rows = documents.len();

    // The expression always clusters the column it receives as one corpus; the
    // backend handles multi-corpus pooling/splitting via a separate column.
    let corpus_indices = vec![0usize; n_rows];

    let cfg = RunConfig {
        embedder_repo_id: kwargs.embedder_model,
        chunking: super::chunking::ChunkingConfig {
            max_tokens: kwargs.max_tokens,
            overlap: kwargs.overlap,
        },
        reduce_dims: kwargs.reduce_dims,
        seed: kwargs.seed,
        cluster: super::cluster::ClusterConfig {
            min_cluster_size: kwargs.min_cluster_size,
            min_samples: kwargs.min_samples,
        },
        ctfidf: super::ctfidf::CtfidfConfig { top_k: kwargs.top_k },
        vectorizer_model_id: kwargs.vectorizer_model,
        lowercase: kwargs.lowercase,
        stopwords: kwargs.stopwords.unwrap_or_default().into_iter().collect(),
    };

    let result = run(&documents, &corpus_indices, &cfg)
        .map_err(|e| PolarsError::ComputeError(format!("topic_modeling failed: {e}").into()))?;

    // Topic id -> (representative_words, x, y) so each row can carry its
    // dominant topic's bubble-chart metadata.
    let topic_meta: HashMap<i32, (&Vec<String>, f32, f32)> = result
        .topics
        .iter()
        .map(|t| (t.id, (&t.representative_words, t.x, t.y)))
        .collect();

    let mut dominant: Vec<i32> = Vec::with_capacity(n_rows);
    let mut xs: Vec<f32> = Vec::with_capacity(n_rows);
    let mut ys: Vec<f32> = Vec::with_capacity(n_rows);

    // Flat columns for the per-row `topic_distribution` list-of-struct, built
    // once then sliced per row (the tokenize/concordance pattern — O(1) slices
    // instead of a fresh StructChunked per row).
    let mut dist_topic_ids: Vec<i32> = Vec::new();
    let mut dist_props: Vec<f32> = Vec::new();
    let mut dist_spans: Vec<(usize, usize)> = Vec::with_capacity(n_rows);

    // Flat column for the per-row `representative_words` list-of-string.
    let mut word_flat: Vec<String> = Vec::new();
    let mut word_spans: Vec<(usize, usize)> = Vec::with_capacity(n_rows);

    // `result.documents` is in input order (doc_index 0..n_rows); iterate in
    // lock-step with the input rows.
    for doc in &result.documents {
        let topic = doc.dominant_topic;
        dominant.push(topic);

        let (words, x, y) = match topic_meta.get(&topic) {
            Some((words, x, y)) => (Some(*words), *x, *y),
            // Outliers (topic == -1) and any unmapped id get empty metadata and
            // origin coords (the backend's existing default for missing coords).
            None => (None, 0.0_f32, 0.0_f32),
        };
        xs.push(x);
        ys.push(y);

        let dist_start = dist_topic_ids.len();
        for (tid, prop) in &doc.topic_distribution {
            dist_topic_ids.push(*tid);
            dist_props.push(*prop);
        }
        dist_spans.push((dist_start, dist_topic_ids.len()));

        let word_start = word_flat.len();
        if let Some(words) = words {
            for w in words.iter() {
                word_flat.push(w.clone());
            }
        }
        word_spans.push((word_start, word_flat.len()));
    }

    // Build the shared inner struct for `topic_distribution` once.
    let dist_inner = StructChunked::from_series(
        PlSmallStr::EMPTY,
        dist_topic_ids.len(),
        [
            Series::new("topic_id".into(), dist_topic_ids),
            Series::new("proportion".into(), dist_props),
        ]
        .iter(),
    )?
    .into_series();

    let mut dist_builder = AnonymousOwnedListBuilder::new(
        "topic_distribution".into(),
        n_rows,
        Some(distribution_struct_type()),
    );
    for (start, end) in dist_spans {
        if end == start {
            dist_builder.append_empty();
        } else {
            let slice = dist_inner.slice(start as i64, end - start);
            dist_builder.append_series(&slice).map_err(|e| {
                PolarsError::ComputeError(format!("topic_distribution list build: {e}").into())
            })?;
        }
    }
    let dist_list = dist_builder.finish().into_series();

    // Build the shared inner string series for `representative_words` once.
    let word_inner = Series::new(PlSmallStr::EMPTY, word_flat);
    let mut word_builder =
        AnonymousOwnedListBuilder::new("representative_words".into(), n_rows, Some(DataType::String));
    for (start, end) in word_spans {
        if end == start {
            word_builder.append_empty();
        } else {
            let slice = word_inner.slice(start as i64, end - start);
            word_builder.append_series(&slice).map_err(|e| {
                PolarsError::ComputeError(format!("representative_words list build: {e}").into())
            })?;
        }
    }
    let word_list = word_builder.finish().into_series();

    let n_topics = vec![result.n_topics as u32; n_rows];
    let n_chunks = vec![result.n_chunks as u32; n_rows];

    let fields = [
        Series::new("dominant_topic".into(), dominant),
        dist_list,
        word_list,
        Series::new("x".into(), xs),
        Series::new("y".into(), ys),
        Series::new("n_topics".into(), n_topics),
        Series::new("n_chunks".into(), n_chunks),
    ];
    let out = StructChunked::from_series(ca.name().clone(), n_rows, fields.iter())?.into_series();
    Ok(out)
}
