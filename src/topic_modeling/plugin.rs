//! Scalar Polars expression wrapping the whole topic-modeling pipeline.
//!
//! Clustering needs the complete document column, so one invocation returns one
//! run-level struct. Document outcomes and complete topic metadata live in
//! separate nested lists; topic metadata is never inferred from dominant-topic
//! rows.

use polars::chunked_array::builder::{AnonymousOwnedListBuilder, ListBuilderTrait};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::{run, RunConfig, TopicModelingResult};

#[derive(Deserialize)]
struct TopicModelingKwargs {
    embedder_model: Option<String>,
    cache: Option<String>,
    segmentation_method: super::chunking::SegmentationMethod,
    max_tokens: usize,
    overlap: usize,
    reduce_dims: usize,
    seed: u64,
    min_cluster_size: usize,
    min_samples: Option<usize>,
    vectorizer_model: Option<String>,
    lowercase: bool,
}

fn distribution_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("topic_id".into(), DataType::Int32),
        Field::new("proportion".into(), DataType::Float32),
    ])
}

fn document_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("doc_index".into(), DataType::UInt32),
        Field::new("dominant_topic".into(), DataType::Int32),
        Field::new(
            "topic_distribution".into(),
            DataType::List(Box::new(distribution_struct_type())),
        ),
    ])
}

fn representative_word_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("word".into(), DataType::String),
        Field::new("occurrence_count".into(), DataType::UInt64),
    ])
}

fn topic_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("id".into(), DataType::Int32),
        Field::new(
            "representative_words".into(),
            DataType::List(Box::new(representative_word_struct_type())),
        ),
        Field::new("x".into(), DataType::Float32),
        Field::new("y".into(), DataType::Float32),
    ])
}

fn stage_timing_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("stage".into(), DataType::String),
        Field::new("elapsed_ms".into(), DataType::Float64),
    ])
}

fn topic_modeling_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let dtype = DataType::Struct(vec![
        Field::new(
            "documents".into(),
            DataType::List(Box::new(document_struct_type())),
        ),
        Field::new(
            "topics".into(),
            DataType::List(Box::new(topic_struct_type())),
        ),
        Field::new("n_chunks".into(), DataType::UInt32),
        Field::new("truncated_segment_count".into(), DataType::UInt32),
        Field::new(
            "stage_timings_ms".into(),
            DataType::List(Box::new(stage_timing_struct_type())),
        ),
    ]);
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

fn build_list_from_spans(
    name: &str,
    inner: &Series,
    spans: &[(usize, usize)],
    inner_type: DataType,
) -> PolarsResult<Series> {
    let mut builder = AnonymousOwnedListBuilder::new(name.into(), spans.len(), Some(inner_type));
    for &(start, end) in spans {
        if start == end {
            builder.append_empty();
        } else {
            builder.append_series(&inner.slice(start as i64, end - start))?;
        }
    }
    Ok(builder.finish().into_series())
}

fn build_single_list(name: &str, inner: &Series, inner_type: DataType) -> PolarsResult<Series> {
    let mut builder = AnonymousOwnedListBuilder::new(name.into(), 1, Some(inner_type));
    if inner.is_empty() {
        builder.append_empty();
    } else {
        builder.append_series(inner)?;
    }
    Ok(builder.finish().into_series())
}

#[polars_expr(output_type_func=topic_modeling_output)]
pub fn topic_modeling(inputs: &[Series], kwargs: TopicModelingKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let documents: Vec<String> = ca
        .into_iter()
        .map(|opt| opt.unwrap_or("").to_string())
        .collect();

    let cfg = RunConfig {
        embedder_repo_id: kwargs.embedder_model,
        embedding_cache_path: kwargs.cache,
        chunking: super::chunking::ChunkingConfig {
            method: kwargs.segmentation_method,
            max_tokens: kwargs.max_tokens,
            overlap: kwargs.overlap,
        },
        reduce_dims: kwargs.reduce_dims,
        seed: kwargs.seed,
        cluster: super::cluster::ClusterConfig {
            min_cluster_size: kwargs.min_cluster_size,
            min_samples: kwargs.min_samples,
        },
        vectorizer_model_id: kwargs.vectorizer_model,
        lowercase: kwargs.lowercase,
    };

    let result = run(&documents, &cfg).map_err(|error| {
        PolarsError::ComputeError(format!("topic_modeling failed: {error:#}").into())
    })?;

    topic_modeling_result_to_series(ca.name().clone(), &result)
}

fn topic_modeling_result_to_series(
    name: PlSmallStr,
    result: &TopicModelingResult,
) -> PolarsResult<Series> {
    let mut distribution_topic_ids = Vec::new();
    let mut distribution_proportions = Vec::new();
    let mut distribution_spans = Vec::with_capacity(result.documents.len());
    for document in &result.documents {
        let start = distribution_topic_ids.len();
        for &(topic_id, proportion) in &document.topic_distribution {
            distribution_topic_ids.push(topic_id);
            distribution_proportions.push(proportion);
        }
        distribution_spans.push((start, distribution_topic_ids.len()));
    }
    let distribution_inner = StructChunked::from_series(
        PlSmallStr::EMPTY,
        distribution_topic_ids.len(),
        [
            Series::new("topic_id".into(), distribution_topic_ids),
            Series::new("proportion".into(), distribution_proportions),
        ]
        .iter(),
    )?
    .into_series();
    let distributions = build_list_from_spans(
        "topic_distribution",
        &distribution_inner,
        &distribution_spans,
        distribution_struct_type(),
    )?;
    let document_inner = StructChunked::from_series(
        PlSmallStr::EMPTY,
        result.documents.len(),
        [
            Series::new(
                "doc_index".into(),
                result
                    .documents
                    .iter()
                    .map(|document| document.doc_index as u32)
                    .collect::<Vec<_>>(),
            ),
            Series::new(
                "dominant_topic".into(),
                result
                    .documents
                    .iter()
                    .map(|document| document.dominant_topic)
                    .collect::<Vec<_>>(),
            ),
            distributions,
        ]
        .iter(),
    )?
    .into_series();
    let document_list = build_single_list("documents", &document_inner, document_struct_type())?;

    let mut words = Vec::new();
    let mut occurrence_counts = Vec::new();
    let mut word_spans = Vec::with_capacity(result.topics.len());
    for topic in &result.topics {
        let start = words.len();
        for representative_word in &topic.representative_words {
            words.push(representative_word.word.as_str());
            occurrence_counts.push(representative_word.occurrence_count as u64);
        }
        word_spans.push((start, words.len()));
    }
    let word_inner = StructChunked::from_series(
        PlSmallStr::EMPTY,
        words.len(),
        [
            Series::new("word".into(), words),
            Series::new("occurrence_count".into(), occurrence_counts),
        ]
        .iter(),
    )?
    .into_series();
    let representative_words = build_list_from_spans(
        "representative_words",
        &word_inner,
        &word_spans,
        representative_word_struct_type(),
    )?;
    let topic_inner = StructChunked::from_series(
        PlSmallStr::EMPTY,
        result.topics.len(),
        [
            Series::new(
                "id".into(),
                result
                    .topics
                    .iter()
                    .map(|topic| topic.id)
                    .collect::<Vec<_>>(),
            ),
            representative_words,
            Series::new(
                "x".into(),
                result
                    .topics
                    .iter()
                    .map(|topic| topic.x)
                    .collect::<Vec<_>>(),
            ),
            Series::new(
                "y".into(),
                result
                    .topics
                    .iter()
                    .map(|topic| topic.y)
                    .collect::<Vec<_>>(),
            ),
        ]
        .iter(),
    )?
    .into_series();
    let topic_list = build_single_list("topics", &topic_inner, topic_struct_type())?;

    let timing_inner = StructChunked::from_series(
        PlSmallStr::EMPTY,
        result.stage_timings_ms.len(),
        [
            Series::new(
                "stage".into(),
                result
                    .stage_timings_ms
                    .iter()
                    .map(|timing| timing.stage.as_str())
                    .collect::<Vec<_>>(),
            ),
            Series::new(
                "elapsed_ms".into(),
                result
                    .stage_timings_ms
                    .iter()
                    .map(|timing| timing.elapsed_ms)
                    .collect::<Vec<_>>(),
            ),
        ]
        .iter(),
    )?
    .into_series();
    let timing_list = build_single_list(
        "stage_timings_ms",
        &timing_inner,
        stage_timing_struct_type(),
    )?;

    let fields = [
        document_list,
        topic_list,
        Series::new("n_chunks".into(), [result.n_chunks as u32]),
        Series::new(
            "truncated_segment_count".into(),
            [result.truncated_segment_count as u32],
        ),
        timing_list,
    ];
    Ok(StructChunked::from_series(name, 1, fields.iter())?.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topic_modeling::ctfidf::RepresentativeWord;
    use crate::topic_modeling::{DocumentResult, TopicInfo};

    #[test]
    fn output_contains_separate_run_level_document_and_topic_lists() {
        let output = topic_modeling_output(&[Field::new("text".into(), DataType::String)])
            .expect("topic output dtype");
        let DataType::Struct(fields) = output.dtype() else {
            panic!("topic output must be a struct")
        };
        assert_eq!(
            fields,
            &vec![
                Field::new(
                    "documents".into(),
                    DataType::List(Box::new(document_struct_type())),
                ),
                Field::new(
                    "topics".into(),
                    DataType::List(Box::new(topic_struct_type())),
                ),
                Field::new("n_chunks".into(), DataType::UInt32),
                Field::new("truncated_segment_count".into(), DataType::UInt32),
                Field::new(
                    "stage_timings_ms".into(),
                    DataType::List(Box::new(stage_timing_struct_type())),
                ),
            ]
        );
    }

    #[test]
    fn scalar_result_preserves_topic_metadata_when_topic_never_dominates() {
        let result = TopicModelingResult {
            documents: vec![DocumentResult {
                doc_index: 0,
                dominant_topic: 0,
                topic_distribution: vec![(0, 0.6), (1, 0.4)],
            }],
            topics: vec![
                TopicInfo {
                    id: 0,
                    representative_words: Vec::new(),
                    x: 0.0,
                    y: 0.0,
                },
                TopicInfo {
                    id: 1,
                    representative_words: vec![RepresentativeWord {
                        word: "hidden".to_string(),
                        occurrence_count: 2,
                        score: 1.0,
                    }],
                    x: 1.0,
                    y: 1.0,
                },
            ],
            n_chunks: 2,
            truncated_segment_count: 0,
            stage_timings_ms: Vec::new(),
        };

        let series = topic_modeling_result_to_series("topic".into(), &result)
            .expect("serialize topic result");
        let topics = series
            .struct_()
            .expect("outer struct")
            .field_by_name("topics")
            .expect("topics field");
        let topic_rows = topics
            .list()
            .expect("topic list")
            .get_as_series(0)
            .expect("scalar topic rows");

        assert_eq!((series.len(), topic_rows.len()), (1, 2));
    }
}
