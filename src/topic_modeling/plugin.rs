//! Scalar Polars expression wrapping the whole topic-modeling pipeline.
//!
//! Clustering needs the complete document column, so one invocation returns one
//! run-level struct. Document outcomes and complete topic metadata live in
//! separate nested lists; topic metadata is never inferred from dominant-topic
//! rows.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::{run, RunConfig, TopicModelingResult};
use crate::list_output::list_from_spans;

#[derive(Deserialize)]
struct TopicModelingKwargs {
    embedder_model: Option<String>,
    cache: Option<String>,
    segmentation_method: super::segmentation::SegmentationMethod,
    max_tokens: usize,
    seed: u64,
    min_cluster_size: usize,
    vectorizer_model: Option<String>,
    lowercase: bool,
}

fn coverage_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("topic_id".into(), DataType::Int32),
        Field::new("coverage".into(), DataType::Float32),
    ])
}

fn document_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("doc_index".into(), DataType::UInt32),
        Field::new("dominant_topic".into(), DataType::Int32),
        Field::new(
            "topic_coverage".into(),
            DataType::List(Box::new(coverage_struct_type())),
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
        Field::new("n_segments".into(), DataType::UInt32),
        Field::new("projection_context".into(), DataType::Binary),
    ]);
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

fn build_list_from_spans(
    name: &str,
    inner: &Series,
    spans: &[(usize, usize)],
    _inner_type: DataType,
) -> PolarsResult<Series> {
    list_from_spans(name.into(), inner, spans)
}

fn build_single_list(name: &str, inner: &Series, inner_type: DataType) -> PolarsResult<Series> {
    build_list_from_spans(name, inner, &[(0, inner.len())], inner_type)
}

#[polars_expr(output_type_func=topic_modeling_output)]
pub fn topic_modeling(inputs: &[Series], kwargs: TopicModelingKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let documents = ca
        .iter()
        .map(|value| value.unwrap_or(""))
        .collect::<Vec<_>>();

    let cfg = RunConfig {
        embedder_repo_id: kwargs.embedder_model,
        embedding_cache_path: kwargs.cache,
        segmentation: super::segmentation::SegmentationConfig {
            method: kwargs.segmentation_method,
            max_tokens: kwargs.max_tokens,
        },
        seed: kwargs.seed,
        cluster: super::cluster::ClusterConfig {
            min_cluster_size: kwargs.min_cluster_size,
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
    let mut coverage_topic_ids = Vec::new();
    let mut coverage_values = Vec::new();
    let mut coverage_spans = Vec::with_capacity(result.documents.len());
    for document in &result.documents {
        let start = coverage_topic_ids.len();
        for &(topic_id, coverage) in &document.topic_coverage {
            coverage_topic_ids.push(topic_id);
            coverage_values.push(coverage);
        }
        coverage_spans.push((start, coverage_topic_ids.len()));
    }
    let coverage_inner = StructChunked::from_series(
        PlSmallStr::EMPTY,
        coverage_topic_ids.len(),
        [
            Series::new("topic_id".into(), coverage_topic_ids),
            Series::new("coverage".into(), coverage_values),
        ]
        .iter(),
    )?
    .into_series();
    let coverage = build_list_from_spans(
        "topic_coverage",
        &coverage_inner,
        &coverage_spans,
        coverage_struct_type(),
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
                    .map(|document| {
                        u32::try_from(document.doc_index).map_err(|_| {
                            PolarsError::ComputeError("Topic document index exceeds UInt32".into())
                        })
                    })
                    .collect::<PolarsResult<Vec<_>>>()?,
            ),
            Series::new(
                "dominant_topic".into(),
                result
                    .documents
                    .iter()
                    .map(|document| document.dominant_topic)
                    .collect::<Vec<_>>(),
            ),
            coverage,
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
            occurrence_counts.push(u64::try_from(representative_word.occurrence_count).map_err(
                |_| PolarsError::ComputeError("Topic word count exceeds UInt64".into()),
            )?);
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

    let n_segments = u32::try_from(result.n_segments)
        .map_err(|_| PolarsError::ComputeError("Topic Segment count exceeds UInt32".into()))?;
    let fields = [
        document_list,
        topic_list,
        Series::new("n_segments".into(), [n_segments]),
        Series::new(
            "projection_context".into(),
            [result.projection_context.as_deref()],
        ),
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
                Field::new("n_segments".into(), DataType::UInt32),
                Field::new("projection_context".into(), DataType::Binary),
            ]
        );
    }

    #[test]
    fn scalar_result_preserves_topic_metadata_when_topic_never_dominates() {
        let result = TopicModelingResult {
            documents: vec![DocumentResult {
                doc_index: 0,
                dominant_topic: 0,
                topic_coverage: vec![(0, 0.6), (1, 0.4)],
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
            n_segments: 2,
            projection_context: Some(vec![1, 2, 3]),
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
