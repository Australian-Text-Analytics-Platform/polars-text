#[cfg(feature = "tokenization")]
use crate::concordance::{list_struct_output, struct_series_from_matches, ConcordanceKwargs};
#[cfg(any(feature = "embedding", feature = "tokenization"))]
use crate::list_output::list_from_spans;
#[cfg(feature = "embedding")]
use ldaca_rs::embedding::Embedder;
#[cfg(feature = "tokenization")]
use ldaca_rs::text::{Concordance, ConcordanceOptions, TokenizeOptions, Tokenizer};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
#[cfg(feature = "tokenization")]
use std::path::Path;
#[cfg(feature = "embedding")]
use std::sync::Arc;
fn int_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(input_fields[0].name().clone(), DataType::Int64))
}

#[cfg(feature = "embedding")]
fn embedding_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let dtype = match input_fields[0].dtype() {
        DataType::String => DataType::List(Box::new(DataType::Float32)),
        DataType::List(inner) if inner.as_ref() == &DataType::String => {
            DataType::List(Box::new(DataType::List(Box::new(DataType::Float32))))
        }
        other => {
            return Err(PolarsError::InvalidOperation(
                format!("embedding expects String or List(String), got {other}").into(),
            ));
        }
    };
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

fn count_string_values(
    inputs: &[Series],
    mut count: impl FnMut(&str) -> usize,
) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: Vec<i64> = ca
        .iter()
        .map(|opt_text| {
            i64::try_from(opt_text.map(&mut count).unwrap_or(0))
                .map_err(|_| PolarsError::ComputeError("text count exceeds Int64".into()))
        })
        .collect::<PolarsResult<_>>()?;
    Ok(Series::new(ca.name().clone(), out))
}

#[polars_expr(output_type_func=int_output)]
pub fn word_count(inputs: &[Series]) -> PolarsResult<Series> {
    count_string_values(inputs, ldaca_rs::text::word_count)
}

#[polars_expr(output_type_func=int_output)]
pub fn sentence_count(inputs: &[Series]) -> PolarsResult<Series> {
    count_string_values(inputs, ldaca_rs::text::sentence_count)
}

#[cfg(feature = "tokenization")]
#[polars_expr(output_type_func=list_struct_output)]
pub fn concordance(inputs: &[Series], kwargs: ConcordanceKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let mut spans = Vec::with_capacity(ca.len());
    let mut flat = Vec::new();
    let matcher = Concordance::new(
        &kwargs.search_word,
        ConcordanceOptions {
            left_tokens: usize::try_from(kwargs.num_left_tokens)
                .map_err(|_| PolarsError::ComputeError("left_tokens must be nonnegative".into()))?,
            right_tokens: usize::try_from(kwargs.num_right_tokens).map_err(|_| {
                PolarsError::ComputeError("right_tokens must be nonnegative".into())
            })?,
            regex: kwargs.regex,
            case_sensitive: kwargs.case_sensitive,
            ignore_punctuation: kwargs.ignore_punctuation,
        },
    )
    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    for opt_text in ca.iter() {
        let start = flat.len();
        let text = match opt_text {
            Some(value) => value,
            None => {
                spans.push((start, start));
                continue;
            }
        };

        let matches = matcher
            .find(text)
            .map_err(|e| PolarsError::ComputeError(format!("Concordance failed: {e}").into()))?;
        flat.extend(matches);
        spans.push((start, flat.len()));
    }
    list_from_spans(
        ca.name().clone(),
        &struct_series_from_matches(flat)?,
        &spans,
    )
}

#[cfg(feature = "tokenization")]
#[derive(serde::Deserialize)]
struct TokenizeKwargs {
    lowercase: bool,
    remove_punct: bool,
    model_id: String,
    #[serde(default)]
    cache: Option<String>,
}

#[cfg(feature = "tokenization")]
fn token_offset_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("token".into(), DataType::String),
        Field::new("start".into(), DataType::Int64),
        Field::new("end".into(), DataType::Int64),
    ])
}

#[cfg(feature = "tokenization")]
fn list_token_struct_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(token_offset_struct_type())),
    ))
}

/// Build one StructChunked from the flat (token, start, end) columns of
/// **all rows**. Used by the tokenize plugin so the per-row loop only has
/// to remember each row's `[start_idx, end_idx)` slice into the flat struct
/// rather than allocate three fresh Series + a fresh StructChunked per row.
#[cfg(feature = "tokenization")]
fn flat_struct_series_from_tokens(
    tok_col: Vec<String>,
    start_col: Vec<i64>,
    end_col: Vec<i64>,
) -> PolarsResult<Series> {
    let n = tok_col.len();
    let fields = [
        Series::new("token".into(), tok_col),
        Series::new("start".into(), start_col),
        Series::new("end".into(), end_col),
    ];
    Ok(StructChunked::from_series(PlSmallStr::EMPTY, n, fields.iter())?.into_series())
}

#[cfg(feature = "tokenization")]
fn build_token_list_series(
    name: PlSmallStr,
    row_count: usize,
    row_spans: Vec<(usize, usize)>,
    tok_col: Vec<String>,
    start_col: Vec<i64>,
    end_col: Vec<i64>,
) -> PolarsResult<Series> {
    let inner = flat_struct_series_from_tokens(tok_col, start_col, end_col)
        .map_err(|e| PolarsError::ComputeError(format!("Struct build failed: {e}").into()))?;

    if row_spans.len() != row_count {
        return Err(PolarsError::ComputeError(
            "token list row count does not match span count".into(),
        ));
    }
    list_from_spans(name, &inner, &row_spans)
}

#[cfg(feature = "embedding")]
#[derive(serde::Deserialize)]
struct EmbeddingKwargs {
    embedder_model: Option<String>,
    #[serde(default)]
    cache: Option<String>,
    #[serde(default)]
    batch_size: Option<usize>,
}

#[cfg(feature = "embedding")]
#[polars_expr(output_type_func=embedding_output)]
pub fn embedding(inputs: &[Series], kwargs: EmbeddingKwargs) -> PolarsResult<Series> {
    let embedder = Embedder::load(kwargs.embedder_model.as_deref())
        .map_err(|e| PolarsError::ComputeError(format!("Embedder init failed: {e:#}").into()))?;
    let batch_size = kwargs.batch_size.filter(|value| *value > 0).unwrap_or(32);

    let cache_path = kwargs.cache.as_deref();

    match inputs[0].dtype() {
        DataType::String => embed_string_series(&inputs[0], &embedder, batch_size, cache_path),
        DataType::List(inner) if inner.as_ref() == &DataType::String => {
            embed_list_string_series(&inputs[0], &embedder, batch_size, cache_path)
        }
        other => Err(PolarsError::InvalidOperation(
            format!("embedding expects String or List(String), got {other}").into(),
        )),
    }
}

#[cfg(feature = "embedding")]
fn encode_embedding_batches(
    embedder: &Embedder,
    texts: &[String],
    batch_size: usize,
    cache_path: Option<&str>,
) -> PolarsResult<Vec<Arc<Vec<f32>>>> {
    let result = if let Some(path) = cache_path {
        embedder.encode_cached(std::path::Path::new(path), texts, batch_size)
    } else {
        embedder
            .encode_batches(texts, batch_size)
            .map(|rows| rows.into_iter().map(Arc::new).collect())
    };
    result.map_err(|e| PolarsError::ComputeError(format!("Embedding failed: {e:#}").into()))
}

#[cfg(feature = "embedding")]
fn build_embedding_vector_list(
    name: PlSmallStr,
    row_spans: Vec<(usize, usize)>,
    flat: Vec<f32>,
) -> PolarsResult<Series> {
    list_from_spans(name, &Series::new(PlSmallStr::EMPTY, flat), &row_spans)
}

#[cfg(feature = "embedding")]
fn embed_string_series(
    input: &Series,
    embedder: &Embedder,
    batch_size: usize,
    cache_path: Option<&str>,
) -> PolarsResult<Series> {
    let ca = input.str()?;

    let mut texts: Vec<String> = Vec::new();
    let mut row_text_indices: Vec<Option<usize>> = Vec::with_capacity(ca.len());
    for opt_text in ca.iter() {
        match opt_text {
            Some(text) => {
                row_text_indices.push(Some(texts.len()));
                texts.push(text.to_string());
            }
            None => row_text_indices.push(None),
        }
    }

    let vectors = encode_embedding_batches(embedder, &texts, batch_size, cache_path)?;

    let mut flat: Vec<f32> = Vec::new();
    let mut row_spans: Vec<(usize, usize)> = Vec::with_capacity(row_text_indices.len());
    for text_index in row_text_indices {
        let start = flat.len();
        if let Some(index) = text_index {
            flat.extend_from_slice(&vectors[index]);
        }
        row_spans.push((start, flat.len()));
    }

    build_embedding_vector_list(ca.name().clone(), row_spans, flat)
}

#[cfg(feature = "embedding")]
fn embed_list_string_series(
    input: &Series,
    embedder: &Embedder,
    batch_size: usize,
    cache_path: Option<&str>,
) -> PolarsResult<Series> {
    let ca = input.list()?;
    let mut texts: Vec<String> = Vec::new();
    let mut item_text_indices: Vec<Option<usize>> = Vec::new();
    let mut row_item_spans: Vec<(usize, usize)> = Vec::with_capacity(ca.len());

    for opt_inner in ca.amortized_iter() {
        let row_start = item_text_indices.len();
        if let Some(inner) = opt_inner {
            let inner_ca = inner.as_ref().str()?;
            for opt_text in inner_ca.iter() {
                match opt_text {
                    Some(text) => {
                        item_text_indices.push(Some(texts.len()));
                        texts.push(text.to_string());
                    }
                    None => item_text_indices.push(None),
                }
            }
        }
        row_item_spans.push((row_start, item_text_indices.len()));
    }

    let vectors = encode_embedding_batches(embedder, &texts, batch_size, cache_path)?;
    let mut flat: Vec<f32> = Vec::new();
    let mut item_vector_spans: Vec<(usize, usize)> = Vec::with_capacity(item_text_indices.len());
    for text_index in item_text_indices {
        let start = flat.len();
        if let Some(index) = text_index {
            flat.extend_from_slice(&vectors[index]);
        }
        item_vector_spans.push((start, flat.len()));
    }

    let vector_list = build_embedding_vector_list(PlSmallStr::EMPTY, item_vector_spans, flat)?;
    list_from_spans(ca.name().clone(), &vector_list, &row_item_spans)
}

#[cfg(feature = "tokenization")]
#[polars_expr(output_type_func=list_token_struct_output)]
pub fn tokenize(inputs: &[Series], kwargs: TokenizeKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let tokenizer = Tokenizer::load(&kwargs.model_id)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let options = TokenizeOptions {
        lowercase: kwargs.lowercase,
        remove_punctuation: kwargs.remove_punct,
    };
    let (mut tok_col, mut start_col, mut end_col, mut spans) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::with_capacity(ca.len()),
    );
    let mut append = |tokens: Vec<ldaca_rs::text::Token>| {
        let start = tok_col.len();
        for token in tokens {
            tok_col.push(token.token);
            start_col.push(token.start);
            end_col.push(token.end);
        }
        spans.push((start, tok_col.len()));
    };
    if let Some(path) = kwargs.cache.as_deref() {
        let texts = ca.iter().flatten().map(str::to_owned).collect::<Vec<_>>();
        let mut rows =
            ldaca_rs::text::tokenize_cached(&tokenizer, Path::new(path), &texts, options)
                .map_err(|e| {
                    PolarsError::ComputeError(format!("Token cache failed: {e:#}").into())
                })?
                .into_iter();
        for text in ca.iter() {
            append(if text.is_some() {
                rows.next()
                    .ok_or_else(|| PolarsError::ComputeError("token row missing".into()))?
            } else {
                Vec::new()
            });
        }
    } else {
        for text in ca.iter() {
            append(
                text.map(|text| tokenizer.tokenize(text, options))
                    .transpose()
                    .map_err(|e| {
                        PolarsError::ComputeError(format!("Tokenization failed: {e:#}").into())
                    })?
                    .unwrap_or_default(),
            );
        }
    }
    build_token_list_series(
        ca.name().clone(),
        ca.len(),
        spans,
        tok_col,
        start_col,
        end_col,
    )
}

#[cfg(feature = "quotation")]
fn quotation_output(input: &[Field]) -> PolarsResult<Field> {
    crate::quotation::output(input)
}

#[cfg(feature = "quotation")]
#[polars_expr(output_type_func=quotation_output)]
pub(crate) fn quotation(
    inputs: &[Series],
    kwargs: crate::quotation::QuotationKwargs,
) -> PolarsResult<Series> {
    crate::quotation::expression(inputs, kwargs)
}

#[polars_expr(output_type_func=int_output)]
pub fn char_count(inputs: &[Series]) -> PolarsResult<Series> {
    count_string_values(inputs, ldaca_rs::text::char_count)
}
#[polars_expr(output_type=String)]
pub fn clean_text(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    Ok(Series::new(
        ca.name().clone(),
        ca.iter()
            .map(|text| ldaca_rs::text::clean_text(text.unwrap_or("")))
            .collect::<Vec<_>>(),
    ))
}
