//! Polars quotation schema and row conversion.
use ldaca_rs::quotation::{QuotationExtractor, Quote};
use polars::prelude::*;
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
};
thread_local! { static MODEL: RefCell<Option<(PathBuf, QuotationExtractor)>> = const { RefCell::new(None) }; }
fn extract(source: &str, path: &Path) -> PolarsResult<Vec<Quote>> {
    MODEL.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        let path = path.canonicalize().map_err(|e| {
            PolarsError::ComputeError(ldaca_rs::quotation::Error::Io(e).to_string().into())
        })?;
        if slot.as_ref().is_none_or(|(cached, _)| *cached != path) {
            let model = QuotationExtractor::load(&path)
                .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
            *slot = Some((path, model));
        }
        let (_, model) = slot
            .as_mut()
            .ok_or_else(|| PolarsError::ComputeError("quotation model unavailable".into()))?;
        model
            .extract(source)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
    })
}
#[derive(serde::Deserialize)]
pub(crate) struct QuotationKwargs {
    model_path: String,
}

pub(crate) fn output(input: &[Field]) -> PolarsResult<Field> {
    if input[0].dtype() != &DataType::String {
        polars_bail!(InvalidOperation: "quotation expects String input");
    }
    Ok(Field::new(
        input[0].name().clone(),
        DataType::List(Box::new(DataType::Struct(fields()))),
    ))
}

fn fields() -> Vec<Field> {
    [
        ("speaker", DataType::String),
        ("speaker_start_idx", DataType::Int64),
        ("speaker_end_idx", DataType::Int64),
        ("quote", DataType::String),
        ("quote_start_idx", DataType::Int64),
        ("quote_end_idx", DataType::Int64),
        ("verb", DataType::String),
        ("verb_start_idx", DataType::Int64),
        ("verb_end_idx", DataType::Int64),
        ("quote_type", DataType::String),
        ("quote_token_count", DataType::Int64),
        ("is_floating_quote", DataType::Boolean),
        ("quote_row_idx", DataType::Int64),
    ]
    .into_iter()
    .map(|(name, dtype)| Field::new(name.into(), dtype))
    .collect()
}

pub(crate) fn expression(inputs: &[Series], kwargs: QuotationKwargs) -> PolarsResult<Series> {
    let input = inputs[0].str()?;
    let mut quotes = Vec::new();
    let mut spans = Vec::with_capacity(input.len());
    for text in input.iter() {
        let start = quotes.len();
        if let Some(text) = text.filter(|text| !text.trim().is_empty()) {
            quotes.extend(
                extract(text, Path::new(&kwargs.model_path))
                    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?,
            );
        }
        spans.push((start, quotes.len()));
    }
    macro_rules! col {
        ($name:literal, $value:expr) => {
            Series::new($name.into(), quotes.iter().map($value).collect::<Vec<_>>())
        };
    }
    let columns = vec![
        col!("speaker", |q: &Quote| q
            .speaker
            .as_ref()
            .map(|s| s.0.as_str())),
        col!("speaker_start_idx", |q: &Quote| q
            .speaker
            .as_ref()
            .map(|s| s.1)),
        col!("speaker_end_idx", |q: &Quote| q
            .speaker
            .as_ref()
            .map(|s| s.2)),
        col!("quote", |q: &Quote| q.quote.0.as_str()),
        col!("quote_start_idx", |q: &Quote| q.quote.1),
        col!("quote_end_idx", |q: &Quote| q.quote.2),
        col!("verb", |q: &Quote| q.verb.as_ref().map(|s| s.0.as_str())),
        col!("verb_start_idx", |q: &Quote| q.verb.as_ref().map(|s| s.1)),
        col!("verb_end_idx", |q: &Quote| q.verb.as_ref().map(|s| s.2)),
        col!("quote_type", |q: &Quote| q.kind.as_str()),
        col!("quote_token_count", |q: &Quote| q.tokens),
        col!("is_floating_quote", |q: &Quote| q.floating),
        col!("quote_row_idx", |q: &Quote| q.index),
    ];
    let flat = StructChunked::from_series("".into(), quotes.len(), columns.iter())?.into_series();
    crate::list_output::list_from_spans(input.name().clone(), &flat, &spans)
}
