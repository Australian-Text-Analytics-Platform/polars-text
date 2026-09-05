//! English quotation extraction over an owned, thread-confined UDPipe model.
mod bridge;
mod document;
mod normalize;
mod rules;

use document::Document;
use normalize::Normalized;
use polars::prelude::*;
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
};

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("quotation model I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("quotation parser: {0}")]
    Parser(#[from] cxx::Exception),
    #[error("quotation parser returned invalid dependency or source ranges")]
    InvalidParse,
    #[error("quotation model is already borrowed on this thread")]
    Borrow,
}

struct CachedModel {
    path: PathBuf,
    model: cxx::UniquePtr<bridge::ffi::Model>,
}
thread_local! { static MODEL: RefCell<Option<CachedModel>> = const { RefCell::new(None) }; }

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

fn extract(source: &str, path: &Path) -> Result<Vec<rules::Quote>, Error> {
    let normalized = Normalized::new(source);
    let parsed = MODEL.with(|slot| {
        let mut slot = slot.try_borrow_mut().map_err(|_| Error::Borrow)?;
        let path = path.canonicalize()?;
        if slot.as_ref().is_none_or(|cached| cached.path != path) {
            let bytes = std::fs::read(&path)?;
            let model = bridge::ffi::load_model(&bytes)?;
            *slot = Some(CachedModel { path, model });
        }
        let cached = slot.as_mut().ok_or(Error::InvalidParse)?;
        Ok::<_, Error>(cached.model.pin_mut().parse(&normalized.text)?)
    })?;
    let doc = Document::new(&normalized.text, parsed)?;
    Ok(rules::extract(&doc, &normalized, source))
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
        col!("speaker", |q: &rules::Quote| q
            .speaker
            .as_ref()
            .map(|s| s.0.as_str())),
        col!("speaker_start_idx", |q: &rules::Quote| q
            .speaker
            .as_ref()
            .map(|s| s.1)),
        col!("speaker_end_idx", |q: &rules::Quote| q
            .speaker
            .as_ref()
            .map(|s| s.2)),
        col!("quote", |q: &rules::Quote| q.quote.0.as_str()),
        col!("quote_start_idx", |q: &rules::Quote| q.quote.1),
        col!("quote_end_idx", |q: &rules::Quote| q.quote.2),
        col!("verb", |q: &rules::Quote| q
            .verb
            .as_ref()
            .map(|s| s.0.as_str())),
        col!("verb_start_idx", |q: &rules::Quote| q
            .verb
            .as_ref()
            .map(|s| s.1)),
        col!("verb_end_idx", |q: &rules::Quote| q
            .verb
            .as_ref()
            .map(|s| s.2)),
        col!("quote_type", |q: &rules::Quote| q.kind.as_str()),
        col!("quote_token_count", |q: &rules::Quote| q.tokens),
        col!("is_floating_quote", |q: &rules::Quote| q.floating),
        col!("quote_row_idx", |q: &rules::Quote| q.index),
    ];
    let flat = StructChunked::from_series("".into(), quotes.len(), columns.iter())?.into_series();
    crate::list_output::list_from_spans(input.name().clone(), &flat, &spans)
}

#[cfg(test)]
mod tests {
    use super::bridge::ffi;

    #[test]
    fn corrupt_model_is_an_error() {
        assert!(ffi::load_model(b"corrupt").is_err());
    }

    #[test]
    #[ignore = "requires WORDFLOW_TEST_UDPIPE_MODEL; run in the provisioned model job"]
    fn native_load_parse_drop_and_independent_threads() {
        let path = std::env::var("WORDFLOW_TEST_UDPIPE_MODEL")
            .expect("provision WORDFLOW_TEST_UDPIPE_MODEL before running model tests");
        let bytes = std::fs::read(path).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let bytes = &bytes;
                scope.spawn(move || {
                    for _ in 0..3 {
                        let mut model = ffi::load_model(bytes).unwrap();
                        let sentences = model
                            .pin_mut()
                            .parse(
                                "Noise\0 Alice said, \"The project will finish tomorrow morning.\"",
                            )
                            .unwrap();
                        assert!(sentences
                            .iter()
                            .flat_map(|s| &s.words)
                            .any(|w| w.form == "morning"));
                    }
                });
            }
        });
    }
}
