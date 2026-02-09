use polars::chunked_array::builder::{AnonymousOwnedListBuilder, ListBuilderTrait};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::concordance::{
    concordance_for_text,
    concordance_struct_type,
    list_struct_output,
    ConcordanceKwargs,
    struct_series_from_matches,
};
use crate::quotation::{
    quotation_list_output,
    quotation_for_text,
    quotation_struct_type,
    struct_series_from_matches as quotation_struct_series,
};
use crate::tokenizer::{ensure_tokenizer, tokenize_text};

fn string_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(input_fields[0].name().clone(), DataType::String))
}

fn int_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(input_fields[0].name().clone(), DataType::Int64))
}

fn clean_text_value(text: &str) -> String {
    let lowered = text.to_lowercase();
    let mut cleaned = String::with_capacity(lowered.len());
    for ch in lowered.chars() {
        if ch.is_ascii_punctuation() {
            cleaned.push(' ');
        } else if ch.is_ascii_digit() {
            cleaned.push(' ');
        } else {
            cleaned.push(ch);
        }
    }
    let mut normalized = String::new();
    let mut last_space = false;
    for ch in cleaned.chars() {
        if ch.is_whitespace() {
            if !last_space {
                normalized.push(' ');
                last_space = true;
            }
        } else {
            normalized.push(ch);
            last_space = false;
        }
    }
    normalized.trim().to_string()
}

#[polars_expr(output_type_func=string_output)]
pub fn clean_text(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let mut out: Vec<String> = Vec::with_capacity(ca.len());
    for opt_text in ca.into_iter() {
        match opt_text {
            Some(text) => out.push(clean_text_value(text)),
            None => out.push(String::new()),
        }
    }
    Ok(Series::new(ca.name().clone(), out))
}

#[polars_expr(output_type_func=int_output)]
pub fn word_count(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let mut out: Vec<i64> = Vec::with_capacity(ca.len());
    for opt_text in ca.into_iter() {
        let count = match opt_text {
            Some(text) => text.split_whitespace().count() as i64,
            None => 0,
        };
        out.push(count);
    }
    Ok(Series::new(ca.name().clone(), out))
}

#[polars_expr(output_type_func=int_output)]
pub fn char_count(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let mut out: Vec<i64> = Vec::with_capacity(ca.len());
    for opt_text in ca.into_iter() {
        let count = match opt_text {
            Some(text) => text.chars().count() as i64,
            None => 0,
        };
        out.push(count);
    }
    Ok(Series::new(ca.name().clone(), out))
}

#[polars_expr(output_type_func=int_output)]
pub fn sentence_count(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let mut out: Vec<i64> = Vec::with_capacity(ca.len());
    for opt_text in ca.into_iter() {
        let count = match opt_text {
            Some(text) => text
                .split(|c| c == '.' || c == '!' || c == '?')
                .filter(|segment| !segment.trim().is_empty())
                .count() as i64,
            None => 0,
        };
        out.push(count);
    }
    Ok(Series::new(ca.name().clone(), out))
}

#[polars_expr(output_type_func=list_string_output)]
pub fn tokenize(inputs: &[Series], kwargs: TokenizeKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let tokenizer = ensure_tokenizer().map_err(|e| {
        PolarsError::ComputeError(format!("Tokenizer init failed: {e}").into())
    })?;

    let mut out: Vec<Option<Series>> = Vec::with_capacity(ca.len());
    for opt_text in ca.into_iter() {
        let text = match opt_text {
            Some(value) => value,
            None => {
                out.push(Some(Series::new(PlSmallStr::EMPTY, Vec::<String>::new())));
                continue;
            }
        };

        let tokens = tokenize_text(tokenizer, text, kwargs.lowercase, kwargs.remove_punct)
            .map_err(|e| {
                PolarsError::ComputeError(
                    format!("Tokenization failed: {e}").into(),
                )
            })?;
        out.push(Some(Series::new(PlSmallStr::EMPTY, tokens)));
    }

    let mut list = ListChunked::from_iter(out);
    list.rename(ca.name().clone());
    Ok(list.into_series())
}

#[polars_expr(output_type_func=list_struct_output)]
pub fn concordance(inputs: &[Series], kwargs: ConcordanceKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let tokenizer = ensure_tokenizer().map_err(|e| {
        PolarsError::ComputeError(format!("Tokenizer init failed: {e}").into())
    })?;

    let mut builder = AnonymousOwnedListBuilder::new(
        PlSmallStr::EMPTY,
        ca.len(),
        Some(concordance_struct_type()),
    );

    for opt_text in ca.into_iter() {
        let text = match opt_text {
            Some(value) => value,
            None => {
                builder.append_empty();
                continue;
            }
        };

        let matches = concordance_for_text(tokenizer, text, &kwargs).map_err(|e| {
            PolarsError::ComputeError(format!("Concordance failed: {e}").into())
        })?;
        if matches.is_empty() {
            builder.append_empty();
        } else {
            let struct_series = struct_series_from_matches(matches);
            builder.append_series(&struct_series).map_err(|e| {
                PolarsError::ComputeError(format!("Concordance failed: {e}").into())
            })?;
        }
    }

    let mut list = builder.finish();
    list.rename(ca.name().clone());
    Ok(list.into_series())
}

#[polars_expr(output_type_func=quotation_list_output)]
pub fn quotation(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    let mut builder = AnonymousOwnedListBuilder::new(
        PlSmallStr::EMPTY,
        ca.len(),
        Some(quotation_struct_type()),
    );

    for opt_text in ca.into_iter() {
        let text = match opt_text {
            Some(value) => value,
            None => {
                builder.append_empty();
                continue;
            }
        };

        let matches = quotation_for_text(text).map_err(|e| {
            PolarsError::ComputeError(format!("Quotation failed: {e}").into())
        })?;
        if matches.is_empty() {
            builder.append_empty();
        } else {
            let struct_series = quotation_struct_series(matches);
            builder.append_series(&struct_series).map_err(|e| {
                PolarsError::ComputeError(format!("Quotation failed: {e}").into())
            })?;
        }
    }

    let mut list = builder.finish();
    list.rename(ca.name().clone());
    Ok(list.into_series())
}

#[derive(serde::Deserialize)]
struct TokenizeKwargs {
    lowercase: bool,
    remove_punct: bool,
}

fn list_string_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(DataType::String)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text_value_normalizes() {
        let cleaned = clean_text_value("Hello, World! 123");
        assert_eq!(cleaned, "hello world");

        let cleaned = clean_text_value("  Hi--there\t42 ");
        assert_eq!(cleaned, "hi there");
    }

    #[test]
    fn test_list_string_output_type() -> PolarsResult<()> {
        let field = Field::new(PlSmallStr::from("text"), DataType::String);
        let output = list_string_output(&[field])?;
        assert_eq!(output.dtype(), &DataType::List(Box::new(DataType::String)));
        Ok(())
    }
}
