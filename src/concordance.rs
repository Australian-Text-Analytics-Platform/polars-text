use anyhow::Result;
use polars::prelude::*;
use regex::RegexBuilder;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::tokenizer::tokenize_plain_text;

#[derive(Deserialize)]
pub struct ConcordanceKwargs {
    pub search_word: String,
    pub num_left_tokens: i64,
    pub num_right_tokens: i64,
    pub regex: bool,
    pub case_sensitive: bool,
}

pub fn list_struct_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(concordance_struct_type())),
    ))
}

pub fn concordance_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("left_context".into(), DataType::String),
        Field::new("matched_text".into(), DataType::String),
        Field::new("right_context".into(), DataType::String),
        Field::new("start_idx".into(), DataType::Int64),
        Field::new("end_idx".into(), DataType::Int64),
        Field::new("l1".into(), DataType::String),
        Field::new("r1".into(), DataType::String),
    ])
}

fn empty_struct_series() -> Series {
    let fields = vec![
        Series::new("left_context".into(), Vec::<String>::new()),
        Series::new("matched_text".into(), Vec::<String>::new()),
        Series::new("right_context".into(), Vec::<String>::new()),
        Series::new("start_idx".into(), Vec::<i64>::new()),
        Series::new("end_idx".into(), Vec::<i64>::new()),
        Series::new("l1".into(), Vec::<String>::new()),
        Series::new("r1".into(), Vec::<String>::new()),
    ];
    StructChunked::from_series(PlSmallStr::EMPTY, 0, fields.iter())
        .expect("empty struct build should succeed")
        .into_series()
}

fn byte_to_char_idx(text: &str, byte_idx: usize) -> i64 {
    text[..byte_idx].chars().count() as i64
}

fn detokenize(tokens: &[String]) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    tokens.join(" ")
}

pub fn concordance_for_text(
    _tokenizer: &Tokenizer,
    text: &str,
    kwargs: &ConcordanceKwargs,
) -> Result<Vec<Series>> {
    if kwargs.search_word.is_empty() {
        return Ok(Vec::new());
    }

    let pattern = if kwargs.regex {
        kwargs.search_word.clone()
    } else {
        regex::escape(&kwargs.search_word)
    };

    let matcher = RegexBuilder::new(&pattern)
        .case_insensitive(!kwargs.case_sensitive)
        .build()?;

    let mut left_contexts = Vec::new();
    let mut matched_texts = Vec::new();
    let mut right_contexts = Vec::new();
    let mut start_indices = Vec::new();
    let mut end_indices = Vec::new();
    let mut l1_vals = Vec::new();
    let mut r1_vals = Vec::new();

    for m in matcher.find_iter(text) {
        let start_byte = m.start();
        let end_byte = m.end();
        let start_idx = byte_to_char_idx(text, start_byte);
        let end_idx = byte_to_char_idx(text, end_byte);

        let left_text = &text[..start_byte];
        let right_text = &text[end_byte..];

        let left_tokens = tokenize_plain_text(left_text, false, false);
        let right_tokens = tokenize_plain_text(right_text, false, false);

        let left_take = kwargs.num_left_tokens.max(0) as usize;
        let right_take = kwargs.num_right_tokens.max(0) as usize;

        let left_slice = if left_take == 0 {
            Vec::new()
        } else if left_tokens.len() <= left_take {
            left_tokens.clone()
        } else {
            left_tokens[left_tokens.len() - left_take..].to_vec()
        };

        let right_slice = if right_take == 0 {
            Vec::new()
        } else if right_tokens.len() <= right_take {
            right_tokens.clone()
        } else {
            right_tokens[..right_take].to_vec()
        };

        let l1 = left_slice.last().cloned().unwrap_or_default();
        let r1 = right_slice.first().cloned().unwrap_or_default();

        left_contexts.push(detokenize(&left_slice));
        matched_texts.push(m.as_str().to_string());
        right_contexts.push(detokenize(&right_slice));
        start_indices.push(start_idx);
        end_indices.push(end_idx);
        l1_vals.push(l1);
        r1_vals.push(r1);
    }

    if matched_texts.is_empty() {
        return Ok(Vec::new());
    }

    let series = vec![
        Series::new("left_context".into(), left_contexts),
        Series::new("matched_text".into(), matched_texts),
        Series::new("right_context".into(), right_contexts),
        Series::new("start_idx".into(), start_indices),
        Series::new("end_idx".into(), end_indices),
        Series::new("l1".into(), l1_vals),
        Series::new("r1".into(), r1_vals),
    ];

    Ok(series)
}

pub fn struct_series_from_matches(matches: Vec<Series>) -> Series {
    if matches.is_empty() {
        return empty_struct_series();
    }
    let length = matches
        .first()
        .map(|series| series.len())
        .unwrap_or(0);
    StructChunked::from_series(PlSmallStr::EMPTY, length, matches.iter())
        .expect("struct build should succeed")
        .into_series()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_to_char_idx_multibyte() {
        let text = "café";
        let byte_idx = text.find('é').expect("é should be found");
        assert_eq!(byte_to_char_idx(text, byte_idx), 3);

        let text = "hi 🙂 there";
        let byte_idx = text.find('🙂').expect("emoji should be found");
        assert_eq!(byte_to_char_idx(text, byte_idx), 3);
    }

    #[test]
    fn test_detokenize() {
        assert_eq!(detokenize(&[]), "");

        let tokens = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(detokenize(&tokens), "hello world");
    }
}
