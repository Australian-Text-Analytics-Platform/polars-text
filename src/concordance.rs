use crate::offsets::byte_spans_to_char_spans;
use crate::tokenizer::{tokenize_plain_text, tokenize_plain_text_with_offsets};
use anyhow::Result;
use polars::prelude::*;
use regex::RegexBuilder;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct ConcordanceKwargs {
    pub search_word: String,
    pub num_left_tokens: i64,
    pub num_right_tokens: i64,
    pub regex: bool,
    pub case_sensitive: bool,
    #[serde(default)]
    pub remove_punct: bool,
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

fn detokenize(tokens: &[String]) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    tokens.join(" ")
}

fn char_offset_to_byte_offset(text: &str, char_offset: i64) -> usize {
    if char_offset <= 0 {
        return 0;
    }

    text.char_indices()
        .nth(char_offset as usize)
        .map_or(text.len(), |(byte_offset, _)| byte_offset)
}

fn raw_context_windows(
    left_text: &str,
    right_text: &str,
    left_take: usize,
    right_take: usize,
) -> (String, String, String, String) {
    let left_tokens = tokenize_plain_text_with_offsets(left_text, false, true);
    let right_tokens = tokenize_plain_text_with_offsets(right_text, false, true);

    let left_start = left_tokens.len().saturating_sub(left_take);
    let left_slice = if left_take == 0 {
        &left_tokens[0..0]
    } else {
        &left_tokens[left_start..]
    };
    let right_end = right_take.min(right_tokens.len());
    let right_slice = &right_tokens[..right_end];

    let left_context = left_slice
        .first()
        .map_or_else(String::new, |(_, start, _)| {
            left_text[char_offset_to_byte_offset(left_text, *start)..].to_string()
        });
    let right_context = right_slice.last().map_or_else(String::new, |(_, _, end)| {
        right_text[..char_offset_to_byte_offset(right_text, *end)].to_string()
    });
    let l1 = left_slice
        .last()
        .map(|(token, _, _)| token.clone())
        .unwrap_or_default();
    let r1 = right_slice
        .first()
        .map(|(token, _, _)| token.clone())
        .unwrap_or_default();

    (left_context, right_context, l1, r1)
}

pub fn concordance_for_text(text: &str, kwargs: &ConcordanceKwargs) -> Result<Vec<Series>> {
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

    // Collect (start_byte, end_byte, matched_text) for every regex hit, then
    // convert all byte offsets to char offsets in a single forward sweep. The
    // prior per-match `text[..byte_idx].chars().count()` was O(C·M) which
    // dominated CPU on CJK documents with many hits; this is O(C + M).
    let hits: Vec<(usize, usize, String)> = matcher
        .find_iter(text)
        .map(|m| (m.start(), m.end(), m.as_str().to_string()))
        .collect();

    let char_spans = byte_spans_to_char_spans(text, hits.iter().map(|(s, e, _)| (*s, *e)));

    for ((start_byte, end_byte, matched), (start_idx, end_idx)) in
        hits.iter().zip(char_spans.into_iter())
    {
        let start_byte = *start_byte;
        let end_byte = *end_byte;

        let left_text = &text[..start_byte];
        let right_text = &text[end_byte..];

        let left_take = kwargs.num_left_tokens.max(0) as usize;
        let right_take = kwargs.num_right_tokens.max(0) as usize;

        let (left_context, right_context, l1, r1) = if kwargs.remove_punct {
            raw_context_windows(left_text, right_text, left_take, right_take)
        } else {
            let left_tokens = tokenize_plain_text(left_text, false, false);
            let right_tokens = tokenize_plain_text(right_text, false, false);
            let left_start = left_tokens.len().saturating_sub(left_take);
            let left_slice = if left_take == 0 {
                &left_tokens[0..0]
            } else {
                &left_tokens[left_start..]
            };
            let right_end = right_take.min(right_tokens.len());
            let right_slice = &right_tokens[..right_end];

            (
                detokenize(left_slice),
                detokenize(right_slice),
                left_slice.last().cloned().unwrap_or_default(),
                right_slice.first().cloned().unwrap_or_default(),
            )
        };

        left_contexts.push(left_context);
        matched_texts.push(matched.clone());
        right_contexts.push(right_context);
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
    let length = matches.first().map(|series| series.len()).unwrap_or(0);
    StructChunked::from_series(PlSmallStr::EMPTY, length, matches.iter())
        .expect("struct build should succeed")
        .into_series()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detokenize() {
        assert_eq!(detokenize(&[]), "");

        let tokens = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(detokenize(&tokens), "hello world");
    }
}
