use crate::offsets::byte_spans_to_char_spans;
use crate::tokenizer::tokenize_plain_text_with_offsets;
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
    pub ignore_punctuation: bool,
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

fn empty_struct_series() -> PolarsResult<Series> {
    let fields = [
        Series::new("left_context".into(), Vec::<String>::new()),
        Series::new("matched_text".into(), Vec::<String>::new()),
        Series::new("right_context".into(), Vec::<String>::new()),
        Series::new("start_idx".into(), Vec::<i64>::new()),
        Series::new("end_idx".into(), Vec::<i64>::new()),
        Series::new("l1".into(), Vec::<String>::new()),
        Series::new("r1".into(), Vec::<String>::new()),
    ];
    Ok(StructChunked::from_series(PlSmallStr::EMPTY, 0, fields.iter())?.into_series())
}

#[derive(Debug)]
struct SourceToken {
    text: String,
    start: usize,
    end: usize,
}

fn source_tokens(text: &str, ignore_punctuation: bool) -> Result<Vec<SourceToken>> {
    let char_count = text.chars().count();
    tokenize_plain_text_with_offsets(text, false, ignore_punctuation)
        .into_iter()
        .map(|(token, start, end)| {
            let start = usize::try_from(start)
                .map_err(|_| anyhow::anyhow!("tokenizer returned a negative start offset"))?;
            let end = usize::try_from(end)
                .map_err(|_| anyhow::anyhow!("tokenizer returned a negative end offset"))?;
            if start > end || end > char_count {
                anyhow::bail!(
                    "tokenizer returned invalid character span {start}..{end} for {char_count} characters"
                );
            }
            Ok(SourceToken {
                text: token,
                start,
                end,
            })
        })
        .collect()
}

fn offset_fragment_tokens(
    fragment: &str,
    start_offset: usize,
    ignore_punctuation: bool,
) -> Result<Vec<SourceToken>> {
    let mut tokens = source_tokens(fragment, ignore_punctuation)?;
    for token in &mut tokens {
        token.start += start_offset;
        token.end += start_offset;
    }
    Ok(tokens)
}

struct ContextWindow {
    context_start: Option<usize>,
    context_end: Option<usize>,
    adjacent_left: String,
    adjacent_right: String,
}

struct ContextRequest<'a> {
    text: &'a str,
    char_to_byte: &'a [usize],
    tokens: &'a [SourceToken],
    match_chars: (usize, usize),
    complete_tokens: (usize, usize),
    take: (usize, usize),
    ignore_punctuation: bool,
}

fn raw_context_window(request: ContextRequest<'_>) -> Result<ContextWindow> {
    let ContextRequest {
        text,
        char_to_byte,
        tokens,
        match_chars: (start_char, end_char),
        complete_tokens: (left_complete_end, right_complete_start),
        take: (left_take, right_take),
        ignore_punctuation,
    } = request;
    let left_fragment = tokens
        .get(left_complete_end)
        .filter(|token| token.start < start_char && start_char < token.end)
        .map_or(Ok(Vec::new()), |token| {
            let fragment = &text[char_to_byte[token.start]..char_to_byte[start_char]];
            offset_fragment_tokens(fragment, token.start, ignore_punctuation)
        })?;
    let left_fragment_take = left_take.min(left_fragment.len());
    let left_full_take = left_take.saturating_sub(left_fragment_take);
    let left_full_start = left_complete_end.saturating_sub(left_full_take);
    let context_start = if left_take == 0 {
        None
    } else if left_full_start < left_complete_end {
        Some(tokens[left_full_start].start)
    } else {
        left_fragment
            .get(left_fragment.len().saturating_sub(left_fragment_take))
            .map(|token| token.start)
    };
    let adjacent_left = if left_fragment_take > 0 {
        left_fragment
            .last()
            .map(|token| token.text.clone())
            .unwrap_or_default()
    } else {
        tokens
            .get(left_complete_end.saturating_sub(1))
            .filter(|_| left_full_take > 0)
            .map(|token| token.text.clone())
            .unwrap_or_default()
    };

    let right_intersecting = right_complete_start
        .checked_sub(1)
        .and_then(|index| tokens.get(index))
        .filter(|token| token.start < end_char && end_char < token.end);
    let right_fragment = right_intersecting.map_or(Ok(Vec::new()), |token| {
        let fragment = &text[char_to_byte[end_char]..char_to_byte[token.end]];
        offset_fragment_tokens(fragment, end_char, ignore_punctuation)
    })?;
    let right_fragment_take = right_take.min(right_fragment.len());
    let right_full_take = right_take.saturating_sub(right_fragment_take);
    let right_full_end = (right_complete_start + right_full_take).min(tokens.len());
    let context_end = if right_take == 0 {
        None
    } else if right_full_end > right_complete_start {
        Some(tokens[right_full_end - 1].end)
    } else {
        right_fragment
            .get(right_fragment_take.saturating_sub(1))
            .map(|token| token.end)
    };
    let adjacent_right = if right_fragment_take > 0 {
        right_fragment
            .first()
            .map(|token| token.text.clone())
            .unwrap_or_default()
    } else {
        tokens
            .get(right_complete_start)
            .filter(|_| right_full_take > 0)
            .map(|token| token.text.clone())
            .unwrap_or_default()
    };

    Ok(ContextWindow {
        context_start,
        context_end,
        adjacent_left,
        adjacent_right,
    })
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
    let tokens = source_tokens(text, kwargs.ignore_punctuation)?;
    let char_to_byte = text
        .char_indices()
        .map(|(byte_offset, _)| byte_offset)
        .chain(std::iter::once(text.len()))
        .collect::<Vec<_>>();
    let mut left_complete_end = 0;
    let mut right_complete_start = 0;

    for ((start_byte, end_byte, matched), (start_idx, end_idx)) in hits.iter().zip(char_spans) {
        let start_byte = *start_byte;
        let end_byte = *end_byte;
        let start_char = usize::try_from(start_idx)
            .map_err(|_| anyhow::anyhow!("negative concordance start offset"))?;
        let end_char = usize::try_from(end_idx)
            .map_err(|_| anyhow::anyhow!("negative concordance end offset"))?;

        while tokens
            .get(left_complete_end)
            .is_some_and(|token| token.end <= start_char)
        {
            left_complete_end += 1;
        }
        right_complete_start = right_complete_start.max(left_complete_end);
        while tokens
            .get(right_complete_start)
            .is_some_and(|token| token.start < end_char)
        {
            right_complete_start += 1;
        }

        let left_take = kwargs.num_left_tokens.max(0) as usize;
        let right_take = kwargs.num_right_tokens.max(0) as usize;

        let window = raw_context_window(ContextRequest {
            text,
            char_to_byte: &char_to_byte,
            tokens: &tokens,
            match_chars: (start_char, end_char),
            complete_tokens: (left_complete_end, right_complete_start),
            take: (left_take, right_take),
            ignore_punctuation: kwargs.ignore_punctuation,
        })?;
        let left_context = window.context_start.map_or_else(String::new, |start| {
            text[char_to_byte[start]..start_byte].to_string()
        });
        let right_context = window.context_end.map_or_else(String::new, |end| {
            text[end_byte..char_to_byte[end]].to_string()
        });

        left_contexts.push(left_context);
        matched_texts.push(matched.clone());
        right_contexts.push(right_context);
        start_indices.push(start_idx);
        end_indices.push(end_idx);
        l1_vals.push(window.adjacent_left);
        r1_vals.push(window.adjacent_right);
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

pub fn struct_series_from_matches(matches: Vec<Series>) -> PolarsResult<Series> {
    if matches.is_empty() {
        return empty_struct_series();
    }
    let length = matches.first().map(|series| series.len()).unwrap_or(0);
    Ok(StructChunked::from_series(PlSmallStr::EMPTY, length, matches.iter())?.into_series())
}
