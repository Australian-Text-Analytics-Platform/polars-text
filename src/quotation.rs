use std::collections::HashSet;
use std::sync::OnceLock;

use anyhow::Result;
use polars::prelude::*;
use regex::Regex;

use crate::pos_tagging::pos_tags_for_text;

const MIN_QUOTE_TOKEN_COUNT: usize = 3;
const MAX_SPEAKER_TOKENS: usize = 60;
const VERB_WINDOW: i64 = 200;

const BOUNDARY_TOKENS: &[&str] = &[
    ".",
    "?",
    "!",
    ";",
    ",",
    "-",
    "--",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "'",
    "\"",
    "“",
    "”",
    "‘",
    "’",
    "…",
    "…",
];

const TITLE_WORDS: &[&str] = &[
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sir",
    "lady",
    "lord",
    "sen",
    "rep",
    "pres",
    "gov",
    "st",
    "rev",
    "hon",
];

const INVALID_SPEAKER_WORDS: &[&str] = &[
    "i",
    "we",
    "me",
    "us",
    "my",
    "our",
];

#[derive(Clone)]
struct TokenInfo {
    text: String,
    start: i64,
    end: i64,
    pos: String,
    sentence_index: usize,
}

#[derive(Clone)]
struct SentenceInfo {
    start: i64,
    end: i64,
    text: String,
}

#[derive(Clone)]
struct QuoteRecord {
    speaker: String,
    speaker_start_idx: Option<i64>,
    speaker_end_idx: Option<i64>,
    quote: String,
    quote_start_idx: i64,
    quote_end_idx: i64,
    verb: String,
    verb_start_idx: Option<i64>,
    verb_end_idx: Option<i64>,
    quote_type: String,
    quote_token_count: i64,
    is_floating_quote: bool,
    sentence_index: usize,
}

pub fn quotation_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("speaker".into(), DataType::String),
        Field::new("speaker_start_idx".into(), DataType::Int64),
        Field::new("speaker_end_idx".into(), DataType::Int64),
        Field::new("quote".into(), DataType::String),
        Field::new("quote_start_idx".into(), DataType::Int64),
        Field::new("quote_end_idx".into(), DataType::Int64),
        Field::new("verb".into(), DataType::String),
        Field::new("verb_start_idx".into(), DataType::Int64),
        Field::new("verb_end_idx".into(), DataType::Int64),
        Field::new("quote_type".into(), DataType::String),
        Field::new("quote_token_count".into(), DataType::Int64),
        Field::new("is_floating_quote".into(), DataType::Boolean),
    ])
}

pub fn quotation_list_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(quotation_struct_type())),
    ))
}

fn quote_verbs() -> &'static HashSet<String> {
    static VERBS: OnceLock<HashSet<String>> = OnceLock::new();
    VERBS.get_or_init(|| {
        let mut verbs = HashSet::new();
        let contents = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/resources/quote_verb.txt"));
        for line in contents.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                verbs.insert(trimmed.to_lowercase());
            }
        }
        verbs
    })
}

/// Convert a byte offset to a character (code-point) index.
/// Clamps to the nearest valid char boundary so it never panics.
fn byte_to_char_idx(text: &str, byte_idx: usize) -> i64 {
    let safe_idx = clamp_to_char_boundary(text, byte_idx);
    text[..safe_idx].chars().count() as i64
}

/// Return the largest valid char boundary <= `idx`, clamped to `text.len()`.
fn clamp_to_char_boundary(text: &str, idx: usize) -> usize {
    let idx = idx.min(text.len());
    // Walk backwards (at most 3 bytes for UTF-8) to find a char boundary.
    let mut i = idx;
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn prepare_tokens_and_sentences(text: &str) -> (Vec<TokenInfo>, Vec<SentenceInfo>) {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let end = idx + ch.len_utf8();
            if end > start {
                sentences.push(SentenceInfo {
                    start: byte_to_char_idx(text, start) as i64,
                    end: byte_to_char_idx(text, end) as i64,
                    text: text[start..end].to_string(),
                });
            }
            start = end;
        }
    }
    if start < text.len() {
        sentences.push(SentenceInfo {
            start: byte_to_char_idx(text, start) as i64,
            end: byte_to_char_idx(text, text.len()) as i64,
            text: text[start..].to_string(),
        });
    }

    let tokens = match pos_tags_for_text(text) {
        Ok(tags) => tags,
        Err(_) => Vec::new(),
    };

    let mut token_infos = Vec::new();
    for tag in tokens {
        let start = tag.start as usize;
        let end = tag.end as usize;
        if start >= text.len() || end > text.len() || start >= end {
            continue;
        }
        // Use .get() to avoid panicking on non-char-boundary byte offsets
        // (common with emoji, accented characters, and other multi-byte UTF-8).
        let token_text = match text.get(start..end) {
            Some(s) => s.to_string(),
            None => continue, // skip tokens with misaligned byte offsets
        };
        let char_start = byte_to_char_idx(text, start) as i64;
        let char_end = byte_to_char_idx(text, end) as i64;
        let sentence_index = sentences
            .iter()
            .position(|s| char_start >= s.start && char_end <= s.end)
            .unwrap_or(0);
        token_infos.push(TokenInfo {
            text: token_text,
            start: char_start,
            end: char_end,
            pos: tag.tag,
            sentence_index,
        });
    }

    (token_infos, sentences)
}

fn detect_quote_spans(text: &str) -> Vec<(i64, i64)> {
    let quote_re = Regex::new("[\\\"\\u{201C}\\u{201D}\\u{00AB}\\u{00BB}\\u{201E}\\u{201F}']")
        .expect("quote regex");
    // Collect (byte_start, byte_end) for each match so we use the actual
    // match end instead of `start + 1`, which can land mid-character for
    // multi-byte quote symbols like \u{201C} (3 bytes in UTF-8).
    let mut match_bounds: Vec<(usize, usize)> = quote_re
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect();
    match_bounds.sort_by_key(|&(s, _)| s);
    let mut spans = Vec::new();
    let mut iter = match_bounds.into_iter();
    while let Some((open_start, _open_end)) = iter.next() {
        if let Some((_close_start, close_end)) = iter.next() {
            if close_end >= open_start {
                spans.push((
                    byte_to_char_idx(text, open_start),
                    byte_to_char_idx(text, close_end),
                ));
            }
        }
    }
    spans
}

fn count_tokens_in_span(tokens: &[TokenInfo], span: (i64, i64)) -> usize {
    tokens
        .iter()
        .filter(|t| t.start >= span.0 && t.end <= span.1)
        .count()
}

fn is_boundary_token(token: &TokenInfo) -> bool {
    let text = token.text.trim();
    BOUNDARY_TOKENS.contains(&text)
}

fn is_name_token(token: &TokenInfo) -> bool {
    let lower = token.text.to_lowercase();
    if TITLE_WORDS.contains(&lower.as_str()) {
        return true;
    }
    if token.pos == "PROPN" || token.pos == "NNP" || token.pos == "NNPS" {
        return true;
    }
    token.text.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
}

fn is_pronoun_token(token: &TokenInfo) -> bool {
    if token.pos == "PRON" || token.pos == "PRP" || token.pos == "PRP$" {
        return !INVALID_SPEAKER_WORDS.contains(&token.text.to_lowercase().as_str());
    }
    false
}

fn token_distance(token: &TokenInfo, span: (i64, i64)) -> i64 {
    if token.start >= span.1 {
        token.start - span.1
    } else if token.end <= span.0 {
        span.0 - token.end
    } else {
        0
    }
}

fn find_nearest_verb(
    tokens: &[TokenInfo],
    quote_span: (i64, i64),
    sentence_index: Option<usize>,
) -> Option<(String, i64, i64, usize)> {
    let verbs = quote_verbs();
    let mut best: Option<(String, i64, i64, usize, i64)> = None;

    for (idx, token) in tokens.iter().enumerate() {
        if token.start >= quote_span.0 && token.end <= quote_span.1 {
            continue;
        }
        if let Some(sentence_idx) = sentence_index {
            if token.sentence_index != sentence_idx {
                continue;
            }
        }
        let token_lower = token.text.to_lowercase();
        if token_lower == "is" || token_lower == "was" || token_lower == "be" {
            continue;
        }

        let mut is_verb = token.pos.starts_with('V') || token.pos == "VERB" || token.pos == "AUX";
        if verbs.contains(&token_lower) {
            is_verb = true;
        }
        if token_lower == "according" {
            if let Some(next) = tokens.get(idx + 1) {
                if next.text.to_lowercase() == "to" {
                    let dist = token_distance(token, quote_span);
                    if dist <= VERB_WINDOW {
                        return Some((
                            "according to".to_string(),
                            token.start,
                            next.end,
                            idx,
                        ));
                    }
                }
            }
        }

        if !is_verb {
            continue;
        }

        let dist = token_distance(token, quote_span);
        if dist > VERB_WINDOW {
            continue;
        }

        if best
            .as_ref()
            .map(|(_, _, _, _, best_dist)| dist < *best_dist)
            .unwrap_or(true)
        {
            best = Some((token.text.clone(), token.start, token.end, idx, dist));
        }
    }

    best.map(|(verb, start, end, idx, _)| (verb, start, end, idx))
}

fn find_speaker_near(
    text: &str,
    tokens: &[TokenInfo],
    start_idx: usize,
    direction: i32,
) -> (String, Option<i64>, Option<i64>) {
    let mut idx = start_idx as i32 + direction;
    let mut hops = 0usize;
    let mut speaker_tokens = Vec::new();

    while idx >= 0 && (idx as usize) < tokens.len() && hops < MAX_SPEAKER_TOKENS {
        let token = &tokens[idx as usize];
        if is_boundary_token(token) {
            if !speaker_tokens.is_empty() {
                break;
            }
        }
        if is_name_token(token) || is_pronoun_token(token) {
            speaker_tokens.push(token.clone());
        } else if !speaker_tokens.is_empty() {
            break;
        }
        idx += direction;
        hops += 1;
    }

    if speaker_tokens.is_empty() {
        return (String::new(), None, None);
    }

    speaker_tokens.sort_by_key(|t| t.start);
    let start = speaker_tokens.first().unwrap().start;
    let end = speaker_tokens.last().unwrap().end;
    let speaker_text = text
        .chars()
        .skip(start as usize)
        .take((end - start) as usize)
        .collect::<String>();

    (
        speaker_text.trim().to_string(),
        Some(start),
        Some(end),
    )
}

fn compute_quote_type(
    quote_start: i64,
    quote_end: i64,
    verb_start: Option<i64>,
    verb_end: Option<i64>,
    speaker_start: Option<i64>,
    speaker_end: Option<i64>,
) -> String {
    let verb_start = verb_start.unwrap_or(-1);
    let verb_end = verb_end.unwrap_or(-1);
    let speaker_start = speaker_start.unwrap_or(-1);
    let speaker_end = speaker_end.unwrap_or(-1);
    let mut positions: Vec<(char, f64)> = Vec::new();
    if quote_start >= 0 {
        positions.push(('Q', quote_start as f64));
    }
    let content_mid = if quote_end > quote_start {
        (quote_start + quote_end) as f64 / 2.0
    } else {
        quote_start as f64
    };
    positions.push(('C', content_mid));
    if quote_end >= 0 {
        positions.push(('q', quote_end as f64));
    }
    if verb_start >= 0 && verb_end >= 0 {
        positions.push(('V', (verb_start + verb_end) as f64 / 2.0));
    }
    if speaker_start >= 0 && speaker_end >= 0 {
        positions.push(('S', (speaker_start + speaker_end) as f64 / 2.0));
    }
    positions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    positions
        .iter()
        .map(|(c, _)| if *c == 'q' { 'Q' } else { *c })
        .collect()
}

fn inherit_floating_quotes(records: &mut Vec<QuoteRecord>, sentences: &[SentenceInfo]) {
    if records.is_empty() {
        return;
    }
    records.sort_by_key(|r| r.quote_start_idx);
    let mut last_structured: Option<QuoteRecord> = None;

    for record in records.iter_mut() {
        let sentence = sentences.get(record.sentence_index);
        let sentence_starts_with_quote = sentence
            .map(|s| s.text.trim_start().starts_with('"') || s.text.trim_start().starts_with('“'))
            .unwrap_or(false);

        let can_inherit = last_structured
            .as_ref()
            .map(|s| !s.speaker.is_empty())
            .unwrap_or(false)
            && record.speaker.is_empty()
            && record.verb.is_empty()
            && sentence_starts_with_quote
            && last_structured
                .as_ref()
                .map(|s| record.sentence_index.saturating_sub(s.sentence_index) <= 5)
                .unwrap_or(false);

        if can_inherit {
            if let Some(last) = &last_structured {
                record.speaker = last.speaker.clone();
                record.speaker_start_idx = last.speaker_start_idx;
                record.speaker_end_idx = last.speaker_end_idx;
                record.is_floating_quote = true;
            }
        }

        if !record.speaker.is_empty() || !record.verb.is_empty() {
            last_structured = Some(record.clone());
        }
    }
}

fn deduplicate_quotes(records: Vec<QuoteRecord>) -> Vec<QuoteRecord> {
    let mut sorted = records;
    sorted.sort_by_key(|r| r.quote_start_idx);
    let mut result: Vec<QuoteRecord> = Vec::new();
    for record in sorted {
        let mut keep = true;
        for existing in result.iter_mut() {
            let overlaps = record.quote_start_idx <= existing.quote_end_idx
                && record.quote_end_idx >= existing.quote_start_idx;
            if overlaps {
                let existing_len = existing.quote_end_idx - existing.quote_start_idx;
                let new_len = record.quote_end_idx - record.quote_start_idx;
                if new_len > existing_len {
                    *existing = record.clone();
                }
                keep = false;
                break;
            }
        }
        if keep {
            result.push(record);
        }
    }
    result
}

fn build_struct_series(records: Vec<QuoteRecord>) -> Vec<Series> {
    let mut speaker = Vec::new();
    let mut speaker_start_idx = Vec::new();
    let mut speaker_end_idx = Vec::new();
    let mut quote = Vec::new();
    let mut quote_start_idx = Vec::new();
    let mut quote_end_idx = Vec::new();
    let mut verb = Vec::new();
    let mut verb_start_idx = Vec::new();
    let mut verb_end_idx = Vec::new();
    let mut quote_type = Vec::new();
    let mut quote_token_count = Vec::new();
    let mut is_floating_quote = Vec::new();

    for record in records {
        speaker.push(if record.speaker.is_empty() { None } else { Some(record.speaker) });
        speaker_start_idx.push(record.speaker_start_idx);
        speaker_end_idx.push(record.speaker_end_idx);
        quote.push(record.quote);
        quote_start_idx.push(record.quote_start_idx);
        quote_end_idx.push(record.quote_end_idx);
        verb.push(if record.verb.is_empty() { None } else { Some(record.verb) });
        verb_start_idx.push(record.verb_start_idx);
        verb_end_idx.push(record.verb_end_idx);
        quote_type.push(record.quote_type);
        quote_token_count.push(record.quote_token_count);
        is_floating_quote.push(record.is_floating_quote);
    }

    vec![
        Series::new("speaker".into(), speaker),
        Series::new("speaker_start_idx".into(), speaker_start_idx),
        Series::new("speaker_end_idx".into(), speaker_end_idx),
        Series::new("quote".into(), quote),
        Series::new("quote_start_idx".into(), quote_start_idx),
        Series::new("quote_end_idx".into(), quote_end_idx),
        Series::new("verb".into(), verb),
        Series::new("verb_start_idx".into(), verb_start_idx),
        Series::new("verb_end_idx".into(), verb_end_idx),
        Series::new("quote_type".into(), quote_type),
        Series::new("quote_token_count".into(), quote_token_count),
        Series::new("is_floating_quote".into(), is_floating_quote),
    ]
}

pub fn quotation_for_text(text: &str) -> Result<Vec<Series>> {
    let quote_spans = detect_quote_spans(text);
    if quote_spans.is_empty() {
        return Ok(Vec::new());
    }

    let (tokens, sentences) = prepare_tokens_and_sentences(text);
    let mut records: Vec<QuoteRecord> = Vec::new();

    let span_count = quote_spans.len();
    for (qs, qe) in quote_spans.iter().copied() {
        let sentence_idx = tokens
            .iter()
            .find(|t| t.start <= qs && t.end >= qs)
            .map(|t| t.sentence_index)
            .unwrap_or(0);

        let verb = find_nearest_verb(&tokens, (qs, qe), Some(sentence_idx))
            .or_else(|| find_nearest_verb(&tokens, (qs, qe), None));

        let (verb_text, verb_start, verb_end, verb_token_idx) = match verb {
            Some(v) => v,
            None => (String::new(), -1, -1, 0),
        };

        let quote_text = text
            .chars()
            .skip(qs as usize)
            .take((qe - qs) as usize)
            .collect::<String>();

        let quote_token_count = count_tokens_in_span(&tokens, (qs, qe));

        if quote_token_count < MIN_QUOTE_TOKEN_COUNT && span_count > 1 {
            continue;
        }

        let (speaker, speaker_start, speaker_end) = if verb_text.is_empty() {
            (String::new(), None, None)
        } else {
            let (speaker_text, s_start, s_end) = find_speaker_near(
                text,
                &tokens,
                verb_token_idx,
                if verb_start > (qs + qe) / 2 { -1 } else { 1 },
            );
            (speaker_text, s_start, s_end)
        };

        let quote_type = compute_quote_type(
            qs,
            qe,
            if verb_start < 0 { None } else { Some(verb_start) },
            if verb_end < 0 { None } else { Some(verb_end) },
            speaker_start,
            speaker_end,
        );

        records.push(QuoteRecord {
            speaker,
            speaker_start_idx: speaker_start,
            speaker_end_idx: speaker_end,
            quote: quote_text,
            quote_start_idx: qs,
            quote_end_idx: qe,
            verb: verb_text,
            verb_start_idx: if verb_start < 0 { None } else { Some(verb_start) },
            verb_end_idx: if verb_end < 0 { None } else { Some(verb_end) },
            quote_type,
            quote_token_count: quote_token_count as i64,
            is_floating_quote: false,
            sentence_index: sentence_idx,
        });
    }

    if records.len() > 1 {
        records = deduplicate_quotes(records);
    }
    inherit_floating_quotes(&mut records, &sentences);

    Ok(build_struct_series(records))
}

pub fn struct_series_from_matches(matches: Vec<Series>) -> PolarsResult<Series> {
    if matches.is_empty() {
        return Ok(Series::new_empty(PlSmallStr::EMPTY, &quotation_struct_type()));
    }
    Ok(
        StructChunked::from_series(PlSmallStr::EMPTY, matches[0].len(), matches.iter())?
            .into_series(),
    )
}
