use polars::chunked_array::builder::{AnonymousOwnedListBuilder, ListBuilderTrait};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::concordance::{
    concordance_for_text, concordance_struct_type, list_struct_output, struct_series_from_matches,
    ConcordanceKwargs,
};
use crate::tokenizer::ensure_tokenizer_for_model;

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

fn map_string_values(
    inputs: &[Series],
    mut map: impl FnMut(&str) -> String,
) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: Vec<String> = ca
        .into_iter()
        .map(|opt_text| opt_text.map(&mut map).unwrap_or_default())
        .collect();
    Ok(Series::new(ca.name().clone(), out))
}

fn count_string_values(
    inputs: &[Series],
    mut count: impl FnMut(&str) -> i64,
) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: Vec<i64> = ca
        .into_iter()
        .map(|opt_text| opt_text.map(&mut count).unwrap_or(0))
        .collect();
    Ok(Series::new(ca.name().clone(), out))
}

#[polars_expr(output_type_func=string_output)]
pub fn clean_text(inputs: &[Series]) -> PolarsResult<Series> {
    map_string_values(inputs, clean_text_value)
}

/// Heuristic test for "writing-system characters that are their own word
/// boundary" — covers Han ideographs (used by Chinese and parts of
/// Japanese/Korean), Hiragana, Katakana, and Hangul syllables. Punctuation
/// and spaces are intentionally excluded because they aren't words.
fn is_cjk_word_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF      // CJK Unified Ideographs
            | 0x3400..=0x4DBF // CJK Extension A
            | 0x20000..=0x2A6DF // CJK Extension B
            | 0x3040..=0x309F // Hiragana
            | 0x30A0..=0x30FF // Katakana
            | 0xAC00..=0xD7AF // Hangul Syllables
    )
}

#[polars_expr(output_type_func=int_output)]
pub fn word_count(inputs: &[Series]) -> PolarsResult<Series> {
    // Phase 3.4: ``split_whitespace`` returns 1 for pure-CJK text because
    // CJK orthography has no inter-word whitespace. Detect that case and
    // count each CJK character as one word — a coarse heuristic that gives
    // a meaningful non-zero count on Chinese / Japanese corpora without a
    // tokenizer round-trip. For real word-level counts post-Tokenise, use
    // ``pl.col(derived_tokens_col).list.len()``. English / whitespace-
    // tokenised flows are byte-identical: any text with internal whitespace
    // still goes through ``split_whitespace``.
    count_string_values(inputs, |text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            0
        } else if trimmed.chars().any(|c| c.is_whitespace()) {
            trimmed.split_whitespace().count() as i64
        } else if trimmed.chars().all(is_cjk_word_char) {
            trimmed.chars().count() as i64
        } else {
            // Mixed CJK + non-CJK with no whitespace (e.g. CJK followed by
            // Latin punctuation): treat the whole run as one word, matching
            // the existing semantics.
            1
        }
    })
}

#[polars_expr(output_type_func=int_output)]
pub fn char_count(inputs: &[Series]) -> PolarsResult<Series> {
    count_string_values(inputs, |text| text.chars().count() as i64)
}

/// Sentence terminators covered by ``sentence_count``. ASCII ``. ! ?``
/// plus the full-width variants used in Chinese and Japanese writing, plus
/// a few common non-Latin terminators we've seen in real corpora. Add
/// future scripts here rather than at the call site so the contract stays
/// in one place.
fn is_sentence_terminator(ch: char) -> bool {
    matches!(
        ch,
        '.' | '!' | '?'           // ASCII
            | '。' | '！' | '？'    // CJK full-width (Chinese, Japanese)
            | '۔'                  // Arabic full stop
            | '؟'                  // Arabic question mark
            | '।' | '॥' // Devanagari danda / double danda
    )
}

#[polars_expr(output_type_func=int_output)]
pub fn sentence_count(inputs: &[Series]) -> PolarsResult<Series> {
    // Phase 3.3: terminator set is Unicode-aware. EN flows are byte-identical
    // because the ASCII terminators are still in the set; CJK now splits on
    // ``。！？`` correctly instead of returning 1 for an entire paragraph.
    count_string_values(inputs, |text| {
        text.split(is_sentence_terminator)
            .filter(|segment| !segment.trim().is_empty())
            .count() as i64
    })
}

#[polars_expr(output_type_func=list_struct_output)]
pub fn concordance(inputs: &[Series], kwargs: ConcordanceKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

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

        let matches = concordance_for_text(text, &kwargs)
            .map_err(|e| PolarsError::ComputeError(format!("Concordance failed: {e}").into()))?;
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

#[derive(serde::Deserialize)]
struct TokenizeKwargs {
    lowercase: bool,
    remove_punct: bool,
    #[serde(default)]
    model_id: Option<String>,
}

fn token_offset_struct_type() -> DataType {
    DataType::Struct(vec![
        Field::new("token".into(), DataType::String),
        Field::new("start".into(), DataType::Int64),
        Field::new("end".into(), DataType::Int64),
    ])
}

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
fn flat_struct_series_from_tokens(
    tok_col: Vec<String>,
    start_col: Vec<i64>,
    end_col: Vec<i64>,
) -> PolarsResult<Series> {
    let n = tok_col.len();
    let fields = vec![
        Series::new("token".into(), tok_col),
        Series::new("start".into(), start_col),
        Series::new("end".into(), end_col),
    ];
    Ok(StructChunked::from_series(PlSmallStr::EMPTY, n, fields.iter())?.into_series())
}

#[polars_expr(output_type_func=list_token_struct_output)]
pub fn tokenize(inputs: &[Series], kwargs: TokenizeKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let backend = ensure_tokenizer_for_model(kwargs.model_id.as_deref())
        .map_err(|e| PolarsError::ComputeError(format!("Tokenizer init failed: {e}").into()))?;

    // Phase 1: tokenise every row, accumulating into flat per-field Vecs
    // and recording each row's [start, end) span into the flat struct.
    // Pre-allocate optimistically — chosen to dominate on multi-hundred-
    // token CJK rows where the prior per-row allocation pattern hurt most.
    let estimated_tokens = ca.len().saturating_mul(32);
    let mut tok_col: Vec<String> = Vec::with_capacity(estimated_tokens);
    let mut start_col: Vec<i64> = Vec::with_capacity(estimated_tokens);
    let mut end_col: Vec<i64> = Vec::with_capacity(estimated_tokens);
    let mut row_spans: Vec<(usize, usize)> = Vec::with_capacity(ca.len());

    for opt_text in ca.into_iter() {
        let span_start = tok_col.len();
        match opt_text {
            Some(text) => {
                let tokens = backend
                    .tokenize_text_with_offsets(text, kwargs.lowercase, kwargs.remove_punct)
                    .map_err(|e| {
                        PolarsError::ComputeError(format!("Tokenization failed: {e}").into())
                    })?;
                for (t, s, e) in tokens {
                    tok_col.push(t);
                    start_col.push(s);
                    end_col.push(e);
                }
            }
            None => {
                // Null input maps to an empty list element (matches the
                // prior behaviour, which also folded null and empty into
                // ``append_empty``).
            }
        }
        row_spans.push((span_start, tok_col.len()));
    }

    // Phase 2: build the inner struct ONCE from the flat columns.
    let inner = flat_struct_series_from_tokens(tok_col, start_col, end_col)
        .map_err(|e| PolarsError::ComputeError(format!("Struct build failed: {e}").into()))?;

    // Phase 3: emit list rows by slicing the shared inner struct.
    // ``Series::slice`` is O(1) — it just shifts the chunk's offset/length —
    // so per-row cost here is bounded by what ``AnonymousOwnedListBuilder``
    // must do internally to materialise the list, not by additional struct
    // construction.
    let mut builder = AnonymousOwnedListBuilder::new(
        PlSmallStr::EMPTY,
        ca.len(),
        Some(token_offset_struct_type()),
    );
    for (s, e) in row_spans {
        if e == s {
            builder.append_empty();
        } else {
            let slice = inner.slice(s as i64, e - s);
            builder.append_series(&slice).map_err(|err| {
                PolarsError::ComputeError(format!("List builder failed: {err}").into())
            })?;
        }
    }

    let mut list = builder.finish();
    list.rename(ca.name().clone());
    Ok(list.into_series())
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
}
