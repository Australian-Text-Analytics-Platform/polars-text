use polars::chunked_array::builder::{AnonymousOwnedListBuilder, ListBuilderTrait};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::concordance::{
    concordance_for_text, concordance_struct_type, list_struct_output, struct_series_from_matches,
    ConcordanceKwargs,
};
use crate::tokenizer::ensure_tokenizer_for_model;
use crate::tokens_cache_io::{
    list_bucket_files, load_cache_map, resolve_cache_dir, write_delta, CONTENT_HASH_COLUMN,
};

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
    let ca = inputs[0].str()?;
    let mut out: Vec<i64> = Vec::with_capacity(ca.len());
    for opt_text in ca.into_iter() {
        let count = match opt_text {
            Some(text) => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    0
                } else if trimmed.chars().any(|c| c.is_whitespace()) {
                    trimmed.split_whitespace().count() as i64
                } else if trimmed.chars().all(is_cjk_word_char) {
                    trimmed.chars().count() as i64
                } else {
                    // Mixed CJK + non-CJK with no whitespace (e.g. CJK
                    // followed by Latin punctuation): treat the whole run
                    // as one word, matching the existing semantics.
                    1
                }
            }
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
            | '।' | '॥'             // Devanagari danda / double danda
    )
}

#[polars_expr(output_type_func=int_output)]
pub fn sentence_count(inputs: &[Series]) -> PolarsResult<Series> {
    // Phase 3.3: terminator set is Unicode-aware. EN flows are byte-identical
    // because the ASCII terminators are still in the set; CJK now splits on
    // ``。！？`` correctly instead of returning 1 for an entire paragraph.
    let ca = inputs[0].str()?;
    let mut out: Vec<i64> = Vec::with_capacity(ca.len());
    for opt_text in ca.into_iter() {
        let count = match opt_text {
            Some(text) => text
                .split(is_sentence_terminator)
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
    let backend = ensure_tokenizer_for_model(kwargs.model_id.as_deref())
        .map_err(|e| PolarsError::ComputeError(format!("Tokenizer init failed: {e}").into()))?;

    let mut out: Vec<Option<Series>> = Vec::with_capacity(ca.len());
    for opt_text in ca.into_iter() {
        let text = match opt_text {
            Some(value) => value,
            None => {
                out.push(Some(Series::new(PlSmallStr::EMPTY, Vec::<String>::new())));
                continue;
            }
        };

        let tokens = backend
            .tokenize_text(text, false, kwargs.lowercase, kwargs.remove_punct)
            .map_err(|e| PolarsError::ComputeError(format!("Tokenization failed: {e}").into()))?;
        out.push(Some(Series::new(PlSmallStr::EMPTY, tokens)));
    }

    let mut list = ListChunked::from_iter(out);
    list.rename(ca.name().clone());
    Ok(list.into_series())
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

fn list_string_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(DataType::String)),
    ))
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

/// Two-input variant — names the output after the FIRST input (the
/// source text column), ignoring the second (the precomputed hash). Used
/// by `tokenize_with_cache_lookup`, which takes (source, hash) but
/// returns just the tokens list. Keeping a separate function (instead
/// of reusing `list_token_struct_output`) is just future-proofing —
/// today both behave identically because both name the output after
/// index 0, but having a dedicated function makes the
/// "second input is metadata, not column source" intent explicit.
fn list_token_struct_output_from_two_inputs(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(token_offset_struct_type())),
    ))
}

fn struct_series_from_tokens(tokens: Vec<(String, i64, i64)>) -> Series {
    let mut tok_col: Vec<String> = Vec::with_capacity(tokens.len());
    let mut start_col: Vec<i64> = Vec::with_capacity(tokens.len());
    let mut end_col: Vec<i64> = Vec::with_capacity(tokens.len());
    for (t, s, e) in tokens {
        tok_col.push(t);
        start_col.push(s);
        end_col.push(e);
    }
    let n = tok_col.len();
    let fields = vec![
        Series::new("token".into(), tok_col),
        Series::new("start".into(), start_col),
        Series::new("end".into(), end_col),
    ];
    StructChunked::from_series(PlSmallStr::EMPTY, n, fields.iter())
        .expect("struct build should succeed")
        .into_series()
}

#[polars_expr(output_type_func=list_token_struct_output)]
pub fn tokenize_with_offsets(inputs: &[Series], kwargs: TokenizeKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let backend = ensure_tokenizer_for_model(kwargs.model_id.as_deref())
        .map_err(|e| PolarsError::ComputeError(format!("Tokenizer init failed: {e}").into()))?;

    let mut builder = AnonymousOwnedListBuilder::new(
        PlSmallStr::EMPTY,
        ca.len(),
        Some(token_offset_struct_type()),
    );

    for opt_text in ca.into_iter() {
        let text = match opt_text {
            Some(value) => value,
            None => {
                builder.append_empty();
                continue;
            }
        };

        let tokens = backend
            .tokenize_text_with_offsets(text, kwargs.lowercase, kwargs.remove_punct)
            .map_err(|e| PolarsError::ComputeError(format!("Tokenize with offsets failed: {e}").into()))?;

        if tokens.is_empty() {
            builder.append_empty();
        } else {
            let struct_series = struct_series_from_tokens(tokens);
            builder.append_series(&struct_series).map_err(|e| {
                PolarsError::ComputeError(format!("List builder failed: {e}").into())
            })?;
        }
    }

    let mut list = builder.finish();
    list.rename(ca.name().clone());
    Ok(list.into_series())
}

/// Kwargs for the lazy tokens-cache lookup expression. Stays in sync
/// with `polars_text.tokenize_with_cache_lookup` on the Python side
/// (see `polars_text/functions.py`). All fields are baked into the
/// serialised `.plbin` plan; none of them carry absolute paths — that
/// is the central portability win of the lazy design.
#[derive(serde::Deserialize)]
struct TokenizeWithCacheKwargs {
    // Tokeniser knobs — same semantics as `TokenizeKwargs`. The
    // expression delegates per-row tokenisation to the same backend
    // `tokenize_with_offsets` uses, so output is byte-identical on
    // cache miss.
    lowercase: bool,
    remove_punct: bool,
    #[serde(default)]
    model_id: Option<String>,

    // Cache plumbing. The bucket filename is a stable hash of
    // (model, params) — same value Python's `tokens_cache.cache_filename`
    // produces, so both sides read/write the same files. `user_id`
    // selects the per-user subdir under `LDACA_TOKENS_CACHE_DIR`.
    bucket_filename: String,
    user_id: String,

    // Controls whether a missing `LDACA_TOKENS_CACHE_DIR` env at
    // execution time is fatal. Set to true in production analysis
    // workers (bootstrap sets the env); leave false in tests so the
    // fallback /tmp path keeps the test suite hermetic.
    #[serde(default = "default_require_env")]
    require_env_cache_dir: bool,
}

fn default_require_env() -> bool {
    false
}

/// Lazy tokens-cache lookup expression.
///
/// Takes two inputs:
///
/// * `inputs[0]` — the source text column (`Utf8`).
/// * `inputs[1]` — the precomputed content-hash column (`UInt64`),
///   produced on the Python side by `pl.col(source).hash()`. We accept
///   it precomputed (instead of recomputing inside Rust) so the cache
///   lookup is guaranteed bit-compatible with whatever hash function
///   the Python eager pipeline uses to populate the same bucket — and
///   so Phase 2.5 plan migration preserves cache hits without us having
///   to match polars' internal hash defaults from the Rust side.
///
/// Per-row semantics:
///
/// 1. Look the precomputed hash up in the per-bucket cache map (loaded
///    once per expression invocation from disk under
///    `LDACA_TOKENS_CACHE_DIR / user_id / tokens / <bucket>{,__delta__*}.parquet`).
/// 2. Cache hit → emit the cached `List<Struct<token, start, end>>` row.
///    Cache miss → call the same `tokenize_text_with_offsets` backend
///    that `tokenize_with_offsets` uses, emit the result AND queue the
///    (hash, tokens) pair for a fresh delta-file write.
/// 3. If any rows missed, write `<bucket>__delta__<uuid>.parquet`
///    under an advisory `flock` so concurrent workers don't race over
///    the same bytes.
///
/// `is_elementwise = false` on the Python wrapper side — the expression
/// maintains state across rows within a batch (the in-memory cache map),
/// so polars must not split a batch in ways that violate that invariant.
#[polars_expr(output_type_func=list_token_struct_output_from_two_inputs)]
pub fn tokenize_with_cache_lookup(
    inputs: &[Series],
    kwargs: TokenizeWithCacheKwargs,
) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "tokenize_with_cache_lookup: expected 2 inputs (source, precomputed_hash); \
             pass `pl.col(source).hash()` as the second argument from the Python wrapper."
                .into(),
        ));
    }
    let ca = inputs[0].str()?;
    let hashes = inputs[1].u64().map_err(|_| {
        PolarsError::ComputeError(
            "tokenize_with_cache_lookup: second input must be UInt64 (the precomputed \
             content hash from `pl.col(source).hash()`)"
                .into(),
        )
    })?;
    let cache_dir = resolve_cache_dir(&kwargs.user_id, kwargs.require_env_cache_dir)?;
    let bucket_files = list_bucket_files(&cache_dir, &kwargs.bucket_filename);
    let cache_map = load_cache_map(&bucket_files)?;

    let mut builder = AnonymousOwnedListBuilder::new(
        PlSmallStr::EMPTY,
        ca.len(),
        Some(token_offset_struct_type()),
    );

    // Rows that missed the cache and need to be tokenised + written
    // back as a new delta. Held until the row scan finishes so a
    // single delta file captures the whole batch.
    let mut miss_hashes: Vec<u64> = Vec::new();
    let mut miss_token_lists: Vec<Series> = Vec::new();

    // Lazily ensure the tokeniser backend exists — only on first miss.
    // Cache-only hot paths (every row is a hit) never pay the cost of
    // loading the tokeniser at all.
    let mut backend_cell: Option<std::sync::Arc<crate::tokenizer::TokenizerBackend>> = None;

    for idx in 0..ca.len() {
        let Some(text) = ca.get(idx) else {
            builder.append_empty();
            continue;
        };
        let Some(h) = hashes.get(idx) else {
            // Null hash — shouldn't happen for non-null strings, but
            // belt-and-braces: tokenise without caching this row.
            let tokens =
                tokenise_with_lazy_backend(&mut backend_cell, kwargs.model_id.as_deref(), text, kwargs.lowercase, kwargs.remove_punct)?;
            push_tokens_row(&mut builder, tokens)?;
            continue;
        };

        if let Some(cached) = cache_map.get(&h) {
            // Cache HIT — push the cached row from the in-memory map.
            // Falls through to the miss path on schema mismatch; this
            // future-proofs against a cache parquet with a different
            // dtype after a polars upgrade.
            match cached {
                AnyValue::List(s) => {
                    builder.append_series(s).map_err(|e| {
                        PolarsError::ComputeError(
                            format!("tokenize_with_cache_lookup: cached row append failed: {e}")
                                .into(),
                        )
                    })?;
                    continue;
                }
                AnyValue::Null => {
                    builder.append_empty();
                    continue;
                }
                _ => {
                    // Unexpected dtype — fall through to recompute.
                }
            }
        }

        // Cache MISS — tokenise from scratch, push to output, and queue
        // for the post-loop delta write.
        let tokens = tokenise_with_lazy_backend(
            &mut backend_cell,
            kwargs.model_id.as_deref(),
            text,
            kwargs.lowercase,
            kwargs.remove_punct,
        )?;
        let token_series = if tokens.is_empty() {
            builder.append_empty();
            // Even empty token lists belong in the cache — recomputing
            // them is non-zero work and identical-input → identical-output
            // is a hard invariant we depend on.
            empty_tokens_struct_series()
        } else {
            let s = struct_series_from_tokens(tokens);
            builder.append_series(&s).map_err(|e| {
                PolarsError::ComputeError(
                    format!("tokenize_with_cache_lookup: tokenise row append failed: {e}").into(),
                )
            })?;
            s
        };
        miss_hashes.push(h);
        miss_token_lists.push(token_series);
    }

    // Persist misses as a fresh delta parquet so subsequent collects on
    // this plan (or on related plans sharing the same bucket) get the
    // cached value back without re-tokenising.
    if !miss_hashes.is_empty() {
        write_misses_to_delta(
            &cache_dir,
            &kwargs.bucket_filename,
            miss_hashes,
            miss_token_lists,
        )?;
    }

    let mut list = builder.finish();
    list.rename(ca.name().clone());
    Ok(list.into_series())
}

/// Empty `Struct<token, start, end>` series — used as the "no tokens"
/// payload for empty-text cache rows so the cache parquet schema is
/// uniform whether or not a given input row produced tokens.
fn empty_tokens_struct_series() -> Series {
    let tok = Series::new_empty("token".into(), &DataType::String);
    let start = Series::new_empty("start".into(), &DataType::Int64);
    let end = Series::new_empty("end".into(), &DataType::Int64);
    StructChunked::from_series(PlSmallStr::EMPTY, 0, [tok, start, end].iter())
        .expect("empty struct build should succeed")
        .into_series()
}

/// Tokeniser-backend lazy init — only loads the model on first miss so
/// cache-only hot paths skip the cost entirely.
fn tokenise_with_lazy_backend(
    backend_cell: &mut Option<std::sync::Arc<crate::tokenizer::TokenizerBackend>>,
    model_id: Option<&str>,
    text: &str,
    lowercase: bool,
    remove_punct: bool,
) -> PolarsResult<Vec<(String, i64, i64)>> {
    if backend_cell.is_none() {
        let b = ensure_tokenizer_for_model(model_id)
            .map_err(|e| PolarsError::ComputeError(format!("Tokenizer init failed: {e}").into()))?;
        *backend_cell = Some(b);
    }
    let backend = backend_cell.as_ref().expect("just inserted");
    backend
        .tokenize_text_with_offsets(text, lowercase, remove_punct)
        .map_err(|e| PolarsError::ComputeError(format!("Tokenize with offsets failed: {e}").into()))
}

/// Append a row of pre-computed tokens to the list builder. Extracted so
/// the null-hash branch above can reuse the same struct-encoding path.
fn push_tokens_row(
    builder: &mut AnonymousOwnedListBuilder,
    tokens: Vec<(String, i64, i64)>,
) -> PolarsResult<()> {
    if tokens.is_empty() {
        builder.append_empty();
        return Ok(());
    }
    let s = struct_series_from_tokens(tokens);
    builder.append_series(&s).map_err(|e| {
        PolarsError::ComputeError(format!("tokens row append failed: {e}").into())
    })
}

/// Encode the queued misses as a 2-column DataFrame (`hash`, `tokens`)
/// matching `TOKENS_CACHE_SCHEMA` and append to the bucket as a fresh
/// delta parquet. No-op if `miss_hashes` is empty — caller guarantees.
fn write_misses_to_delta(
    cache_dir: &std::path::Path,
    bucket_filename: &str,
    miss_hashes: Vec<u64>,
    miss_token_lists: Vec<Series>,
) -> PolarsResult<()> {
    let hash_series = Series::new(CONTENT_HASH_COLUMN.into(), miss_hashes);
    let mut list_builder = AnonymousOwnedListBuilder::new(
        "tokens".into(),
        miss_token_lists.len(),
        Some(token_offset_struct_type()),
    );
    for s in &miss_token_lists {
        list_builder.append_series(s).map_err(|e| {
            PolarsError::ComputeError(
                format!("tokenize_with_cache_lookup: delta-row encode failed: {e}").into(),
            )
        })?;
    }
    let tokens_series = list_builder.finish().into_series();
    let mut df = DataFrame::new_infer_height(vec![hash_series.into(), tokens_series.into()])?;
    write_delta(cache_dir, bucket_filename, &mut df)?;
    Ok(())
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
