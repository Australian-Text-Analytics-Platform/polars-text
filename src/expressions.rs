#[cfg(feature = "tokenization")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "tokenization")]
use std::path::Path;

#[cfg(feature = "tokenization")]
use anyhow::{Context, Result as AnyhowResult};
#[cfg(feature = "tokenization")]
use duckdb::{params, Connection, Error as DuckDbError};
#[cfg(any(feature = "embedding", feature = "tokenization"))]
use polars::chunked_array::builder::{AnonymousOwnedListBuilder, ListBuilderTrait};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[cfg(feature = "tokenization")]
use crate::cache::{get_or_insert_text_values, hash_text, TextCacheTable};
#[cfg(feature = "tokenization")]
use crate::concordance::{
    concordance_for_text, concordance_struct_type, list_struct_output, struct_series_from_matches,
    ConcordanceKwargs,
};
#[cfg(feature = "tokenization")]
use crate::tokenizer::{ensure_tokenizer_for_model, TokenizerBackend};
#[cfg(feature = "embedding")]
use crate::topic_modeling::embedding::{ensure_embedder, Embedder};
#[cfg(feature = "embedding")]
use crate::topic_modeling::embedding_cache::{get_or_insert_embeddings, CacheScope};

fn string_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(input_fields[0].name().clone(), DataType::String))
}

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

#[cfg(feature = "tokenization")]
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

#[cfg(feature = "tokenization")]
#[derive(serde::Deserialize)]
struct TokenizeKwargs {
    lowercase: bool,
    remove_punct: bool,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    cache: Option<String>,
}

#[cfg(feature = "tokenization")]
const TOKEN_CACHE_SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS token_cache (
    model VARCHAR NOT NULL,
    params_hash VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL,
    tokens VARCHAR[] NOT NULL,
    start_offsets BIGINT[] NOT NULL,
    end_offsets BIGINT[] NOT NULL,
    PRIMARY KEY (model, params_hash, content_hash)
)
"#;

#[cfg(feature = "tokenization")]
#[derive(serde::Serialize)]
struct TokenCacheParams {
    lowercase: bool,
    remove_punct: bool,
}

#[cfg(feature = "tokenization")]
#[derive(Clone)]
struct TokenCacheEntry {
    tokens: Vec<String>,
    starts: Vec<i64>,
    ends: Vec<i64>,
}

#[cfg(feature = "tokenization")]
impl TokenCacheEntry {
    fn from_offsets(offsets: Vec<(String, i64, i64)>) -> Self {
        let mut tokens = Vec::with_capacity(offsets.len());
        let mut starts = Vec::with_capacity(offsets.len());
        let mut ends = Vec::with_capacity(offsets.len());
        for (token, start, end) in offsets {
            tokens.push(token);
            starts.push(start);
            ends.push(end);
        }
        Self {
            tokens,
            starts,
            ends,
        }
    }

    fn len(&self) -> usize {
        self.tokens
            .len()
            .min(self.starts.len())
            .min(self.ends.len())
    }

    fn append_to(
        &self,
        tok_col: &mut Vec<String>,
        start_col: &mut Vec<i64>,
        end_col: &mut Vec<i64>,
    ) {
        for index in 0..self.len() {
            tok_col.push(self.tokens[index].clone());
            start_col.push(self.starts[index]);
            end_col.push(self.ends[index]);
        }
    }
}

#[cfg(feature = "tokenization")]
struct TokenCacheTable<'a> {
    model_id: &'a str,
    params_hash: &'a str,
}

pub(crate) type TokenCacheDebugRow =
    (String, String, String, Vec<String>, Vec<i64>, Vec<i64>);

#[cfg(feature = "tokenization")]
pub(crate) fn debug_token_cache_snapshot(
    path: &Path,
) -> AnyhowResult<(Vec<String>, Vec<TokenCacheDebugRow>)> {
    let conn = Connection::open(path).with_context(|| format!("open cache {}", path.display()))?;

    let mut schema_stmt = conn
        .prepare("DESCRIBE token_cache")
        .context("prepare token cache schema snapshot")?;
    let schema_rows = schema_stmt
        .query_map([], |row| row.get::<_, String>(0))
        .context("query token cache schema snapshot")?;
    let mut columns = Vec::new();
    for column in schema_rows {
        columns.push(column.context("read token cache schema row")?);
    }

    let mut row_stmt = conn
        .prepare(
            r#"
            SELECT model, params_hash, content_hash,
                   to_json(tokens), to_json(start_offsets), to_json(end_offsets)
            FROM token_cache
            ORDER BY content_hash
            "#,
        )
        .context("prepare token cache row snapshot")?;
    let row_items = row_stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })
        .context("query token cache row snapshot")?;

    let mut rows = Vec::new();
    for row in row_items {
        let (model, params_hash, content_hash, tokens_json, starts_json, ends_json) =
            row.context("read token cache snapshot row")?;
        rows.push((
            model,
            params_hash,
            content_hash,
            serde_json::from_str(&tokens_json).context("decode token cache snapshot tokens")?,
            serde_json::from_str(&starts_json).context("decode token cache snapshot starts")?,
            serde_json::from_str(&ends_json).context("decode token cache snapshot ends")?,
        ));
    }

    Ok((columns, rows))
}

#[cfg(feature = "tokenization")]
impl TextCacheTable for TokenCacheTable<'_> {
    type Value = TokenCacheEntry;

    fn schema_sql(&self) -> &'static str {
        TOKEN_CACHE_SCHEMA_SQL
    }

    fn fetch_cached(
        &self,
        conn: &Connection,
        hashes: &[String],
    ) -> AnyhowResult<HashMap<String, Self::Value>> {
        let mut out = HashMap::new();
        let mut stmt = conn
            .prepare(
                r#"
                                SELECT to_json(tokens), to_json(start_offsets), to_json(end_offsets)
                FROM token_cache
                WHERE model = ?
                  AND params_hash = ?
                  AND content_hash = ?
                "#,
            )
            .context("prepare token cache lookup")?;

        for hash in hashes.iter().collect::<HashSet<_>>() {
            let row_result =
                stmt.query_row(params![self.model_id, self.params_hash, hash], |row| {
                    let tokens_json: String = row.get(0)?;
                    let starts_json: String = row.get(1)?;
                    let ends_json: String = row.get(2)?;
                    Ok((tokens_json, starts_json, ends_json))
                });
            match row_result {
                Ok((tokens_json, starts_json, ends_json)) => {
                    let entry = TokenCacheEntry {
                        tokens: serde_json::from_str(&tokens_json)
                            .context("decode token cache tokens")?,
                        starts: serde_json::from_str(&starts_json)
                            .context("decode token cache starts")?,
                        ends: serde_json::from_str(&ends_json)
                            .context("decode token cache ends")?,
                    };
                    out.insert(hash.clone(), entry);
                }
                Err(DuckDbError::QueryReturnedNoRows) => {}
                Err(err) => return Err(err).context("read token cache row"),
            }
        }

        Ok(out)
    }

    fn persist_new(
        &self,
        conn: &Connection,
        entries: &[(String, Self::Value)],
    ) -> AnyhowResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut stmt = conn
            .prepare(
                r#"
                INSERT OR IGNORE INTO token_cache
                (model, params_hash, content_hash, tokens, start_offsets, end_offsets)
                VALUES (?, ?, ?, ?::VARCHAR[], ?::BIGINT[], ?::BIGINT[])
                "#,
            )
            .context("prepare token cache insert")?;

        for (hash, entry) in entries {
            let tokens_json =
                serde_json::to_string(&entry.tokens).context("encode token cache tokens")?;
            let starts_json =
                serde_json::to_string(&entry.starts).context("encode token cache starts")?;
            let ends_json =
                serde_json::to_string(&entry.ends).context("encode token cache ends")?;
            stmt.execute(params![
                self.model_id,
                self.params_hash,
                hash,
                tokens_json,
                starts_json,
                ends_json,
            ])
            .context("insert token cache row")?;
        }
        Ok(())
    }
}

#[cfg(feature = "tokenization")]
fn token_params_hash(lowercase: bool, remove_punct: bool) -> AnyhowResult<String> {
    let params = TokenCacheParams {
        lowercase,
        remove_punct,
    };
    Ok(hash_text(&serde_json::to_string(&params)?))
}

#[cfg(feature = "tokenization")]
fn tokenize_uncached_entries(
    backend: &TokenizerBackend,
    texts: &[String],
    lowercase: bool,
    remove_punct: bool,
) -> AnyhowResult<Vec<TokenCacheEntry>> {
    texts
        .iter()
        .map(|text| {
            backend
                .tokenize_text_with_offsets(text, lowercase, remove_punct)
                .map(TokenCacheEntry::from_offsets)
        })
        .collect()
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
    let fields = vec![
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

    let mut builder = AnonymousOwnedListBuilder::new(
        PlSmallStr::EMPTY,
        row_count,
        Some(token_offset_struct_type()),
    );
    for (start, end) in row_spans {
        if end == start {
            builder.append_empty();
        } else {
            let slice = inner.slice(start as i64, end - start);
            builder.append_series(&slice).map_err(|err| {
                PolarsError::ComputeError(format!("List builder failed: {err}").into())
            })?;
        }
    }

    let mut list = builder.finish();
    list.rename(name);
    Ok(list.into_series())
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
    let embedder = ensure_embedder(kwargs.embedder_model.as_deref())
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
) -> PolarsResult<Vec<Vec<f32>>> {
    if let Some(cache_path) = cache_path {
        let scope = CacheScope {
            model_id: embedder.model_id(),
            revision: embedder.model_revision(),
            provider_id: embedder.provider_id(),
        };
        return get_or_insert_embeddings(
            std::path::Path::new(cache_path),
            scope,
            texts,
            |misses| {
                encode_uncached_embedding_batches(embedder, misses, batch_size)
                    .map_err(|err| anyhow::anyhow!("{err}"))
            },
        )
        .map_err(|e| PolarsError::ComputeError(format!("Embedding cache failed: {e}").into()));
    }

    encode_uncached_embedding_batches(embedder, texts, batch_size)
}

#[cfg(feature = "embedding")]
fn encode_uncached_embedding_batches(
    embedder: &Embedder,
    texts: &[String],
    batch_size: usize,
) -> PolarsResult<Vec<Vec<f32>>> {
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(batch_size) {
        let encoded = embedder
            .encode(chunk)
            .map_err(|e| PolarsError::ComputeError(format!("Embedding failed: {e:#}").into()))?;
        vectors.extend(encoded);
    }
    Ok(vectors)
}

#[cfg(feature = "embedding")]
fn build_embedding_vector_list(
    name: PlSmallStr,
    row_spans: Vec<(usize, usize)>,
    flat: Vec<f32>,
) -> PolarsResult<Series> {
    let inner = Series::new(PlSmallStr::EMPTY, flat);
    let mut builder =
        AnonymousOwnedListBuilder::new(PlSmallStr::EMPTY, row_spans.len(), Some(DataType::Float32));
    for (start, end) in row_spans {
        if end == start {
            builder.append_empty();
        } else {
            let slice = inner.slice(start as i64, end - start);
            builder.append_series(&slice).map_err(|err| {
                PolarsError::ComputeError(format!("Embedding list build failed: {err}").into())
            })?;
        }
    }

    let mut list = builder.finish();
    list.rename(name);
    Ok(list.into_series())
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
    for opt_text in ca.into_iter() {
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

    for opt_inner in ca.into_iter() {
        let row_start = item_text_indices.len();
        if let Some(inner) = opt_inner {
            let inner_ca = inner.str()?;
            for opt_text in inner_ca.into_iter() {
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
    let mut builder = AnonymousOwnedListBuilder::new(
        PlSmallStr::EMPTY,
        row_item_spans.len(),
        Some(DataType::List(Box::new(DataType::Float32))),
    );
    for (start, end) in row_item_spans {
        if end == start {
            builder.append_empty();
        } else {
            let slice = vector_list.slice(start as i64, end - start);
            builder.append_series(&slice).map_err(|err| {
                PolarsError::ComputeError(
                    format!("Nested embedding list build failed: {err}").into(),
                )
            })?;
        }
    }

    let mut list = builder.finish();
    list.rename(ca.name().clone());
    Ok(list.into_series())
}

#[cfg(feature = "tokenization")]
#[polars_expr(output_type_func=list_token_struct_output)]
pub fn tokenize(inputs: &[Series], kwargs: TokenizeKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let backend = ensure_tokenizer_for_model(kwargs.model_id.as_deref())
        .map_err(|e| PolarsError::ComputeError(format!("Tokenizer init failed: {e}").into()))?;

    if let Some(cache_path) = kwargs.cache.as_deref() {
        let mut texts = Vec::new();
        let mut row_text_indices = Vec::with_capacity(ca.len());
        for opt_text in ca.into_iter() {
            match opt_text {
                Some(text) => {
                    row_text_indices.push(Some(texts.len()));
                    texts.push(text.to_string());
                }
                None => row_text_indices.push(None),
            }
        }

        let params_hash = token_params_hash(kwargs.lowercase, kwargs.remove_punct)
            .map_err(|e| PolarsError::ComputeError(format!("Token cache failed: {e:#}").into()))?;
        let table = TokenCacheTable {
            model_id: kwargs.model_id.as_deref().unwrap_or_default(),
            params_hash: &params_hash,
        };
        let entries = get_or_insert_text_values(Path::new(cache_path), &table, &texts, |misses| {
            tokenize_uncached_entries(
                backend.as_ref(),
                misses,
                kwargs.lowercase,
                kwargs.remove_punct,
            )
        })
        .map_err(|e| PolarsError::ComputeError(format!("Token cache failed: {e:#}").into()))?;

        let estimated_tokens = ca.len().saturating_mul(32);
        let mut tok_col = Vec::with_capacity(estimated_tokens);
        let mut start_col = Vec::with_capacity(estimated_tokens);
        let mut end_col = Vec::with_capacity(estimated_tokens);
        let mut row_spans = Vec::with_capacity(ca.len());
        for text_index in row_text_indices {
            let span_start = tok_col.len();
            if let Some(index) = text_index {
                entries[index].append_to(&mut tok_col, &mut start_col, &mut end_col);
            }
            row_spans.push((span_start, tok_col.len()));
        }

        return build_token_list_series(
            ca.name().clone(),
            ca.len(),
            row_spans,
            tok_col,
            start_col,
            end_col,
        );
    }

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

    build_token_list_series(
        ca.name().clone(),
        ca.len(),
        row_spans,
        tok_col,
        start_col,
        end_col,
    )
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
