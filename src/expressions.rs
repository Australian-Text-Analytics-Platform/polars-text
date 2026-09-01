#[cfg(feature = "tokenization")]
use anyhow::{Context, Result as AnyhowResult};
#[cfg(feature = "tokenization")]
use duckdb::{params, Connection};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
#[cfg(feature = "tokenization")]
use std::collections::HashMap;
#[cfg(feature = "tokenization")]
use std::path::Path;
#[cfg(any(feature = "embedding", feature = "tokenization"))]
use std::sync::Arc;
use unicode_segmentation::UnicodeSegmentation;

#[cfg(feature = "tokenization")]
use crate::cache::{get_or_insert_text_values, hash_text, stage_requested_hashes, TextCacheTable};
#[cfg(feature = "tokenization")]
use crate::concordance::{
    concordance_for_text, list_struct_output, struct_series_from_matches, ConcordanceKwargs,
};
#[cfg(any(feature = "embedding", feature = "tokenization"))]
use crate::list_output::list_from_spans;
#[cfg(feature = "tokenization")]
use crate::tokenizer::{ensure_tokenizer_for_model, tokenizer_cache_fingerprint, TokenizerBackend};
#[cfg(feature = "embedding")]
use crate::topic_modeling::embedding::{ensure_embedder, Embedder};
#[cfg(feature = "embedding")]
use crate::topic_modeling::embedding_cache::{get_or_insert_embeddings, CacheScope};

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

fn count_string_values(
    inputs: &[Series],
    mut count: impl FnMut(&str) -> i64,
) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: Vec<i64> = ca
        .iter()
        .map(|opt_text| opt_text.map(&mut count).unwrap_or(0))
        .collect();
    Ok(Series::new(ca.name().clone(), out))
}

#[polars_expr(output_type_func=int_output)]
pub fn word_count(inputs: &[Series]) -> PolarsResult<Series> {
    count_string_values(inputs, |text| text.unicode_words().count() as i64)
}

#[polars_expr(output_type_func=int_output)]
pub fn sentence_count(inputs: &[Series]) -> PolarsResult<Series> {
    count_string_values(inputs, |text| {
        text.unicode_sentences()
            .filter(|sentence| !sentence.trim().is_empty())
            .count() as i64
    })
}

#[cfg(feature = "tokenization")]
#[polars_expr(output_type_func=list_struct_output)]
pub fn concordance(inputs: &[Series], kwargs: ConcordanceKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let mut spans = Vec::with_capacity(ca.len());
    let mut flat = struct_series_from_matches(Vec::new())?;

    for opt_text in ca.iter() {
        let start = flat.len();
        let text = match opt_text {
            Some(value) => value,
            None => {
                spans.push((start, start));
                continue;
            }
        };

        let matches = concordance_for_text(text, &kwargs)
            .map_err(|e| PolarsError::ComputeError(format!("Concordance failed: {e}").into()))?;
        let struct_series = struct_series_from_matches(matches)?;
        flat.append(&struct_series)?;
        spans.push((start, flat.len()));
    }
    list_from_spans(ca.name().clone(), &flat, &spans)
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
    fingerprint VARCHAR NOT NULL,
    params_hash VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL,
    tokens VARCHAR[] NOT NULL,
    start_offsets BIGINT[] NOT NULL,
    end_offsets BIGINT[] NOT NULL,
    PRIMARY KEY (model, fingerprint, params_hash, content_hash)
)
"#;

#[cfg(feature = "tokenization")]
#[derive(serde::Serialize)]
struct TokenCacheParams {
    lowercase: bool,
    remove_punct: bool,
}

#[cfg(feature = "tokenization")]
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
    fingerprint: &'a str,
    params_hash: &'a str,
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
    ) -> AnyhowResult<HashMap<String, Arc<Self::Value>>> {
        stage_requested_hashes(conn, hashes)?;
        let mut out = HashMap::new();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT cache.content_hash, to_json(cache.tokens),
                       to_json(cache.start_offsets), to_json(cache.end_offsets)
                FROM token_cache AS cache
                INNER JOIN requested_hashes AS requested USING (content_hash)
                WHERE cache.model = ? AND cache.fingerprint = ? AND cache.params_hash = ?
                "#,
            )
            .context("prepare token cache lookup")?;
        let mut rows = stmt.query(params![self.model_id, self.fingerprint, self.params_hash])?;
        while let Some(row) = rows.next()? {
            let hash: String = row.get(0)?;
            let tokens_json: String = row.get(1)?;
            let starts_json: String = row.get(2)?;
            let ends_json: String = row.get(3)?;
            let entry = TokenCacheEntry {
                tokens: serde_json::from_str(&tokens_json).context("decode token cache tokens")?,
                starts: serde_json::from_str(&starts_json).context("decode token cache starts")?,
                ends: serde_json::from_str(&ends_json).context("decode token cache ends")?,
            };
            out.insert(hash, Arc::new(entry));
        }

        Ok(out)
    }

    fn persist_new(
        &self,
        conn: &Connection,
        entries: &[(String, Arc<Self::Value>)],
    ) -> AnyhowResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        conn.execute_batch(
            "BEGIN; DROP TABLE IF EXISTS staged_token_cache;
             CREATE TEMP TABLE staged_token_cache (
               model VARCHAR, fingerprint VARCHAR, params_hash VARCHAR, content_hash VARCHAR,
               tokens VARCHAR[], start_offsets BIGINT[], end_offsets BIGINT[]
             );",
        )?;
        let result = (|| -> AnyhowResult<()> {
            let mut appender = conn.appender("staged_token_cache")?;
            for (hash, entry) in entries {
                let tokens_json = serde_json::to_string(&entry.tokens)?;
                let starts_json = serde_json::to_string(&entry.starts)?;
                let ends_json = serde_json::to_string(&entry.ends)?;
                appender.append_row(params![
                    self.model_id,
                    self.fingerprint,
                    self.params_hash,
                    hash,
                    tokens_json,
                    starts_json,
                    ends_json,
                ])?;
            }
            appender.flush()?;
            drop(appender);
            conn.execute_batch(
                "INSERT OR IGNORE INTO token_cache
                 SELECT model, fingerprint, params_hash, content_hash, tokens, start_offsets, end_offsets
                 FROM staged_token_cache; COMMIT;",
            )?;
            Ok(())
        })();
        if result.is_err() {
            let _ = conn.execute_batch("ROLLBACK");
        }
        result
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
    let fields = [
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

    if row_spans.len() != row_count {
        return Err(PolarsError::ComputeError(
            "token list row count does not match span count".into(),
        ));
    }
    list_from_spans(name, &inner, &row_spans)
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
) -> PolarsResult<Vec<Arc<Vec<f32>>>> {
    if let Some(cache_path) = cache_path {
        let scope = CacheScope {
            model_id: embedder.model_id(),
            fingerprint: embedder.cache_fingerprint(),
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
        .map(|vectors| vectors.into_iter().map(Arc::new).collect())
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
    list_from_spans(name, &Series::new(PlSmallStr::EMPTY, flat), &row_spans)
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
    for opt_text in ca.iter() {
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

    for opt_inner in ca.amortized_iter() {
        let row_start = item_text_indices.len();
        if let Some(inner) = opt_inner {
            let inner_ca = inner.as_ref().str()?;
            for opt_text in inner_ca.iter() {
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
    list_from_spans(ca.name().clone(), &vector_list, &row_item_spans)
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
        for opt_text in ca.iter() {
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
        let model_id = kwargs.model_id.as_deref().unwrap_or_default();
        let fingerprint = tokenizer_cache_fingerprint(model_id)
            .map_err(|e| PolarsError::ComputeError(format!("Token cache failed: {e:#}").into()))?;
        let table = TokenCacheTable {
            model_id,
            fingerprint: &fingerprint,
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

    for opt_text in ca.iter() {
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
