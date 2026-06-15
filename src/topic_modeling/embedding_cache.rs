//! DuckDB-backed embedding vector cache.
//!
//! Why this exists: ONNX inference is the expensive part of topic modeling and
//! the public `embedding` expression, while the existing tokenization cache must
//! remain a separate `tokens.duckdb` file. This module owns the Rust-side
//! `embeddings.duckdb` schema and keeps cache keys tied to the model, revision,
//! execution-provider label, and text content hash.
//!
//! Used by: `expressions::embedding` and `topic_modeling::run`, with the backend
//! passing a per-user embedding cache path into the pipeline.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use duckdb::{params, Connection, Error as DuckDbError};

use crate::cache::{get_or_insert_text_values, TextCacheTable};

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS embedding_cache (
    model VARCHAR NOT NULL,
    revision VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL,
    dim UBIGINT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, revision, provider, content_hash)
)
"#;

/// Stable dimensions of a cache namespace. The provider is included because
/// acceleration backends can differ slightly in floating-point output.
#[derive(Clone, Copy)]
pub struct CacheScope<'a> {
    pub model_id: &'a str,
    pub revision: &'a str,
    pub provider_id: &'a str,
}

/// Return embeddings for `texts`, computing and inserting only unique misses.
///
/// Flow: create/open the requested DuckDB file, fetch cached blobs for unique
/// text hashes, run the caller's miss encoder, persist new blobs, then expand
/// duplicates back to input order.
pub fn get_or_insert_embeddings(
    path: &Path,
    scope: CacheScope<'_>,
    texts: &[String],
    compute_misses: impl FnOnce(&[String]) -> Result<Vec<Vec<f32>>>,
) -> Result<Vec<Vec<f32>>> {
    let table = EmbeddingCacheTable { scope };
    get_or_insert_text_values(path, &table, texts, compute_misses)
}

struct EmbeddingCacheTable<'a> {
    scope: CacheScope<'a>,
}

impl TextCacheTable for EmbeddingCacheTable<'_> {
    type Value = Vec<f32>;

    fn schema_sql(&self) -> &'static str {
        SCHEMA_SQL
    }

    fn fetch_cached(
        &self,
        conn: &Connection,
        hashes: &[String],
    ) -> Result<HashMap<String, Self::Value>> {
        let mut out = HashMap::new();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT dim, embedding
                FROM embedding_cache
                WHERE model = ?
                  AND revision = ?
                  AND provider = ?
                  AND content_hash = ?
                "#,
            )
            .context("prepare embedding cache lookup")?;

        for hash in hashes.iter().collect::<HashSet<_>>() {
            let row_result = stmt.query_row(
                params![
                    self.scope.model_id,
                    self.scope.revision,
                    self.scope.provider_id,
                    hash
                ],
                |row| {
                    let dim: u64 = row.get(0)?;
                    let blob: Vec<u8> = row.get(1)?;
                    Ok(blob_to_vector(&blob, dim as usize))
                },
            );
            match row_result {
                Ok(Some(vector)) => {
                    out.insert(hash.clone(), vector);
                }
                Ok(None) | Err(DuckDbError::QueryReturnedNoRows) => {}
                Err(err) => return Err(err).context("read embedding cache row"),
            }
        }

        Ok(out)
    }

    fn persist_new(&self, conn: &Connection, entries: &[(String, Self::Value)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut stmt = conn
            .prepare(
                r#"
            INSERT OR IGNORE INTO embedding_cache
            (model, revision, provider, content_hash, dim, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
            )
            .context("prepare embedding cache insert")?;

        for (hash, vector) in entries {
            let blob = vector_to_blob(vector);
            stmt.execute(params![
                self.scope.model_id,
                self.scope.revision,
                self.scope.provider_id,
                hash,
                vector.len() as u64,
                blob,
            ])
            .context("insert embedding cache row")?;
        }
        Ok(())
    }
}

fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vector.len() * std::mem::size_of::<f32>());
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

fn blob_to_vector(blob: &[u8], dim: usize) -> Option<Vec<f32>> {
    if blob.len() != dim * std::mem::size_of::<f32>() {
        return None;
    }
    let mut vector = Vec::with_capacity(dim);
    for chunk in blob.chunks_exact(std::mem::size_of::<f32>()) {
        vector.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(vector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_cache_path(test_name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("polars-text-{test_name}-{unique}.duckdb"))
    }

    #[test]
    fn vector_blob_round_trips_little_endian_f32s() {
        let vector = vec![0.25_f32, -1.5, 3.75];
        let blob = vector_to_blob(&vector);
        assert_eq!(blob_to_vector(&blob, vector.len()).unwrap(), vector);
    }

    #[test]
    fn blob_to_vector_rejects_wrong_dimension() {
        let blob = vector_to_blob(&[1.0_f32, 2.0]);
        assert!(blob_to_vector(&blob, 3).is_none());
    }

    #[test]
    fn cache_computes_only_misses_and_preserves_input_order() -> Result<()> {
        let path = temp_cache_path("embedding-cache-misses");
        let scope = CacheScope {
            model_id: "test-model",
            revision: "test-revision",
            provider_id: "test-provider",
        };

        let first_texts = vec!["alpha".to_string(), "beta".to_string(), "alpha".to_string()];
        let mut first_misses = Vec::new();
        let first = get_or_insert_embeddings(&path, scope, &first_texts, |misses| {
            first_misses = misses.to_vec();
            Ok(misses
                .iter()
                .map(|text| match text.as_str() {
                    "alpha" => vec![1.0, 1.5],
                    "beta" => vec![2.0, 2.5],
                    _ => unreachable!("unexpected first miss {text}"),
                })
                .collect())
        })?;
        assert_eq!(first_misses, vec!["alpha".to_string(), "beta".to_string()]);
        assert_eq!(first, vec![vec![1.0, 1.5], vec![2.0, 2.5], vec![1.0, 1.5]]);

        let second_texts = vec!["beta".to_string(), "gamma".to_string(), "alpha".to_string()];
        let mut second_misses = Vec::new();
        let second = get_or_insert_embeddings(&path, scope, &second_texts, |misses| {
            second_misses = misses.to_vec();
            Ok(misses
                .iter()
                .map(|text| match text.as_str() {
                    "gamma" => vec![3.0, 3.5],
                    _ => unreachable!("cached text was recomputed: {text}"),
                })
                .collect())
        })?;
        assert_eq!(second_misses, vec!["gamma".to_string()]);
        assert_eq!(second, vec![vec![2.0, 2.5], vec![3.0, 3.5], vec![1.0, 1.5]]);

        let conn = Connection::open(&path)?;
        let row_count: u64 =
            conn.query_row("SELECT count(*) FROM embedding_cache", [], |row| row.get(0))?;
        assert_eq!(row_count, 3);
        drop(conn);
        let _ = std::fs::remove_file(path);
        Ok(())
    }
}
