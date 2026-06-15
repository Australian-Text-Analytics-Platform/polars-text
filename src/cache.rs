//! Shared DuckDB-backed text cache primitives.
//!
//! Tokenization and embedding keep separate DuckDB files and value schemas, but
//! both are content-addressed per-text computations. This module owns the common
//! flow: hash text, fetch cached values, compute unique misses outside every DB
//! lock, insert with conflict-safe semantics, then expand to input order.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use duckdb::Connection;
use fs2::FileExt;
use once_cell::sync::Lazy;
use sha2::{Digest, Sha256};

static CACHE_PATH_LOCKS: Lazy<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Stable SHA-256 content hash used by every text cache table.
pub fn hash_text(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Table-specific codec and namespace for one cached value type.
pub trait TextCacheTable {
    type Value: Clone;

    /// SQL that creates this table and its primary key if it does not exist.
    fn schema_sql(&self) -> &'static str;

    /// Fetch cached values for unique text hashes in this table's namespace.
    fn fetch_cached(
        &self,
        conn: &Connection,
        hashes: &[String],
    ) -> Result<HashMap<String, Self::Value>>;

    /// Persist newly computed values. Implementations should use conflict-safe
    /// insert semantics so duplicate miss computation is harmless.
    fn persist_new(&self, conn: &Connection, entries: &[(String, Self::Value)]) -> Result<()>;
}

/// Return values for `texts`, computing and inserting only unique cache misses.
pub fn get_or_insert_text_values<T>(
    path: &Path,
    table: &T,
    texts: &[String],
    compute_misses: impl FnOnce(&[String]) -> Result<Vec<T::Value>>,
) -> Result<Vec<T::Value>>
where
    T: TextCacheTable,
{
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let hashes = texts.iter().map(|text| hash_text(text)).collect::<Vec<_>>();
    let mut cached = with_cache_lock(path, || {
        let conn = open_cache(path, table.schema_sql())?;
        table.fetch_cached(&conn, &hashes)
    })?;

    let (miss_hashes, miss_texts) = unique_misses(texts, &hashes, &cached);
    let computed = if miss_texts.is_empty() {
        Vec::new()
    } else {
        let values = compute_misses(&miss_texts)?;
        if values.len() != miss_texts.len() {
            anyhow::bail!(
                "cache miss encoder returned {} values for {} texts",
                values.len(),
                miss_texts.len()
            );
        }
        values
    };

    if !computed.is_empty() {
        let entries = miss_hashes
            .iter()
            .cloned()
            .zip(computed.iter().cloned())
            .collect::<Vec<_>>();
        with_cache_lock(path, || {
            let conn = open_cache(path, table.schema_sql())?;
            table.persist_new(&conn, &entries)
        })?;
        for (hash, value) in entries {
            cached.insert(hash, value);
        }
    }

    hashes
        .iter()
        .map(|hash| {
            cached
                .get(hash)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("cache missing computed hash {hash}"))
        })
        .collect()
}

fn unique_misses<V: Clone>(
    texts: &[String],
    hashes: &[String],
    cached: &HashMap<String, V>,
) -> (Vec<String>, Vec<String>) {
    let mut seen_misses = HashSet::new();
    let mut miss_hashes = Vec::new();
    let mut miss_texts = Vec::new();
    for (hash, text) in hashes.iter().zip(texts) {
        if !cached.contains_key(hash) && seen_misses.insert(hash.clone()) {
            miss_hashes.push(hash.clone());
            miss_texts.push(text.clone());
        }
    }
    (miss_hashes, miss_texts)
}

fn open_cache(path: &Path, schema_sql: &str) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create cache dir {}", parent.display()))?;
    }
    let conn = Connection::open(path).with_context(|| format!("open cache {}", path.display()))?;
    conn.execute_batch(schema_sql)
        .context("initialize cache schema")?;
    Ok(conn)
}

fn with_cache_lock<T>(path: &Path, action: impl FnOnce() -> Result<T>) -> Result<T> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create cache lock dir {}", parent.display()))?;
    }

    let local_lock = local_path_lock(path)?;
    let _local_guard = local_lock
        .lock()
        .map_err(|_| anyhow::anyhow!("cache path lock poisoned for {}", path.display()))?;

    let lock_path = lock_path_for(path);
    let lock_file = open_lock_file(&lock_path)?;
    lock_file
        .lock_exclusive()
        .with_context(|| format!("lock cache file {}", lock_path.display()))?;
    let result = action();
    let unlock_result = lock_file
        .unlock()
        .with_context(|| format!("unlock cache file {}", lock_path.display()));

    match (result, unlock_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
    }
}

fn local_path_lock(path: &Path) -> Result<Arc<Mutex<()>>> {
    let key = normalize_lock_key(path)?;
    let mut locks = CACHE_PATH_LOCKS
        .lock()
        .map_err(|_| anyhow::anyhow!("cache path lock registry poisoned"))?;
    Ok(Arc::clone(
        locks.entry(key).or_insert_with(|| Arc::new(Mutex::new(()))),
    ))
}

fn normalize_lock_key(path: &Path) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let absolute_parent = if parent.is_absolute() {
        parent.to_path_buf()
    } else {
        std::env::current_dir()?.join(parent)
    };
    Ok(absolute_parent.join(
        path.file_name()
            .ok_or_else(|| anyhow::anyhow!("cache path has no file name: {}", path.display()))?,
    ))
}

fn lock_path_for(path: &Path) -> PathBuf {
    let mut lock_name = path
        .file_name()
        .map(|value| value.to_os_string())
        .unwrap_or_else(|| "cache".into());
    lock_name.push(".lock");
    path.with_file_name(lock_name)
}

fn open_lock_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path)
        .with_context(|| format!("open cache lock file {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duckdb::{params, Error as DuckDbError};
    use std::cell::Cell;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const STRING_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS string_cache (
    namespace VARCHAR NOT NULL,
    content_hash VARCHAR NOT NULL,
    value VARCHAR NOT NULL,
    PRIMARY KEY (namespace, content_hash)
)
"#;

    struct StringTable<'a> {
        namespace: &'a str,
    }

    impl TextCacheTable for StringTable<'_> {
        type Value = String;

        fn schema_sql(&self) -> &'static str {
            STRING_SCHEMA
        }

        fn fetch_cached(
            &self,
            conn: &Connection,
            hashes: &[String],
        ) -> Result<HashMap<String, Self::Value>> {
            let mut out = HashMap::new();
            let mut stmt = conn.prepare(
                "SELECT value FROM string_cache WHERE namespace = ? AND content_hash = ?",
            )?;
            for hash in hashes.iter().collect::<HashSet<_>>() {
                let result = stmt.query_row(params![self.namespace, hash], |row| row.get(0));
                match result {
                    Ok(value) => {
                        out.insert(hash.clone(), value);
                    }
                    Err(DuckDbError::QueryReturnedNoRows) => {}
                    Err(err) => return Err(err).context("read string cache row"),
                }
            }
            Ok(out)
        }

        fn persist_new(&self, conn: &Connection, entries: &[(String, Self::Value)]) -> Result<()> {
            let mut stmt = conn.prepare(
                "INSERT OR IGNORE INTO string_cache (namespace, content_hash, value) VALUES (?, ?, ?)",
            )?;
            for (hash, value) in entries {
                stmt.execute(params![self.namespace, hash, value])?;
            }
            Ok(())
        }
    }

    fn temp_cache_path(test_name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("polars-text-cache-{test_name}-{unique}.duckdb"))
    }

    #[test]
    fn text_hash_is_stable_sha256() {
        assert_eq!(
            hash_text("hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn cache_flow_computes_unique_misses_and_preserves_order() -> Result<()> {
        let path = temp_cache_path("flow");
        let table = StringTable { namespace: "a" };
        let texts = vec!["alpha".to_string(), "beta".to_string(), "alpha".to_string()];
        let mut misses_seen = Vec::new();
        let values = get_or_insert_text_values(&path, &table, &texts, |misses| {
            misses_seen = misses.to_vec();
            Ok(misses.iter().map(|text| format!("value:{text}")).collect())
        })?;

        assert_eq!(misses_seen, vec!["alpha".to_string(), "beta".to_string()]);
        assert_eq!(
            values,
            vec![
                "value:alpha".to_string(),
                "value:beta".to_string(),
                "value:alpha".to_string()
            ]
        );

        let mut second_misses = Vec::new();
        let second = get_or_insert_text_values(&path, &table, &texts, |misses| {
            second_misses = misses.to_vec();
            Ok(misses
                .iter()
                .map(|text| format!("unexpected:{text}"))
                .collect())
        })?;
        assert!(second_misses.is_empty());
        assert_eq!(second, values);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(lock_path_for(&path));
        Ok(())
    }

    #[test]
    fn compute_runs_outside_cache_lock() -> Result<()> {
        let path = temp_cache_path("outside-lock");
        let outer = StringTable { namespace: "outer" };
        let inner = StringTable { namespace: "inner" };
        let nested_completed = Cell::new(false);

        let outer_texts = vec!["outer-text".to_string()];
        let outer_values = get_or_insert_text_values(&path, &outer, &outer_texts, |misses| {
            let inner_texts = vec!["inner-text".to_string()];
            let inner_values =
                get_or_insert_text_values(&path, &inner, &inner_texts, |inner_misses| {
                    Ok(inner_misses
                        .iter()
                        .map(|text| format!("inner:{text}"))
                        .collect())
                })?;
            assert_eq!(inner_values, vec!["inner:inner-text".to_string()]);
            nested_completed.set(true);
            Ok(misses.iter().map(|text| format!("outer:{text}")).collect())
        })?;

        assert!(nested_completed.get());
        assert_eq!(outer_values, vec!["outer:outer-text".to_string()]);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(lock_path_for(&path));
        Ok(())
    }
}
