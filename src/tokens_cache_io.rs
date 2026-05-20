//! On-disk I/O for the lazy tokens-cache lookup expression.
//!
//! The companion of `expressions::tokenize_with_cache_lookup`. Owns:
//!
//! * cache-directory resolution from `LDACA_TOKENS_CACHE_DIR` + `user_id`
//!   at execution time — no absolute paths get baked into the serialised
//!   `.plbin` lazy plan (which is the whole point of the lazy refactor:
//!   plans become cross-machine portable);
//! * listing every parquet file that belongs to one bucket
//!   (`<bucket>.parquet` legacy + `<bucket>__delta__*.parquet` deltas),
//!   matching the layout the Python `tokens_cache.py` writes;
//! * loading those files into an in-memory hash → tokens map for cache
//!   lookup, with last-writer-wins dedup matching the Python read path
//!   (`scan_parquet([all]).unique(subset=[hash], keep="first")`);
//! * appending a fresh delta parquet under an advisory file lock, so
//!   multiple analysis workers operating on the same bucket can't
//!   corrupt each other.
//!
//! See `backend/docs/developer-guide/lazy-tokenisation-refactor.md` §3-§6
//! for the design + concurrency contract this module is the Rust half of.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use polars::prelude::*;

/// Env var the Python side sets to point at the per-install cache base.
/// Resolution: `{LDACA_TOKENS_CACHE_DIR}/{user_id}/tokens/`. Matches the
/// existing convention in `backend/src/ldaca_wordflow/core/tokens_cache.py`
/// so cached buckets written by either side are mutually readable.
pub(crate) const CACHE_ROOT_ENV: &str = "LDACA_TOKENS_CACHE_DIR";

/// Filename infix that marks a delta file — value MUST stay in sync with
/// `tokens_cache.DELTA_INFIX` on the Python side, since both sides glob
/// the same bucket directory.
pub(crate) const DELTA_INFIX: &str = "__delta__";

/// The cached-hash column name — MUST stay in sync with
/// `tokens_cache.CONTENT_HASH_COLUMN`.
pub(crate) const CONTENT_HASH_COLUMN: &str = "__ldaca_content_hash__";

/// Resolve the absolute cache directory for one (user, bucket) tuple
/// from the runtime environment. The directory is created if missing
/// (treating "fresh machine, no cache yet" as a normal cache-miss path
/// rather than an error condition — that's the central portability win
/// of the lazy design).
///
/// If `require_env_cache_dir` is true and the env var is missing, returns
/// an error rather than silently picking a default. Production analysis
/// workers should set the env var in their bootstrap and pass `true` so
/// a misconfigured worker fails loudly instead of silently recomputing on
/// every invocation.
pub(crate) fn resolve_cache_dir(
    user_id: &str,
    require_env: bool,
) -> PolarsResult<PathBuf> {
    let base = match std::env::var(CACHE_ROOT_ENV) {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ if require_env => {
            return Err(PolarsError::ComputeError(
                format!(
                    "tokenize_with_cache_lookup: ${CACHE_ROOT_ENV} is not set. \
                     The Python worker bootstrap must set this to the cache \
                     base directory before any analysis touches a lazy-tokenised \
                     node. Refusing to silently recompute on every collect."
                )
                .into(),
            ));
        }
        // Tests + dev: fall back to a sentinel under /tmp so behaviour is
        // observable but no real cache pollution happens. Production
        // never hits this branch.
        _ => std::env::temp_dir().join("ldaca-tokens-cache"),
    };
    let dir = base.join(user_id).join("tokens");
    std::fs::create_dir_all(&dir).map_err(|e| {
        PolarsError::ComputeError(
            format!(
                "tokenize_with_cache_lookup: failed to mkdir {}: {}",
                dir.display(),
                e
            )
            .into(),
        )
    })?;
    Ok(dir)
}

/// Strip `.parquet` from `bucket_filename` to get the bucket key.
///
/// Bucket key is the stem shared between the legacy single-file cache
/// (`<bucket>.parquet`) and every delta file
/// (`<bucket>__delta__<uuid>.parquet`). The Python side stores the
/// `.parquet` form in `node.derived[col]['cache_filename']` for
/// back-compat; this helper normalises it to bucket-key form.
fn bucket_key_from_filename(filename: &str) -> &str {
    filename
        .strip_suffix(".parquet")
        .unwrap_or(filename)
}

/// List every parquet file in `dir` belonging to the given bucket,
/// ordered oldest-first by mtime so the caller's last-writer-wins dedup
/// matches the Python `tokens_cache_lazyframe` ordering.
///
/// Returns an empty vec if `dir` is missing or the bucket has no files —
/// callers treat both as "everything is a cache miss".
pub(crate) fn list_bucket_files(dir: &Path, bucket_filename: &str) -> Vec<PathBuf> {
    let bucket = bucket_key_from_filename(bucket_filename);
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    let legacy = dir.join(format!("{bucket}.parquet"));
    if let Ok(meta) = std::fs::metadata(&legacy) {
        let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        files.push((mtime, legacy));
    }
    let delta_prefix = format!("{bucket}{DELTA_INFIX}");
    let read_dir = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    for entry in read_dir.flatten() {
        let name = entry.file_name();
        let name_str = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        if !name_str.starts_with(&delta_prefix) || !name_str.ends_with(".parquet") {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        files.push((mtime, entry.path()));
    }
    files.sort_by_key(|(t, _)| *t);
    files.into_iter().map(|(_, p)| p).collect()
}

/// In-memory cached-tokens map keyed by content hash (UInt64).
///
/// The value is the AnyValue for the `tokens` column of that row — a
/// List<Struct<token, start, end>>. We keep it as `AnyValue<'static>`
/// so the per-row append into the output builder is a simple clone
/// rather than re-encoding through the parquet reader on every hit.
pub(crate) type CacheMap = HashMap<u64, AnyValue<'static>>;

/// Read every file in `paths` and union them into an in-memory map.
/// Within a single hash, the FIRST file in `paths` wins — same semantic
/// as the Python `unique(keep="first")` after the oldest-first sort.
pub(crate) fn load_cache_map(paths: &[PathBuf]) -> PolarsResult<CacheMap> {
    let mut map: CacheMap = HashMap::new();
    for path in paths {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                // A bucket file that vanished between list and open is
                // not fatal — just skip it. The lazy design treats
                // missing cache files as cache misses; partial loads
                // are no exception.
                if e.kind() == std::io::ErrorKind::NotFound {
                    continue;
                }
                return Err(PolarsError::ComputeError(
                    format!("tokens_cache_io: open {}: {}", path.display(), e).into(),
                ));
            }
        };
        let df = ParquetReader::new(file).finish().map_err(|e| {
            PolarsError::ComputeError(
                format!("tokens_cache_io: read {}: {}", path.display(), e).into(),
            )
        })?;
        // A bucket parquet must carry exactly the two well-known columns;
        // schema violation is silent in the dedup pass (newer schema
        // → just skip). Future-proofing: if the schema ever gains a
        // column, this loader keeps working without modification.
        let Ok(hash_col) = df.column(CONTENT_HASH_COLUMN) else {
            continue;
        };
        let Ok(tokens_col) = df.column("tokens") else {
            continue;
        };
        let hash_ca = match hash_col.as_materialized_series().u64() {
            Ok(ca) => ca.clone(),
            Err(_) => continue,
        };
        let tokens_series = tokens_col.as_materialized_series();
        for idx in 0..df.height() {
            let h = match hash_ca.get(idx) {
                Some(v) => v,
                None => continue, // null hash — unusable
            };
            map.entry(h).or_insert_with(|| {
                tokens_series
                    .get(idx)
                    .map(|av| av.into_static())
                    .unwrap_or(AnyValue::Null)
            });
        }
    }
    Ok(map)
}

/// Append `new_rows` as a fresh `<bucket>__delta__<uuid>.parquet` under
/// `dir`, guarded by an advisory `flock` on `<bucket>.lock`. On
/// POSIX the lock is `fcntl.flock(LOCK_EX)`; on Windows we fall back
/// to a best-effort marker file (mirror of the Python helper).
///
/// `new_rows` must carry exactly `CONTENT_HASH_COLUMN` (UInt64) and
/// `tokens` (List<Struct<...>>). No-op when `new_rows.height() == 0`.
pub(crate) fn write_delta(
    dir: &Path,
    bucket_filename: &str,
    new_rows: &mut DataFrame,
) -> PolarsResult<PathBuf> {
    if new_rows.height() == 0 {
        // Conventional bucket path — the file may not exist on disk.
        return Ok(dir.join(bucket_filename));
    }
    let bucket = bucket_key_from_filename(bucket_filename);
    let lock_path = dir.join(format!("{bucket}.parquet.lock"));
    let delta_path = dir.join(format!(
        "{bucket}{DELTA_INFIX}{}.parquet",
        uuid::Uuid::new_v4().simple()
    ));

    let _guard = FileLock::acquire(&lock_path)?;

    // tmp + rename so readers using `scan_parquet([...])` never observe
    // a torn file. Mirrors `_atomic_write_parquet` on the Python side.
    let tmp_path = dir.join(format!(
        "{}.tmp.{}",
        delta_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("cache.parquet"),
        uuid::Uuid::new_v4().simple()
    ));
    {
        let file = File::create(&tmp_path).map_err(|e| {
            PolarsError::ComputeError(
                format!("tokens_cache_io: create {}: {}", tmp_path.display(), e).into(),
            )
        })?;
        ParquetWriter::new(file).finish(new_rows).map_err(|e| {
            // Best-effort cleanup of the tmp file on write failure so we
            // don't leak orphan parquets if the disk fills mid-write.
            let _ = std::fs::remove_file(&tmp_path);
            PolarsError::ComputeError(
                format!("tokens_cache_io: write {}: {}", tmp_path.display(), e).into(),
            )
        })?;
    }
    std::fs::rename(&tmp_path, &delta_path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        PolarsError::ComputeError(
            format!(
                "tokens_cache_io: rename {} -> {}: {}",
                tmp_path.display(),
                delta_path.display(),
                e
            )
            .into(),
        )
    })?;

    Ok(delta_path)
}

/// POSIX `fcntl.flock(LOCK_EX)` wrapper, RAII-released on drop.
/// On non-POSIX (Windows) builds, falls back to a marker-file presence
/// check with retries — same best-effort approximation the Python side
/// uses on the Tauri desktop build.
struct FileLock {
    #[cfg(unix)]
    fd: std::os::fd::OwnedFd,
    #[cfg(not(unix))]
    path: PathBuf,
}

impl FileLock {
    #[cfg(unix)]
    fn acquire(lock_path: &Path) -> PolarsResult<Self> {
        use std::os::fd::AsRawFd;
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(lock_path)
            .map_err(|e| {
                PolarsError::ComputeError(
                    format!("tokens_cache_io: open lock {}: {}", lock_path.display(), e)
                        .into(),
                )
            })?;
        // SAFETY: `flock` is the standard POSIX advisory-lock call; the
        // file descriptor stays open for the lifetime of `self` and is
        // closed on drop, which the kernel uses to release the lock.
        let rc = unsafe { libc_flock(file.as_raw_fd(), LOCK_EX) };
        if rc != 0 {
            return Err(PolarsError::ComputeError(
                format!(
                    "tokens_cache_io: flock {}: errno {}",
                    lock_path.display(),
                    std::io::Error::last_os_error()
                )
                .into(),
            ));
        }
        Ok(Self {
            fd: std::os::fd::OwnedFd::from(file),
        })
    }

    #[cfg(not(unix))]
    fn acquire(lock_path: &Path) -> PolarsResult<Self> {
        // Best-effort marker-file lock for Windows. The Tauri build is
        // single-user single-process for its critical path so contention
        // is vanishingly rare; we retry briefly then proceed regardless.
        for _ in 0..50 {
            if !lock_path.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let _ = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(lock_path);
        Ok(Self {
            path: lock_path.to_path_buf(),
        })
    }
}

#[cfg(not(unix))]
impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// Minimal libc shim so we don't take a dep on the full `libc` crate
// just for flock. Same signature as the system call.
#[cfg(unix)]
const LOCK_EX: i32 = 2;

#[cfg(unix)]
extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
}

#[cfg(unix)]
#[allow(non_snake_case)]
unsafe fn libc_flock(fd: i32, op: i32) -> i32 {
    flock(fd, op)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ldaca-tokens-cache-io-{}-{}",
            name,
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn resolve_cache_dir_creates_per_user_subdir_under_env() {
        let base = temp_dir("resolve");
        // Set env, resolve, verify the dir exists and matches base/user/tokens
        std::env::set_var(CACHE_ROOT_ENV, &base);
        let resolved = resolve_cache_dir("alice", false).unwrap();
        assert_eq!(resolved, base.join("alice").join("tokens"));
        assert!(resolved.exists());
        std::env::remove_var(CACHE_ROOT_ENV);
    }

    #[test]
    fn resolve_cache_dir_errors_when_env_missing_and_require_env_true() {
        // Use a known-unset var name in case some other test set it
        std::env::remove_var(CACHE_ROOT_ENV);
        let err = resolve_cache_dir("alice", true).unwrap_err();
        assert!(
            format!("{err}").contains(CACHE_ROOT_ENV),
            "error should mention the env var"
        );
    }

    #[test]
    fn list_bucket_files_orders_oldest_first() {
        let dir = temp_dir("list");
        let bucket = "mymodel__abcdef";
        // Write legacy first (so it gets the earliest mtime), then two
        // deltas with a small sleep between to ensure mtime ordering.
        std::fs::write(dir.join(format!("{bucket}.parquet")), b"legacy").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.join(format!("{bucket}__delta__001.parquet")), b"d1")
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.join(format!("{bucket}__delta__002.parquet")), b"d2")
            .unwrap();
        // Unrelated bucket should be ignored
        std::fs::write(dir.join("other__bucket.parquet"), b"nope").unwrap();
        let files =
            list_bucket_files(&dir, &format!("{bucket}.parquet"));
        assert_eq!(files.len(), 3, "expected 3 files, got {files:?}");
        assert!(files[0].to_string_lossy().ends_with(&format!("{bucket}.parquet")));
        assert!(files[1]
            .to_string_lossy()
            .contains(&format!("{bucket}__delta__001")));
        assert!(files[2]
            .to_string_lossy()
            .contains(&format!("{bucket}__delta__002")));
    }

    #[test]
    fn list_bucket_files_returns_empty_when_dir_missing() {
        let missing = std::env::temp_dir().join(format!(
            "ldaca-no-such-dir-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let files = list_bucket_files(&missing, "bucket.parquet");
        assert!(files.is_empty());
    }

    fn make_cache_df(hashes: &[u64], tokens: &[&str]) -> DataFrame {
        // One row per hash; token list is just one struct with the given
        // text and dummy offsets. Schema matches TOKENS_CACHE_SCHEMA.
        let hash_s = Series::new(CONTENT_HASH_COLUMN.into(), hashes);
        let mut list_builder = polars::chunked_array::builder::AnonymousOwnedListBuilder::new(
            "tokens".into(),
            hashes.len(),
            Some(DataType::Struct(vec![
                Field::new("token".into(), DataType::String),
                Field::new("start".into(), DataType::Int64),
                Field::new("end".into(), DataType::Int64),
            ])),
        );
        for tok in tokens {
            let tok_s = Series::new("token".into(), vec![tok.to_string()]);
            let start_s = Series::new("start".into(), vec![0i64]);
            let end_s = Series::new("end".into(), vec![tok.len() as i64]);
            let struct_s = StructChunked::from_series(
                PlSmallStr::EMPTY,
                1,
                [tok_s, start_s, end_s].iter(),
            )
            .unwrap()
            .into_series();
            use polars::chunked_array::builder::ListBuilderTrait;
            list_builder.append_series(&struct_s).unwrap();
        }
        let list_s = list_builder.finish().into_series();
        DataFrame::new_infer_height(vec![hash_s.into(), list_s.into()]).unwrap()
    }

    #[test]
    fn write_delta_then_load_round_trips_hashes() {
        let dir = temp_dir("write_load");
        let bucket = "rt__deadbeef.parquet";
        let mut df = make_cache_df(&[100u64, 200, 300], &["alpha", "beta", "gamma"]);
        let path = write_delta(&dir, bucket, &mut df).unwrap();
        assert!(path.exists(), "delta file should exist on disk");

        let files = list_bucket_files(&dir, bucket);
        assert_eq!(files.len(), 1);
        let map = load_cache_map(&files).unwrap();
        assert_eq!(map.len(), 3);
        assert!(map.contains_key(&100));
        assert!(map.contains_key(&200));
        assert!(map.contains_key(&300));
    }

    #[test]
    fn load_cache_map_keeps_first_for_duplicate_hashes() {
        let dir = temp_dir("dedup");
        let bucket = "dd__cafe.parquet";
        // First write: hash 42 → "first"
        let mut df1 = make_cache_df(&[42u64], &["first"]);
        write_delta(&dir, bucket, &mut df1).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Second write: hash 42 → "second" (newer file)
        let mut df2 = make_cache_df(&[42u64], &["second"]);
        write_delta(&dir, bucket, &mut df2).unwrap();

        let files = list_bucket_files(&dir, bucket);
        assert_eq!(files.len(), 2, "expected two delta files");
        let map = load_cache_map(&files).unwrap();
        assert_eq!(map.len(), 1);
        // Verify the FIRST write won by checking the token text inside
        let val = map.get(&42).expect("hash 42 should be present");
        let s = format!("{val:?}");
        assert!(
            s.contains("first") && !s.contains("second"),
            "expected first-writer-wins, got {s}"
        );
    }

    #[test]
    fn write_delta_no_op_when_height_is_zero() {
        let dir = temp_dir("empty");
        let bucket = "z__0000.parquet";
        let empty_hash = Series::new(CONTENT_HASH_COLUMN.into(), Vec::<u64>::new());
        let empty_tokens = Series::new_empty(
            "tokens".into(),
            &DataType::List(Box::new(DataType::Struct(vec![
                Field::new("token".into(), DataType::String),
                Field::new("start".into(), DataType::Int64),
                Field::new("end".into(), DataType::Int64),
            ]))),
        );
        let mut df =
            DataFrame::new_infer_height(vec![empty_hash.into(), empty_tokens.into()]).unwrap();
        let path = write_delta(&dir, bucket, &mut df).unwrap();
        // Conventional bucket path returned, but no file written
        assert!(!path.exists());
        assert!(list_bucket_files(&dir, bucket).is_empty());
    }
}
