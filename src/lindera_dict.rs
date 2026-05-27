//! On-demand Lindera dictionary downloader.
//!
//! Phase 5 (Lindera JA + KO) ships three prebuilt morpheme dicts:
//! IPADIC (~15 MB gzipped) and UniDic (~50 MB gzipped) for Japanese,
//! ko-dic (~34 MB gzipped) for Korean. Those larger dicts are NOT bundled in
//! the polars-text wheel — instead the first JA/KO tokenize call lands here,
//! fetches the matching tarball from
//! the HuggingFace dataset `SIH/lindera-dicts` (overridable via
//! `LDACA_LINDERA_DICT_REPO` for testing), extracts into the per-OS
//! cache dir, and hands the loaded `Tokenizer` back to `tokenizer.rs`
//! for memoization in the existing `REGISTRY`.
//! Chinese Jieba is embedded through Lindera and uses the same tokenizer
//! construction helper without the download/cache path.
//!
//! Cache layout:
//!   <cache-dir>/ldaca/lindera/
//!     ipadic-mecab-2.7.0-bin/   matrix.mtx, dict.da, unk.bin, ...
//!     unidic-mecab-2.1.2-bin/   ...
//!     ko-dic-2.1.1-bin/         ...
//!
//! Subsequent calls hit the in-process REGISTRY first and never re-enter
//! this module — `ensure_lindera_tokenizer` is only called on cache miss.

use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer as LinderaTokenizer;

/// HF dataset repo hosting the prebuilt Lindera dicts. Tests + CI can
/// point at a fixture repo by setting this env var.
const DEFAULT_DICT_REPO: &str = "SIH/lindera-dicts";
const DICT_REPO_ENV: &str = "LDACA_LINDERA_DICT_REPO";
const LOCK_RETRY_COUNT: usize = 300;
const LOCK_RETRY_DELAY: Duration = Duration::from_millis(100);

/// Which prebuilt dict to fetch. Mirrors the model-id constants in
/// `tokenizer.rs` (`LINDERA_JA_IPADIC_MODEL_ID` etc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinderaDict {
    JaIpadic,
    JaUnidic,
    KoDic,
}

impl LinderaDict {
    fn archive_name(&self) -> &'static str {
        match self {
            LinderaDict::JaIpadic => "ipadic-mecab-2.7.0-bin.tar.gz",
            LinderaDict::JaUnidic => "unidic-mecab-2.1.2-bin.tar.gz",
            LinderaDict::KoDic => "ko-dic-2.1.1-bin.tar.gz",
        }
    }

    /// The directory name we expect the tarball to extract to.
    fn cache_subdir(&self) -> &'static str {
        match self {
            LinderaDict::JaIpadic => "ipadic-mecab-2.7.0-bin",
            LinderaDict::JaUnidic => "unidic-mecab-2.1.2-bin",
            LinderaDict::KoDic => "ko-dic-2.1.1-bin",
        }
    }
}

/// `<cache-dir>/ldaca/lindera/`. Resolves via `dirs::cache_dir` so it
/// lands at the OS-appropriate location (Library/Caches on macOS,
/// ~/.cache on Linux, %LOCALAPPDATA% on Windows).
fn cache_root() -> Result<PathBuf> {
    let base = dirs::cache_dir().context("Could not resolve OS cache directory")?;
    Ok(base.join("ldaca").join("lindera"))
}

struct DictLock {
    path: PathBuf,
}

impl Drop for DictLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn acquire_dict_lock(lock_path: &Path) -> Result<DictLock> {
    for _ in 0..LOCK_RETRY_COUNT {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(lock_path)
        {
            Ok(_) => {
                return Ok(DictLock {
                    path: lock_path.to_path_buf(),
                });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                thread::sleep(LOCK_RETRY_DELAY);
            }
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("Failed to acquire Lindera dict lock {lock_path:?}"));
            }
        }
    }
    bail!("Timed out waiting for Lindera dict lock {lock_path:?}");
}

fn fresh_extract_dir(root: &Path, kind: LinderaDict) -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("System clock is before UNIX_EPOCH")?
        .as_nanos();
    let path = root.join(format!(
        ".{}.extract.{}.{}",
        kind.cache_subdir(),
        std::process::id(),
        nonce
    ));
    fs::create_dir_all(&path)
        .with_context(|| format!("Failed to create temp extract dir {path:?}"))?;
    Ok(path)
}

/// Returns the absolute path to the extracted dict directory. Downloads
/// + extracts from HF on cache miss.
pub fn ensure_dict(kind: LinderaDict) -> Result<PathBuf> {
    let root = cache_root()?;
    let dict_dir = root.join(kind.cache_subdir());

    // Sentinel: lindera dicts always include matrix.mtx. If it's there
    // we treat the extract as complete and short-circuit.
    if dict_dir.join("matrix.mtx").is_file() {
        return Ok(dict_dir);
    }

    fs::create_dir_all(&root).with_context(|| format!("Failed to create cache dir {root:?}"))?;

    let lock_path = root.join(format!(".{}.lock", kind.cache_subdir()));
    let _lock = acquire_dict_lock(&lock_path)?;
    if dict_dir.join("matrix.mtx").is_file() {
        return Ok(dict_dir);
    }

    let repo_id = std::env::var(DICT_REPO_ENV).unwrap_or_else(|_| DEFAULT_DICT_REPO.to_string());
    let api = ApiBuilder::from_env()
        .build()
        .context("Failed to initialize hf-hub client")?;
    let repo = api.dataset(repo_id.clone());
    let archive_path = repo.get(kind.archive_name()).with_context(|| {
        format!(
            "Failed to fetch {archive} from HF dataset {repo_id}",
            archive = kind.archive_name()
        )
    })?;

    let extract_dir = fresh_extract_dir(&root, kind)?;
    extract_tar_gz(&archive_path, &extract_dir)
        .with_context(|| format!("Failed to extract {archive_path:?}"))?;

    let extracted_dict_dir = extract_dir.join(kind.cache_subdir());

    if !extracted_dict_dir.join("matrix.mtx").is_file() {
        let _ = fs::remove_dir_all(&extract_dir);
        bail!(
            "Lindera dict archive {archive} did not extract \
             to {expected:?}/matrix.mtx — repo layout may have changed",
            archive = kind.archive_name(),
            expected = extracted_dict_dir,
        );
    }

    if dict_dir.exists() {
        fs::remove_dir_all(&dict_dir)
            .with_context(|| format!("Failed to remove incomplete dict dir {dict_dir:?}"))?;
    }
    fs::rename(&extracted_dict_dir, &dict_dir)
        .with_context(|| format!("Failed to move {extracted_dict_dir:?} to {dict_dir:?}"))?;
    let _ = fs::remove_dir_all(&extract_dir);
    Ok(dict_dir)
}

fn extract_tar_gz(archive_path: &Path, dest: &Path) -> Result<()> {
    use flate2::read::GzDecoder;
    let f =
        fs::File::open(archive_path).with_context(|| format!("Failed to open {archive_path:?}"))?;
    let gz = GzDecoder::new(f);
    let mut ar = tar::Archive::new(gz);
    ar.unpack(dest).context("tar extract failed")
}

fn lindera_tokenizer_from_uri(uri: &str) -> Result<LinderaTokenizer> {
    let dictionary = load_dictionary(uri)
        .map_err(|e| anyhow::anyhow!("Lindera load_dictionary({uri}) failed: {e}"))?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
    Ok(LinderaTokenizer::new(segmenter))
}

pub fn ensure_embedded_lindera_tokenizer(dictionary_name: &str) -> Result<LinderaTokenizer> {
    let uri = format!("embedded://{dictionary_name}");
    lindera_tokenizer_from_uri(&uri)
}

/// Build a `LinderaTokenizer` for the given dict, downloading + extracting
/// on first use. Caller (in `tokenizer.rs`) wraps the result in the
/// shared `REGISTRY` so subsequent lookups skip both the download and
/// the dict-load step.
pub fn ensure_lindera_tokenizer(kind: LinderaDict) -> Result<LinderaTokenizer> {
    let dict_dir = ensure_dict(kind)?;
    let uri = format!("file://{}", dict_dir.display());
    lindera_tokenizer_from_uri(&uri)
}

#[cfg(test)]
mod tests {
    use super::{acquire_dict_lock, LinderaDict};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn archive_and_subdir_names_are_distinct_per_kind() {
        // Sanity: prevents an accidental copy-paste collision in
        // archive_name / cache_subdir that would route two dicts to
        // the same cache path.
        let all = [
            LinderaDict::JaIpadic,
            LinderaDict::JaUnidic,
            LinderaDict::KoDic,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.archive_name(), b.archive_name());
                assert_ne!(a.cache_subdir(), b.cache_subdir());
            }
        }
    }

    #[test]
    fn dict_lock_is_released_on_drop() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ldaca-lindera-lock-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("temp dir");
        let lock_path = dir.join("dict.lock");

        {
            let _lock = acquire_dict_lock(&lock_path).expect("first lock");
            assert!(lock_path.exists());
        }

        assert!(!lock_path.exists());
        let _lock = acquire_dict_lock(&lock_path).expect("second lock");
        assert!(fs::metadata(&lock_path).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }
}
