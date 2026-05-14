//! On-demand Lindera dictionary downloader.
//!
//! Phase 5 (Lindera JA + KO) ships three prebuilt morpheme dicts:
//! IPADIC (~25 MB) and UniDic (~100 MB) for Japanese, ko-dic (~12 MB) for
//! Korean. Per `docs/pluggable-tokeniser/PLAN.md` they are NOT bundled in
//! the polars-text wheel — instead the first JA/KO tokenize call lands
//! here, fetches the matching tarball from the HuggingFace dataset
//! `ldaca/lindera-dicts` (overridable via `LDACA_LINDERA_DICT_REPO` for
//! testing), extracts into the per-OS cache dir, and hands the loaded
//! `Tokenizer` back to `tokenizer.rs` for memoization in the existing
//! `REGISTRY`.
//!
//! Cache layout:
//!   <cache-dir>/ldaca/lindera/
//!     ipadic-mecab-2.7.0-bin/   matrix.mtx, dict.da, unk.bin, ...
//!     unidic-mecab-2.1.2-bin/   ...
//!     ko-dic-2.1.1-bin/         ...
//!
//! Subsequent calls hit the in-process REGISTRY first and never re-enter
//! this module — `ensure_lindera_tokenizer` is only called on cache miss.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer as LinderaTokenizer;

/// HF dataset repo hosting the prebuilt Lindera dicts. Tests + CI can
/// point at a fixture repo by setting this env var.
const DEFAULT_DICT_REPO: &str = "ldaca/lindera-dicts";
const DICT_REPO_ENV: &str = "LDACA_LINDERA_DICT_REPO";

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

    fs::create_dir_all(&root)
        .with_context(|| format!("Failed to create cache dir {root:?}"))?;

    let repo_id =
        std::env::var(DICT_REPO_ENV).unwrap_or_else(|_| DEFAULT_DICT_REPO.to_string());
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

    extract_tar_gz(&archive_path, &root)
        .with_context(|| format!("Failed to extract {archive_path:?}"))?;

    if !dict_dir.join("matrix.mtx").is_file() {
        bail!(
            "Lindera dict archive {archive} did not extract \
             to {expected:?}/matrix.mtx — repo layout may have changed",
            archive = kind.archive_name(),
            expected = dict_dir,
        );
    }
    Ok(dict_dir)
}

fn extract_tar_gz(archive_path: &Path, dest: &Path) -> Result<()> {
    use flate2::read::GzDecoder;
    let f = fs::File::open(archive_path)
        .with_context(|| format!("Failed to open {archive_path:?}"))?;
    let gz = GzDecoder::new(f);
    let mut ar = tar::Archive::new(gz);
    ar.unpack(dest).context("tar extract failed")
}

/// Build a `LinderaTokenizer` for the given dict, downloading + extracting
/// on first use. Caller (in `tokenizer.rs`) wraps the result in the
/// shared `REGISTRY` so subsequent lookups skip both the download and
/// the dict-load step.
pub fn ensure_lindera_tokenizer(kind: LinderaDict) -> Result<LinderaTokenizer> {
    let dict_dir = ensure_dict(kind)?;
    let uri = format!("file://{}", dict_dir.display());
    let dictionary = load_dictionary(&uri)
        .map_err(|e| anyhow::anyhow!("Lindera load_dictionary({uri}) failed: {e}"))?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
    Ok(LinderaTokenizer::new(segmenter))
}

#[cfg(test)]
mod tests {
    use super::LinderaDict;

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
}
