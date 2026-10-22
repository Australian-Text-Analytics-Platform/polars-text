//! ONNX Runtime sentence embeddings for topic modeling and Polars expressions.
//!
//! Why this exists: embedding is the dominant cost in native topic modeling, so
//! the pipeline now uses ONNX Runtime instead of the previous CPU-only Candle
//! path. Model management stays automatic: callers provide only a Hugging Face
//! model id, and this module uses `hf-hub`'s default cache/download behavior to
//! fetch tokenizer/config/ONNX artifacts when they are missing locally.
//!
//! Contract: only repositories with ONNX artifacts are supported. If a repo has
//! only safetensors/PyTorch weights, loading fails with a clear error rather
//! than attempting conversion at runtime.
//!
//! Called by: `topic_modeling::run` for Topic Segment embeddings and, through
//! the expression plugin, `.text.embedding`.
//!
//! Execution (polars-text 0.6.8, Wordflow issue 412): every call sorts its
//! texts by token count and runs batches of 32 rows, so a batch holds texts of
//! similar length and pads little. On macOS the two curated models run on Core
//! ML (GPU and Neural Engine) through static-shape sessions: one compiled
//! session per length bucket (32, 64, 128 and 256 tokens, capped by the
//! model's limit), each created on first use and compiled once into a cache
//! next to the Hugging Face cache. A batch goes to the smallest bucket that
//! holds its longest text. Dynamic-shape Core ML was 5x slower than the CPU
//! (2026-10-07: 50 partitions, re-planned per batch shape); a fixed shape runs
//! the whole graph on Core ML, about 3x faster than all CPU cores on an M5 Pro.
//! Other models, Linux and Windows embed on the CPU with dynamic padding. When
//! Core ML cannot be prepared the model falls back to the CPU and keeps the
//! reason (`provider_fallback_reason`).

use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use anyhow::{Context, Result};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, ValueType};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokenizers::{Encoding, Tokenizer, TruncationParams};

/// Canonical Sentence Transformers default with a published ONNX artifact.
pub const DEFAULT_EMBEDDER_REPO_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// Multilingual model offered beside the default (Wordflow issue 413); its
/// limit is 128 tokens.
pub const MULTILINGUAL_EMBEDDER_REPO_ID: &str =
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
/// v3 (0.6.8): length-sorted batches and static Core ML sessions on macOS.
const EMBEDDING_PIPELINE_VERSION: &str = "sentence-transformers-v3";

const EMBEDDING_THREADS_ENV: &str = "POLARS_TEXT_EMBEDDING_THREADS";
/// `cpu` keeps a Mac on the CPU (for comparisons and troubleshooting).
const EMBEDDING_PROVIDER_ENV: &str = "POLARS_TEXT_EMBEDDING_PROVIDER";

/// Rows per ONNX run; static sessions are compiled for exactly this many rows.
const ENCODE_BATCH_ROWS: usize = 32;
/// Token lengths of the static sessions, capped by the model's limit.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const STATIC_TOKEN_BUCKETS: [usize; 4] = [32, 64, 128, 256];
const CPU_PROVIDER_ID: &str = "CPUExecutionProvider";

/// A loaded ONNX sentence encoder. `Session::run` requires `&mut self`, so each
/// session sits behind a mutex while cheap tokenizer work remains lock-free.
pub struct Embedder {
    backend: Backend,
    /// Truncates at `max_length`; never pads (rows are padded per batch).
    tokenizer: Tokenizer,
    pad_id: i64,
    pad_type_id: i64,
    input_names: Vec<String>,
    output: EmbeddingOutput,
    pooling: PoolingConfig,
    normalize: bool,
    embedding_dim: usize,
    max_length: usize,
    model_id: String,
    provider_id: String,
    provider_fallback_reason: Mutex<Option<String>>,
    cache_fingerprint: String,
}

/// How batches run: one dynamic-shape CPU session, or static Core ML buckets.
enum Backend {
    Dynamic(Mutex<Session>),
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Static(StaticSessions),
}

/// Static-shape sessions for one model, one per length bucket, created lazily.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct StaticSessions {
    onnx_path: PathBuf,
    cache_root: Option<PathBuf>,
    profile: Option<PathBuf>,
    buckets: Vec<StaticBucket>,
    /// CPU session for a bucket whose static session could not be prepared.
    cpu: OnceLock<std::result::Result<Mutex<Session>, String>>,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
struct StaticBucket {
    tokens: usize,
    session: OnceLock<std::result::Result<Mutex<Session>, String>>,
}

/// Which sessions `load_embedder_with` prepares.
#[derive(Debug, Clone)]
enum SessionPlan {
    /// Static Core ML buckets where validated, otherwise the CPU.
    Preferred {
        cache_root: Option<PathBuf>,
        profile: Option<PathBuf>,
    },
    /// The CPU only, with dynamic padding.
    Cpu { profile: Option<PathBuf> },
}

struct PreparedBackend {
    backend: Backend,
    provider_id: String,
    fallback_reason: Option<String>,
}

#[derive(Debug, Clone)]
enum EmbeddingOutput {
    TokenEmbeddings(String),
    SentenceEmbedding(String),
}

impl EmbeddingOutput {
    fn name(&self) -> &str {
        match self {
            Self::TokenEmbeddings(name) | Self::SentenceEmbedding(name) => name,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct SentenceTransformerModule {
    idx: usize,
    path: String,
    #[serde(rename = "type")]
    module_type: String,
}

#[derive(Debug, Clone, Deserialize)]
struct PoolingConfig {
    word_embedding_dimension: usize,
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
}

impl PoolingConfig {
    fn enabled_count(&self) -> usize {
        [
            self.pooling_mode_cls_token,
            self.pooling_mode_max_tokens,
            self.pooling_mode_mean_tokens,
            self.pooling_mode_mean_sqrt_len_tokens,
            self.pooling_mode_weightedmean_tokens,
            self.pooling_mode_lasttoken,
        ]
        .into_iter()
        .filter(|enabled| *enabled)
        .count()
    }
}

struct PinnedRepo {
    repo: ApiRepo,
    snapshot_root: PathBuf,
    revision: String,
}

impl PinnedRepo {
    fn get(&self, filename: &str) -> Result<PathBuf> {
        let cached = self.snapshot_root.join(filename);
        if cached.exists() {
            Ok(cached)
        } else {
            self.repo
                .get(filename)
                .with_context(|| format!("fetch {filename} at revision {}", self.revision))
        }
    }
}

impl Embedder {
    /// Output embedding dimensionality when it can be read from `config.json`.
    /// Some ONNX repos omit this field; those still encode correctly and infer
    /// the real dimensionality from ORT output at runtime.
    pub fn dim(&self) -> usize {
        self.embedding_dim
    }

    /// Hugging Face repo id used for model download and embedding cache keys.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Configured execution policy, used for cache keys and diagnostics.
    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    /// Why Core ML could not be prepared and the CPU was used instead, if so.
    pub fn provider_fallback_reason(&self) -> Option<String> {
        self.provider_fallback_reason
            .lock()
            .ok()
            .and_then(|reason| reason.clone())
    }

    /// Immutable Hugging Face snapshot used for every model artifact.
    /// Complete embedding-pipeline identity used by persistent caches.
    pub fn cache_fingerprint(&self) -> &str {
        &self.cache_fingerprint
    }

    /// Canonical maximum input length declared by the model.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// A tokenizer clone with truncation/padding disabled for segment-size measurement.
    pub fn sizing_tokenizer(&self) -> Tokenizer {
        tokenizer_for_sizing(&self.tokenizer)
    }

    /// Encode `texts` into row vectors (the model's pooling and normalization),
    /// in input order.
    ///
    /// Texts are tokenized once, sorted by token count and run in batches of
    /// `ENCODE_BATCH_ROWS`, so each batch pads only to its own longest text
    /// (or, on Core ML, to the smallest bucket that holds it).
    pub fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let encodings = self
            .tokenizer
            .encode_batch(refs, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode_batch failed: {e}"))?;
        let lengths = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .collect::<Vec<_>>();

        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        for batch in length_sorted_batches(&lengths, ENCODE_BATCH_ROWS) {
            let rows = batch
                .iter()
                .map(|&index| &encodings[index])
                .collect::<Vec<_>>();
            let longest = batch.iter().map(|&index| lengths[index]).max().unwrap_or(0);
            let encoded = if longest == 0 {
                vec![vec![0.0; self.embedding_dim]; rows.len()]
            } else {
                self.encode_rows(&rows, longest)?
            };
            anyhow::ensure!(
                encoded.len() == rows.len(),
                "embedding batch returned {} vectors for {} texts",
                encoded.len(),
                rows.len()
            );
            for (&index, vector) in batch.iter().zip(encoded) {
                vectors[index] = Some(vector);
            }
        }
        vectors
            .into_iter()
            .map(|vector| vector.context("embedding batch left a text without a vector"))
            .collect()
    }

    /// Run one batch of tokenized rows on the session its length belongs to.
    fn encode_rows(&self, rows: &[&Encoding], longest: usize) -> Result<Vec<Vec<f32>>> {
        match &self.backend {
            Backend::Dynamic(session) => self.run_session(session, rows, rows.len(), longest),
            Backend::Static(sessions) => {
                let bucket = bucket_index(
                    &sessions
                        .buckets
                        .iter()
                        .map(|bucket| bucket.tokens)
                        .collect::<Vec<_>>(),
                    longest,
                )
                .with_context(|| {
                    format!("no static embedding bucket holds {longest} tokens")
                })?;
                match sessions.bucket_session(bucket) {
                    Ok(session) => self.run_session(
                        session,
                        rows,
                        ENCODE_BATCH_ROWS,
                        sessions.buckets[bucket].tokens,
                    ),
                    Err(reason) => {
                        self.record_fallback(&reason);
                        let session = sessions.cpu_session()?;
                        self.run_session(session, rows, rows.len(), longest)
                    }
                }
            }
        }
    }

    fn record_fallback(&self, reason: &str) {
        if let Ok(mut recorded) = self.provider_fallback_reason.lock() {
            if recorded.is_none() {
                eprintln!(
                    "polars-text: Core ML embedding unavailable for {}, using the CPU: {reason}",
                    self.model_id
                );
                *recorded = Some(reason.to_owned());
            }
        }
    }

    /// Pad `rows` to `batch` rows of `seq` tokens and run them. Padding rows
    /// repeat the first row and are dropped from the result.
    fn run_session(
        &self,
        session: &Mutex<Session>,
        rows: &[&Encoding],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let first = rows.first().context("embedding batch has no rows")?;
        anyhow::ensure!(
            rows.len() <= batch,
            "embedding batch of {} rows exceeds {batch}",
            rows.len()
        );
        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        let mut token_types = Vec::with_capacity(batch * seq);
        for row in 0..batch {
            let enc = rows.get(row).unwrap_or(first);
            let length = enc.get_ids().len();
            anyhow::ensure!(
                length <= seq,
                "embedding row of {length} tokens exceeds the batch width {seq}"
            );
            let padding = seq - length;
            ids.extend(enc.get_ids().iter().map(|id| i64::from(*id)));
            ids.extend(std::iter::repeat_n(self.pad_id, padding));
            mask.extend(enc.get_attention_mask().iter().map(|id| i64::from(*id)));
            mask.extend(std::iter::repeat_n(0, padding));
            token_types.extend(enc.get_type_ids().iter().map(|id| i64::from(*id)));
            token_types.extend(std::iter::repeat_n(self.pad_type_id, padding));
        }

        let shape = [batch, seq];
        let input_ids = Tensor::from_array((shape, ids)).context("build input_ids tensor")?;
        let attention_mask =
            Tensor::from_array((shape, mask.clone())).context("build attention_mask tensor")?;
        let token_type_ids =
            Tensor::from_array((shape, token_types)).context("build token_type_ids tensor")?;

        let mut inputs = ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        };
        if self.input_names.iter().any(|name| name == "token_type_ids") {
            inputs.push((
                Cow::from("token_type_ids"),
                SessionInputValue::from(token_type_ids),
            ));
        }

        let mut session = session
            .lock()
            .map_err(|_| anyhow::anyhow!("embedder session lock poisoned"))?;
        let outputs = session
            .run(inputs)
            .context("ONNX embedding inference failed")?;
        let output = outputs.get(self.output.name()).ok_or_else(|| {
            anyhow::anyhow!(
                "ONNX inference did not return declared output {}",
                self.output.name()
            )
        })?;
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .context("extract ONNX embedding tensor")?;
        let dims: Vec<usize> = shape.iter().map(|dim| *dim as usize).collect();
        let mut vectors = embeddings_from_output(
            &self.output,
            &self.pooling,
            self.normalize,
            OutputBatch {
                dims: &dims,
                data,
                batch,
                sequence_length: seq,
                attention_mask: &mask,
            },
        )?;
        vectors.truncate(rows.len());
        Ok(vectors)
    }
}

/// Input positions sorted by token count (ties keep input order), in batches.
fn length_sorted_batches(lengths: &[usize], rows: usize) -> Vec<Vec<usize>> {
    let mut order = (0..lengths.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| lengths[index]);
    order.chunks(rows.max(1)).map(<[usize]>::to_vec).collect()
}

/// The smallest bucket that holds `tokens` (buckets ascend).
fn bucket_index(buckets: &[usize], tokens: usize) -> Option<usize> {
    buckets.iter().position(|&bucket| bucket >= tokens)
}

/// Static bucket lengths for a model: the standard buckets below its limit,
/// plus the limit itself so the longest texts always have a bucket.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn static_buckets(max_length: usize) -> Vec<usize> {
    let mut buckets = STATIC_TOKEN_BUCKETS
        .into_iter()
        .filter(|&tokens| tokens < max_length)
        .collect::<Vec<_>>();
    buckets.push(max_length);
    buckets
}

/// Static Core ML is validated only for the two curated models in their plain
/// fp32 `model.onnx` export (not the quantized artifacts).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn static_shape_validated(model_id: &str, artifact: &Path) -> bool {
    artifact.file_name().is_some_and(|name| name == "model.onnx")
        && matches!(
            model_id,
            DEFAULT_EMBEDDER_REPO_ID | MULTILINGUAL_EMBEDDER_REPO_ID
        )
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn static_provider_id(buckets: &[usize]) -> String {
    let tokens = buckets
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("/");
    format!("CoreMLExecutionProvider:MLProgram:ALL:{ENCODE_BATCH_ROWS}x{tokens}+{CPU_PROVIDER_ID}")
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
impl StaticSessions {
    fn new(
        onnx_path: &Path,
        cache_root: Option<PathBuf>,
        profile: Option<PathBuf>,
        buckets: &[usize],
    ) -> Self {
        Self {
            onnx_path: onnx_path.to_path_buf(),
            cache_root,
            profile,
            buckets: buckets
                .iter()
                .map(|&tokens| StaticBucket {
                    tokens,
                    session: OnceLock::new(),
                })
                .collect(),
            cpu: OnceLock::new(),
        }
    }

    /// The bucket's session, compiled (or loaded from the cache) on first use.
    fn bucket_session(&self, index: usize) -> std::result::Result<&Mutex<Session>, String> {
        let bucket = &self.buckets[index];
        bucket
            .session
            .get_or_init(|| {
                build_static_session(
                    &self.onnx_path,
                    bucket.tokens,
                    self.cache_root.as_deref(),
                    self.profile
                        .as_ref()
                        .map(|dir| dir.join(format!("static-{}", bucket.tokens)))
                        .as_deref(),
                )
                .map(Mutex::new)
                .map_err(|error| format!("{error:#}"))
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn cpu_session(&self) -> Result<&Mutex<Session>> {
        self.cpu
            .get_or_init(|| {
                build_cpu_session(
                    &self.onnx_path,
                    self.profile.as_ref().map(|dir| dir.join("cpu")).as_deref(),
                )
                .map(Mutex::new)
                .map_err(|error| format!("{error:#}"))
            })
            .as_ref()
            .map_err(|error| anyhow::anyhow!("CPU fallback session failed: {error}"))
    }

    /// Any prepared session, for reading the graph's inputs and outputs.
    fn metadata_session(&self) -> Option<&Mutex<Session>> {
        self.buckets
            .iter()
            .find_map(|bucket| bucket.session.get().and_then(|session| session.as_ref().ok()))
    }
}

impl Backend {
    fn metadata_session(&self) -> Result<&Mutex<Session>> {
        match self {
            Self::Dynamic(session) => Ok(session),
            Self::Static(sessions) => sessions
                .metadata_session()
                .context("no static embedding session was prepared"),
        }
    }
}

fn tokenizer_for_sizing(tokenizer: &Tokenizer) -> Tokenizer {
    let mut tok = tokenizer.clone();
    let _ = tok.with_truncation(None);
    tok.with_padding(None);
    tok
}

struct OutputBatch<'a> {
    dims: &'a [usize],
    data: &'a [f32],
    batch: usize,
    sequence_length: usize,
    attention_mask: &'a [i64],
}

fn embeddings_from_output(
    output: &EmbeddingOutput,
    pooling: &PoolingConfig,
    normalize: bool,
    batch: OutputBatch<'_>,
) -> Result<Vec<Vec<f32>>> {
    let OutputBatch {
        dims,
        data,
        batch,
        sequence_length: seq,
        attention_mask: mask,
    } = batch;
    let expected_dimension = pooling
        .word_embedding_dimension
        .checked_mul(pooling.enabled_count())
        .ok_or_else(|| anyhow::anyhow!("metadata-derived embedding dimension overflow"))?;
    let mut rows = match (output, dims) {
        (EmbeddingOutput::SentenceEmbedding(_), [out_batch, dim]) if *out_batch == batch => {
            if *dim != expected_dimension {
                anyhow::bail!(
                    "sentence embedding output width {dim} does not match metadata-derived width {expected_dimension}"
                );
            }
            if data.len() != batch * dim {
                anyhow::bail!(
                    "sentence embedding tensor length {} does not match shape [{batch}, {dim}]",
                    data.len()
                );
            }
            data.chunks_exact(*dim).map(<[f32]>::to_vec).collect()
        }
        (EmbeddingOutput::TokenEmbeddings(_), [out_batch, out_seq, dim])
            if *out_batch == batch && *out_seq == seq =>
        {
            pool_token_embeddings(data, batch, seq, *dim, mask, pooling)?
        }
        other => anyhow::bail!(
            "ONNX output {} has incompatible shape {:?} for the declared Sentence Transformers graph",
            output.name(),
            other.1
        ),
    };
    if normalize {
        normalize_nested_rows(&mut rows);
    }
    Ok(rows)
}

fn pool_token_embeddings(
    hidden: &[f32],
    batch: usize,
    seq: usize,
    dim: usize,
    mask: &[i64],
    pooling: &PoolingConfig,
) -> Result<Vec<Vec<f32>>> {
    if hidden.len() != batch * seq * dim {
        anyhow::bail!(
            "hidden tensor length {} does not match shape [{batch}, {seq}, {dim}]",
            hidden.len()
        );
    }
    if mask.len() != batch * seq {
        anyhow::bail!(
            "attention mask length {} does not match shape [{batch}, {seq}]",
            mask.len()
        );
    }

    if pooling.word_embedding_dimension != dim {
        anyhow::bail!(
            "pooling metadata expects hidden size {}, but ONNX returned {dim}",
            pooling.word_embedding_dimension
        );
    }
    let mode_count = pooling.enabled_count();
    if mode_count == 0 {
        anyhow::bail!("pooling metadata enables no supported pooling mode");
    }

    let mut rows = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut pooled = Vec::with_capacity(dim * mode_count);
        let mask_row = &mask[row * seq..(row + 1) * seq];
        if pooling.pooling_mode_cls_token {
            let offset = row * seq * dim;
            pooled.extend_from_slice(&hidden[offset..offset + dim]);
        }
        if pooling.pooling_mode_max_tokens {
            let mut values = vec![f32::NEG_INFINITY; dim];
            for (token, &mask_value) in mask_row.iter().enumerate() {
                if mask_value == 0 {
                    continue;
                }
                let offset = (row * seq + token) * dim;
                for col in 0..dim {
                    values[col] = values[col].max(hidden[offset + col]);
                }
            }
            for value in &mut values {
                if !value.is_finite() {
                    *value = 0.0;
                }
            }
            pooled.extend(values);
        }

        let mut sum = vec![0.0_f32; dim];
        let mut token_count = 0.0_f32;
        let mut weighted_sum = vec![0.0_f32; dim];
        let mut weight_total = 0.0_f32;
        for (token, &raw_mask_value) in mask_row.iter().enumerate() {
            let mask_value = raw_mask_value as f32;
            if mask_value == 0.0 {
                continue;
            }
            token_count += mask_value;
            let weight = (token + 1) as f32 * mask_value;
            weight_total += weight;
            let offset = (row * seq + token) * dim;
            for col in 0..dim {
                sum[col] += hidden[offset + col] * mask_value;
                weighted_sum[col] += hidden[offset + col] * weight;
            }
        }
        if pooling.pooling_mode_mean_tokens {
            let denom = token_count.max(1e-9);
            pooled.extend(sum.iter().map(|value| value / denom));
        }
        if pooling.pooling_mode_mean_sqrt_len_tokens {
            let denom = token_count.sqrt().max(1e-9);
            pooled.extend(sum.iter().map(|value| value / denom));
        }
        if pooling.pooling_mode_weightedmean_tokens {
            let denom = weight_total.max(1e-9);
            pooled.extend(weighted_sum.iter().map(|value| value / denom));
        }
        if pooling.pooling_mode_lasttoken {
            if let Some(token) = mask_row.iter().rposition(|value| *value != 0) {
                let offset = (row * seq + token) * dim;
                pooled.extend_from_slice(&hidden[offset..offset + dim]);
            } else {
                pooled.resize(pooled.len() + dim, 0.0);
            }
        }
        rows.push(pooled);
    }
    Ok(rows)
}

fn normalize_nested_rows(rows: &mut [Vec<f32>]) {
    for row in rows {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in row {
                *value /= norm;
            }
        }
    }
}

static REGISTRY: OnceLock<RwLock<HashMap<String, Arc<Embedder>>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<String, Arc<Embedder>>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Get-or-load the embedder for `repo_id` (default ONNX model if `None`).
pub fn ensure_embedder(repo_id: Option<&str>) -> Result<Arc<Embedder>> {
    let key = repo_id
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(DEFAULT_EMBEDDER_REPO_ID)
        .to_string();

    {
        let map = registry()
            .read()
            .map_err(|_| anyhow::anyhow!("embedder registry lock poisoned"))?;
        if let Some(embedder) = map.get(&key) {
            return Ok(Arc::clone(embedder));
        }
    }
    let mut map = registry()
        .write()
        .map_err(|_| anyhow::anyhow!("embedder registry lock poisoned"))?;
    if let Some(embedder) = map.get(&key) {
        return Ok(Arc::clone(embedder));
    }
    let embedder = Arc::new(load_embedder(&key)?);
    map.insert(key, Arc::clone(&embedder));
    Ok(embedder)
}

type SizingTokenizers = RwLock<HashMap<String, Arc<Tokenizer>>>;
static SIZING_TOKENIZERS: OnceLock<SizingTokenizers> = OnceLock::new();

/// The model's tokenizer for measuring text (no truncation, no padding),
/// loaded without the ONNX graph, so measuring never compiles a Core ML model.
/// Reuses an embedder that is already loaded.
#[cfg_attr(not(feature = "topic-modeling"), allow(dead_code))]
pub fn ensure_sizing_tokenizer(repo_id: Option<&str>) -> Result<Arc<Tokenizer>> {
    let key = repo_id
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(DEFAULT_EMBEDDER_REPO_ID)
        .to_string();
    if let Some(embedder) = registry()
        .read()
        .map_err(|_| anyhow::anyhow!("embedder registry lock poisoned"))?
        .get(&key)
    {
        return Ok(Arc::new(embedder.sizing_tokenizer()));
    }
    let tokenizers = SIZING_TOKENIZERS.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(tokenizer) = tokenizers
        .read()
        .map_err(|_| anyhow::anyhow!("sizing tokenizer registry lock poisoned"))?
        .get(&key)
    {
        return Ok(Arc::clone(tokenizer));
    }
    let tokenizer = Arc::new(tokenizer_for_sizing(&load_model_metadata(&key)?.tokenizer));
    tokenizers
        .write()
        .map_err(|_| anyhow::anyhow!("sizing tokenizer registry lock poisoned"))?
        .entry(key)
        .or_insert_with(|| Arc::clone(&tokenizer));
    Ok(tokenizer)
}

fn load_embedder(repo_id: &str) -> Result<Embedder> {
    let plan = if provider_override_is_cpu() {
        SessionPlan::Cpu { profile: None }
    } else {
        SessionPlan::Preferred {
            cache_root: Some(coreml_cache_root()),
            profile: None,
        }
    };
    load_embedder_with(repo_id, &plan)
}

fn provider_override_is_cpu() -> bool {
    env::var(EMBEDDING_PROVIDER_ENV)
        .is_ok_and(|value| value.trim().eq_ignore_ascii_case("cpu"))
}

/// Compiled Core ML models live next to the downloaded models, in the same
/// Hugging Face cache (`HF_HOME`), so clearing that cache clears them too.
fn coreml_cache_root() -> PathBuf {
    hf_hub::Cache::from_env().path().join("coreml")
}

/// A model's Sentence Transformers metadata and tokenizer, without the ONNX
/// graph: enough to size text with the model's tokenizer.
struct ModelMetadata {
    repo: PinnedRepo,
    revision: String,
    pooling: PoolingConfig,
    normalize: bool,
    hidden_size: usize,
    mode_count: usize,
    max_length: usize,
    /// Truncates at `max_length`; padding as the repository declares it.
    tokenizer: Tokenizer,
}

fn load_model_metadata(repo_id: &str) -> Result<ModelMetadata> {
    let api = ApiBuilder::from_env()
        .build()
        .context("failed to init hf-hub api for embedder")?;
    let unpinned = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    let modules_path = unpinned
        .get("modules.json")
        .with_context(|| format!("fetch Sentence Transformers modules.json for {repo_id}"))?;
    let (snapshot_root, revision) = snapshot_from_path(&modules_path)?;
    let repo = PinnedRepo {
        repo: api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.clone(),
        )),
        snapshot_root,
        revision: revision.clone(),
    };

    let modules: Vec<SentenceTransformerModule> = read_json_path(&modules_path)?;
    let (transformer_path, pooling_path, normalize) = validate_module_graph(&modules, repo_id)?;
    let config_file = module_file(transformer_path, "config.json");
    let tokenizer_file = module_file(transformer_path, "tokenizer.json");
    let sentence_config_file = module_file(transformer_path, "sentence_bert_config.json");
    let pooling_file = module_file(pooling_path, "config.json");

    let config_path = repo.get(&config_file)?;
    let tokenizer_path = repo
        .get(&tokenizer_file)
        .or_else(|_| repo.get("tokenizer.json"))?;
    let pooling: PoolingConfig = read_json_path(&repo.get(&pooling_file)?)?;
    let hidden_size = read_required_usize(&config_path, "hidden_size")?;
    if hidden_size != pooling.word_embedding_dimension {
        anyhow::bail!(
            "embedding model {repo_id} declares hidden_size {hidden_size}, but pooling expects {}",
            pooling.word_embedding_dimension
        );
    }
    let mode_count = pooling.enabled_count();
    if mode_count == 0 {
        anyhow::bail!("embedding model {repo_id} enables no supported pooling mode");
    }
    let max_length = resolve_max_length(
        &repo,
        &sentence_config_file,
        &module_file(transformer_path, "tokenizer_config.json"),
        &config_path,
    )?;
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer for {repo_id}: {e}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("configure truncation: {e}"))?;
    Ok(ModelMetadata {
        repo,
        revision,
        pooling,
        normalize,
        hidden_size,
        mode_count,
        max_length,
        tokenizer,
    })
}

fn load_embedder_with(repo_id: &str, plan: &SessionPlan) -> Result<Embedder> {
    let ModelMetadata {
        repo,
        revision,
        pooling,
        normalize,
        hidden_size,
        mode_count,
        max_length,
        mut tokenizer,
    } = load_model_metadata(repo_id)?;
    let (onnx_path, onnx_artifact) = resolve_onnx_artifact(&repo, repo_id)?;

    // Batches are padded by `run_session` (to the batch's longest row or the
    // static bucket), with the tokenizer's own pad token when it declares one.
    let (pad_id, pad_type_id) = pad_ids(&tokenizer);
    tokenizer.with_padding(None);

    let PreparedBackend {
        backend,
        provider_id,
        fallback_reason,
    } = prepare_backend(&onnx_path, repo_id, max_length, plan)?;
    if let Some(reason) = &fallback_reason {
        eprintln!("polars-text: Core ML embedding unavailable for {repo_id}, using the CPU: {reason}");
    }
    let embedding_dim = hidden_size
        .checked_mul(mode_count)
        .ok_or_else(|| anyhow::anyhow!("embedding dimension overflow for {repo_id}"))?;
    let (input_names, output) = {
        let session = backend
            .metadata_session()?
            .lock()
            .map_err(|_| anyhow::anyhow!("embedder session lock poisoned"))?;
        let input_names = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect::<Vec<_>>();
        (input_names, select_embedding_output(&session, embedding_dim)?)
    };
    let cache_fingerprint = embedding_fingerprint(
        &revision,
        &onnx_artifact,
        &pooling,
        normalize,
        max_length,
        embedding_dim,
        &provider_id,
    );

    Ok(Embedder {
        backend,
        tokenizer,
        pad_id,
        pad_type_id,
        input_names,
        output,
        pooling,
        normalize,
        embedding_dim,
        max_length,
        model_id: repo_id.to_string(),
        provider_id,
        provider_fallback_reason: Mutex::new(fallback_reason),
        cache_fingerprint,
    })
}

fn snapshot_from_path(path: &Path) -> Result<(PathBuf, String)> {
    let snapshot_root = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Hugging Face cache path has no snapshot parent"))?;
    let snapshots_dir = snapshot_root
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str());
    if snapshots_dir != Some("snapshots") {
        anyhow::bail!(
            "Hugging Face artifact {} is not stored in an immutable snapshot",
            path.display()
        );
    }
    let revision = snapshot_root
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Hugging Face snapshot has no revision"))?;
    Ok((snapshot_root.to_path_buf(), revision.to_string()))
}

fn read_json_path<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("read model metadata {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("parse model metadata {}", path.display()))
}

fn module_file(module_path: &str, filename: &str) -> String {
    if module_path.is_empty() {
        filename.to_string()
    } else {
        format!("{module_path}/{filename}")
    }
}

fn validate_module_graph<'a>(
    modules: &'a [SentenceTransformerModule],
    repo_id: &str,
) -> Result<(&'a str, &'a str, bool)> {
    if modules.len() < 2 || modules.len() > 3 {
        anyhow::bail!(
            "unsupported Sentence Transformers graph for {repo_id}: expected Transformer, Pooling, and optional Normalize"
        );
    }
    for (expected, module) in modules.iter().enumerate() {
        if module.idx != expected {
            anyhow::bail!("unsupported non-contiguous module graph for {repo_id}");
        }
        let path = Path::new(&module.path);
        if path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
                && !module.path.is_empty()
        {
            anyhow::bail!(
                "unsupported Sentence Transformers module path {:?} for {repo_id}",
                module.path
            );
        }
    }
    if !is_standard_module(&modules[0].module_type, "Transformer")
        || !is_standard_module(&modules[1].module_type, "Pooling")
    {
        anyhow::bail!(
            "unsupported Sentence Transformers graph for {repo_id}: first modules must be Transformer and Pooling"
        );
    }
    let normalize = match modules.get(2) {
        Some(module) if is_standard_module(&module.module_type, "Normalize") => true,
        Some(module) => anyhow::bail!(
            "unsupported Sentence Transformers module {} for {repo_id}",
            module.module_type
        ),
        None => false,
    };
    Ok((&modules[0].path, &modules[1].path, normalize))
}

fn is_standard_module(module_type: &str, name: &str) -> bool {
    module_type.starts_with("sentence_transformers.models.")
        && module_type.rsplit('.').next() == Some(name)
}

fn read_required_usize(path: &Path, key: &str) -> Result<usize> {
    let value: serde_json::Value = read_json_path(path)?;
    valid_usize(value.get(key))
        .ok_or_else(|| anyhow::anyhow!("model metadata {} has no valid {key}", path.display()))
}

fn valid_usize(value: Option<&serde_json::Value>) -> Option<usize> {
    value
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0 && *value < 1_000_000)
}

fn resolve_max_length(
    repo: &PinnedRepo,
    sentence_config_file: &str,
    tokenizer_config_file: &str,
    model_config_path: &Path,
) -> Result<usize> {
    let sentence = repo
        .get(sentence_config_file)
        .ok()
        .map(|path| read_json_path(&path))
        .transpose()?;
    let tokenizer = repo
        .get(tokenizer_config_file)
        .ok()
        .map(|path| read_json_path(&path))
        .transpose()?;
    let model = read_json_path(model_config_path)?;
    canonical_max_length(sentence.as_ref(), tokenizer.as_ref(), &model)
}

fn canonical_max_length(
    sentence: Option<&serde_json::Value>,
    tokenizer: Option<&serde_json::Value>,
    model: &serde_json::Value,
) -> Result<usize> {
    sentence
        .and_then(|value| valid_usize(value.get("max_seq_length")))
        .or_else(|| tokenizer.and_then(|value| valid_usize(value.get("model_max_length"))))
        .or_else(|| valid_usize(model.get("max_position_embeddings")))
        .ok_or_else(|| {
            anyhow::anyhow!("embedding model metadata has no valid canonical maximum length")
        })
}

fn select_embedding_output(session: &Session, embedding_dim: usize) -> Result<EmbeddingOutput> {
    let rank = |dtype: &ValueType| match dtype {
        ValueType::Tensor { shape, .. } => Some(shape.len()),
        _ => None,
    };
    for candidate in ["token_embeddings", "last_hidden_state"] {
        if let Some(output) = session
            .outputs()
            .iter()
            .find(|output| output.name() == candidate && rank(output.dtype()) == Some(3))
        {
            return Ok(EmbeddingOutput::TokenEmbeddings(output.name().to_string()));
        }
    }
    if let Some(output) = session
        .outputs()
        .iter()
        .find(|output| output.name() == "sentence_embedding" && rank(output.dtype()) == Some(2))
    {
        if let ValueType::Tensor { shape, .. } = output.dtype() {
            let declared_dim = shape.get(1).copied().unwrap_or(-1);
            if declared_dim > 0 && declared_dim as usize != embedding_dim {
                anyhow::bail!(
                    "ONNX sentence_embedding width {declared_dim} does not match metadata-derived width {embedding_dim}"
                );
            }
        }
        return Ok(EmbeddingOutput::SentenceEmbedding(
            output.name().to_string(),
        ));
    }
    let outputs = session
        .outputs()
        .iter()
        .map(|output| format!("{}:{:?}", output.name(), output.dtype()))
        .collect::<Vec<_>>()
        .join(", ");
    anyhow::bail!(
        "ONNX model exposes no supported named sentence or token embedding output ({outputs})"
    )
}

fn embedding_fingerprint(
    revision: &str,
    artifact: &str,
    pooling: &PoolingConfig,
    normalize: bool,
    max_length: usize,
    embedding_dim: usize,
    provider: &str,
) -> String {
    let mut hasher = Sha256::new();
    for value in [
        EMBEDDING_PIPELINE_VERSION.to_string(),
        revision.to_string(),
        artifact.to_string(),
        pooling.word_embedding_dimension.to_string(),
        pooling.pooling_mode_cls_token.to_string(),
        pooling.pooling_mode_max_tokens.to_string(),
        pooling.pooling_mode_mean_tokens.to_string(),
        pooling.pooling_mode_mean_sqrt_len_tokens.to_string(),
        pooling.pooling_mode_weightedmean_tokens.to_string(),
        pooling.pooling_mode_lasttoken.to_string(),
        normalize.to_string(),
        max_length.to_string(),
        embedding_dim.to_string(),
        provider.to_string(),
    ] {
        hasher.update(value.as_bytes());
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

fn resolve_onnx_artifact(repo: &PinnedRepo, repo_id: &str) -> Result<(PathBuf, String)> {
    const CANDIDATES: &[&str] = &[
        "onnx/model.onnx",
        "model.onnx",
        "onnx/model_quantized.onnx",
        "onnx/model_qint8_avx512.onnx",
        "onnx/model_quantized_uint8.onnx",
    ];

    let mut errors = Vec::new();
    for candidate in CANDIDATES {
        match repo.get(candidate) {
            Ok(path) => {
                ensure_external_onnx_data(repo, candidate);
                return Ok((path, (*candidate).to_string()));
            }
            Err(err) => errors.push(format!("{candidate}: {err}")),
        }
    }

    anyhow::bail!(
        "unsupported embedding model {repo_id}: only Hugging Face repositories with ONNX artifacts are supported; tried {} ({})",
        CANDIDATES.join(", "),
        errors.join("; ")
    )
}

fn ensure_external_onnx_data(repo: &PinnedRepo, onnx_file: &str) {
    let companion = format!("{onnx_file}_data");
    let _ = repo.get(&companion);
}

/// Prepare the sessions `plan` asks for. On macOS a validated model gets
/// static Core ML buckets; the smallest bucket is compiled straight away so a
/// Mac that cannot run Core ML falls back to the CPU here, with the reason,
/// rather than in the middle of a run.
fn prepare_backend(
    onnx_path: &Path,
    model_id: &str,
    max_length: usize,
    plan: &SessionPlan,
) -> Result<PreparedBackend> {
    let (cache_root, profile) = match plan {
        SessionPlan::Cpu { profile } => {
            return Ok(PreparedBackend {
                backend: Backend::Dynamic(Mutex::new(build_cpu_session(
                    onnx_path,
                    profile.as_ref().map(|dir| dir.join("cpu")).as_deref(),
                )?)),
                provider_id: CPU_PROVIDER_ID.to_owned(),
                fallback_reason: None,
            })
        }
        SessionPlan::Preferred {
            cache_root,
            profile,
        } => (cache_root, profile),
    };
    #[cfg(target_os = "macos")]
    if static_shape_validated(model_id, onnx_path) {
        let buckets = static_buckets(max_length);
        let sessions =
            StaticSessions::new(onnx_path, cache_root.clone(), profile.clone(), &buckets);
        return match sessions.bucket_session(0) {
            Ok(_) => Ok(PreparedBackend {
                backend: Backend::Static(sessions),
                provider_id: static_provider_id(&buckets),
                fallback_reason: None,
            }),
            Err(reason) => Ok(PreparedBackend {
                backend: Backend::Dynamic(Mutex::new(
                    build_cpu_session(
                        onnx_path,
                        profile.as_ref().map(|dir| dir.join("cpu")).as_deref(),
                    )
                    .with_context(|| {
                        format!("Core ML preparation failed ({reason}); CPU fallback failed")
                    })?,
                )),
                provider_id: CPU_PROVIDER_ID.to_owned(),
                fallback_reason: Some(reason),
            }),
        };
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (model_id, max_length, cache_root);
    Ok(PreparedBackend {
        backend: Backend::Dynamic(Mutex::new(build_cpu_session(
            onnx_path,
            profile.as_ref().map(|dir| dir.join("cpu")).as_deref(),
        )?)),
        provider_id: CPU_PROVIDER_ID.to_owned(),
        fallback_reason: None,
    })
}

/// Session options shared by every session: full graph optimization, no
/// memory-pattern planning, and the thread cap when one is set.
fn base_session_builder() -> Result<SessionBuilder> {
    let mut builder = Session::builder()
        .context("create ONNX Runtime session builder")?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| anyhow::anyhow!("configure ONNX graph optimization: {err:?}"))?
        .with_memory_pattern(false)
        .map_err(|err| anyhow::anyhow!("configure ONNX memory pattern: {err:?}"))?;

    // Without the variable ONNX Runtime uses every physical core. A server can
    // cap each embedding job (for example 8 of 32 cores) by setting it; the
    // number given is used, not always 1.
    if let Some(threads) = embedding_threads() {
        builder = builder
            .with_intra_threads(threads)
            .map_err(|err| anyhow::anyhow!("configure ONNX intra-op threads: {err:?}"))?
            .with_inter_threads(1)
            .map_err(|err| anyhow::anyhow!("configure ONNX inter-op threads: {err:?}"))?;
    }
    Ok(builder)
}

fn profiled(builder: SessionBuilder, profile: Option<&Path>) -> Result<SessionBuilder> {
    match profile {
        Some(path) => builder
            .with_profiling(path)
            .map_err(|err| anyhow::anyhow!("configure ONNX profiling: {err:?}")),
        None => Ok(builder),
    }
}

/// Dynamic-shape session on ONNX Runtime's CPU provider (all platforms).
fn build_cpu_session(onnx_path: &Path, profile: Option<&Path>) -> Result<Session> {
    profiled(base_session_builder()?, profile)?
        .commit_from_file(onnx_path)
        .with_context(|| format!("load ONNX model {}", onnx_path.display()))
}

/// Static-shape Core ML session for `ENCODE_BATCH_ROWS` x `tokens`, compiled
/// as an ML Program for every compute unit (CPU, GPU, Neural Engine) and
/// cached under `cache_root`, one directory per artifact, shape and runtime.
#[cfg(target_os = "macos")]
fn build_static_session(
    onnx_path: &Path,
    tokens: usize,
    cache_root: Option<&Path>,
    profile: Option<&Path>,
) -> Result<Session> {
    use ort::ep;

    let policy = format!("CoreMLExecutionProvider:MLProgram:ALL:{ENCODE_BATCH_ROWS}x{tokens}");
    let mut provider = ep::CoreML::default()
        .with_model_format(ep::coreml::ModelFormat::MLProgram)
        .with_compute_units(ep::coreml::ComputeUnits::All);
    if let Some(root) = cache_root {
        let cache = root.join(coreml_cache_key(onnx_path, &policy, ort::info()));
        std::fs::create_dir_all(&cache).with_context(|| {
            format!("prepare the compiled CoreML cache {}", cache.display())
        })?;
        provider = provider.with_model_cache_dir(cache.to_string_lossy());
    }
    let mut builder = base_session_builder()?
        .with_execution_providers([provider.build().error_on_failure()])
        .map_err(|err| anyhow::anyhow!("configure the Core ML execution provider: {err:?}"))?;
    for (name, value) in [
        ("batch_size", ENCODE_BATCH_ROWS),
        ("sequence_length", tokens),
    ] {
        builder = builder
            .with_dimension_override(name, value as i64)
            .map_err(|err| anyhow::anyhow!("fix the embedding input {name}: {err:?}"))?;
    }
    let session = profiled(builder, profile)?
        .commit_from_file(onnx_path)
        .with_context(|| format!("compile {} for Core ML at {policy}", onnx_path.display()))?;
    let expected = [ENCODE_BATCH_ROWS as i64, tokens as i64];
    anyhow::ensure!(
        session.inputs().iter().all(|input| matches!(
            input.dtype(),
            ValueType::Tensor { shape, .. } if shape.as_ref() == expected
        )),
        "ONNX inputs do not take the fixed shape {ENCODE_BATCH_ROWS}x{tokens}"
    );
    Ok(session)
}

#[cfg(not(target_os = "macos"))]
fn build_static_session(
    _onnx_path: &Path,
    _tokens: usize,
    _cache_root: Option<&Path>,
    _profile: Option<&Path>,
) -> Result<Session> {
    anyhow::bail!("static Core ML sessions exist only on macOS")
}

/// Cache directory name for one compiled model: artifact path (which carries
/// the snapshot revision), execution policy with shape, and ONNX Runtime build.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn coreml_cache_key(path: &Path, policy: &str, runtime: &str) -> String {
    let mut hash = Sha256::new();
    for part in ["coreml-cache-v1", &path.to_string_lossy(), policy, runtime] {
        hash.update(part.as_bytes());
        hash.update([0]);
    }
    format!("{:x}", hash.finalize())
}

/// Pad token and type ids: the tokenizer's own padding settings when it has
/// them, else its `[PAD]`/`<pad>` token, else 0. Padded positions are masked,
/// so the value only matters to models that derive positions from it.
fn pad_ids(tokenizer: &Tokenizer) -> (i64, i64) {
    if let Some(padding) = tokenizer.get_padding() {
        return (i64::from(padding.pad_id), i64::from(padding.pad_type_id));
    }
    let pad_id = ["[PAD]", "<pad>"]
        .into_iter()
        .find_map(|token| tokenizer.token_to_id(token))
        .unwrap_or(0);
    (i64::from(pad_id), 0)
}

fn embedding_threads() -> Option<usize> {
    env::var(EMBEDDING_THREADS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
}

/// Worker threads for native CPU work beside ONNX Runtime (segmentation):
/// every available CPU, capped by `POLARS_TEXT_EMBEDDING_THREADS` when set.
#[cfg_attr(not(feature = "topic-modeling"), allow(dead_code))]
pub(crate) fn computation_threads() -> usize {
    bounded_threads(
        std::thread::available_parallelism().map_or(1, usize::from),
        embedding_threads(),
    )
}

#[cfg_attr(not(feature = "topic-modeling"), allow(dead_code))]
fn bounded_threads(available: usize, cap: Option<usize>) -> usize {
    cap.map_or(available, |cap| cap.min(available)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::{PaddingParams, PaddingStrategy};

    fn mean_pooling() -> PoolingConfig {
        PoolingConfig {
            word_embedding_dimension: 2,
            pooling_mode_cls_token: false,
            pooling_mode_max_tokens: false,
            pooling_mode_mean_tokens: true,
            pooling_mode_mean_sqrt_len_tokens: false,
            pooling_mode_weightedmean_tokens: false,
            pooling_mode_lasttoken: false,
        }
    }

    #[test]
    fn mean_pool_normalize_matches_hand_computation() {
        // batch=1, seq=2, dim=2. Second token masked out, so the pooled vector
        // equals the first token's vector, then L2-normalized.
        let hidden = vec![3.0_f32, 4.0, 100.0, 100.0];
        let mask = vec![1_i64, 0];
        let out = embeddings_from_output(
            &EmbeddingOutput::TokenEmbeddings("last_hidden_state".to_string()),
            &mean_pooling(),
            true,
            OutputBatch {
                dims: &[1, 2, 2],
                data: &hidden,
                batch: 1,
                sequence_length: 2,
                attention_mask: &mask,
            },
        )
        .unwrap();
        let row = &out[0];
        assert!((row[0] - 0.6).abs() < 1e-5, "got {}", row[0]);
        assert!((row[1] - 0.8).abs() < 1e-5, "got {}", row[1]);
    }

    #[test]
    fn embeddings_from_2d_output_normalizes_rows() {
        let out = embeddings_from_output(
            &EmbeddingOutput::SentenceEmbedding("sentence_embedding".to_string()),
            &mean_pooling(),
            true,
            OutputBatch {
                dims: &[1, 2],
                data: &[3.0, 4.0],
                batch: 1,
                sequence_length: 1,
                attention_mask: &[1],
            },
        )
        .unwrap();
        assert!((out[0][0] - 0.6).abs() < 1e-5, "got {}", out[0][0]);
        assert!((out[0][1] - 0.8).abs() < 1e-5, "got {}", out[0][1]);
    }

    #[test]
    fn embeddings_from_2d_output_rejects_metadata_width_mismatch() {
        let error = embeddings_from_output(
            &EmbeddingOutput::SentenceEmbedding("sentence_embedding".to_string()),
            &mean_pooling(),
            false,
            OutputBatch {
                dims: &[1, 3],
                data: &[1.0, 2.0, 3.0],
                batch: 1,
                sequence_length: 1,
                attention_mask: &[1],
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("metadata-derived width"));
    }

    #[test]
    fn pooling_concatenates_standard_modes_in_sentence_transformers_order() {
        let pooling = PoolingConfig {
            word_embedding_dimension: 2,
            pooling_mode_cls_token: true,
            pooling_mode_max_tokens: true,
            pooling_mode_mean_tokens: true,
            pooling_mode_mean_sqrt_len_tokens: true,
            pooling_mode_weightedmean_tokens: true,
            pooling_mode_lasttoken: true,
        };
        let row = pool_token_embeddings(&[1.0, 2.0, 3.0, 4.0], 1, 2, 2, &[1, 1], &pooling)
            .unwrap()
            .remove(0);

        let expected = [
            1.0,
            2.0,
            3.0,
            4.0,
            2.0,
            3.0,
            4.0 / 2.0_f32.sqrt(),
            6.0 / 2.0_f32.sqrt(),
            7.0 / 3.0,
            10.0 / 3.0,
            3.0,
            4.0,
        ];
        assert!(
            row.iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1e-5),
            "got {row:?}"
        );
    }

    #[test]
    fn sizing_tokenizer_clears_padding() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        let model = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            ..Default::default()
        }));

        assert_eq!(tokenizer.encode("hello", false).unwrap().get_ids().len(), 8);
        let sizing = tokenizer_for_sizing(&tokenizer);
        assert_eq!(sizing.encode("hello", false).unwrap().get_ids().len(), 1);
    }

    #[test]
    fn sentence_transformer_max_length_has_precedence() {
        let sentence = serde_json::json!({"max_seq_length": 256});
        let tokenizer = serde_json::json!({"model_max_length": 384});
        let model = serde_json::json!({"max_position_embeddings": 512});
        assert_eq!(
            canonical_max_length(Some(&sentence), Some(&tokenizer), &model).unwrap(),
            256
        );
        assert_eq!(
            canonical_max_length(None, Some(&tokenizer), &model).unwrap(),
            384
        );
        assert!(canonical_max_length(None, None, &serde_json::json!({})).is_err());
    }

    #[test]
    fn module_graph_rejects_dense_and_non_contiguous_modules() {
        let transformer = SentenceTransformerModule {
            idx: 0,
            path: String::new(),
            module_type: "sentence_transformers.models.Transformer".to_string(),
        };
        let pooling = SentenceTransformerModule {
            idx: 1,
            path: "1_Pooling".to_string(),
            module_type: "sentence_transformers.models.Pooling".to_string(),
        };
        assert!(validate_module_graph(&[transformer.clone(), pooling.clone()], "test").is_ok());

        let dense = SentenceTransformerModule {
            idx: 2,
            path: "2_Dense".to_string(),
            module_type: "sentence_transformers.models.Dense".to_string(),
        };
        assert!(
            validate_module_graph(&[transformer.clone(), pooling.clone(), dense], "test")
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );

        let mut non_contiguous = pooling;
        non_contiguous.idx = 3;
        assert!(validate_module_graph(&[transformer, non_contiguous], "test").is_err());
    }

    #[test]
    fn embedding_fingerprint_changes_with_revision_and_provider() {
        let pooling = mean_pooling();
        let first =
            embedding_fingerprint("sha-a", "onnx/model.onnx", &pooling, true, 256, 2, "cpu");
        let revision =
            embedding_fingerprint("sha-b", "onnx/model.onnx", &pooling, true, 256, 2, "cpu");
        let provider =
            embedding_fingerprint("sha-a", "onnx/model.onnx", &pooling, true, 256, 2, "coreml");
        assert_ne!(first, revision);
        assert_ne!(first, provider);
    }

    #[test]
    fn worker_threads_obey_available_processors_and_explicit_caps() {
        assert_eq!(bounded_threads(8, None), 8);
        assert_eq!(bounded_threads(8, Some(2)), 2);
        assert_eq!(bounded_threads(2, Some(32)), 2);
        assert_eq!(bounded_threads(0, None), 1);
    }

    #[test]
    fn batches_sort_by_token_count_and_keep_input_order_for_ties() {
        assert_eq!(
            length_sorted_batches(&[5, 1, 3, 1, 9], 2),
            vec![vec![1, 3], vec![2, 0], vec![4]]
        );
        assert!(length_sorted_batches(&[], 32).is_empty());
        let lengths = (0..70).map(|index| 70 - index).collect::<Vec<_>>();
        let batches = length_sorted_batches(&lengths, ENCODE_BATCH_ROWS);
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![32, 32, 6]
        );
        let mut every = batches.concat();
        every.sort_unstable();
        assert_eq!(every, (0..70).collect::<Vec<_>>());
    }

    #[test]
    fn a_batch_goes_to_the_smallest_bucket_that_holds_it() {
        let buckets = static_buckets(256);
        assert_eq!(buckets, vec![32, 64, 128, 256]);
        assert_eq!(bucket_index(&buckets, 1), Some(0));
        assert_eq!(bucket_index(&buckets, 32), Some(0));
        assert_eq!(bucket_index(&buckets, 33), Some(1));
        assert_eq!(bucket_index(&buckets, 129), Some(3));
        assert_eq!(bucket_index(&buckets, 256), Some(3));
        assert_eq!(bucket_index(&buckets, 257), None);
    }

    #[test]
    fn buckets_stop_at_the_model_limit() {
        assert_eq!(static_buckets(128), vec![32, 64, 128]);
        assert_eq!(static_buckets(100), vec![32, 64, 100]);
        assert_eq!(static_buckets(512), vec![32, 64, 128, 256, 512]);
        assert_eq!(static_buckets(16), vec![16]);
    }

    #[test]
    fn static_core_ml_is_limited_to_the_curated_fp32_models() {
        let artifact = Path::new("/models/onnx/model.onnx");
        assert!(static_shape_validated(DEFAULT_EMBEDDER_REPO_ID, artifact));
        assert!(static_shape_validated(
            MULTILINGUAL_EMBEDDER_REPO_ID,
            Path::new("/models/model.onnx")
        ));
        assert!(!static_shape_validated("unvalidated/model", artifact));
        assert!(!static_shape_validated(
            DEFAULT_EMBEDDER_REPO_ID,
            Path::new("/models/onnx/model_quantized.onnx")
        ));
        assert_eq!(
            static_provider_id(&[32, 64, 128]),
            "CoreMLExecutionProvider:MLProgram:ALL:32x32/64/128+CPUExecutionProvider"
        );
    }

    #[test]
    fn compiled_model_cache_separates_artifacts_policies_and_runtime_builds() {
        let key = coreml_cache_key(
            Path::new("/models/snapshots/sha-a/model.onnx"),
            "all:32x256",
            "ort-a",
        );
        assert_eq!(
            key,
            coreml_cache_key(
                Path::new("/models/snapshots/sha-a/model.onnx"),
                "all:32x256",
                "ort-a"
            )
        );
        for (path, policy, runtime) in [
            ("/models/snapshots/sha-b/model.onnx", "all:32x256", "ort-a"),
            ("/models/snapshots/sha-a/model.onnx", "gpu:32x256", "ort-a"),
            ("/models/snapshots/sha-a/model.onnx", "all:32x64", "ort-a"),
            ("/models/snapshots/sha-a/model.onnx", "all:32x256", "ort-b"),
        ] {
            assert_ne!(key, coreml_cache_key(Path::new(path), policy, runtime));
        }
    }

    #[test]
    fn pad_ids_prefer_the_tokenizers_own_padding() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("[PAD]".to_string(), 3);
        let model = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        assert_eq!(pad_ids(&tokenizer), (3, 0));
        tokenizer.with_padding(Some(PaddingParams {
            pad_id: 7,
            pad_type_id: 1,
            ..Default::default()
        }));
        assert_eq!(pad_ids(&tokenizer), (7, 1));
    }

    impl Embedder {
        /// End profiling on every prepared session and return the execution
        /// providers its nodes ran on.
        fn executed_providers(&self) -> Result<std::collections::BTreeSet<String>> {
            let sessions: Vec<&Mutex<Session>> = match &self.backend {
                Backend::Dynamic(session) => vec![session],
                Backend::Static(sessions) => sessions
                    .buckets
                    .iter()
                    .filter_map(|bucket| bucket.session.get())
                    .chain(sessions.cpu.get())
                    .filter_map(|session| session.as_ref().ok())
                    .collect(),
            };
            let mut used = std::collections::BTreeSet::new();
            for session in sessions {
                let profile = session
                    .lock()
                    .map_err(|_| anyhow::anyhow!("session lock poisoned"))?
                    .end_profiling()?;
                let events: Vec<serde_json::Value> =
                    serde_json::from_slice(&std::fs::read(profile)?)?;
                used.extend(
                    events
                        .iter()
                        .filter_map(|event| event["args"]["provider"].as_str().map(str::to_owned)),
                );
            }
            Ok(used)
        }

        /// Token lengths of the static buckets compiled so far.
        fn prepared_buckets(&self) -> Vec<usize> {
            match &self.backend {
                Backend::Dynamic(_) => Vec::new(),
                Backend::Static(sessions) => sessions
                    .buckets
                    .iter()
                    .filter(|bucket| bucket.session.get().is_some_and(|session| session.is_ok()))
                    .map(|bucket| bucket.tokens)
                    .collect(),
            }
        }
    }

    fn assert_vectors_agree(expected: &[Vec<f32>], actual: &[Vec<f32>]) {
        assert_eq!(expected.len(), actual.len());
        for (left, right) in expected.iter().zip(actual) {
            assert_eq!(left.len(), right.len());
            assert!(right.iter().all(|value| value.is_finite()));
            let dot = left
                .iter()
                .zip(right)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>();
            let norm = |values: &[f32]| {
                values
                    .iter()
                    .map(|x| f64::from(*x).powi(2))
                    .sum::<f64>()
                    .sqrt()
            };
            let cosine = dot / (norm(left) * norm(right));
            assert!(cosine > 0.9999, "cosine {cosine}");
            let largest = left
                .iter()
                .zip(right)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(largest < 1e-3, "largest difference {largest}");
        }
    }

    /// Acceptance for Wordflow issue 412 on a real Mac: static Core ML runs
    /// the graph on Core ML, agrees with the CPU on blank, Unicode, long,
    /// duplicate and partial-batch input in any order, and falls back to the
    /// CPU when the compiled-model cache is unusable. Elsewhere it checks that
    /// the preferred plan is the CPU. Downloads both curated models.
    #[test]
    #[ignore = "downloads both curated models; run on an Apple Silicon Mac to accept Core ML"]
    fn preferred_sessions_agree_with_the_cpu_and_fall_back() -> Result<()> {
        let scratch = tempfile::tempdir()?;
        let texts = (0..73)
            .map(|index| match index % 6 {
                0 => String::new(),
                1 | 4 => "Affordable homes and lower rent.".to_string(),
                2 => "医生讨论健康。 Café culture and Unicode text.".to_string(),
                3 => format!(
                    "Record {index}. {}",
                    "Doctors treat patients in hospitals. ".repeat(50)
                ),
                _ => format!(
                    "Record {index}. {}",
                    "Reliable public transport runs on time. ".repeat(index % 9 + 1)
                ),
            })
            .collect::<Vec<_>>();
        let reversed = texts.iter().rev().cloned().collect::<Vec<_>>();
        for (index, model) in [DEFAULT_EMBEDDER_REPO_ID, MULTILINGUAL_EMBEDDER_REPO_ID]
            .into_iter()
            .enumerate()
        {
            let profiles = scratch.path().join(format!("profiles-{index}"));
            for plan in ["plain", "preferred"] {
                std::fs::create_dir_all(profiles.join(plan))?;
            }
            let cpu = load_embedder_with(
                model,
                &SessionPlan::Cpu {
                    profile: Some(profiles.join("plain")),
                },
            )?;
            let expected = cpu.encode(&texts)?;
            assert_eq!(cpu.provider_id(), CPU_PROVIDER_ID);

            let preferred = load_embedder_with(
                model,
                &SessionPlan::Preferred {
                    cache_root: Some(scratch.path().join("coreml")),
                    profile: Some(profiles.join("preferred")),
                },
            )?;
            for count in [0, 1, 31, 32, 33, 65, 73] {
                assert_vectors_agree(&expected[..count], &preferred.encode(&texts[..count])?);
            }
            let mut expected_reversed = expected.clone();
            expected_reversed.reverse();
            assert_vectors_agree(&expected_reversed, &preferred.encode(&reversed)?);
            // One small batch sized for each bucket, so every bucket compiles.
            for tokens in static_buckets(cpu.max_length()) {
                let sized = vec!["house ".repeat(tokens.saturating_sub(4).max(1)); 3];
                assert_vectors_agree(&cpu.encode(&sized)?, &preferred.encode(&sized)?);
            }
            assert_eq!(
                cpu.executed_providers()?,
                [CPU_PROVIDER_ID.to_string()].into()
            );
            let used = preferred.executed_providers()?;
            eprintln!(
                "EMBEDDING {model}: policy={} buckets={:?} providers={used:?}",
                preferred.provider_id(),
                preferred.prepared_buckets()
            );

            #[cfg(target_os = "macos")]
            {
                assert!(preferred.provider_id().starts_with("CoreMLExecutionProvider"));
                assert_eq!(preferred.provider_fallback_reason(), None);
                assert!(used.contains("CoreMLExecutionProvider"), "{used:?}");
                assert_eq!(preferred.prepared_buckets().len(), static_buckets(preferred.max_length()).len());
                assert_ne!(preferred.cache_fingerprint(), cpu.cache_fingerprint());

                let blocked = scratch.path().join(format!("blocked-{index}"));
                std::fs::write(&blocked, "not a directory")?;
                let fallback = load_embedder_with(
                    model,
                    &SessionPlan::Preferred {
                        cache_root: Some(blocked),
                        profile: None,
                    },
                )?;
                assert_eq!(fallback.provider_id(), CPU_PROVIDER_ID);
                assert!(fallback
                    .provider_fallback_reason()
                    .is_some_and(|reason| reason.contains("compiled CoreML cache")));
                assert_eq!(fallback.cache_fingerprint(), cpu.cache_fingerprint());
                assert_vectors_agree(&expected, &fallback.encode(&texts)?);
            }
            #[cfg(not(target_os = "macos"))]
            {
                assert_eq!(preferred.provider_id(), CPU_PROVIDER_ID);
                assert_eq!(used, [CPU_PROVIDER_ID.to_string()].into());
            }
        }
        Ok(())
    }
}
