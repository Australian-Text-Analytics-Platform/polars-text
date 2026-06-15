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
//! Called by: `topic_modeling::run` for chunk embeddings and, through the
//! expression plugin, `polars_text.functions.embedding` / `.text.embedding`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use once_cell::sync::OnceCell;
use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Default ONNX embedder. This model is popular, has published ONNX weights,
/// and its model card documents mean pooling + L2 normalization for sentence
/// embeddings. It is English-focused; a multilingual ONNX default can replace
/// it after benchmarking.
pub const DEFAULT_EMBEDDER_REPO_ID: &str = "onnx-community/all-MiniLM-L6-v2-ONNX";

/// Hard truncation cap for the embedding tokenizer. Chunks are already sized to
/// the chunk budget, but capping here protects against a pathological single
/// token blob exceeding the model's positional range.
const EMBED_TRUNCATION_MAX: usize = 512;

const EMBEDDING_THREADS_ENV: &str = "POLARS_TEXT_EMBEDDING_THREADS";

/// A loaded ONNX sentence encoder. `Session::run` requires `&mut self`, so the
/// session sits behind a mutex while cheap tokenizer work remains lock-free.
pub struct Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    input_names: Vec<String>,
    output_name: String,
    hidden_size: usize,
    model_id: String,
    provider_id: String,
    model_revision: String,
}

impl Embedder {
    /// Output embedding dimensionality when it can be read from `config.json`.
    /// Some ONNX repos omit this field; those still encode correctly and infer
    /// the real dimensionality from ORT output at runtime.
    pub fn dim(&self) -> usize {
        self.hidden_size
    }

    /// Hugging Face repo id used for model download and embedding cache keys.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Execution-provider label used for cache keys and diagnostics.
    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    /// Model revision label used for cache keys. Currently `main` because the
    /// public API only accepts a model id and relies on hf-hub defaults.
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    /// A tokenizer clone with truncation/padding disabled for chunk-size measurement.
    pub fn sizing_tokenizer(&self) -> Tokenizer {
        tokenizer_for_sizing(&self.tokenizer)
    }

    /// Encode `texts` into mean-pooled, L2-normalized row vectors.
    pub fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut tokenizer = self.tokenizer.clone();
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let encodings = tokenizer
            .encode_batch(refs, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode_batch failed: {e}"))?;

        let batch = encodings.len();
        let seq = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        if seq == 0 {
            return Ok(vec![vec![0.0; self.hidden_size]; batch]);
        }

        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        let mut token_types = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().map(|id| i64::from(*id)));
            mask.extend(enc.get_attention_mask().iter().map(|id| i64::from(*id)));
            token_types.extend(enc.get_type_ids().iter().map(|id| i64::from(*id)));
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

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("embedder session lock poisoned"))?;
        let outputs = session
            .run(inputs)
            .context("ONNX embedding inference failed")?;
        let output = if outputs.contains_key(self.output_name.as_str()) {
            &outputs[self.output_name.as_str()]
        } else {
            &outputs[0]
        };
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .context("extract ONNX embedding tensor")?;
        let dims: Vec<usize> = shape.iter().map(|dim| *dim as usize).collect();
        embeddings_from_output(&dims, data, batch, seq, &mask)
    }
}

fn tokenizer_for_sizing(tokenizer: &Tokenizer) -> Tokenizer {
    let mut tok = tokenizer.clone();
    let _ = tok.with_truncation(None);
    tok.with_padding(None);
    tok
}

fn embeddings_from_output(
    dims: &[usize],
    data: &[f32],
    batch: usize,
    seq: usize,
    mask: &[i64],
) -> Result<Vec<Vec<f32>>> {
    match dims {
        [out_batch, dim] if *out_batch == batch => Ok(normalize_rows(data, batch, *dim)),
        [out_batch, out_seq, dim] if *out_batch == batch && *out_seq == seq => {
            mean_pool_normalize(data, batch, seq, *dim, mask)
        }
        other => anyhow::bail!(
            "unsupported ONNX embedding output shape {:?}; expected [batch, dim] or [batch, seq, dim]",
            other
        ),
    }
}

/// Mean-pool `(batch, seq, dim)` hidden states over a `(batch, seq)` mask, then
/// L2-normalize each row. Kept pure so tests do not need ONNX Runtime.
fn mean_pool_normalize(
    hidden: &[f32],
    batch: usize,
    seq: usize,
    dim: usize,
    mask: &[i64],
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

    let mut rows = vec![vec![0.0_f32; dim]; batch];
    for row in 0..batch {
        let mut count = 0.0_f32;
        for token in 0..seq {
            let mask_value = mask[row * seq + token] as f32;
            if mask_value == 0.0 {
                continue;
            }
            count += mask_value;
            let offset = (row * seq + token) * dim;
            for col in 0..dim {
                rows[row][col] += hidden[offset + col] * mask_value;
            }
        }
        let denom = count.max(1e-9);
        for value in &mut rows[row] {
            *value /= denom;
        }
    }
    Ok(normalize_nested_rows(rows))
}

fn normalize_rows(data: &[f32], batch: usize, dim: usize) -> Vec<Vec<f32>> {
    let rows = data
        .chunks(dim)
        .take(batch)
        .map(|row| row.to_vec())
        .collect();
    normalize_nested_rows(rows)
}

fn normalize_nested_rows(mut rows: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    for row in &mut rows {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in row {
                *value /= norm;
            }
        }
    }
    rows
}

static REGISTRY: OnceCell<RwLock<HashMap<String, Arc<Embedder>>>> = OnceCell::new();

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

fn load_embedder(repo_id: &str) -> Result<Embedder> {
    let api = ApiBuilder::from_env()
        .build()
        .context("failed to init hf-hub api for embedder")?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json").context("fetch config.json")?;
    let tokenizer_path = repo.get("tokenizer.json").context("fetch tokenizer.json")?;
    let onnx_path = resolve_onnx_artifact(&repo, repo_id)?;

    let hidden_size = read_hidden_size(&config_path).unwrap_or(0);
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer for {repo_id}: {e}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: EMBED_TRUNCATION_MAX,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("configure truncation: {e}"))?;

    let (session, provider_id) = build_session(&onnx_path)?;
    let input_names = session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect::<Vec<_>>();
    let output_name = session
        .outputs()
        .first()
        .map(|output| output.name().to_string())
        .ok_or_else(|| anyhow::anyhow!("ONNX model has no outputs"))?;

    Ok(Embedder {
        session: Mutex::new(session),
        tokenizer,
        input_names,
        output_name,
        hidden_size,
        model_id: repo_id.to_string(),
        provider_id,
        model_revision: "main".to_string(),
    })
}

fn resolve_onnx_artifact(repo: &ApiRepo, repo_id: &str) -> Result<PathBuf> {
    const CANDIDATES: &[&str] = &[
        "model.onnx",
        "onnx/model.onnx",
        "onnx/model_quantized.onnx",
        "onnx/model_qint8_avx512.onnx",
        "onnx/model_quantized_uint8.onnx",
    ];

    let mut errors = Vec::new();
    for candidate in CANDIDATES {
        match repo.get(candidate) {
            Ok(path) => {
                ensure_external_onnx_data(repo, candidate);
                return Ok(path);
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

fn ensure_external_onnx_data(repo: &ApiRepo, onnx_file: &str) {
    let companion = format!("{onnx_file}_data");
    let _ = repo.get(&companion);
}

fn read_hidden_size(config_path: &Path) -> Option<usize> {
    let config = std::fs::read_to_string(config_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&config).ok()?;
    value
        .get("hidden_size")?
        .as_u64()
        .map(|value| value as usize)
}

fn build_session(onnx_path: &Path) -> Result<(Session, String)> {
    let provider_id = planned_provider_id();
    let mut builder = Session::builder()
        .context("create ONNX Runtime session builder")?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| anyhow::anyhow!("configure ONNX graph optimization: {err:?}"))?
        .with_memory_pattern(false)
        .map_err(|err| anyhow::anyhow!("configure ONNX memory pattern: {err:?}"))?;

    if embedding_threads().is_some() {
        builder = builder
            .with_intra_threads(1)
            .map_err(|err| anyhow::anyhow!("configure ONNX intra-op threads: {err:?}"))?
            .with_inter_threads(1)
            .map_err(|err| anyhow::anyhow!("configure ONNX inter-op threads: {err:?}"))?;
    }

    let providers = execution_providers();
    if !providers.is_empty() {
        builder = builder
            .with_execution_providers(providers)
            .map_err(|err| anyhow::anyhow!("configure ONNX execution providers: {err:?}"))?;
    }

    let session = builder
        .commit_from_file(onnx_path)
        .with_context(|| format!("load ONNX model {}", onnx_path.display()))?;
    Ok((session, provider_id))
}

fn execution_providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
    let mut providers = Vec::new();
    #[cfg(target_os = "windows")]
    {
        providers.push(ep::DirectML::default().build());
    }
    providers.push(xnnpack_provider());
    providers
}

fn xnnpack_provider() -> ort::ep::ExecutionProviderDispatch {
    let provider = if let Some(threads) = embedding_threads().and_then(NonZeroUsize::new) {
        ep::XNNPACK::default().with_intra_op_num_threads(threads)
    } else {
        ep::XNNPACK::default()
    };
    provider.build()
}

fn planned_provider_id() -> String {
    let mut providers = Vec::new();
    #[cfg(target_os = "windows")]
    providers.push("DmlExecutionProvider");
    providers.push("XnnpackExecutionProvider");
    providers.push("CPUExecutionProvider");
    providers.join("+")
}

fn embedding_threads() -> Option<usize> {
    env::var(EMBEDDING_THREADS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
}

/// Sorted list of loaded embedder repo ids. Mirrors `tokenizer::loaded_model_ids`
/// for the prefetch/introspection PyO3 surface.
pub fn loaded_embedder_ids() -> Vec<String> {
    let Ok(map) = registry().read() else {
        return Vec::new();
    };
    let mut ids: Vec<String> = map.keys().cloned().collect();
    ids.sort();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use tokenizers::models::wordlevel::WordLevel;

    #[test]
    fn mean_pool_normalize_matches_hand_computation() {
        // batch=1, seq=2, dim=2. Second token masked out, so the pooled vector
        // equals the first token's vector, then L2-normalized.
        let hidden = vec![3.0_f32, 4.0, 100.0, 100.0];
        let mask = vec![1_i64, 0];
        let out = mean_pool_normalize(&hidden, 1, 2, 2, &mask).unwrap();
        let row = &out[0];
        assert!((row[0] - 0.6).abs() < 1e-5, "got {}", row[0]);
        assert!((row[1] - 0.8).abs() < 1e-5, "got {}", row[1]);
    }

    #[test]
    fn embeddings_from_2d_output_normalizes_rows() {
        let out = embeddings_from_output(&[1, 2], &[3.0, 4.0], 1, 1, &[1]).unwrap();
        assert!((out[0][0] - 0.6).abs() < 1e-5, "got {}", out[0][0]);
        assert!((out[0][1] - 0.8).abs() < 1e-5, "got {}", out[0][1]);
    }

    #[test]
    fn loaded_embedder_ids_is_sorted() {
        let ids = loaded_embedder_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
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
}
