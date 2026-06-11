//! Candle-based multilingual sentence embeddings for topic modeling.
//!
//! Why this exists: the pipeline must embed text chunks **offline and
//! in-process** with no inference server or ONNX runtime. Phase 0 verified that
//! candle + `paraphrase-multilingual-MiniLM-L12-v2` reproduces Python
//! sentence-transformers to 4 decimals while keeping the binary small, so this
//! module wraps that approach behind a small registry + batch encoder.
//!
//! What it does: loads a BERT-family sentence-transformer (config + tokenizer +
//! safetensors via `hf-hub`, so weights come from the local HF cache after the
//! first download), then encodes batches of strings into mean-pooled,
//! L2-normalized vectors. Normalization means downstream Euclidean distance in
//! `cluster`/`reduce` is monotonic with cosine similarity — the property the
//! clustering relies on.
//!
//! Registry: loaded models are cached per process keyed by repo id, mirroring
//! `tokenizer::REGISTRY`, so repeated topic-modeling runs in one worker reuse
//! the weights instead of reloading hundreds of MB each call.
//!
//! Called by: `topic_modeling::run` (orchestrator) to embed chunks, and the
//! chunker borrows a truncation-disabled clone of the same tokenizer for sizing.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use once_cell::sync::OnceCell;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Default offline multilingual embedder. BERT architecture, 384-dim output,
/// strong cross-lingual behavior (Phase 0: EN<->ZH cosine 0.99). Chosen over
/// English MiniLM-L6 so CJK and other languages embed in the same space.
pub const DEFAULT_EMBEDDER_REPO_ID: &str =
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";

/// Hard truncation cap for the embedding tokenizer. Chunks are already sized to
/// the chunk budget, but capping here protects against a pathological single
/// "token" blob exceeding the model's positional range.
const EMBED_TRUNCATION_MAX: usize = 512;

/// A loaded encoder: the candle model plus the tokenizer used to feed it and
/// the device it runs on. Held behind an `Arc` in the registry; `forward` only
/// needs `&self`, so concurrent encodes share one copy of the weights.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    hidden_size: usize,
}

impl Embedder {
    /// Output embedding dimensionality (e.g. 384). Used by callers that need to
    /// pre-size matrices and by tests asserting the model loaded as expected.
    pub fn dim(&self) -> usize {
        self.hidden_size
    }

    /// A tokenizer clone with truncation **disabled**, for the chunker's size
    /// measurement. The chunker must see true token counts so it can pack to the
    /// chunk budget; truncation would make every long string look budget-sized.
    pub fn sizing_tokenizer(&self) -> Tokenizer {
        let mut tok = self.tokenizer.clone();
        let _ = tok.with_truncation(None);
        tok
    }

    /// Encode `texts` into mean-pooled, L2-normalized row vectors.
    ///
    /// Flow: tokenize the whole batch with batch-longest padding, build the
    /// `(batch, seq)` id/mask tensors, run the BERT forward pass, then mean-pool
    /// token states over the attention mask and L2-normalize each row. Empty
    /// input returns an empty vector. Padding is applied per call so a short
    /// batch is not padded to some global maximum.
    pub fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut tokenizer = self.tokenizer.clone();
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let encodings = tokenizer
            .encode_batch(refs, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode_batch failed: {e}"))?;

        let batch = encodings.len();
        let seq = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        if seq == 0 {
            return Ok(vec![vec![0.0; self.hidden_size]; batch]);
        }
        let mut ids = Vec::with_capacity(batch * seq);
        let mut mask = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            ids.extend_from_slice(enc.get_ids());
            mask.extend_from_slice(enc.get_attention_mask());
        }
        let token_ids = Tensor::from_vec(ids, (batch, seq), &self.device)?;
        let attn = Tensor::from_vec(mask, (batch, seq), &self.device)?;
        let token_type_ids = token_ids.zeros_like()?;

        let hidden = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attn))
            .context("BERT forward pass failed")?;
        let pooled = mean_pool_normalize(&hidden, &attn)?;
        Ok(pooled.to_vec2()?)
    }
}

/// Mean-pool `(batch, seq, dim)` hidden states over a `(batch, seq)` 0/1 mask,
/// then L2-normalize each row. Pulled out of `encode` so it can be unit-tested
/// against a hand-computed expectation without loading a model.
fn mean_pool_normalize(hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, _seq, dim) = hidden.dims3()?;
    let mask_f = mask
        .to_dtype(DTYPE)?
        .unsqueeze(2)?
        .broadcast_as(hidden.shape())?;
    let summed = (hidden * &mask_f)?.sum(1)?;
    let counts = mask_f.sum(1)?.clamp(1e-9, f32::INFINITY as f64)?;
    let mean = (summed / counts)?;
    let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?.broadcast_as((batch, dim))?;
    Ok((mean / norm)?)
}

static REGISTRY: OnceCell<RwLock<HashMap<String, Arc<Embedder>>>> = OnceCell::new();

fn registry() -> &'static RwLock<HashMap<String, Arc<Embedder>>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Get-or-load the embedder for `repo_id` (default model if `None`).
///
/// Double-checked against the registry so a second caller for the same model
/// waits on the write lock once, then reuses the cached `Arc`. The first load
/// downloads/reads config + tokenizer + safetensors from the HF cache.
pub fn ensure_embedder(repo_id: Option<&str>) -> Result<Arc<Embedder>> {
    let key = repo_id
        .filter(|v| !v.trim().is_empty())
        .unwrap_or(DEFAULT_EMBEDDER_REPO_ID)
        .to_string();

    {
        let map = registry()
            .read()
            .map_err(|_| anyhow::anyhow!("embedder registry lock poisoned"))?;
        if let Some(e) = map.get(&key) {
            return Ok(Arc::clone(e));
        }
    }
    let mut map = registry()
        .write()
        .map_err(|_| anyhow::anyhow!("embedder registry lock poisoned"))?;
    if let Some(e) = map.get(&key) {
        return Ok(Arc::clone(e));
    }
    let embedder = Arc::new(load_embedder(&key)?);
    map.insert(key, Arc::clone(&embedder));
    Ok(embedder)
}

/// Load config/tokenizer/weights for `repo_id` and build a candle `BertModel`.
/// The tokenizer is configured with a truncation cap so the embedder never
/// feeds an over-long sequence into the model's fixed positional embeddings.
fn load_embedder(repo_id: &str) -> Result<Embedder> {
    let api = ApiBuilder::from_env()
        .build()
        .context("failed to init hf-hub api for embedder")?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    let config_path = repo.get("config.json").context("fetch config.json")?;
    let tokenizer_path = repo.get("tokenizer.json").context("fetch tokenizer.json")?;
    let weights_path = repo
        .get("model.safetensors")
        .context("fetch model.safetensors")?;

    let config: Config = serde_json::from_str(
        &std::fs::read_to_string(&config_path).context("read config.json")?,
    )
    .context("parse BERT config.json")?;
    let hidden_size = config.hidden_size;

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer for {repo_id}: {e}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: EMBED_TRUNCATION_MAX,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("configure truncation: {e}"))?;

    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
            .context("mmap safetensors")?
    };
    let model = BertModel::load(vb, &config).context("build BertModel")?;

    Ok(Embedder {
        model,
        tokenizer,
        device,
        hidden_size,
    })
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

    #[test]
    fn mean_pool_normalize_matches_hand_computation() {
        // batch=1, seq=2, dim=2. Second token masked out, so the pooled vector
        // equals the first token's vector, then L2-normalized.
        let device = Device::Cpu;
        let hidden =
            Tensor::from_vec(vec![3.0f32, 4.0, 100.0, 100.0], (1, 2, 2), &device).unwrap();
        let mask = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device).unwrap();
        let out = mean_pool_normalize(&hidden, &mask).unwrap();
        let row = &out.to_vec2::<f32>().unwrap()[0];
        // (3,4) normalized -> (0.6, 0.8).
        assert!((row[0] - 0.6).abs() < 1e-5, "got {}", row[0]);
        assert!((row[1] - 0.8).abs() < 1e-5, "got {}", row[1]);
    }

    #[test]
    fn loaded_embedder_ids_is_sorted() {
        let ids = loaded_embedder_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }
}
