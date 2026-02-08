use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result};
use candle_core::{Device, Module, Tensor};
use candle_nn::{linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use once_cell::sync::OnceCell;
use serde::Deserialize;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

const POS_MODEL_ID: &str = "vblagoje/bert-english-uncased-finetuned-pos";
const POS_MODEL_REVISION: &str = "main";
const TOKENIZER_MODEL_ID: &str = "bert-base-uncased";

#[derive(Clone, Debug)]
pub struct PosToken {
    pub tag: String,
    pub start: i64,
    pub end: i64,
}

struct PosBundle {
    model: BertModel,
    classifier: candle_nn::Linear,
    tokenizer: Tokenizer,
    device: Device,
    id2label: HashMap<i64, String>,
}

static POS_BUNDLE: OnceCell<Mutex<PosBundle>> = OnceCell::new();

#[derive(Deserialize)]
struct PosConfig {
    id2label: HashMap<String, String>,
    num_labels: usize,
    hidden_size: usize,
}

fn load_tokenizer() -> Result<Tokenizer> {
    let api = ApiBuilder::from_env().build().context("Failed to init hf-hub")?;
    let repo = Repo::with_revision(
        TOKENIZER_MODEL_ID.to_string(),
        RepoType::Model,
        "main".to_string(),
    );
    let api = api.repo(repo);
    let tokenizer_path = api
        .get("tokenizer.json")
        .context("Failed to fetch tokenizer.json")?;
    Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))
}

fn build_pos_bundle() -> Result<PosBundle> {
    let device = Device::Cpu;
    let api = ApiBuilder::from_env().build().context("Failed to init hf-hub")?;
    let repo = Repo::with_revision(
        POS_MODEL_ID.to_string(),
        RepoType::Model,
        POS_MODEL_REVISION.to_string(),
    );
    let api = api.repo(repo);

    let config_path = api.get("config.json").context("Failed to fetch config.json")?;
    let weights_path = api
        .get("model.safetensors")
        .context("Failed to fetch model.safetensors")?;

    let config_contents = std::fs::read_to_string(config_path)
        .context("Failed to read config.json")?;
    let config: Config = serde_json::from_str(&config_contents)
        .context("Failed to parse BERT config.json")?;

    let pos_config: PosConfig = serde_json::from_str(&config_contents)
        .context("Failed to parse POS config.json")?;

    let tokenizer = load_tokenizer()?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
            .context("Failed to map model weights")?
    };
    let model = BertModel::load(vb.pp("bert"), &config)
        .context("Failed to load BERT model")?;
    let classifier = linear(pos_config.hidden_size, pos_config.num_labels, vb.pp("classifier"))
        .context("Failed to load classifier")?;

    let mut id2label = HashMap::new();
    for (key, value) in pos_config.id2label {
        if let Ok(idx) = key.parse::<i64>() {
            id2label.insert(idx, value);
        }
    }

    Ok(PosBundle {
        model,
        classifier,
        tokenizer,
        device,
        id2label,
    })
}

fn with_pos_bundle<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&mut PosBundle) -> Result<R>,
{
    let mutex = POS_BUNDLE
        .get_or_try_init(|| build_pos_bundle().map(Mutex::new))?;
    let mut guard = mutex
        .lock()
        .map_err(|_| anyhow::anyhow!("POS bundle mutex poisoned"))?;
    f(&mut *guard)
}

fn decode_tags(logits: Tensor) -> Result<Vec<i64>> {
    let logits = logits.to_vec2::<f32>()?;
    let mut tags = Vec::with_capacity(logits.len());
    for token_logits in logits {
        let mut best = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, val) in token_logits.iter().enumerate() {
            if *val > best_val {
                best = idx as i64;
                best_val = *val;
            }
        }
        tags.push(best);
    }
    Ok(tags)
}

pub fn pos_tags_for_text(text: &str) -> Result<Vec<PosToken>> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    with_pos_bundle(|bundle| {
        let device = &bundle.device;
        let mut tokenizer = bundle.tokenizer.clone();

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;

        let ids = Tensor::new(encoding.get_ids(), device)?;
        let attention = Tensor::new(encoding.get_attention_mask(), device)?;
        let token_type = ids.zeros_like()?;

        let ids = ids.unsqueeze(0)?;
        let attention = attention.unsqueeze(0)?;
        let token_type = token_type.unsqueeze(0)?;

        let embeddings = bundle
            .model
            .forward(&ids, &token_type, Some(&attention))
            .context("POS model forward failed")?;

        let logits = bundle.classifier.forward(&embeddings)?;
        let logits = logits.squeeze(0)?;

        let tag_ids = decode_tags(logits)?;
        let offsets = encoding.get_offsets();
        let tokens = encoding.get_tokens();

        let mut results = Vec::new();
        for ((_, (start, end)), tag_id) in tokens.iter().zip(offsets.iter()).zip(tag_ids) {
            if *start == 0 && *end == 0 {
                continue;
            }
            let label = bundle
                .id2label
                .get(&tag_id)
                .cloned()
                .unwrap_or_else(|| "X".to_string());
            results.push(PosToken {
                tag: label,
                start: *start as i64,
                end: *end as i64,
            });
        }

        Ok(results)
    })
}
