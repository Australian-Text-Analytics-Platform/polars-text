use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::Dbscan;
use ndarray::Array2;
use once_cell::sync::OnceCell;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
use crate::tokenizer::tokenize_plain_text;

const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const DEFAULT_REVISION: &str = "main";

struct CandleBundle {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

static CANDLE_BUNDLE: OnceCell<Mutex<CandleBundle>> = OnceCell::new();

fn build_candle_bundle() -> Result<CandleBundle> {
    let device = Device::Cpu;
    let api = ApiBuilder::from_env().build().context("Failed to initialize hf-hub client")?;
    let repo = Repo::with_revision(
        DEFAULT_MODEL_ID.to_string(),
        RepoType::Model,
        DEFAULT_REVISION.to_string(),
    );
    let api = api.repo(repo);

    let config_path = api.get("config.json").context("Failed to fetch config.json")?;
    let tokenizer_path = api
        .get("tokenizer.json")
        .context("Failed to fetch tokenizer.json")?;
    let weights_path = api
        .get("model.safetensors")
        .context("Failed to fetch model.safetensors")?;

    let config_contents = std::fs::read_to_string(config_path)
        .context("Failed to read config.json")?;
    let config: Config = serde_json::from_str(&config_contents)
        .context("Failed to parse config.json")?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)
            .context("Failed to map model weights")?
    };
    let model = BertModel::load(vb, &config).context("Failed to load BERT model")?;

    Ok(CandleBundle {
        model,
        tokenizer,
        device,
    })
}

fn with_candle_bundle<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&CandleBundle) -> Result<R>,
{
    let mutex = CANDLE_BUNDLE
        .get_or_try_init(|| build_candle_bundle().map(Mutex::new))?;
    let mut guard = mutex
        .lock()
        .map_err(|_| anyhow::anyhow!("Candle model mutex poisoned"))?;
    f(&mut *guard)
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

fn embed_with_candle(bundle: &CandleBundle, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

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

    let tokens = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("Tokenizer batch encode failed: {e}"))?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let ids = tokens.get_ids().to_vec();
            Tensor::new(ids.as_slice(), device).map_err(anyhow::Error::from)
        })
        .collect::<Result<Vec<_>>>()?;

    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let mask = tokens.get_attention_mask().to_vec();
            Tensor::new(mask.as_slice(), device).map_err(anyhow::Error::from)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;
    let token_type_ids = token_ids.zeros_like()?;

    let embeddings = bundle
        .model
        .forward(&token_ids, &token_type_ids, Some(&attention_mask))
        .context("Candle model forward failed")?;

    let attention_mask_for_pooling = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
    let sum_mask = attention_mask_for_pooling.sum(1)?;
    let pooled = (embeddings.broadcast_mul(&attention_mask_for_pooling)?).sum(1)?;
    let pooled = pooled.broadcast_div(&sum_mask)?;
    let normalized = normalize_l2(&pooled)?;

    normalized
        .to_vec2::<f32>()
        .context("Failed to convert embeddings to Vec<Vec<f32>>")
}

fn l2_normalize(mut embeddings: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    for embedding in embeddings.iter_mut() {
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in embedding.iter_mut() {
                *v /= norm;
            }
        }
    }
    embeddings
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let diff = av - bv;
        sum += diff * diff;
    }
    sum.sqrt()
}

fn estimate_eps(embeddings: &[Vec<f32>], min_points: usize) -> f32 {
    let n = embeddings.len();
    if n <= 1 {
        return 0.5;
    }
    let k = min_points.max(2).min(n - 1);
    let mut kth_distances = Vec::with_capacity(n);
    for i in 0..n {
        let mut distances = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j {
                continue;
            }
            distances.push(euclidean_distance(&embeddings[i], &embeddings[j]));
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        kth_distances.push(distances[k - 1]);
    }
    kth_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    kth_distances[kth_distances.len() / 2].max(1e-6)
}

fn cluster_embeddings(
    embeddings: Vec<Vec<f32>>,
    min_points: usize,
    eps: Option<f32>,
) -> Result<(Vec<i32>, Vec<Vec<f32>>, f32)> {
    if embeddings.is_empty() {
        return Ok((Vec::new(), Vec::new(), eps.unwrap_or(0.0)));
    }

    let normalized = l2_normalize(embeddings);
    let n_samples = normalized.len();
    let dim = normalized[0].len();

    let flat: Vec<f64> = normalized
        .iter()
        .flat_map(|v| v.iter().map(|&x| x as f64))
        .collect();

    let records = Array2::from_shape_vec((n_samples, dim), flat)
        .map_err(|e| anyhow::anyhow!("Failed to build embedding matrix: {e}"))?;
    let dataset = DatasetBase::from(records);

    let eps_value = eps.unwrap_or_else(|| estimate_eps(&normalized, min_points));

    let model = Dbscan::params(min_points).tolerance(eps_value as f64);
    let result = model
        .transform(dataset)
        .map_err(|e| anyhow::anyhow!("DBSCAN failed: {e}"))?;

    let labels: Vec<i32> = result
        .targets()
        .iter()
        .map(|label: &Option<usize>| (*label).map(|v| v as i32).unwrap_or(-1))
        .collect();

    Ok((labels, normalized, eps_value))
}

fn tokenize_terms(_bundle: &CandleBundle, text: &str) -> Result<Vec<String>> {
    let tokens = tokenize_plain_text(text, true, true);
    Ok(tokens)
}

fn build_topic_labels(
    bundle: &CandleBundle,
    texts_by_topic: &HashMap<i32, Vec<String>>,
    max_terms: usize,
) -> Result<HashMap<i32, String>> {
    let mut labels = HashMap::new();
    for (topic_id, texts) in texts_by_topic {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for token in tokenize_terms(bundle, text)? {
                *counts.entry(token).or_insert(0) += 1;
            }
        }
        let mut terms: Vec<(String, usize)> = counts.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1));
        let top_terms: Vec<String> = terms
            .into_iter()
            .take(max_terms.max(1))
            .map(|(t, _)| t)
            .collect();
        let label = format!("topic_{topic_id}: {}", top_terms.join(", "));
        labels.insert(*topic_id, label);
    }
    Ok(labels)
}

fn topic_modeling_inner(
    texts: &[String],
    min_points: usize,
    eps: Option<f32>,
    max_terms: usize,
) -> Result<(HashMap<i32, String>, Vec<i32>)> {
    with_candle_bundle(|bundle| {
        let embeddings = embed_with_candle(bundle, texts)?;
        let (labels, _normalized, _eps_used) =
            cluster_embeddings(embeddings, min_points.max(2), eps)?;

        let mut texts_by_topic: HashMap<i32, Vec<String>> = HashMap::new();
        for (text, label) in texts.iter().zip(labels.iter()) {
            if *label >= 0 {
                texts_by_topic
                    .entry(*label)
                    .or_default()
                    .push(text.clone());
            }
        }

        let mut topic_labels = build_topic_labels(bundle, &texts_by_topic, max_terms)?;
        topic_labels.insert(-1, "noise".to_string());

        Ok((topic_labels, labels))
    })
}

pub fn topic_modeling_py(
    py: Python<'_>,
    texts: Vec<String>,
    min_points: usize,
    eps: Option<f32>,
    max_terms: usize,
    _seed: u64,
) -> PyResult<(Py<PyAny>, Vec<Vec<(i64, f32)>>)> {
    let mut valid_texts = Vec::new();
    let mut valid_indices = Vec::new();
    let mut assignments: Vec<Option<i32>> = vec![None; texts.len()];

    for (idx, text) in texts.iter().enumerate() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            valid_texts.push(trimmed.to_string());
            valid_indices.push(idx);
        }
    }

    let topics_dict = PyDict::new(py);

    if !valid_texts.is_empty() {
        let (topics, labels) = py
            .detach(|| topic_modeling_inner(&valid_texts, min_points, eps, max_terms))
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{e}")))?;

        for (label, idx) in labels.iter().zip(valid_indices.iter()) {
            assignments[*idx] = Some(*label);
        }

        for (topic_id, label) in topics {
            topics_dict
                .set_item(topic_id, label)
                .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{e}")))?;
        }
    }

    let doc_topics: Vec<Vec<(i64, f32)>> = assignments
        .iter()
        .map(|assignment| match assignment {
            Some(label) => vec![(*label as i64, 1.0_f32)],
            None => Vec::new(),
        })
        .collect();

    Ok((topics_dict.into_pyobject(py)?.into(), doc_topics))
}
