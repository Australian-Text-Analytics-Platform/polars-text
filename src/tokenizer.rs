use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use once_cell::sync::OnceCell;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer};

pub const DEFAULT_TOKENIZER_MODEL: &str = "bert-base-uncased";
const DEFAULT_TOKENIZER_REVISION: &str = "main";

static REGISTRY: OnceCell<RwLock<HashMap<String, Arc<Tokenizer>>>> = OnceCell::new();

const SPECIAL_TOKENS: &[&str] = &["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];

fn registry() -> &'static RwLock<HashMap<String, Arc<Tokenizer>>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn ensure_tokenizer_for_model(model_id: Option<&str>) -> Result<Arc<Tokenizer>> {
    let key = model_id.unwrap_or(DEFAULT_TOKENIZER_MODEL);

    if let Some(tok) = registry().read().unwrap().get(key) {
        return Ok(Arc::clone(tok));
    }

    let mut map = registry().write().unwrap();
    if let Some(tok) = map.get(key) {
        return Ok(Arc::clone(tok));
    }
    let tokenizer = load_tokenizer_from_hub(key)?;
    let arc = Arc::new(tokenizer);
    map.insert(key.to_string(), Arc::clone(&arc));
    Ok(arc)
}

fn load_tokenizer_from_hub(model_id: &str) -> Result<Tokenizer> {
    let api = ApiBuilder::from_env()
        .build()
        .context("Failed to initialize hf-hub client")?;
    let repo = Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        DEFAULT_TOKENIZER_REVISION.to_string(),
    );
    let api = api.repo(repo);
    let tokenizer_path = api
        .get("tokenizer.json")
        .with_context(|| format!("Failed to fetch tokenizer.json for {model_id}"))?;
    Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer {model_id}: {e}"))
}

pub fn loaded_model_ids() -> Vec<String> {
    let mut ids: Vec<String> = registry().read().unwrap().keys().cloned().collect();
    ids.sort();
    ids
}

pub fn tokenize_text(
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
    lowercase: bool,
    remove_punctuation: bool,
) -> Result<Vec<String>> {
    let processed = if lowercase {
        text.to_lowercase()
    } else {
        text.to_string()
    };

    let encoding = tokenizer
        .encode(processed, add_special_tokens)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;

    let mut tokens: Vec<String> = encoding.get_tokens().iter().cloned().collect();
    if remove_punctuation {
        tokens.retain(|tok| tok.chars().any(|ch| ch.is_alphanumeric()));
    }
    Ok(tokens)
}

pub fn tokenize_plain_text(
    text: &str,
    lowercase: bool,
    remove_punctuation: bool,
) -> Vec<String> {
    let processed = if lowercase {
        text.to_lowercase()
    } else {
        text.to_string()
    };

    let mut pre_tokenized = PreTokenizedString::from(processed.as_str());
    if BertPreTokenizer.pre_tokenize(&mut pre_tokenized).is_err() {
        return Vec::new();
    }

    let special_set = SPECIAL_TOKENS;
    pre_tokenized
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .filter_map(|(token, _span, _)| {
            if remove_punctuation && !token.chars().any(|ch| ch.is_alphanumeric()) {
                return None;
            }

            let token_upper = token.to_ascii_uppercase();
            let bracketed = format!("[{token_upper}]");
            if special_set.contains(&token_upper.as_str()) || special_set.contains(&bracketed.as_str()) {
                return None;
            }

            if token.is_empty() {
                return None;
            }

            Some(token.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{loaded_model_ids, tokenize_plain_text};

    #[test]
    fn test_tokenize_plain_text_drops_special_tokens() {
        let input = "[CLS] hello [SEP] [PAD] [UNK]";
        let tokens = tokenize_plain_text(input, true, true);
        assert_eq!(tokens, vec!["hello"]);
    }

    #[test]
    fn test_loaded_model_ids_returns_sorted_vec() {
        // Without forcing a load, the registry may be empty or populated by other
        // tests in this binary. Just verify the call shape and ordering invariant.
        let ids = loaded_model_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }
}
