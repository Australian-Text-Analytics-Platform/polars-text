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
pub const JIEBA_MODEL_ID: &str = "jieba";

const SPECIAL_TOKENS: &[&str] = &["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];

/// A model-specific tokenizer fronting either a HuggingFace tokenizer
/// (BERT-family WordPiece, XLM-R SentencePiece, etc.) or `jieba-rs`
/// for word-level Chinese segmentation.
pub enum TokenizerBackend {
    HuggingFace(Tokenizer),
    Jieba(jieba_rs::Jieba),
}

impl TokenizerBackend {
    pub fn tokenize_text(
        &self,
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

        let mut tokens: Vec<String> = match self {
            TokenizerBackend::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode(processed, add_special_tokens)
                    .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;
                encoding.get_tokens().iter().cloned().collect()
            }
            TokenizerBackend::Jieba(jb) => jb
                .cut(&processed, true)
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        };

        if remove_punctuation {
            tokens.retain(|tok| tok.chars().any(|ch| ch.is_alphanumeric()));
        }
        Ok(tokens)
    }
}

static REGISTRY: OnceCell<RwLock<HashMap<String, Arc<TokenizerBackend>>>> = OnceCell::new();

fn registry() -> &'static RwLock<HashMap<String, Arc<TokenizerBackend>>> {
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn ensure_tokenizer_for_model(model_id: Option<&str>) -> Result<Arc<TokenizerBackend>> {
    let key = model_id.unwrap_or(DEFAULT_TOKENIZER_MODEL);

    if let Some(tok) = registry().read().unwrap().get(key) {
        return Ok(Arc::clone(tok));
    }

    let mut map = registry().write().unwrap();
    if let Some(tok) = map.get(key) {
        return Ok(Arc::clone(tok));
    }
    let backend = load_backend(key)?;
    let arc = Arc::new(backend);
    map.insert(key.to_string(), Arc::clone(&arc));
    Ok(arc)
}

fn load_backend(model_id: &str) -> Result<TokenizerBackend> {
    if model_id == JIEBA_MODEL_ID {
        Ok(TokenizerBackend::Jieba(jieba_rs::Jieba::new()))
    } else {
        load_hf_tokenizer(model_id).map(TokenizerBackend::HuggingFace)
    }
}

fn load_hf_tokenizer(model_id: &str) -> Result<Tokenizer> {
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
    use super::{loaded_model_ids, tokenize_plain_text, TokenizerBackend, JIEBA_MODEL_ID};

    #[test]
    fn test_tokenize_plain_text_drops_special_tokens() {
        let input = "[CLS] hello [SEP] [PAD] [UNK]";
        let tokens = tokenize_plain_text(input, true, true);
        assert_eq!(tokens, vec!["hello"]);
    }

    #[test]
    fn test_loaded_model_ids_returns_sorted_vec() {
        let ids = loaded_model_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn test_jieba_backend_segments_chinese_words() {
        let jb = jieba_rs::Jieba::new();
        let backend = TokenizerBackend::Jieba(jb);
        let tokens = backend
            .tokenize_text("今天天气很好", false, false, true)
            .unwrap();
        // Jieba should produce word-level tokens, not chars. With remove_punct=true
        // the punctuation filter still passes Chinese characters (they are
        // is_alphanumeric per Unicode). The key invariant is that at least one
        // token is multi-character — that is the distinguishing feature vs the
        // bert-base-chinese char-level fallback.
        assert!(
            tokens.iter().any(|t| t.chars().count() > 1),
            "expected at least one multi-char (word-level) token from Jieba, got {tokens:?}"
        );
    }

    #[test]
    fn test_jieba_model_id_constant() {
        assert_eq!(JIEBA_MODEL_ID, "jieba");
    }
}
