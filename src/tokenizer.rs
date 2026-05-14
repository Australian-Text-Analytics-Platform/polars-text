use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use lindera::tokenizer::Tokenizer as LinderaTokenizer;
use once_cell::sync::OnceCell;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer};

pub const DEFAULT_TOKENIZER_MODEL: &str = "bert-base-uncased";
const DEFAULT_TOKENIZER_REVISION: &str = "main";
pub const JIEBA_MODEL_ID: &str = "jieba";
/// Opaque model IDs for the three Lindera dict variants. Phase 5 routes
/// these through `load_backend` to a downloader (lindera_dict.rs) that
/// fetches the binary dict on first use; subsequent calls hit the
/// in-memory REGISTRY. The frontend selector surfaces JA's IPADIC vs
/// UniDic choice; KO has only ko-dic so it's a single ID.
pub const LINDERA_JA_IPADIC_MODEL_ID: &str = "lindera-ja-ipadic";
pub const LINDERA_JA_UNIDIC_MODEL_ID: &str = "lindera-ja-unidic";
pub const LINDERA_KO_DIC_MODEL_ID: &str = "lindera-ko-dic";

const SPECIAL_TOKENS: &[&str] = &["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];

/// A model-specific tokenizer fronting one of:
/// - a HuggingFace tokenizer (BERT-family WordPiece, XLM-R SentencePiece, etc.)
/// - `jieba-rs` for word-level Chinese segmentation
/// - Lindera for morpheme-level Japanese (IPADIC / UniDic) or Korean (ko-dic)
pub enum TokenizerBackend {
    HuggingFace(Tokenizer),
    Jieba(jieba_rs::Jieba),
    Lindera(LinderaTokenizer),
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
            TokenizerBackend::Lindera(tok) => tok
                .tokenize(&processed)
                .map_err(|e| anyhow::anyhow!("Lindera tokenize failed: {e}"))?
                .into_iter()
                .map(|t| t.surface.to_string())
                .collect(),
        };

        if remove_punctuation {
            tokens.retain(|tok| tok.chars().any(|ch| ch.is_alphanumeric()));
        }
        Ok(tokens)
    }

    /// Tokenize and emit `(token, start, end)` triples where `start` and `end`
    /// are **character** offsets (1 char per Hanzi, not 3 bytes), into the
    /// `lowercase`-applied processed text. Used by Phase 2's persisted tokens
    /// column so downstream tools can locate each token's span in the source.
    pub fn tokenize_text_with_offsets(
        &self,
        text: &str,
        lowercase: bool,
        remove_punctuation: bool,
    ) -> Result<Vec<(String, i64, i64)>> {
        let processed = if lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let raw: Vec<(String, i64, i64)> = match self {
            TokenizerBackend::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode(processed.as_str(), false)
                    .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;
                let toks = encoding.get_tokens();
                let offsets = encoding.get_offsets();
                toks.iter()
                    .zip(offsets.iter())
                    .map(|(tok, (b_start, b_end))| {
                        let cs = byte_to_char_idx(&processed, *b_start) as i64;
                        let ce = byte_to_char_idx(&processed, *b_end) as i64;
                        (tok.clone(), cs, ce)
                    })
                    .collect()
            }
            TokenizerBackend::Jieba(jb) => jb
                .tokenize(&processed, jieba_rs::TokenizeMode::Default, true)
                .into_iter()
                .map(|t| (t.word.to_string(), t.start as i64, t.end as i64))
                .collect(),
            TokenizerBackend::Lindera(tok) => {
                let toks = tok
                    .tokenize(&processed)
                    .map_err(|e| anyhow::anyhow!("Lindera tokenize failed: {e}"))?;
                // Lindera emits byte offsets; the rest of polars-text speaks in
                // char offsets (matches Jieba + the HF arm above), so translate
                // through the same helper before handing back to the caller.
                toks.into_iter()
                    .map(|t| {
                        let cs = byte_to_char_idx(&processed, t.byte_start) as i64;
                        let ce = byte_to_char_idx(&processed, t.byte_end) as i64;
                        (t.surface.to_string(), cs, ce)
                    })
                    .collect()
            }
        };

        let result: Vec<(String, i64, i64)> = raw
            .into_iter()
            .filter(|(tok, _, _)| {
                !remove_punctuation || tok.chars().any(|ch| ch.is_alphanumeric())
            })
            .collect();

        Ok(result)
    }
}

fn byte_to_char_idx(text: &str, byte_idx: usize) -> usize {
    text.char_indices().take_while(|(b, _)| *b < byte_idx).count()
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
        return Ok(TokenizerBackend::Jieba(jieba_rs::Jieba::new()));
    }
    if let Some(kind) = lindera_dict_for_model_id(model_id) {
        let tok = crate::lindera_dict::ensure_lindera_tokenizer(kind)?;
        return Ok(TokenizerBackend::Lindera(tok));
    }
    load_hf_tokenizer(model_id).map(TokenizerBackend::HuggingFace)
}

fn lindera_dict_for_model_id(model_id: &str) -> Option<crate::lindera_dict::LinderaDict> {
    match model_id {
        LINDERA_JA_IPADIC_MODEL_ID => Some(crate::lindera_dict::LinderaDict::JaIpadic),
        LINDERA_JA_UNIDIC_MODEL_ID => Some(crate::lindera_dict::LinderaDict::JaUnidic),
        LINDERA_KO_DIC_MODEL_ID => Some(crate::lindera_dict::LinderaDict::KoDic),
        _ => None,
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

    #[test]
    fn test_jieba_offsets_reconstruct_chinese() {
        let jb = jieba_rs::Jieba::new();
        let backend = TokenizerBackend::Jieba(jb);
        let text = "我爱中国";
        let toks = backend
            .tokenize_text_with_offsets(text, false, false)
            .unwrap();
        for (tok, start, end) in &toks {
            // Char-slice the text using the (start, end) char positions and
            // verify it equals the emitted token string.
            let extracted: String = text
                .chars()
                .skip(*start as usize)
                .take((*end - *start) as usize)
                .collect();
            assert_eq!(
                *tok, extracted,
                "Jieba offset mismatch for {tok:?} ({start}..{end})"
            );
        }
    }
}
