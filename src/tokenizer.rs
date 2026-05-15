use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use lindera::tokenizer::Tokenizer as LinderaTokenizer;
use once_cell::sync::OnceCell;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer};

use crate::offsets::byte_spans_to_char_spans;

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
    /// Whether `to_lowercase()` is a semantically meaningful preprocessing step.
    /// Jieba (zh) and Lindera (ja/ko) operate on scripts with no case, so
    /// running the full Unicode case-fold table walk + heap allocation per row
    /// is pure waste. The Python wrapper also defaults to `lowercase=False`
    /// for CJK models — this is a safety net for any caller that forgets.
    fn case_aware(&self) -> bool {
        matches!(self, TokenizerBackend::HuggingFace(_))
    }

    /// Apply the requested case-folding only when meaningful. Returns a
    /// `Cow<str>` so that the no-op branch borrows the original text
    /// (zero allocations) instead of paying for a `text.to_string()` clone.
    fn preprocess<'a>(&self, text: &'a str, lowercase: bool) -> Cow<'a, str> {
        if lowercase && self.case_aware() {
            Cow::Owned(text.to_lowercase())
        } else {
            Cow::Borrowed(text)
        }
    }

    pub fn tokenize_text(
        &self,
        text: &str,
        add_special_tokens: bool,
        lowercase: bool,
        remove_punctuation: bool,
    ) -> Result<Vec<String>> {
        let processed = self.preprocess(text, lowercase);

        let mut tokens: Vec<String> = match self {
            TokenizerBackend::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode(processed.as_ref(), add_special_tokens)
                    .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;
                encoding.get_tokens().iter().cloned().collect()
            }
            TokenizerBackend::Jieba(jb) => jb
                .cut(processed.as_ref(), true)
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            TokenizerBackend::Lindera(tok) => tok
                .tokenize(processed.as_ref())
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
        let processed = self.preprocess(text, lowercase);
        let processed_ref: &str = processed.as_ref();

        let raw: Vec<(String, i64, i64)> = match self {
            TokenizerBackend::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode(processed_ref, false)
                    .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;
                let toks = encoding.get_tokens();
                let offsets = encoding.get_offsets();
                let char_spans = byte_spans_to_char_spans(
                    processed_ref,
                    offsets.iter().map(|(s, e)| (*s, *e)),
                );
                toks.iter()
                    .zip(char_spans.into_iter())
                    .map(|(tok, (cs, ce))| (tok.clone(), cs, ce))
                    .collect()
            }
            TokenizerBackend::Jieba(jb) => jb
                .tokenize(processed_ref, jieba_rs::TokenizeMode::Default, true)
                .into_iter()
                .map(|t| (t.word.to_string(), t.start as i64, t.end as i64))
                .collect(),
            TokenizerBackend::Lindera(tok) => {
                let toks = tok
                    .tokenize(processed_ref)
                    .map_err(|e| anyhow::anyhow!("Lindera tokenize failed: {e}"))?;
                // Lindera emits byte offsets; the rest of polars-text speaks in
                // char offsets (matches Jieba + the HF arm above), so translate
                // through the batch helper before handing back to the caller.
                // Single forward sweep over char_indices — O(C + N) instead of
                // the prior O(C·N) per-token byte_to_char_idx walk.
                let char_spans = byte_spans_to_char_spans(
                    processed_ref,
                    toks.iter().map(|t| (t.byte_start, t.byte_end)),
                );
                toks.into_iter()
                    .zip(char_spans.into_iter())
                    .map(|(t, (cs, ce))| (t.surface.to_string(), cs, ce))
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
    use super::{
        lindera_dict_for_model_id, loaded_model_ids, tokenize_plain_text,
        TokenizerBackend, JIEBA_MODEL_ID, LINDERA_JA_IPADIC_MODEL_ID,
        LINDERA_JA_UNIDIC_MODEL_ID, LINDERA_KO_DIC_MODEL_ID,
    };
    use crate::lindera_dict::LinderaDict;

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

    #[test]
    fn test_lindera_model_id_constants_match_dict_kinds() {
        // Phase 5: the three Lindera model IDs route to the matching
        // LinderaDict variant in load_backend. Catches a copy-paste
        // mistake where two ids map to the same kind.
        assert_eq!(
            lindera_dict_for_model_id(LINDERA_JA_IPADIC_MODEL_ID),
            Some(LinderaDict::JaIpadic)
        );
        assert_eq!(
            lindera_dict_for_model_id(LINDERA_JA_UNIDIC_MODEL_ID),
            Some(LinderaDict::JaUnidic)
        );
        assert_eq!(
            lindera_dict_for_model_id(LINDERA_KO_DIC_MODEL_ID),
            Some(LinderaDict::KoDic)
        );
    }

    #[test]
    fn test_jieba_lowercase_flag_is_noop_for_cjk() {
        // CJK has no case, so the lowercase flag must produce the same
        // tokens whether on or off. Before the case_aware() guard, the
        // backend paid a full to_lowercase() Unicode walk + heap clone
        // per row for nothing; now it short-circuits to a borrow.
        let jb = jieba_rs::Jieba::new();
        let backend = TokenizerBackend::Jieba(jb);
        let text = "今天天气很好";
        let tokens_lower = backend.tokenize_text(text, false, true, false).unwrap();
        let tokens_plain = backend.tokenize_text(text, false, false, false).unwrap();
        assert_eq!(tokens_lower, tokens_plain);

        let offsets_lower = backend
            .tokenize_text_with_offsets(text, true, false)
            .unwrap();
        let offsets_plain = backend
            .tokenize_text_with_offsets(text, false, false)
            .unwrap();
        assert_eq!(offsets_lower, offsets_plain);
    }

    #[test]
    fn test_case_aware_flag() {
        // Pin the backend-classification contract so future additions
        // (e.g. a fastText backend for CJK) explicitly opt in or out.
        let jb = jieba_rs::Jieba::new();
        assert!(!TokenizerBackend::Jieba(jb).case_aware());
    }

    #[test]
    fn test_lindera_dict_for_unknown_model_id_returns_none() {
        // Anything that isn't one of the three opaque ids falls through
        // to the HF path in load_backend — verified here by asserting
        // None for plausible adjacent strings.
        assert_eq!(lindera_dict_for_model_id("lindera-ja"), None);
        assert_eq!(lindera_dict_for_model_id("ja-ipadic"), None);
        assert_eq!(lindera_dict_for_model_id("bert-base-uncased"), None);
        assert_eq!(lindera_dict_for_model_id("jieba"), None);
        assert_eq!(lindera_dict_for_model_id(""), None);
    }
}
