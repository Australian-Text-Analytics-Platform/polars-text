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
pub const PLAIN_WORDS_EN_MODEL_ID: &str = "plain_words_en";
pub const JIEBA_MODEL_ID: &str = "jieba";
/// Opaque model IDs for the three downloaded Lindera dict variants. Phase 5 routes
/// these through `load_backend` to a downloader (lindera_dict.rs) that
/// fetches the binary dict on first use; subsequent calls hit the
/// in-memory REGISTRY. The frontend selector surfaces JA's IPADIC vs
/// UniDic choice; KO has only ko-dic so it's a single ID.
pub const LINDERA_JA_IPADIC_MODEL_ID: &str = "lindera-ja-ipadic";
pub const LINDERA_JA_UNIDIC_MODEL_ID: &str = "lindera-ja-unidic";
pub const LINDERA_KO_DIC_MODEL_ID: &str = "lindera-ko-dic";

const SPECIAL_TOKENS: &[&str] = &["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];

fn keep_token_text(token: &str, remove_punctuation: bool) -> bool {
    !remove_punctuation || token.chars().any(|ch| ch.is_alphanumeric())
}

/// A model-specific tokenizer fronting one of:
/// - a HuggingFace tokenizer (BERT-family WordPiece, XLM-R SentencePiece, etc.)
/// - Lindera for Chinese Jieba word segmentation, Japanese (IPADIC / UniDic),
///   or Korean (ko-dic)
pub enum TokenizerBackend {
    PlainWordsEn,
    HuggingFace(Tokenizer),
    Lindera(LinderaTokenizer),
}

struct TokenRecord {
    token: String,
    start: i64,
    end: i64,
}

fn is_special_token_text(token: &str) -> bool {
    let token_upper = token.to_ascii_uppercase();
    if SPECIAL_TOKENS.contains(&token_upper.as_str()) {
        return true;
    }

    let bracketed = format!("[{token_upper}]");
    SPECIAL_TOKENS.contains(&bracketed.as_str())
}

fn plain_word_records(text: &str, remove_punctuation: bool) -> Vec<TokenRecord> {
    let mut pre_tokenized = PreTokenizedString::from(text);
    if BertPreTokenizer.pre_tokenize(&mut pre_tokenized).is_err() {
        return Vec::new();
    }

    let splits = pre_tokenized.get_splits(OffsetReferential::Original, OffsetType::Byte);
    let char_spans = byte_spans_to_char_spans(
        text,
        splits.iter().map(|(_, (start, end), _)| (*start, *end)),
    );

    splits
        .into_iter()
        .zip(char_spans)
        .filter_map(|((token, _span, _), (start, end))| {
            if remove_punctuation && !token.chars().any(|ch| ch.is_alphanumeric()) {
                return None;
            }

            if is_special_token_text(token) || token.is_empty() {
                return None;
            }

            Some(TokenRecord {
                token: token.to_string(),
                start,
                end,
            })
        })
        .collect()
}

impl TokenizerBackend {
    /// Whether `to_lowercase()` is a semantically meaningful preprocessing step.
    /// Jieba (zh) and Lindera (ja/ko) operate on scripts with no case, so
    /// running the full Unicode case-fold table walk + heap allocation per row
    /// is pure waste. The Python wrapper also defaults to `lowercase=False`
    /// for CJK models — this is a safety net for any caller that forgets.
    fn case_aware(&self) -> bool {
        matches!(
            self,
            TokenizerBackend::PlainWordsEn | TokenizerBackend::HuggingFace(_)
        )
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
        Ok(self
            .tokenize_records(text, add_special_tokens, lowercase, remove_punctuation)?
            .into_iter()
            .map(|record| record.token)
            .collect())
    }

    fn tokenize_records(
        &self,
        text: &str,
        add_special_tokens: bool,
        lowercase: bool,
        remove_punctuation: bool,
    ) -> Result<Vec<TokenRecord>> {
        let processed = self.preprocess(text, lowercase);
        let processed_ref: &str = processed.as_ref();

        let mut records: Vec<TokenRecord> = match self {
            TokenizerBackend::PlainWordsEn => plain_word_records(processed_ref, remove_punctuation),
            TokenizerBackend::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode(processed_ref, add_special_tokens)
                    .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;
                let toks = encoding.get_tokens();
                let offsets = encoding.get_offsets();
                let char_spans =
                    byte_spans_to_char_spans(processed_ref, offsets.iter().map(|(s, e)| (*s, *e)));
                toks.iter()
                    .zip(char_spans.into_iter())
                    .map(|(tok, (start, end))| TokenRecord {
                        token: tok.clone(),
                        start,
                        end,
                    })
                    .collect()
            }
            TokenizerBackend::Lindera(tok) => tok
                .tokenize(processed_ref)
                .map_err(|e| anyhow::anyhow!("Lindera tokenize failed: {e}"))?
                .into_iter()
                .map(|t| TokenRecord {
                    token: t.surface.to_string(),
                    start: t.byte_start as i64,
                    end: t.byte_end as i64,
                })
                .collect(),
        };

        if matches!(self, TokenizerBackend::Lindera(_)) {
            let char_spans = byte_spans_to_char_spans(
                processed_ref,
                records
                    .iter()
                    .map(|record| (record.start as usize, record.end as usize)),
            );
            for (record, (start, end)) in records.iter_mut().zip(char_spans) {
                record.start = start;
                record.end = end;
            }
        }

        records.retain(|record| keep_token_text(&record.token, remove_punctuation));
        Ok(records)
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
        let result: Vec<(String, i64, i64)> = self
            .tokenize_records(text, false, lowercase, remove_punctuation)?
            .into_iter()
            .map(|record| (record.token, record.start, record.end))
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

    {
        let map = registry()
            .read()
            .map_err(|_| anyhow::anyhow!("tokenizer registry lock poisoned"))?;
        if let Some(tok) = map.get(key) {
            return Ok(Arc::clone(tok));
        }
    }

    let mut map = registry()
        .write()
        .map_err(|_| anyhow::anyhow!("tokenizer registry lock poisoned"))?;
    if let Some(tok) = map.get(key) {
        return Ok(Arc::clone(tok));
    }
    let backend = load_backend(key)?;
    let arc = Arc::new(backend);
    map.insert(key.to_string(), Arc::clone(&arc));
    Ok(arc)
}

fn load_backend(model_id: &str) -> Result<TokenizerBackend> {
    if model_id == PLAIN_WORDS_EN_MODEL_ID {
        return Ok(TokenizerBackend::PlainWordsEn);
    }
    if model_id == JIEBA_MODEL_ID {
        let tok = crate::lindera_dict::ensure_embedded_lindera_tokenizer(JIEBA_MODEL_ID)?;
        return Ok(TokenizerBackend::Lindera(tok));
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
    let Ok(map) = registry().read() else {
        return Vec::new();
    };
    let mut ids: Vec<String> = map.keys().cloned().collect();
    ids.sort();
    ids
}

pub fn tokenize_plain_text(text: &str, lowercase: bool, remove_punctuation: bool) -> Vec<String> {
    TokenizerBackend::PlainWordsEn
        .tokenize_text(text, false, lowercase, remove_punctuation)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{
        ensure_tokenizer_for_model, lindera_dict_for_model_id, loaded_model_ids,
        tokenize_plain_text, TokenizerBackend, JIEBA_MODEL_ID, LINDERA_JA_IPADIC_MODEL_ID,
        LINDERA_JA_UNIDIC_MODEL_ID, LINDERA_KO_DIC_MODEL_ID, PLAIN_WORDS_EN_MODEL_ID,
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
        let backend = ensure_tokenizer_for_model(Some(JIEBA_MODEL_ID)).unwrap();
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
    fn test_plain_words_en_model_id_constant() {
        assert_eq!(PLAIN_WORDS_EN_MODEL_ID, "plain_words_en");
    }

    #[test]
    fn test_plain_words_en_backend_matches_plain_helper() {
        let backend = TokenizerBackend::PlainWordsEn;
        let text = "Hello, [UNK] ##sta Queensland";
        let tokens = backend.tokenize_text(text, false, true, true).unwrap();
        assert_eq!(tokens, tokenize_plain_text(text, true, true));
    }

    #[test]
    fn test_plain_words_en_offsets_reconstruct_english() {
        let backend = TokenizerBackend::PlainWordsEn;
        let text = "Hello, Queensland";
        let toks = backend
            .tokenize_text_with_offsets(text, true, true)
            .unwrap();
        let text_lc = text.to_lowercase();
        for (tok, start, end) in &toks {
            let extracted: String = text_lc
                .chars()
                .skip(*start as usize)
                .take((*end - *start) as usize)
                .collect();
            assert_eq!(*tok, extracted);
        }
    }

    #[test]
    fn test_jieba_offsets_reconstruct_chinese() {
        let backend = ensure_tokenizer_for_model(Some(JIEBA_MODEL_ID)).unwrap();
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
        let backend = ensure_tokenizer_for_model(Some(JIEBA_MODEL_ID)).unwrap();
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
        let backend = ensure_tokenizer_for_model(Some(JIEBA_MODEL_ID)).unwrap();
        assert!(!backend.case_aware());
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
