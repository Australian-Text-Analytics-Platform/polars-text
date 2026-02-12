use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use once_cell::sync::OnceCell;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer};

const DEFAULT_TOKENIZER_MODEL: &str = "bert-base-uncased";
const DEFAULT_TOKENIZER_REVISION: &str = "main";

static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();

const SPECIAL_TOKENS: &[&str] = &["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];

pub fn ensure_tokenizer() -> Result<&'static Tokenizer> {
    TOKENIZER.get_or_try_init(|| {
        let api = ApiBuilder::from_env()
            .build()
            .context("Failed to initialize hf-hub client")?;
        let repo = Repo::with_revision(
            DEFAULT_TOKENIZER_MODEL.to_string(),
            RepoType::Model,
            DEFAULT_TOKENIZER_REVISION.to_string(),
        );
        let api = api.repo(repo);
        let tokenizer_path = api
            .get("tokenizer.json")
            .context("Failed to fetch tokenizer.json")?;
        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))
    })
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
    use super::tokenize_plain_text;

    #[test]
    fn test_tokenize_plain_text_drops_special_tokens() {
        let input = "[CLS] hello [SEP] [PAD] [UNK]";
        let tokens = tokenize_plain_text(input, true, true);
        assert_eq!(tokens, vec!["hello"]);
    }
}
