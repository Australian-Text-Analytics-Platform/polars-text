use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use once_cell::sync::OnceCell;
use tokenizers::Tokenizer;

const DEFAULT_TOKENIZER_MODEL: &str = "bert-base-uncased";
const DEFAULT_TOKENIZER_REVISION: &str = "main";

static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();

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
    lowercase: bool,
    remove_punct: bool,
) -> Result<Vec<String>> {
    let processed = if lowercase {
        text.to_lowercase()
    } else {
        text.to_string()
    };

    let encoding = tokenizer
        .encode(processed, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?;

    let mut tokens: Vec<String> = encoding.get_tokens().iter().cloned().collect();
    if remove_punct {
        tokens.retain(|tok| tok.chars().any(|ch| ch.is_alphanumeric()));
    }
    Ok(tokens)
}
