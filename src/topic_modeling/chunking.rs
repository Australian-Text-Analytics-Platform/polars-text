//! Recursive/semantic chunking for the topic-modeling pipeline.
//!
//! Why this exists: today's BERTopic path embeds whole documents with a
//! 256-token model and silently truncates anything longer, so long documents
//! lose most of their content before clustering. The Rust pipeline instead
//! splits every document into token-budgeted chunks and embeds the chunks, so a
//! long document contributes several points (and ultimately a multi-topic
//! distribution) while a short document is simply one chunk. The whole pipeline
//! runs one uniform path — short text is not special-cased, it just yields a
//! single chunk here.
//!
//! Standard used: recursive/semantic splitting via the `text-splitter` crate,
//! which descends paragraph -> sentence -> word -> token boundaries and packs
//! units up to a token budget (the embedding model's context window). This is
//! the industry-standard recursive chunking strategy; it never cuts a sentence
//! mid-word and only falls back to harder boundaries when a unit overflows the
//! budget. A small token overlap keeps topical context across chunk seams.
//!
//! Called by: `topic_modeling::run` (orchestrator) before embedding.

use anyhow::{Context, Result};
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::Tokenizer;

/// One chunk of a source document. `doc_index` ties the chunk back to its
/// document so `rollup` can aggregate chunk topic assignments into a
/// per-document distribution; `chunk_index` is the chunk's ordinal within that
/// document (0-based) for stable ordering and debugging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub doc_index: usize,
    pub chunk_index: usize,
    pub text: String,
}

/// Chunking knobs. `max_tokens` is the per-chunk token budget and should match
/// the embedding model's context window (paraphrase-multilingual-MiniLM-L12-v2
/// = 128 tokens trained, 512 hard cap; we default to 256 for a balance of
/// context and chunk count). `overlap` carries a few tokens across chunk seams
/// so a topic spanning a boundary is not split blind; ~10-20% of `max_tokens`
/// is the usual range. Overlap must be strictly less than `max_tokens`.
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub max_tokens: usize,
    pub overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            overlap: 32,
        }
    }
}

/// Split every document into token-budgeted chunks.
///
/// Flow: build one `TextSplitter` whose size is measured by the model's HF
/// tokenizer, then for each document collect its semantic chunks. A
/// non-whitespace document always yields at least one chunk (a short document
/// yields exactly one). Whitespace-only/empty documents yield zero chunks; the
/// rollup stage maps those to an all-outlier distribution rather than crashing.
///
/// The `tokenizer` must have truncation disabled by the caller — otherwise the
/// sizer would cap chunk sizes at the tokenizer's truncation limit instead of
/// `max_tokens`. `embedding` loads the tokenizer and clears truncation before
/// handing a clone here.
pub fn chunk_documents(
    docs: &[String],
    tokenizer: Tokenizer,
    cfg: &ChunkingConfig,
) -> Result<Vec<Chunk>> {
    if cfg.max_tokens == 0 {
        anyhow::bail!("chunking max_tokens must be > 0");
    }
    let overlap = cfg.overlap.min(cfg.max_tokens.saturating_sub(1));
    let chunk_config = ChunkConfig::new(cfg.max_tokens)
        .with_sizer(tokenizer)
        .with_overlap(overlap)
        .context("invalid chunk overlap configuration")?;
    let splitter = TextSplitter::new(chunk_config);

    let mut chunks = Vec::with_capacity(docs.len());
    for (doc_index, doc) in docs.iter().enumerate() {
        if doc.trim().is_empty() {
            continue;
        }
        for (chunk_index, piece) in splitter.chunks(doc).enumerate() {
            chunks.push(Chunk {
                doc_index,
                chunk_index,
                text: piece.to_string(),
            });
        }
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::ensure_tokenizer_for_model;

    /// Build a real HF tokenizer for sizing. Uses the plain BERT tokenizer the
    /// crate already ships access to; the exact vocabulary does not matter for
    /// boundary tests, only that token counting is consistent.
    fn sizing_tokenizer() -> Tokenizer {
        // bert-base-uncased is already in the dev HF cache; load via the crate's
        // HF loader path by going through a fresh Tokenizer file is overkill, so
        // we construct a whitespace-ish tokenizer from the multilingual model is
        // also heavy. Instead reuse the registry's HF tokenizer loader.
        let backend = ensure_tokenizer_for_model(Some("huggingface:bert-base-uncased"))
            .expect("load bert tokenizer for test");
        // The registry stores a TokenizerBackend, not a raw Tokenizer, so for
        // the chunker test we load the raw tokenizer file directly instead.
        let _ = backend;
        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .build()
            .unwrap();
        let path = api
            .model("bert-base-uncased".to_string())
            .get("tokenizer.json")
            .unwrap();
        let mut tok = Tokenizer::from_file(path).unwrap();
        tok.with_truncation(None).unwrap();
        tok
    }

    #[test]
    #[ignore = "needs HF tokenizer download; run manually"]
    fn short_document_yields_single_chunk() {
        let docs = vec!["A short sentence about cats.".to_string()];
        let cfg = ChunkingConfig {
            max_tokens: 64,
            overlap: 0,
        };
        let chunks = chunk_documents(&docs, sizing_tokenizer(), &cfg).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].doc_index, 0);
        assert_eq!(chunks[0].chunk_index, 0);
        assert!(chunks[0].text.contains("cats"));
    }

    #[test]
    #[ignore = "needs HF tokenizer download; run manually"]
    fn empty_documents_produce_no_chunks() {
        let docs = vec!["   ".to_string(), "".to_string()];
        let cfg = ChunkingConfig::default();
        let chunks = chunk_documents(&docs, sizing_tokenizer(), &cfg).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    #[ignore = "needs HF tokenizer download; run manually"]
    fn long_document_splits_into_multiple_ordered_chunks() {
        // Build a clearly-over-budget document: many distinct sentences.
        let body = (0..40)
            .map(|i| format!("Sentence number {i} discusses an entirely separate topic."))
            .collect::<Vec<_>>()
            .join(" ");
        let docs = vec![body];
        let cfg = ChunkingConfig {
            max_tokens: 32,
            overlap: 4,
        };
        let chunks = chunk_documents(&docs, sizing_tokenizer(), &cfg).unwrap();
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        // chunk_index is contiguous and 0-based within the single document.
        for (expected, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.doc_index, 0);
            assert_eq!(chunk.chunk_index, expected);
        }
    }
}
