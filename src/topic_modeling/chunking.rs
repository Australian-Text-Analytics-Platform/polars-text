//! Paragraph/sentence/length chunking for the topic-modeling pipeline.
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
//! Strategy: split paragraphs first, split oversized paragraphs into sentences,
//! then split oversized sentences by token length. A small overlap carries
//! trailing units into the next chunk when it fits, keeping topical context
//! across chunk boundaries without hiding where the hard fallbacks happen.
//!
//! Called by: `topic_modeling::run` (orchestrator) before embedding.

use anyhow::Result;
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

/// Chunking knobs. `max_tokens` is the per-chunk token budget and should stay
/// within the embedding model's context window; we default to 256 for a balance
/// of context and chunk count. `overlap` carries a few tokens across chunk seams
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

#[derive(Debug, Clone)]
struct ChunkUnit {
    text: String,
    tokens: usize,
}

/// Split every document into token-budgeted chunks.
///
/// Flow: split each document into paragraphs, split only oversized paragraphs
/// into sentences, split only oversized sentences by token length, then pack the
/// resulting units up to `max_tokens` with a bounded overlap. A
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

    let mut chunks = Vec::with_capacity(docs.len());
    for (doc_index, doc) in docs.iter().enumerate() {
        chunks.extend(chunk_document(doc_index, doc, &tokenizer, cfg)?);
    }
    Ok(chunks)
}

fn chunk_document(
    doc_index: usize,
    doc: &str,
    tokenizer: &Tokenizer,
    cfg: &ChunkingConfig,
) -> Result<Vec<Chunk>> {
    chunk_document_with_counter(doc_index, doc, cfg, &mut |text| {
        count_tokens(tokenizer, text)
    })
}

fn chunk_document_with_counter(
    doc_index: usize,
    doc: &str,
    cfg: &ChunkingConfig,
    count_tokens: &mut impl FnMut(&str) -> Result<usize>,
) -> Result<Vec<Chunk>> {
    if doc.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut units = Vec::new();
    for paragraph in split_paragraphs(doc) {
        append_sized_units(&paragraph, cfg.max_tokens, count_tokens, &mut units)?;
    }

    let chunk_texts = pack_units(units, cfg);
    Ok(chunk_texts
        .into_iter()
        .enumerate()
        .map(|(chunk_index, text)| Chunk {
            doc_index,
            chunk_index,
            text,
        })
        .collect())
}

fn append_sized_units(
    text: &str,
    max_tokens: usize,
    count_tokens: &mut impl FnMut(&str) -> Result<usize>,
    out: &mut Vec<ChunkUnit>,
) -> Result<()> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    let tokens = count_tokens(trimmed)?;
    if tokens <= max_tokens {
        out.push(ChunkUnit {
            text: trimmed.to_string(),
            tokens,
        });
        return Ok(());
    }

    let sentences = split_sentences(trimmed);
    if sentences.len() > 1 {
        for sentence in sentences {
            append_sentence_or_length_units(&sentence, max_tokens, count_tokens, out)?;
        }
    } else {
        append_length_units(trimmed, max_tokens, count_tokens, out)?;
    }
    Ok(())
}

fn append_sentence_or_length_units(
    sentence: &str,
    max_tokens: usize,
    count_tokens: &mut impl FnMut(&str) -> Result<usize>,
    out: &mut Vec<ChunkUnit>,
) -> Result<()> {
    let tokens = count_tokens(sentence)?;
    if tokens <= max_tokens {
        out.push(ChunkUnit {
            text: sentence.to_string(),
            tokens,
        });
    } else {
        append_length_units(sentence, max_tokens, count_tokens, out)?;
    }
    Ok(())
}

fn append_length_units(
    text: &str,
    max_tokens: usize,
    count_tokens: &mut impl FnMut(&str) -> Result<usize>,
    out: &mut Vec<ChunkUnit>,
) -> Result<()> {
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.len() > 1 {
        let mut word_units = Vec::new();
        append_packed_segments(&words, " ", max_tokens, count_tokens, &mut word_units)?;
        for unit in word_units {
            if unit.tokens <= max_tokens {
                out.push(unit);
            } else {
                append_length_units(&unit.text, max_tokens, count_tokens, out)?;
            }
        }
    } else {
        let chars = text.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        let refs = chars.iter().map(String::as_str).collect::<Vec<_>>();
        append_packed_segments(&refs, "", max_tokens, count_tokens, out)?;
    }
    Ok(())
}

fn append_packed_segments(
    segments: &[&str],
    separator: &str,
    max_tokens: usize,
    count_tokens: &mut impl FnMut(&str) -> Result<usize>,
    out: &mut Vec<ChunkUnit>,
) -> Result<()> {
    let mut current = String::new();
    for segment in segments {
        let candidate = if current.is_empty() {
            (*segment).to_string()
        } else {
            format!("{current}{separator}{segment}")
        };
        let candidate_tokens = count_tokens(&candidate)?;
        if candidate_tokens <= max_tokens || current.is_empty() {
            current = candidate;
            continue;
        }

        let current_tokens = count_tokens(&current)?;
        out.push(ChunkUnit {
            text: current,
            tokens: current_tokens,
        });
        current = (*segment).to_string();
    }

    if !current.trim().is_empty() {
        let tokens = count_tokens(&current)?;
        out.push(ChunkUnit {
            text: current,
            tokens,
        });
    }
    Ok(())
}

fn pack_units(units: Vec<ChunkUnit>, cfg: &ChunkingConfig) -> Vec<String> {
    let overlap = cfg.overlap.min(cfg.max_tokens.saturating_sub(1));
    let mut chunks = Vec::new();
    let mut current: Vec<ChunkUnit> = Vec::new();
    let mut current_tokens = 0usize;

    for unit in units {
        if !current.is_empty() && current_tokens + unit.tokens > cfg.max_tokens {
            chunks.push(join_units(&current));
            current = overlap_suffix(&current, overlap, cfg.max_tokens);
            current_tokens = current.iter().map(|unit| unit.tokens).sum();
            while !current.is_empty() && current_tokens + unit.tokens > cfg.max_tokens {
                current_tokens -= current.remove(0).tokens;
            }
        }
        current_tokens += unit.tokens;
        current.push(unit);
    }

    if !current.is_empty() {
        chunks.push(join_units(&current));
    }
    chunks
}

fn overlap_suffix(units: &[ChunkUnit], overlap: usize, max_tokens: usize) -> Vec<ChunkUnit> {
    if overlap == 0 {
        return Vec::new();
    }
    let mut selected = Vec::new();
    let mut tokens = 0usize;
    for unit in units.iter().rev() {
        if tokens + unit.tokens >= max_tokens {
            break;
        }
        selected.push(unit.clone());
        tokens += unit.tokens;
        if tokens >= overlap {
            break;
        }
    }
    selected.reverse();
    selected
}

fn join_units(units: &[ChunkUnit]) -> String {
    units
        .iter()
        .map(|unit| unit.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn split_paragraphs(text: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current.is_empty() {
                paragraphs.push(current.trim().to_string());
                current.clear();
            }
            continue;
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(trimmed);
    }
    if !current.is_empty() {
        paragraphs.push(current.trim().to_string());
    }
    paragraphs
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    for (idx, ch) in text.char_indices() {
        if is_sentence_terminator(ch) {
            let end = idx + ch.len_utf8();
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            start = end;
        }
    }
    let trailing = text[start..].trim();
    if !trailing.is_empty() {
        sentences.push(trailing.to_string());
    }
    sentences
}

fn is_sentence_terminator(ch: char) -> bool {
    matches!(
        ch,
        '.' | '!' | '?' | '。' | '！' | '？' | '۔' | '؟' | '।' | '॥'
    )
}

fn count_tokens(tokenizer: &Tokenizer, text: &str) -> Result<usize> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().len())
        .map_err(|err| anyhow::anyhow!("chunk token count failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_with_counter(
        doc: &str,
        cfg: ChunkingConfig,
        mut count_tokens: impl FnMut(&str) -> Result<usize>,
    ) -> Vec<String> {
        chunk_document_with_counter(0, doc, &cfg, &mut count_tokens)
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.text)
            .collect()
    }

    fn word_tokens(text: &str) -> Result<usize> {
        Ok(text.split_whitespace().count())
    }

    fn char_tokens(text: &str) -> Result<usize> {
        Ok(text.chars().filter(|ch| !ch.is_whitespace()).count())
    }

    #[test]
    fn short_document_yields_single_chunk() {
        let cfg = ChunkingConfig {
            max_tokens: 64,
            overlap: 0,
        };
        let chunks = chunk_with_counter("A short sentence about cats.", cfg, word_tokens);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("cats"));
    }

    #[test]
    fn empty_documents_produce_no_chunks() {
        let cfg = ChunkingConfig::default();
        let chunks = chunk_with_counter("   ", cfg, word_tokens);
        assert!(chunks.is_empty());
    }

    #[test]
    fn paragraph_boundaries_are_first_split() {
        let cfg = ChunkingConfig {
            max_tokens: 3,
            overlap: 0,
        };
        let chunks = chunk_with_counter("alpha beta.\n\ngamma delta.", cfg, word_tokens);
        assert_eq!(chunks, vec!["alpha beta.", "gamma delta."]);
    }

    #[test]
    fn sentence_boundaries_split_oversized_paragraphs() {
        let cfg = ChunkingConfig {
            max_tokens: 3,
            overlap: 0,
        };
        let chunks = chunk_with_counter("alpha beta. gamma delta.", cfg, word_tokens);
        assert_eq!(chunks, vec!["alpha beta.", "gamma delta."]);
    }

    #[test]
    fn long_sentence_falls_back_to_token_length_chunks() {
        let cfg = ChunkingConfig {
            max_tokens: 2,
            overlap: 0,
        };
        let chunks = chunk_with_counter("one two three four five", cfg, word_tokens);
        assert_eq!(chunks, vec!["one two", "three four", "five"]);
    }

    #[test]
    fn overlap_carries_suffix_units_into_next_chunk() {
        let cfg = ChunkingConfig {
            max_tokens: 4,
            overlap: 2,
        };
        let chunks = chunk_with_counter("a b. c d. e f.", cfg, word_tokens);
        assert_eq!(chunks, vec!["a b. c d.", "c d. e f."]);
    }

    #[test]
    fn cjk_sentence_punctuation_is_respected() {
        let cfg = ChunkingConfig {
            max_tokens: 5,
            overlap: 0,
        };
        let chunks = chunk_with_counter("你好世界。再见世界！", cfg, char_tokens);
        assert_eq!(chunks, vec!["你好世界。", "再见世界！"]);
    }
}
