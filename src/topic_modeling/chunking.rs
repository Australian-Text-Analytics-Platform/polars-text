//! Topic Segment construction for the topic-modeling pipeline.
//!
//! A Topic Segment is the unit embedded and clustered by the native topic-
//! modelling pipeline. Segmenting before embedding lets long documents
//! contribute several points and ultimately a multi-topic distribution.
//!
//! Automatic mode splits blank-line blocks, then Unicode sentences, then token-
//! length units, and packs them with a bounded overlap. Paragraph and Sentence
//! modes preserve their semantic boundaries and right-truncate oversized units.
//!
//! Called by: `topic_modeling::run` (orchestrator) before embedding.

use anyhow::{bail, Result};
use serde::Deserialize;
use tokenizers::Tokenizer;
use unicode_segmentation::UnicodeSegmentation;

/// Selects how source documents become Topic Segments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentationMethod {
    /// Hierarchically split and pack text up to the configured token budget.
    #[default]
    Automatic,
    /// Treat every non-empty newline-delimited line as one segment.
    Paragraph,
    /// Treat every Unicode UAX #29 sentence as one segment.
    Sentence,
}

/// One Topic Segment of a source document. `doc_index` ties the segment back to its
/// document so `rollup` can aggregate chunk topic assignments into a
/// per-document distribution; `chunk_index` is the chunk's ordinal within that
/// document (0-based) for stable ordering and debugging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub doc_index: usize,
    pub chunk_index: usize,
    pub text: String,
}

/// Segmentation knobs. `max_tokens` is the per-segment token budget and should stay
/// within the embedding model's context window; we default to 256 for a balance
/// of context and chunk count. `overlap` carries a few tokens across chunk seams
/// so a topic spanning a boundary is not split blind; ~10-20% of `max_tokens`
/// is the usual range. Overlap must be strictly less than `max_tokens`.
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub method: SegmentationMethod,
    pub max_tokens: usize,
    pub overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            method: SegmentationMethod::Automatic,
            max_tokens: 256,
            overlap: 32,
        }
    }
}

/// Topic Segments and run-level information produced during segmentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkingResult {
    pub chunks: Vec<Chunk>,
    pub truncated_count: usize,
}

#[derive(Debug, Clone)]
struct ChunkUnit {
    text: String,
    tokens: usize,
}

/// Split every document into token-budgeted Topic Segments.
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
) -> Result<ChunkingResult> {
    if cfg.max_tokens == 0 {
        bail!("chunking max_tokens must be > 0");
    }

    let mut chunks = Vec::with_capacity(docs.len());
    let mut truncated_count = 0;
    for (doc_index, doc) in docs.iter().enumerate() {
        match cfg.method {
            SegmentationMethod::Automatic => {
                chunks.extend(chunk_document_automatic(doc_index, doc, &tokenizer, cfg)?);
            }
            SegmentationMethod::Paragraph => {
                let units = doc.lines().map(str::trim).filter(|line| !line.is_empty());
                append_semantic_segments(
                    doc_index,
                    units,
                    &tokenizer,
                    cfg.max_tokens,
                    &mut chunks,
                    &mut truncated_count,
                )?;
            }
            SegmentationMethod::Sentence => {
                let units = doc.unicode_sentences().map(str::trim);
                append_semantic_segments(
                    doc_index,
                    units,
                    &tokenizer,
                    cfg.max_tokens,
                    &mut chunks,
                    &mut truncated_count,
                )?;
            }
        }
    }
    Ok(ChunkingResult {
        chunks,
        truncated_count,
    })
}

fn chunk_document_automatic(
    doc_index: usize,
    doc: &str,
    tokenizer: &Tokenizer,
    cfg: &ChunkingConfig,
) -> Result<Vec<Chunk>> {
    chunk_document_with_counter(doc_index, doc, cfg, &mut |text| {
        count_tokens(tokenizer, text)
    })
}

fn append_semantic_segments<'a>(
    doc_index: usize,
    units: impl Iterator<Item = &'a str>,
    tokenizer: &Tokenizer,
    max_tokens: usize,
    chunks: &mut Vec<Chunk>,
    truncated_count: &mut usize,
) -> Result<()> {
    let mut chunk_index = 0;
    for unit in units.filter(|unit| !unit.is_empty()) {
        let (text, was_truncated) = truncate_to_token_prefix(unit, tokenizer, max_tokens)?;
        if text.is_empty() {
            continue;
        }
        chunks.push(Chunk {
            doc_index,
            chunk_index,
            text,
        });
        chunk_index += 1;
        *truncated_count += usize::from(was_truncated);
    }
    Ok(())
}

fn truncate_to_token_prefix(
    text: &str,
    tokenizer: &Tokenizer,
    max_tokens: usize,
) -> Result<(String, bool)> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|err| anyhow::anyhow!("segment tokenization failed: {err}"))?;
    if encoding.len() <= max_tokens {
        return Ok((text.to_string(), false));
    }

    let end = encoding
        .get_offsets()
        .get(max_tokens - 1)
        .map(|(_, end)| *end)
        .ok_or_else(|| anyhow::anyhow!("segment tokenizer returned no truncation boundary"))?;
    if end == 0 || end > text.len() || !text.is_char_boundary(end) {
        bail!("segment tokenizer returned an invalid UTF-8 truncation boundary");
    }
    Ok((text[..end].trim_end().to_string(), true))
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
    text.unicode_sentences()
        .map(str::trim)
        .filter(|sentence| !sentence.is_empty())
        .map(str::to_string)
        .collect()
}

fn count_tokens(tokenizer: &Tokenizer, text: &str) -> Result<usize> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().len())
        .map_err(|err| anyhow::anyhow!("chunk token count failed: {err}"))
}

#[cfg(test)]
mod tests {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    use super::*;

    fn whitespace_tokenizer(words: &[&str]) -> Tokenizer {
        let vocab = std::iter::once((String::from("[UNK]"), 0))
            .chain(
                words
                    .iter()
                    .enumerate()
                    .map(|(index, word)| ((*word).to_string(), (index + 1) as u32)),
            )
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token(String::from("[UNK]"))
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
    }

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
            method: SegmentationMethod::Automatic,
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
            method: SegmentationMethod::Automatic,
            max_tokens: 3,
            overlap: 0,
        };
        let chunks = chunk_with_counter("alpha beta.\n\ngamma delta.", cfg, word_tokens);
        assert_eq!(chunks, vec!["alpha beta.", "gamma delta."]);
    }

    #[test]
    fn sentence_boundaries_split_oversized_paragraphs() {
        let cfg = ChunkingConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 3,
            overlap: 0,
        };
        let chunks = chunk_with_counter("alpha beta. Gamma delta.", cfg, word_tokens);
        assert_eq!(chunks, vec!["alpha beta.", "Gamma delta."]);
    }

    #[test]
    fn long_sentence_falls_back_to_token_length_chunks() {
        let cfg = ChunkingConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 2,
            overlap: 0,
        };
        let chunks = chunk_with_counter("one two three four five", cfg, word_tokens);
        assert_eq!(chunks, vec!["one two", "three four", "five"]);
    }

    #[test]
    fn overlap_carries_suffix_units_into_next_chunk() {
        let cfg = ChunkingConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 4,
            overlap: 2,
        };
        let chunks = chunk_with_counter("a b. C d. E f.", cfg, word_tokens);
        assert_eq!(chunks, vec!["a b. C d.", "C d. E f."]);
    }

    #[test]
    fn cjk_sentence_punctuation_is_respected() {
        let cfg = ChunkingConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 5,
            overlap: 0,
        };
        let chunks = chunk_with_counter("你好世界。再见世界！", cfg, char_tokens);
        assert_eq!(chunks, vec!["你好世界。", "再见世界！"]);
    }

    #[test]
    fn paragraph_mode_keeps_one_nonempty_line_and_truncates_its_tail() {
        let tokenizer = whitespace_tokenizer(&["one", "two", "three", "four", "five"]);
        let cfg = ChunkingConfig {
            method: SegmentationMethod::Paragraph,
            max_tokens: 2,
            overlap: 0,
        };

        let result = chunk_documents(
            &[String::from("one two three\n\n four five")],
            tokenizer,
            &cfg,
        )
        .unwrap();
        let texts = result
            .chunks
            .into_iter()
            .map(|chunk| chunk.text)
            .collect::<Vec<_>>();

        assert_eq!(
            (texts, result.truncated_count),
            (vec![String::from("one two"), String::from("four five")], 1)
        );
    }

    #[test]
    fn sentence_mode_uses_unicode_sentence_boundaries() {
        let tokenizer = whitespace_tokenizer(&["Value", "is", "3", "14", "Next", "item"]);
        let cfg = ChunkingConfig {
            method: SegmentationMethod::Sentence,
            max_tokens: 20,
            overlap: 0,
        };

        let result = chunk_documents(
            &[String::from("Value is 3.14. Next item.")],
            tokenizer,
            &cfg,
        )
        .unwrap();
        let texts = result
            .chunks
            .into_iter()
            .map(|chunk| chunk.text)
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["Value is 3.14.", "Next item."]);
    }

    #[test]
    fn sentence_mode_uses_uax_abbreviation_boundaries_without_heuristics() {
        let tokenizer = whitespace_tokenizer(&["Mr", "Fox", "jumped"]);
        let cfg = ChunkingConfig {
            method: SegmentationMethod::Sentence,
            max_tokens: 20,
            overlap: 0,
        };

        let result = chunk_documents(&[String::from("Mr. Fox jumped.")], tokenizer, &cfg).unwrap();
        let texts = result
            .chunks
            .into_iter()
            .map(|chunk| chunk.text)
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["Mr.", "Fox jumped."]);
    }

    #[test]
    fn semantic_truncation_preserves_utf8_boundaries() {
        let tokenizer = whitespace_tokenizer(&["猫", "狗", "鳥"]);
        let cfg = ChunkingConfig {
            method: SegmentationMethod::Paragraph,
            max_tokens: 2,
            overlap: 0,
        };

        let result = chunk_documents(&[String::from("猫 狗 鳥")], tokenizer, &cfg).unwrap();

        assert_eq!(result.chunks[0].text, "猫 狗");
        assert_eq!(result.truncated_count, 1);
    }
}
