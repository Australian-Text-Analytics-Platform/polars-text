//! Topic Segment construction for the topic-modeling pipeline.
//!
//! A Topic Segment is the unit embedded and clustered by the native topic-
//! modelling pipeline. Segmenting before embedding lets long documents
//! contribute several points and ultimately multi-Topic Coverage.
//!
//! All three modes keep semantic units whole when they fit the token budget and
//! never pack several units into one segment:
//! - Automatic: paragraphs, then sentences within an oversized paragraph.
//!   Paragraphs are blank-line blocks when the document has any blank line
//!   (single newlines inside them are line wrapping); otherwise every non-empty
//!   line is a paragraph.
//! - Line: every non-empty line, then sentences within an oversized line.
//! - Sentence: every Unicode UAX #29 sentence.
//!
//! A unit that is still over budget is split at the clause punctuation nearest
//! its middle, recursively, so pieces stay nearly equal and end at natural
//! pauses. Only a run with no usable punctuation is cut at the token budget
//! (preferring word starts), and a tiny leftover from such a cut is dropped.
//! Segments with no letters or digits (a stray quote mark, a lone full stop)
//! carry no topic content and are dropped in every mode.
//!
//! Called by: `topic_modeling::run` (orchestrator) before embedding.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tokenizers::Tokenizer;
use unicode_segmentation::UnicodeSegmentation;

/// A leftover chunk from a punctuation-free budget cut is dropped when it has
/// fewer content tokens than this (capped at half the budget, so tiny budgets
/// never discard everything).
const MIN_FRAGMENT_TOKENS: usize = 4;

/// Selects how source documents become Topic Segments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentationMethod {
    /// Paragraphs first, then sentences, then punctuation-balanced pieces.
    #[default]
    Automatic,
    /// Treat every non-empty newline-delimited line as one segment.
    Line,
    /// Treat every Unicode UAX #29 sentence as one segment.
    Sentence,
}

/// One non-overlapping source span embedded and clustered as a Topic Segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicSegment {
    pub doc_index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub owned_character_count: usize,
}

impl TopicSegment {
    pub fn text<'a>(&self, document: &'a str) -> Result<&'a str> {
        document
            .get(self.start_byte..self.end_byte)
            .with_context(|| {
                format!(
                    "Topic Segment byte range {}..{} is invalid for document {}",
                    self.start_byte, self.end_byte, self.doc_index
                )
            })
    }
}

/// Segmentation knobs. `max_tokens` is the per-segment embedding-model budget.
#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    pub method: SegmentationMethod,
    pub max_tokens: usize,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            method: SegmentationMethod::Automatic,
            max_tokens: 256,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TokenSpan {
    start: usize,
    end: usize,
}

/// Shared inputs for splitting one document's units into segments.
struct DocumentSplitter<'a> {
    doc_index: usize,
    doc: &'a str,
    tokens: &'a [TokenSpan],
    max_tokens: usize,
}

/// Split every document into token-budgeted Topic Segments.
///
/// Flow: tokenize each document once, select the configured semantic units,
/// and split only oversized units. A document with any letters or digits
/// yields at least one segment. Whitespace-only/empty documents yield zero
/// segments; the rollup stage maps those to an empty coverage value with `-1`
/// dominant.
///
/// The `tokenizer` must have truncation and padding disabled by the caller:
/// otherwise the sizer would cap segment sizes at the tokenizer's truncation
/// limit, or count padding as special tokens. `embedding` loads the tokenizer
/// and clears both before handing a clone here.
pub fn segment_documents(
    docs: &[&str],
    tokenizer: &Tokenizer,
    cfg: &SegmentationConfig,
) -> Result<Vec<TopicSegment>> {
    if cfg.max_tokens == 0 {
        bail!("segmentation max_tokens must be > 0");
    }

    let mut segments = Vec::with_capacity(docs.len());
    for (doc_index, &doc) in docs.iter().enumerate() {
        let (tokens, special_token_count) = token_spans(doc, tokenizer)?;
        let content_token_budget = cfg
            .max_tokens
            .checked_sub(special_token_count)
            .filter(|budget| *budget > 0)
            .with_context(|| {
                format!(
                    "segmentation max_tokens {} cannot fit the model's {special_token_count} special tokens",
                    cfg.max_tokens
                )
            })?;
        let splitter = DocumentSplitter {
            doc_index,
            doc,
            tokens: &tokens,
            max_tokens: content_token_budget,
        };
        match cfg.method {
            SegmentationMethod::Automatic => {
                for (start, end) in paragraph_ranges(doc) {
                    splitter.split_block(start, end, &mut segments)?;
                }
            }
            SegmentationMethod::Line => {
                for (start, end) in line_ranges(doc) {
                    splitter.split_block(start, end, &mut segments)?;
                }
            }
            SegmentationMethod::Sentence => {
                for (start, sentence) in doc.split_sentence_bound_indices() {
                    splitter.split_unit(start, start + sentence.len(), &mut segments)?;
                }
            }
        }
    }
    Ok(segments)
}

fn token_spans(text: &str, tokenizer: &Tokenizer) -> Result<(Vec<TokenSpan>, usize)> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|error| anyhow::anyhow!("segment tokenization failed: {error}"))?;
    if encoding.get_offsets().len() != encoding.get_special_tokens_mask().len() {
        bail!("segment tokenizer returned inconsistent offsets and special-token mask");
    }
    let special_token_count = encoding
        .get_special_tokens_mask()
        .iter()
        .filter(|&&special| special != 0)
        .count();
    let tokens = encoding
        .get_offsets()
        .iter()
        .zip(encoding.get_special_tokens_mask())
        .filter_map(|(&(start, end), &special)| (special == 0).then_some(TokenSpan { start, end }))
        .collect::<Vec<_>>();
    if tokens.iter().any(|token| {
        token.start >= token.end
            || token.end > text.len()
            || !text.is_char_boundary(token.start)
            || !text.is_char_boundary(token.end)
    }) || tokens
        .windows(2)
        .any(|pair| pair[0].start > pair[1].start || pair[0].end > pair[1].end)
    {
        bail!("segment tokenizer returned invalid or unordered byte offsets");
    }
    if tokens
        .len()
        .checked_add(special_token_count)
        .context("segment token count overflow")?
        != encoding.get_ids().len()
    {
        bail!("segment tokenizer returned a non-special token without a source span");
    }
    Ok((tokens, special_token_count))
}

impl DocumentSplitter<'_> {
    /// Emits a paragraph or line whole when it fits, otherwise its sentences.
    fn split_block(
        &self,
        start: usize,
        end: usize,
        segments: &mut Vec<TopicSegment>,
    ) -> Result<()> {
        let Some((start, end)) = trimmed_range(self.doc, start, end)? else {
            return Ok(());
        };
        if tokens_in_range(self.tokens, start, end).len() <= self.max_tokens {
            return self.push(start, end, segments);
        }
        let block = self
            .doc
            .get(start..end)
            .context("segmentation received an invalid source range")?;
        for (relative_start, sentence) in block.split_sentence_bound_indices() {
            let sentence_start = start + relative_start;
            self.split_unit(sentence_start, sentence_start + sentence.len(), segments)?;
        }
        Ok(())
    }

    /// Emits a sentence-level unit whole when it fits; otherwise splits it at
    /// the clause punctuation nearest its middle, recursively, and falls back
    /// to budget cuts only when no punctuation is usable.
    fn split_unit(&self, start: usize, end: usize, segments: &mut Vec<TopicSegment>) -> Result<()> {
        let Some((start, end)) = trimmed_range(self.doc, start, end)? else {
            return Ok(());
        };
        let unit_tokens = tokens_in_range(self.tokens, start, end);
        if unit_tokens.len() <= self.max_tokens {
            return self.push(start, end, segments);
        }
        match self.middle_punctuation_split(start, end, unit_tokens) {
            Some(split) => {
                self.split_unit(start, split, segments)?;
                self.split_unit(split, end, segments)
            }
            None => self.cut_by_budget(start, end, unit_tokens, segments),
        }
    }

    /// Byte offset just after the clause punctuation whose left side holds the
    /// token count closest to half the unit. Both sides must keep at least the
    /// tiny-fragment minimum, so an opening "However," is never split off.
    fn middle_punctuation_split(
        &self,
        start: usize,
        end: usize,
        unit_tokens: &[TokenSpan],
    ) -> Option<usize> {
        let unit = self.doc.get(start..end)?;
        let half = unit_tokens.len() as f64 / 2.0;
        let min_side = self.min_fragment_tokens();
        let mut best: Option<(f64, usize)> = None;
        let mut chars = unit.char_indices().peekable();
        while let Some((offset, character)) = chars.next() {
            if !is_clause_punctuation(character) {
                continue;
            }
            let mut split = start + offset + character.len_utf8();
            // Keep closing quotes and brackets with the clause they close.
            while let Some(&(next_offset, next)) = chars.peek() {
                if !is_closing_mark(next) {
                    break;
                }
                split = start + next_offset + next.len_utf8();
                chars.next();
            }
            let followed_by_space = self.doc[split..end]
                .chars()
                .next()
                .is_some_and(char::is_whitespace);
            if !(followed_by_space || is_wide_punctuation(character)) {
                continue;
            }
            let left = unit_tokens.partition_point(|token| token.end <= split);
            if left < min_side || unit_tokens.len() - left < min_side {
                continue;
            }
            let distance = (left as f64 - half).abs();
            if best.is_none_or(|(best_distance, _)| distance < best_distance) {
                best = Some((distance, split));
            }
        }
        best.map(|(_, split)| split)
    }

    /// Cuts a punctuation-free run into budget-sized chunks that start at word
    /// starts where possible, dropping a tiny leftover.
    fn cut_by_budget(
        &self,
        start: usize,
        end: usize,
        unit_tokens: &[TokenSpan],
        segments: &mut Vec<TopicSegment>,
    ) -> Result<()> {
        let min_fragment = self.min_fragment_tokens();
        let mut first = 0usize;
        while first < unit_tokens.len() {
            let mut next = (first + self.max_tokens).min(unit_tokens.len());
            if next < unit_tokens.len() {
                if let Some(word_start) = (first + 1..=next)
                    .rev()
                    .find(|&index| self.starts_word(unit_tokens[index].start))
                {
                    next = word_start;
                }
            }
            let span_start = if first == 0 {
                start
            } else {
                unit_tokens[first].start
            };
            let span_end = unit_tokens.get(next).map_or(end, |token| token.start);
            let is_leftover = next == unit_tokens.len() && first > 0;
            if !(is_leftover && next - first < min_fragment) {
                if let Some((span_start, span_end)) = trimmed_range(self.doc, span_start, span_end)?
                {
                    self.push(span_start, span_end, segments)?;
                }
            }
            first = next;
        }
        Ok(())
    }

    /// Smallest piece worth keeping, capped at half the budget so tiny budgets
    /// never discard everything.
    fn min_fragment_tokens(&self) -> usize {
        MIN_FRAGMENT_TOKENS.min(self.max_tokens.div_ceil(2)).max(1)
    }

    fn starts_word(&self, byte: usize) -> bool {
        byte == 0
            || self.doc[..byte]
                .chars()
                .next_back()
                .is_some_and(char::is_whitespace)
    }

    /// Records a segment unless it has no letters or digits.
    fn push(&self, start: usize, end: usize, segments: &mut Vec<TopicSegment>) -> Result<()> {
        let text = self
            .doc
            .get(start..end)
            .context("Topic Segment received an invalid source range")?;
        if !text.chars().any(char::is_alphanumeric) {
            return Ok(());
        }
        segments.push(TopicSegment {
            doc_index: self.doc_index,
            start_byte: start,
            end_byte: end,
            owned_character_count: text.chars().count(),
        });
        Ok(())
    }
}

/// Punctuation that ends a clause and makes a natural split point.
fn is_clause_punctuation(character: char) -> bool {
    matches!(
        character,
        ',' | ';' | ':' | '.' | '!' | '?' | '\u{2014}' | '\u{2013}' | '\u{2026}'
    ) || is_wide_punctuation(character)
}

/// Full-width CJK punctuation, which is not followed by a space.
fn is_wide_punctuation(character: char) -> bool {
    matches!(
        character,
        '\u{3001}' | '\u{3002}' | '\u{FF0C}' | '\u{FF1B}' | '\u{FF1A}' | '\u{FF01}' | '\u{FF1F}'
    )
}

fn is_closing_mark(character: char) -> bool {
    matches!(
        character,
        '"' | '\''
            | ')'
            | ']'
            | '}'
            | '\u{201D}'
            | '\u{2019}'
            | '\u{300D}'
            | '\u{300F}'
            | '\u{FF09}'
    )
}

/// Paragraph ranges for Automatic mode: blank-line blocks when the document has
/// a blank line, otherwise its non-empty lines.
fn paragraph_ranges(text: &str) -> Vec<(usize, usize)> {
    let lines = line_ranges(text);
    let has_blank_line = lines
        .iter()
        .any(|&(start, end)| text[start..end].trim().is_empty())
        && lines
            .iter()
            .filter(|&&(start, end)| !text[start..end].trim().is_empty())
            .count()
            > 1;
    if !has_blank_line {
        return lines;
    }
    let mut paragraphs = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    for (start, end) in lines {
        if text[start..end].trim().is_empty() {
            if let Some(paragraph) = open.take() {
                paragraphs.push(paragraph);
            }
        } else {
            open = Some(open.map_or((start, end), |(first, _)| (first, end)));
        }
    }
    paragraphs.extend(open);
    paragraphs
}

fn line_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for line in text.split_inclusive('\n') {
        let end = start + line.strip_suffix('\n').map_or(line.len(), str::len);
        ranges.push((start, end));
        start += line.len();
    }
    if start < text.len() {
        ranges.push((start, text.len()));
    }
    ranges
}

fn tokens_in_range(tokens: &[TokenSpan], start: usize, end: usize) -> &[TokenSpan] {
    let first = tokens.partition_point(|token| token.end <= start);
    let last = tokens.partition_point(|token| token.start < end);
    &tokens[first.min(last)..last]
}

fn trimmed_range(text: &str, start: usize, end: usize) -> Result<Option<(usize, usize)>> {
    let value = text
        .get(start..end)
        .with_context(|| format!("invalid source byte range {start}..{end}"))?;
    let leading = value.len() - value.trim_start().len();
    let trailing = value.len() - value.trim_end().len();
    let start = start + leading;
    let end = end.saturating_sub(trailing);
    Ok((start < end).then_some((start, end)))
}

#[cfg(test)]
mod tests {
    use tokenizers::models::bpe::BPE;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
    use tokenizers::processors::bert::BertProcessing;

    use super::*;

    fn whitespace_tokenizer(words: &[&str]) -> Tokenizer {
        let model = WordLevel::builder()
            .vocab(
                std::iter::once((String::from("[UNK]"), 0))
                    .chain(
                        words
                            .iter()
                            .enumerate()
                            .map(|(index, word)| ((*word).to_string(), (index + 1) as u32)),
                    )
                    .collect(),
            )
            .unk_token(String::from("[UNK]"))
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer
    }

    fn word_segments(doc: &str, cfg: SegmentationConfig) -> Vec<String> {
        let words = doc.split_whitespace().collect::<Vec<_>>();
        let tokenizer = whitespace_tokenizer(&words);
        segment_documents(&[doc], &tokenizer, &cfg)
            .unwrap()
            .into_iter()
            .map(|segment| segment.text(doc).unwrap().to_string())
            .collect()
    }

    fn character_tokenizer(text: &str) -> Tokenizer {
        let vocab: ahash::AHashMap<String, u32> = std::iter::once((String::from("[UNK]"), 0))
            .chain(
                text.chars()
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .enumerate()
                    .map(|(index, character)| (character.to_string(), (index + 1) as u32)),
            )
            .collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token(String::from("[UNK]"))
            .build()
            .unwrap();
        Tokenizer::new(model)
    }

    fn bert_whitespace_tokenizer(words: &[&str]) -> Tokenizer {
        let mut tokenizer = whitespace_tokenizer(
            &["[CLS]", "[SEP]"]
                .into_iter()
                .chain(words.iter().copied())
                .collect::<Vec<_>>(),
        );
        let cls = tokenizer.token_to_id("[CLS]").unwrap();
        let sep = tokenizer.token_to_id("[SEP]").unwrap();
        tokenizer.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".into(), sep),
            ("[CLS]".into(), cls),
        )));
        tokenizer
    }

    #[test]
    fn short_document_yields_single_segment() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 64,
        };
        let segments = word_segments("A short sentence about cats.", cfg);
        assert_eq!(segments.len(), 1);
        assert!(segments[0].contains("cats"));
    }

    #[test]
    fn empty_documents_produce_no_segments() {
        let cfg = SegmentationConfig::default();
        let segments = word_segments("   ", cfg);
        assert!(segments.is_empty());
    }

    #[test]
    fn paragraph_boundaries_are_first_split() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 3,
        };
        let segments = word_segments("alpha beta.\n\ngamma delta.", cfg);
        assert_eq!(segments, vec!["alpha beta.", "gamma delta."]);
    }

    #[test]
    fn sentence_boundaries_split_oversized_paragraphs() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 3,
        };
        let segments = word_segments("alpha beta. Gamma delta.", cfg);
        assert_eq!(segments, vec!["alpha beta.", "Gamma delta."]);
    }

    #[test]
    fn long_sentence_falls_back_to_token_length_segments() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 2,
        };
        let segments = word_segments("one two three four five", cfg);
        assert_eq!(segments, vec!["one two", "three four", "five"]);
    }

    #[test]
    fn segment_budget_includes_model_special_tokens() {
        let tokenizer = bert_whitespace_tokenizer(&["one", "two", "three", "four", "five"]);
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 4,
        };
        let doc = "one two three four five";
        let result = segment_documents(&[doc], &tokenizer, &cfg).unwrap();
        let texts = result
            .iter()
            .map(|segment| segment.text(doc).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["one two", "three four", "five"]);
        assert!(texts.iter().all(|text| {
            tokenizer.encode(*text, true).unwrap().get_ids().len() <= cfg.max_tokens
        }));
    }

    #[test]
    fn automatic_segments_preserve_interior_source_spacing() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 2,
        };
        let segments = word_segments("one \t two   three four", cfg);
        assert_eq!(segments, vec!["one \t two", "three four"]);
    }

    #[test]
    fn automatic_keeps_sentences_separate_instead_of_packing() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 4,
        };
        let segments = word_segments("a b. C d. E f.", cfg);
        assert_eq!(segments, vec!["a b.", "C d.", "E f."]);
    }

    #[test]
    fn automatic_uses_single_newlines_as_paragraphs_without_blank_lines() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 64,
        };
        let segments = word_segments("alpha beta.\ngamma delta.", cfg);
        assert_eq!(segments, vec!["alpha beta.", "gamma delta."]);
    }

    #[test]
    fn automatic_treats_single_newlines_as_wrapping_when_blank_lines_exist() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 64,
        };
        let segments = word_segments("alpha\nbeta.\n\ngamma delta.", cfg);
        assert_eq!(segments, vec!["alpha\nbeta.", "gamma delta."]);
    }

    #[test]
    fn oversized_units_split_at_the_middle_punctuation() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Sentence,
            max_tokens: 6,
        };
        let segments = word_segments("one two, three four, five six, seven eight.", cfg);
        assert_eq!(
            segments,
            vec!["one two, three four,", "five six, seven eight."]
        );
    }

    #[test]
    fn punctuation_near_an_edge_does_not_split_off_a_fragment() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Sentence,
            max_tokens: 8,
        };
        let segments = word_segments("However, w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13", cfg);
        assert!(
            segments.iter().all(|segment| segment != "However,"),
            "{segments:?}"
        );
    }

    #[test]
    fn punctuation_free_cuts_drop_a_tiny_leftover() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Sentence,
            max_tokens: 8,
        };
        let segments = word_segments("w1 w2 w3 w4 w5 w6 w7 w8 w9", cfg);
        assert_eq!(segments, vec!["w1 w2 w3 w4 w5 w6 w7 w8"]);
    }

    #[test]
    fn segments_without_letters_or_digits_are_dropped() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Line,
            max_tokens: 64,
        };
        let segments = word_segments("hello world\n\"\n.", cfg);
        assert_eq!(segments, vec!["hello world"]);
    }

    #[test]
    fn cjk_sentence_punctuation_is_respected() {
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Automatic,
            max_tokens: 5,
        };
        let doc = "你好世界。再见世界！";
        let tokenizer = character_tokenizer(doc);
        let segments = segment_documents(&[doc], &tokenizer, &cfg)
            .unwrap()
            .into_iter()
            .map(|segment| segment.text(doc).unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(segments, vec!["你好世界。", "再见世界！"]);
    }

    #[test]
    fn line_mode_recursively_splits_oversized_lines_without_losing_the_tail() {
        let tokenizer = whitespace_tokenizer(&["one", "two", "three", "four", "five"]);
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Line,
            max_tokens: 2,
        };

        let doc = "one two three\n\n four five";
        let result = segment_documents(&[doc], &tokenizer, &cfg).unwrap();
        let texts = result
            .into_iter()
            .map(|segment| segment.text(doc).unwrap().to_string())
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["one two", "three", "four five"]);
    }

    #[test]
    fn sentence_mode_uses_unicode_sentence_boundaries() {
        let tokenizer = whitespace_tokenizer(&["Value", "is", "3", "14", "Next", "item"]);
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Sentence,
            max_tokens: 20,
        };

        let doc = "Value is 3.14. Next item.";
        let result = segment_documents(&[doc], &tokenizer, &cfg).unwrap();
        let texts = result
            .into_iter()
            .map(|segment| segment.text(doc).unwrap().to_string())
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["Value is 3.14.", "Next item."]);
    }

    #[test]
    fn sentence_mode_uses_uax_abbreviation_boundaries_without_heuristics() {
        let tokenizer = whitespace_tokenizer(&["Mr", "Fox", "jumped"]);
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Sentence,
            max_tokens: 20,
        };

        let doc = "Mr. Fox jumped.";
        let result = segment_documents(&[doc], &tokenizer, &cfg).unwrap();
        let texts = result
            .into_iter()
            .map(|segment| segment.text(doc).unwrap().to_string())
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["Mr.", "Fox jumped."]);
    }

    #[test]
    fn line_mode_preserves_utf8_and_all_oversized_content() {
        let tokenizer = whitespace_tokenizer(&["猫", "狗", "鳥"]);
        let cfg = SegmentationConfig {
            method: SegmentationMethod::Line,
            max_tokens: 2,
        };

        let doc = "猫 狗 鳥";
        let result = segment_documents(&[doc], &tokenizer, &cfg).unwrap();
        let texts = result
            .iter()
            .map(|segment| segment.text(doc).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(texts, vec!["猫 狗", "鳥"]);
    }
}
