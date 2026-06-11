//! c-TF-IDF topic labeling: pick the words that best characterize each topic.
//!
//! Why this exists: a topic is just a set of chunks until it has human-readable
//! keywords. BERTopic's class-based TF-IDF (c-TF-IDF) treats each topic as one
//! "document" (the concatenation of its chunks) and scores a term by how
//! frequent it is *within* the topic versus *across* the whole corpus, so terms
//! that are common everywhere (and thus uninformative) are down-weighted without
//! needing a hand-tuned stopword list — though we still accept one for the
//! residual function words a multilingual tokenizer leaves behind.
//!
//! Formula (BERTopic's `ClassTfidfTransformer`):
//!   tf(t, c)  = count(t in c) / total_words(c)          (within-topic frequency)
//!   idf(t)    = ln(1 + A / f(t))                         (A = avg words/topic,
//!                                                          f(t) = corpus freq)
//!   score     = tf(t, c) * idf(t)
//! Top-`k` scored terms per topic become its representative words.
//!
//! Determinism: the math is a pure function of the term counts; ties are broken
//! alphabetically so the same counts always yield the same ordering. This is the
//! part we unit-test. The tokenization helper depends on downloaded model files
//! and is exercised by the manual harness, not CI.
//!
//! Called by: `topic_modeling::run` after clustering, once chunk texts are
//! grouped by topic.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;

use crate::tokenizer::{ensure_tokenizer_for_model, TokenizerBackend};

/// Labeling knobs. `top_k` is how many representative words to keep per topic
/// (maps to the backend's `representative_words_count`).
#[derive(Debug, Clone)]
pub struct CtfidfConfig {
    pub top_k: usize,
}

impl Default for CtfidfConfig {
    fn default() -> Self {
        Self { top_k: 10 }
    }
}

/// Score terms per topic with c-TF-IDF and return the top-`k` `(word, score)`
/// pairs for each topic, highest score first.
///
/// `per_topic_counts[i]` is the term→count map for topic `i` (already tokenized
/// and stopword-filtered by the caller). This is the deterministic core.
///
/// Flow: derive per-topic word totals and the corpus-wide term frequency, then
/// for every term in every topic compute `tf * idf`, sort each topic's terms by
/// score (alphabetical tie-break for stable output), and truncate to `top_k`.
pub fn ctfidf_scores(
    per_topic_counts: &[HashMap<String, usize>],
    cfg: &CtfidfConfig,
) -> Vec<Vec<(String, f32)>> {
    let n_topics = per_topic_counts.len();
    if n_topics == 0 {
        return Vec::new();
    }

    // Words per topic and the average (A) used in the idf term.
    let words_per_topic: Vec<usize> = per_topic_counts
        .iter()
        .map(|counts| counts.values().sum())
        .collect();
    let total_words: usize = words_per_topic.iter().sum();
    let avg_words = total_words as f64 / n_topics as f64;

    // Corpus-wide frequency of each term across all topics.
    let mut corpus_freq: HashMap<&str, usize> = HashMap::new();
    for counts in per_topic_counts {
        for (term, &c) in counts {
            *corpus_freq.entry(term.as_str()).or_insert(0) += c;
        }
    }

    per_topic_counts
        .iter()
        .zip(&words_per_topic)
        .map(|(counts, &words)| {
            if words == 0 {
                return Vec::new();
            }
            let mut scored: Vec<(String, f32)> = counts
                .iter()
                .map(|(term, &count)| {
                    let tf = count as f64 / words as f64;
                    let f_t = corpus_freq[term.as_str()] as f64;
                    let idf = (1.0 + avg_words / f_t).ln();
                    (term.clone(), (tf * idf) as f32)
                })
                .collect();
            // Highest score first; alphabetical tie-break keeps output stable.
            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            scored.truncate(cfg.top_k);
            scored
        })
        .collect()
}

/// Tokenize each topic's concatenated text into a term→count map using the
/// shared multilingual `TokenizerBackend`, dropping stopwords and (via the
/// backend) punctuation. Mirrors the current Python pipeline's lindera-based
/// "vectorizer corpora" so CJK topics get word-segmented, not split per byte.
///
/// `model_id` selects the segmentation backend (`lindera:jieba` for Chinese,
/// `native:plain_words_en` for English, a HF id for WordPiece, etc.); `None`
/// uses the registry default. Returns one map per input topic, aligned by index.
pub fn count_topic_terms(
    topic_texts: &[String],
    model_id: Option<&str>,
    lowercase: bool,
    stopwords: &HashSet<String>,
) -> Result<Vec<HashMap<String, usize>>> {
    let backend: Arc<TokenizerBackend> = ensure_tokenizer_for_model(model_id)?;
    topic_texts
        .iter()
        .map(|text| {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for tok in backend.tokenize_text(text, false, lowercase, true)? {
                if stopwords.contains(&tok) {
                    continue;
                }
                *counts.entry(tok).or_insert(0) += 1;
            }
            Ok(counts)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(pairs: &[(&str, usize)]) -> HashMap<String, usize> {
        pairs.iter().map(|(t, c)| (t.to_string(), *c)).collect()
    }

    /// A term unique to one topic must outscore a term shared by every topic,
    /// because the idf factor collapses toward zero for ubiquitous terms. Hand
    /// values make this deterministic and CI-safe.
    #[test]
    fn distinctive_terms_outrank_ubiquitous_terms() {
        // "shared" appears in both topics; "alpha"/"beta" are topic-specific.
        let topic_a = counts(&[("alpha", 5), ("shared", 5)]);
        let topic_b = counts(&[("beta", 5), ("shared", 5)]);
        let res = ctfidf_scores(&[topic_a, topic_b], &CtfidfConfig { top_k: 5 });

        assert_eq!(res.len(), 2);
        // Topic A's top word is its distinctive term, not the shared one.
        assert_eq!(res[0][0].0, "alpha");
        assert_eq!(res[1][0].0, "beta");
        // The shared term scores strictly lower within each topic.
        let a_alpha = res[0].iter().find(|(t, _)| t == "alpha").unwrap().1;
        let a_shared = res[0].iter().find(|(t, _)| t == "shared").unwrap().1;
        assert!(a_alpha > a_shared);
    }

    #[test]
    fn respects_top_k_and_is_sorted_descending() {
        let topic = counts(&[("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
        let res = ctfidf_scores(&[topic], &CtfidfConfig { top_k: 2 });
        assert_eq!(res[0].len(), 2);
        assert!(res[0][0].1 >= res[0][1].1);
    }

    #[test]
    fn empty_topic_yields_no_words() {
        let res = ctfidf_scores(&[HashMap::new()], &CtfidfConfig::default());
        assert_eq!(res, vec![Vec::<(String, f32)>::new()]);
    }
}
