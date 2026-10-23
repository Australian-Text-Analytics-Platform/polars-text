//! c-TF-IDF topic labeling: pick the words that best characterize each topic.
//!
//! Why this exists: a topic is just a set of Topic Segments until it has human-readable
//! keywords. BERTopic's class-based TF-IDF (c-TF-IDF) treats each topic as one
//! "document" (the concatenation of its Topic Segments) and scores a term by how
//! frequent it is *within* the topic versus *across* the whole corpus, so terms
//! that are common everywhere (and thus uninformative) are down-weighted without
//! needing a hand-tuned stopword list. User stopwords are a presentation concern
//! and do not alter this model output.
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
//! Called by: `topic_modeling::run` after clustering. Counts are accumulated
//! directly from non-overlapping Topic Segment source views.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::tokenizer::{ensure_tokenizer_for_model, TokenizerBackend};

/// Fixed candidate capacity retained for presentation-time filtering.
pub const REPRESENTATIVE_WORD_CANDIDATE_LIMIT: usize = 100;

/// A representative term in c-TF-IDF order.
#[derive(Debug, Clone, Serialize)]
pub struct RepresentativeWord {
    pub word: String,
    pub occurrence_count: usize,
    /// Used only while ranking. The plugin and serialized pipeline result omit it.
    #[serde(skip)]
    pub(crate) score: f32,
}

/// Score terms per topic with c-TF-IDF and return the fixed candidate set for
/// each topic, highest score first.
///
/// `per_topic_counts[i]` is the term→count map for topic `i` (already tokenized
/// by the caller). This is the deterministic core.
///
/// Flow: derive per-topic word totals and the corpus-wide term frequency, then
/// for every term in every topic compute `tf * idf`, sort each topic's terms by
/// score (alphabetical tie-break for stable output), and truncate to the fixed
/// representative-word candidate limit.
pub fn representative_words(
    per_topic_counts: &[HashMap<String, usize>],
) -> Vec<Vec<RepresentativeWord>> {
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
            let mut scored: Vec<RepresentativeWord> = counts
                .iter()
                .map(|(term, &count)| {
                    let tf = count as f64 / words as f64;
                    let f_t = corpus_freq[term.as_str()] as f64;
                    let idf = (1.0 + avg_words / f_t).ln();
                    RepresentativeWord {
                        word: term.clone(),
                        occurrence_count: count,
                        score: (tf * idf) as f32,
                    }
                })
                .collect();
            // Highest score first; alphabetical tie-break keeps output stable.
            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.word.cmp(&b.word))
            });
            scored.truncate(REPRESENTATIVE_WORD_CANDIDATE_LIMIT);
            scored
        })
        .collect()
}

/// Tokenize each topic's concatenated text into a term→count map using the
/// shared multilingual `TokenizerBackend`, dropping punctuation via the
/// backend. Mirrors the current Python pipeline's lindera-based
/// "vectorizer corpora" so CJK topics get word-segmented, not split per byte.
///
/// `model_id` selects the segmentation backend (`lindera:jieba` for Chinese,
/// `native:plain_words_en` for English, a HF id for WordPiece, etc.). Returns
/// one map per input topic, aligned by index.
pub fn count_topic_terms<'a>(
    topic_count: usize,
    assigned_segments: impl IntoIterator<Item = (i32, &'a str)>,
    model_id: &str,
    lowercase: bool,
) -> Result<Vec<HashMap<String, usize>>> {
    use rayon::prelude::*;

    let backend: Arc<TokenizerBackend> = ensure_tokenizer_for_model(model_id)?;
    let mut segments = Vec::new();
    for (label, text) in assigned_segments {
        if label < 0 {
            continue;
        }
        let topic = usize::try_from(label)
            .map_err(|_| anyhow::anyhow!("Topic label {label} is invalid"))?;
        anyhow::ensure!(
            topic < topic_count,
            "Topic label {label} is outside the topic count"
        );
        segments.push((topic, text));
    }
    // Tokenising every segment is most of this step, so it runs on the same
    // processors as segmentation; counts are summed, so the order of the
    // parts doesn't change the result (Wordflow issue 431).
    let workers = super::embedding::computation_threads()
        .min(segments.len())
        .max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("start the topic word workers")?;
    let part_size = segments.len().div_ceil(workers * 4).max(1);
    let parts: Vec<Result<Vec<HashMap<String, usize>>>> = pool.install(|| {
        segments
            .par_chunks(part_size)
            .map(|part| {
                let mut per_topic = vec![HashMap::new(); topic_count];
                for &(topic, text) in part {
                    let counts: &mut HashMap<String, usize> = &mut per_topic[topic];
                    for token in backend.tokenize_text(text, lowercase, true)? {
                        *counts.entry(token).or_insert(0) += 1;
                    }
                }
                Ok(per_topic)
            })
            .collect()
    });
    let mut per_topic = vec![HashMap::new(); topic_count];
    for part in parts {
        for (total, counts) in per_topic.iter_mut().zip(part?) {
            for (token, count) in counts {
                *total.entry(token).or_insert(0) += count;
            }
        }
    }
    Ok(per_topic)
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
        let res = representative_words(&[topic_a, topic_b]);

        assert_eq!(res.len(), 2);
        // Topic A's top word is its distinctive term, not the shared one.
        assert_eq!(res[0][0].word, "alpha");
        assert_eq!(res[1][0].word, "beta");
        // The shared term scores strictly lower within each topic.
        let a_alpha = res[0]
            .iter()
            .find(|term| term.word == "alpha")
            .unwrap()
            .score;
        let a_shared = res[0]
            .iter()
            .find(|term| term.word == "shared")
            .unwrap()
            .score;
        assert!(a_alpha > a_shared);
    }

    #[test]
    fn retains_counts_and_truncates_to_fixed_candidate_limit() {
        let topic = (0..105)
            .map(|index| (format!("term-{index:03}"), index + 1))
            .collect::<HashMap<_, _>>();
        let res = representative_words(&[topic]);
        assert_eq!(res[0].len(), REPRESENTATIVE_WORD_CANDIDATE_LIMIT);
        assert!(res[0][0].score >= res[0][1].score);
        assert!(res[0].iter().all(|term| term.occurrence_count > 0));
        assert_eq!(res[0][0].occurrence_count, 105);
    }

    #[test]
    fn breaks_equal_score_ties_alphabetically() {
        let res = representative_words(&[counts(&[("beta", 2), ("alpha", 2)])]);
        assert_eq!(
            res[0]
                .iter()
                .map(|term| term.word.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );
    }

    /// The parallel count (Wordflow issue 431) equals one sequential pass,
    /// including outliers skipped and words repeated across parts.
    #[test]
    fn parallel_counts_match_a_single_pass() {
        use crate::tokenizer::{ensure_tokenizer_for_model, PLAIN_WORDS_EN_MODEL_ID};

        let words = [
            "housing", "budget", "cricket", "reef", "the", "council", "final",
        ];
        let texts: Vec<String> = (0..2_000)
            .map(|i| {
                (0..(i % 9 + 1))
                    .map(|j| words[(i * 7 + j * 3) % words.len()])
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        let labels: Vec<i32> = (0..texts.len()).map(|i| (i % 5) as i32 - 1).collect();
        let got = count_topic_terms(
            4,
            labels.iter().copied().zip(texts.iter().map(String::as_str)),
            PLAIN_WORDS_EN_MODEL_ID,
            true,
        )
        .unwrap();
        let backend = ensure_tokenizer_for_model(PLAIN_WORDS_EN_MODEL_ID).unwrap();
        let mut want = vec![HashMap::new(); 4];
        for (label, text) in labels.iter().zip(&texts) {
            if *label < 0 {
                continue;
            }
            for token in backend.tokenize_text(text, true, true).unwrap() {
                *want[*label as usize].entry(token).or_insert(0usize) += 1;
            }
        }
        assert_eq!(got, want);
        assert!(count_topic_terms(2, [(3, "reef")], PLAIN_WORDS_EN_MODEL_ID, true).is_err());
    }

    #[test]
    fn empty_topic_yields_no_words() {
        let res = representative_words(&[HashMap::new()]);
        assert_eq!(res.len(), 1);
        assert!(res[0].is_empty());
    }
}
