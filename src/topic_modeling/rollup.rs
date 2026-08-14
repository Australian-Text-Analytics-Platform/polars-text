//! Roll chunk-level topic assignments up to per-document topic distributions.
//!
//! Why this exists: clustering labels *chunks*, but the product surfaces
//! *documents*. A long document legitimately spans several topics, so instead of
//! collapsing it to a single id (the old BERTopic behavior that the long-text
//! redesign explicitly removes), we report the normalized mix of topics across
//! its segments. Each segment contributes the Unicode-character length of its
//! retained text. A short document yields one segment and therefore a near-one-
//! hot distribution — the same code path, no length branching.
//!
//! Outlier handling: HDBSCAN's `-1` chunks are kept in the proportions so they
//! sum to 1 over every weighted segment of the document (an all-outlier document is honest
//! about being unclusterable), but `dominant_topic` prefers a real topic and
//! only falls back to `-1` when the document has no clustered chunk at all.
//!
//! Determinism: pure counting; distributions are emitted in ascending topic-id
//! order and dominant-topic ties break to the smaller id, so identical input
//! always produces identical output. Fully unit-tested.
//!
//! Called by: `topic_modeling::run` after clustering, feeding the per-document
//! payload.

use std::cmp::Reverse;
use std::collections::BTreeMap;

use crate::topic_modeling::cluster::OUTLIER_LABEL;

/// One topic's share of a document, proportions summing to 1 across the
/// document's retained segment characters. `topic_id` may be `OUTLIER_LABEL`
/// (`-1`).
#[derive(Debug, Clone, PartialEq)]
pub struct TopicProportion {
    pub topic_id: i32,
    pub proportion: f32,
}

/// Per-document result: the full distribution plus the single dominant topic
/// kept for coloring, back-compat detach columns, and the exact-count controls.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentTopics {
    pub topic_distribution: Vec<TopicProportion>,
    pub dominant_topic: i32,
}

/// Aggregate chunk labels into one `DocumentTopics` per document.
///
/// `n_docs` is the document count (some documents may own zero chunks, e.g.
/// empty/whitespace input — those get an empty distribution and `-1` dominant).
/// `segment_doc_index[i]` is the owning document of segment `i`;
/// `segment_labels[i]` is its topic; and `segment_weights[i]` is the Unicode-
/// character length of its retained text. The three segment slices must have
/// the same length.
///
/// Flow: sum each document's segment weights by label, divide by the total
/// retained-character weight to get proportions, choose the highest-weight
/// non-outlier topic as dominant, and emit the distribution sorted by topic id.
pub fn rollup(
    n_docs: usize,
    segment_doc_index: &[usize],
    segment_labels: &[i32],
    segment_weights: &[usize],
) -> Vec<DocumentTopics> {
    debug_assert_eq!(segment_doc_index.len(), segment_labels.len());
    debug_assert_eq!(segment_doc_index.len(), segment_weights.len());

    // Per-document topic weights in ascending-id order (BTreeMap = stable output).
    let mut per_doc: Vec<BTreeMap<i32, usize>> = vec![BTreeMap::new(); n_docs];
    for ((&doc, &label), &weight) in segment_doc_index
        .iter()
        .zip(segment_labels)
        .zip(segment_weights)
    {
        if doc < n_docs {
            *per_doc[doc].entry(label).or_insert(0) += weight;
        }
    }

    per_doc
        .into_iter()
        .map(|weights| {
            let total: usize = weights.values().sum();
            if total == 0 {
                return DocumentTopics {
                    topic_distribution: Vec::new(),
                    dominant_topic: OUTLIER_LABEL,
                };
            }

            let distribution: Vec<TopicProportion> = weights
                .iter()
                .map(|(&topic_id, &weight)| TopicProportion {
                    topic_id,
                    proportion: weight as f32 / total as f32,
                })
                .collect();

            // Dominant = most-represented real topic; outliers only win if the
            // document has no clustered chunk at all. BTreeMap iteration is
            // ascending, so the first max found is the smallest-id winner.
            let dominant_topic = weights
                .iter()
                .filter(|(&id, _)| id != OUTLIER_LABEL)
                .min_by_key(|(&id, &weight)| (Reverse(weight), id))
                .map(|(&id, _)| id)
                .unwrap_or(OUTLIER_LABEL);

            DocumentTopics {
                topic_distribution: distribution,
                dominant_topic,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_doc_gets_multi_topic_distribution() {
        // Document 0 has 4 chunks: topics 0,0,1,-1.
        let docs = rollup(1, &[0, 0, 0, 0], &[0, 0, 1, OUTLIER_LABEL], &[1, 1, 1, 1]);
        assert_eq!(docs.len(), 1);
        let d = &docs[0];
        assert_eq!(d.dominant_topic, 0); // topic 0 has the most chunks
                                         // Proportions: topic 0 = 0.5, topic 1 = 0.25, outlier = 0.25, sorted asc.
        assert_eq!(d.topic_distribution.len(), 3);
        assert_eq!(d.topic_distribution[0].topic_id, OUTLIER_LABEL);
        let sum: f32 = d.topic_distribution.iter().map(|p| p.proportion).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn short_doc_is_near_one_hot() {
        let docs = rollup(1, &[0], &[2], &[5]);
        assert_eq!(docs[0].dominant_topic, 2);
        assert_eq!(
            docs[0].topic_distribution,
            vec![TopicProportion {
                topic_id: 2,
                proportion: 1.0
            }]
        );
    }

    #[test]
    fn document_with_no_chunks_is_outlier() {
        // Two docs, only doc 1 has a chunk; doc 0 is empty.
        let docs = rollup(2, &[1], &[0], &[3]);
        assert_eq!(docs[0].dominant_topic, OUTLIER_LABEL);
        assert!(docs[0].topic_distribution.is_empty());
        assert_eq!(docs[1].dominant_topic, 0);
    }

    #[test]
    fn all_outlier_document_falls_back_to_outlier_dominant() {
        let docs = rollup(1, &[0, 0], &[OUTLIER_LABEL, OUTLIER_LABEL], &[2, 3]);
        assert_eq!(docs[0].dominant_topic, OUTLIER_LABEL);
        assert_eq!(docs[0].topic_distribution[0].topic_id, OUTLIER_LABEL);
        assert!((docs[0].topic_distribution[0].proportion - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unequal_segment_lengths_control_distribution_and_dominance() {
        let docs = rollup(1, &[0, 0], &[0, 1], &[2, 8]);
        assert_eq!(docs[0].dominant_topic, 1);
        assert_eq!(docs[0].topic_distribution[0].proportion, 0.2);
        assert_eq!(docs[0].topic_distribution[1].proportion, 0.8);
    }

    #[test]
    fn outlier_weight_is_included_in_normalization_but_not_dominance() {
        let docs = rollup(1, &[0, 0, 0], &[OUTLIER_LABEL, 0, 1], &[8, 1, 1]);
        assert_eq!(docs[0].dominant_topic, 0);
        let sum: f32 = docs[0]
            .topic_distribution
            .iter()
            .map(|entry| entry.proportion)
            .sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_dominance_ties_choose_smaller_topic_id() {
        let docs = rollup(1, &[0, 0], &[4, 2], &[3, 3]);
        assert_eq!(docs[0].dominant_topic, 2);
    }

    #[test]
    fn repeated_overlap_text_contributes_weight_for_each_segment() {
        let segment_texts = ["alpha overlap", "overlap beta"];
        let weights = segment_texts
            .iter()
            .map(|text| text.chars().count())
            .collect::<Vec<_>>();
        let docs = rollup(1, &[0, 0], &[0, 1], &weights);
        let expected = weights[0] as f32 / weights.iter().sum::<usize>() as f32;
        assert_eq!(docs[0].topic_distribution[0].proportion, expected);
    }
}
