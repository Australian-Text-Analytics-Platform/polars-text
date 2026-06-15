//! Roll chunk-level topic assignments up to per-document topic distributions.
//!
//! Why this exists: clustering labels *chunks*, but the product surfaces
//! *documents*. A long document legitimately spans several topics, so instead of
//! collapsing it to a single id (the old BERTopic behavior that the long-text
//! redesign explicitly removes), we report the normalized mix of topics across
//! its chunks. A short document yields one chunk and therefore a near-one-hot
//! distribution — the same code path, no length branching.
//!
//! Outlier handling: HDBSCAN's `-1` chunks are kept in the proportions so they
//! sum to 1 over every chunk of the document (an all-outlier document is honest
//! about being unclusterable), but `dominant_topic` prefers a real topic and
//! only falls back to `-1` when the document has no clustered chunk at all.
//!
//! Determinism: pure counting; distributions are emitted in ascending topic-id
//! order and dominant-topic ties break to the smaller id, so identical input
//! always produces identical output. Fully unit-tested.
//!
//! Called by: `topic_modeling::run` after clustering, feeding the per-document
//! payload and the per-corpus soft sizes used by the bubble chart.

use std::collections::BTreeMap;

use crate::topic_modeling::cluster::OUTLIER_LABEL;

/// One topic's share of a document, proportions summing to 1 across the
/// document's chunks. `topic_id` may be `OUTLIER_LABEL` (`-1`).
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
/// `chunk_doc_index[i]` is the owning document of chunk `i`; `chunk_labels[i]`
/// is its topic. The two chunk slices must be the same length.
///
/// Flow: tally each document's chunk labels, divide by the chunk count to get
/// proportions, choose the highest-proportion non-outlier topic as dominant, and
/// emit the distribution sorted by topic id.
pub fn rollup(
    n_docs: usize,
    chunk_doc_index: &[usize],
    chunk_labels: &[i32],
) -> Vec<DocumentTopics> {
    debug_assert_eq!(chunk_doc_index.len(), chunk_labels.len());

    // Per-document topic counts in ascending-id order (BTreeMap = stable output).
    let mut per_doc: Vec<BTreeMap<i32, usize>> = vec![BTreeMap::new(); n_docs];
    for (&doc, &label) in chunk_doc_index.iter().zip(chunk_labels) {
        if doc < n_docs {
            *per_doc[doc].entry(label).or_insert(0) += 1;
        }
    }

    per_doc
        .into_iter()
        .map(|counts| {
            let total: usize = counts.values().sum();
            if total == 0 {
                return DocumentTopics {
                    topic_distribution: Vec::new(),
                    dominant_topic: OUTLIER_LABEL,
                };
            }

            let distribution: Vec<TopicProportion> = counts
                .iter()
                .map(|(&topic_id, &count)| TopicProportion {
                    topic_id,
                    proportion: count as f32 / total as f32,
                })
                .collect();

            // Dominant = most-represented real topic; outliers only win if the
            // document has no clustered chunk at all. BTreeMap iteration is
            // ascending, so the first max found is the smallest-id winner.
            let dominant_topic = counts
                .iter()
                .filter(|(&id, _)| id != OUTLIER_LABEL)
                .max_by_key(|(_, &c)| c)
                .map(|(&id, _)| id)
                .unwrap_or(OUTLIER_LABEL);

            DocumentTopics {
                topic_distribution: distribution,
                dominant_topic,
            }
        })
        .collect()
}

/// Sum document proportions into per-corpus, per-topic "soft sizes" — the bubble
/// chart's redefined `size[]`. Outlier mass is dropped (the bubble chart shows
/// real topics only); chunk counts live elsewhere in `meta`.
///
/// `doc_corpus[d]` is document `d`'s corpus index; `n_corpora`/`n_topics` bound
/// the output. Returns `sizes[corpus][topic_id]`.
pub fn corpus_topic_sizes(
    docs: &[DocumentTopics],
    doc_corpus: &[usize],
    n_corpora: usize,
    n_topics: usize,
) -> Vec<Vec<f32>> {
    let mut sizes = vec![vec![0.0f32; n_topics]; n_corpora];
    for (doc, &corpus) in docs.iter().zip(doc_corpus) {
        if corpus >= n_corpora {
            continue;
        }
        for tp in &doc.topic_distribution {
            if tp.topic_id == OUTLIER_LABEL {
                continue;
            }
            let t = tp.topic_id as usize;
            if t < n_topics {
                sizes[corpus][t] += tp.proportion;
            }
        }
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_doc_gets_multi_topic_distribution() {
        // Document 0 has 4 chunks: topics 0,0,1,-1.
        let docs = rollup(1, &[0, 0, 0, 0], &[0, 0, 1, OUTLIER_LABEL]);
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
        let docs = rollup(1, &[0], &[2]);
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
        let docs = rollup(2, &[1], &[0]);
        assert_eq!(docs[0].dominant_topic, OUTLIER_LABEL);
        assert!(docs[0].topic_distribution.is_empty());
        assert_eq!(docs[1].dominant_topic, 0);
    }

    #[test]
    fn all_outlier_document_falls_back_to_outlier_dominant() {
        let docs = rollup(1, &[0, 0], &[OUTLIER_LABEL, OUTLIER_LABEL]);
        assert_eq!(docs[0].dominant_topic, OUTLIER_LABEL);
        assert_eq!(docs[0].topic_distribution[0].topic_id, OUTLIER_LABEL);
        assert!((docs[0].topic_distribution[0].proportion - 1.0).abs() < 1e-6);
    }

    #[test]
    fn soft_sizes_sum_proportions_per_corpus_excluding_outliers() {
        // Corpus 0: doc with 0.5/0.5 across topics 0 and 1.
        // Corpus 1: doc fully on topic 0, plus outlier mass that must be ignored.
        let docs = vec![
            DocumentTopics {
                topic_distribution: vec![
                    TopicProportion {
                        topic_id: 0,
                        proportion: 0.5,
                    },
                    TopicProportion {
                        topic_id: 1,
                        proportion: 0.5,
                    },
                ],
                dominant_topic: 0,
            },
            DocumentTopics {
                topic_distribution: vec![
                    TopicProportion {
                        topic_id: OUTLIER_LABEL,
                        proportion: 0.25,
                    },
                    TopicProportion {
                        topic_id: 0,
                        proportion: 0.75,
                    },
                ],
                dominant_topic: 0,
            },
        ];
        let sizes = corpus_topic_sizes(&docs, &[0, 1], 2, 2);
        assert_eq!(sizes[0], vec![0.5, 0.5]);
        assert_eq!(sizes[1], vec![0.75, 0.0]); // outlier 0.25 dropped
    }
}
