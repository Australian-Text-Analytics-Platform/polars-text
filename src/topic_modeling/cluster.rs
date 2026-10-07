//! HDBSCAN clustering of reduced Topic Segment embeddings into topics.
//!
//! Why this exists: after PaCMAP reduction, Topic Segments that discuss the same
//! thing sit close together; HDBSCAN turns those density peaks into topics and,
//! crucially, leaves genuinely off-topic segments as noise (label `-1`) instead of
//! forcing every point into a cluster. That noise handling is why BERTopic uses
//! HDBSCAN rather than k-means, and it carries straight over here.
//!
//! Determinism: HDBSCAN is deterministic given identical input, so topic
//! assignments are reproducible without a seed (PaCMAP upstream supplies the
//! seeded randomness).
//!
//! Distance metric: embeddings are L2-normalized upstream, so Euclidean distance
//! is monotonic with cosine distance and we can use the Euclidean metric the
//! crate provides directly.
//!
//! Called by: `topic_modeling::run` after `reduce`, on the reduced segment points.

use anyhow::Result;
use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams};

/// Outlier/noise label emitted by HDBSCAN for segments that belong to no topic.
/// Mirrors BERTopic's `-1` outlier topic so the rest of the pipeline (rollup,
/// payload, frontend) can treat it the same way.
pub const OUTLIER_LABEL: i32 = -1;

/// Auto only intervenes when one topic is the main topic of more than this
/// share of the documents. Excess-of-mass selection can pick one blob that
/// swallows the corpus. Documents, not segments, are what researchers read,
/// and the segment count depends on the segmentation mode (Wordflow issue 336).
const AUTO_DOMINANT_SHARE: f64 = 0.5;

/// Bounded number of Auto refinement passes; each pass descends one level into
/// the dominant topic.
const AUTO_MAX_PASSES: usize = 8;

/// Result of clustering: one label per input point. Labels are contiguous
/// `0..n_topics` for real topics, or `OUTLIER_LABEL` for noise. `n_topics` is
/// the count of distinct non-outlier labels, precomputed for the orchestrator.
/// `max_cluster_size` is the cap that produced these labels, if any: the
/// user's fixed Max topic size, or the size Auto settled on (`None` when Auto
/// found no dominant topic or kept the uncapped result).
#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub labels: Vec<i32>,
    pub n_topics: usize,
    pub max_cluster_size: Option<usize>,
    /// What Auto decided, or `None` with a fixed Max topic size.
    pub auto: Option<AutoOutcome>,
}

/// The document each clustered point belongs to and the characters it owns,
/// so Auto can find each document's main topic as the rollup does.
#[derive(Debug, Clone, Copy)]
pub struct PointDocuments<'a> {
    pub doc_indices: &'a [usize],
    pub weights: &'a [usize],
}

/// Auto's decision, reported so the run can explain it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoDecision {
    /// No topic was the main topic of more than half of the documents.
    NotNeeded,
    /// The dominant topic was split with a cap (`max_cluster_size`).
    Split,
    /// Splitting it left most of its segments as outliers, so it was kept.
    Kept,
}

impl AutoDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotNeeded => "not_needed",
            Self::Split => "split",
            Self::Kept => "kept",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoOutcome {
    pub decision: AutoDecision,
    /// Share of documents whose main topic was the largest one before Auto.
    pub document_share: f64,
}

/// Cluster `points` into topics.
///
/// Flow: with a fixed Max topic size, run HDBSCAN once with that cap (never
/// below `min_cluster_size + 1`). With Auto (`None`), run uncapped; while one
/// topic is the main topic of more than half of the documents, re-run with the
/// cap just below that topic's segment count and keep the capped labels only
/// when `accept_capped` approves them. Without `documents`, every point counts
/// as its own document. The crate's labels are already contiguous from zero,
/// which projection and rollup rely on for indexing.
pub fn cluster(
    points: &[Vec<f32>],
    min_cluster_size: usize,
    max_cluster_size: Option<usize>,
    documents: Option<PointDocuments<'_>>,
) -> Result<ClusterResult> {
    let n = points.len();
    if n < 2 {
        return Ok(ClusterResult {
            labels: vec![OUTLIER_LABEL; n],
            n_topics: 0,
            max_cluster_size: None,
            auto: None,
        });
    }
    let min_cluster_size = min_cluster_size.clamp(2, n);
    if let Some(requested) = max_cluster_size {
        return run_hdbscan(
            points,
            min_cluster_size,
            Some(requested.max(min_cluster_size + 1)),
        );
    }

    let own_documents: Vec<usize>;
    let unit_weights: Vec<usize>;
    let documents = match documents {
        Some(documents) => documents,
        None => {
            own_documents = (0..n).collect();
            unit_weights = vec![1; n];
            PointDocuments {
                doc_indices: &own_documents,
                weights: &unit_weights,
            }
        }
    };
    let mut best = run_hdbscan(points, min_cluster_size, None)?;
    let initial_share =
        main_topic_by_documents(&best.labels, documents).map_or(0.0, |(_, share)| share);
    let mut decision = if initial_share > AUTO_DOMINANT_SHARE {
        AutoDecision::Kept
    } else {
        AutoDecision::NotNeeded
    };
    for _ in 0..AUTO_MAX_PASSES {
        let Some((dominant, share)) = main_topic_by_documents(&best.labels, documents) else {
            break;
        };
        if share <= AUTO_DOMINANT_SHARE {
            break;
        }
        let dominant_size = best
            .labels
            .iter()
            .filter(|&&label| label == dominant)
            .count();
        let cap = dominant_size.saturating_sub(1).max(min_cluster_size + 1);
        if cap >= dominant_size {
            break;
        }
        let capped = run_hdbscan(points, min_cluster_size, Some(cap))?;
        if !accept_capped(&best, &capped, dominant) {
            break;
        }
        best = capped;
        decision = AutoDecision::Split;
    }
    best.auto = Some(AutoOutcome {
        decision,
        document_share: initial_share,
    });
    Ok(best)
}

/// Segments in the largest topic of `labels` (outliers excluded), the number a
/// fixed Max topic size is compared with.
pub fn largest_topic_size(labels: &[i32]) -> Option<usize> {
    let mut sizes = std::collections::HashMap::<i32, usize>::new();
    for &label in labels.iter().filter(|&&label| label != OUTLIER_LABEL) {
        *sizes.entry(label).or_default() += 1;
    }
    sizes.into_values().max()
}

/// The topic that is the main topic of the most documents, and the share of
/// documents (with any segment) it is the main topic of. A document's main
/// topic is the label owning most of its characters, as in the rollup; a
/// document whose largest share is outliers counts for no topic.
fn main_topic_by_documents(labels: &[i32], documents: PointDocuments<'_>) -> Option<(i32, f64)> {
    let mut per_document: std::collections::HashMap<usize, std::collections::HashMap<i32, usize>> =
        std::collections::HashMap::new();
    for ((&label, &document), &weight) in labels
        .iter()
        .zip(documents.doc_indices)
        .zip(documents.weights)
    {
        *per_document
            .entry(document)
            .or_default()
            .entry(label)
            .or_default() += weight;
    }
    if per_document.is_empty() {
        return None;
    }
    let mut documents_per_topic = std::collections::HashMap::<i32, usize>::new();
    for weights in per_document.values() {
        let main = weights
            .iter()
            .max_by_key(|&(&label, &weight)| (weight, std::cmp::Reverse(label)))
            .map(|(&label, _)| label);
        if let Some(label) = main.filter(|&label| label != OUTLIER_LABEL) {
            *documents_per_topic.entry(label).or_default() += 1;
        }
    }
    documents_per_topic
        .into_iter()
        .max_by_key(|&(label, count)| (count, std::cmp::Reverse(label)))
        .map(|(label, count)| (label, count as f64 / per_document.len() as f64))
}

fn run_hdbscan(
    points: &[Vec<f32>],
    min_cluster_size: usize,
    max_cluster_size: Option<usize>,
) -> Result<ClusterResult> {
    let n = points.len();
    let mut builder = HdbscanHyperParams::builder()
        .min_cluster_size(min_cluster_size)
        .dist_metric(DistanceMetric::Euclidean);
    let applied = max_cluster_size.filter(|&cap| cap < n);
    if let Some(cap) = applied {
        builder = builder.max_cluster_size(cap);
    }
    let clusterer = Hdbscan::new(points, builder.build());
    let labels = clusterer
        .cluster_par()
        .map_err(|e| anyhow::anyhow!("HDBSCAN clustering failed: {e}"))?;
    let n_topics = labels
        .iter()
        .filter(|&&l| l != OUTLIER_LABEL)
        .collect::<std::collections::HashSet<_>>()
        .len();
    Ok(ClusterResult {
        labels,
        n_topics,
        max_cluster_size: applied,
        auto: None,
    })
}

/// Accepts a capped re-clustering only when it finds more topics and keeps at
/// least half of the previously dominant topic's points in topics, so a
/// genuine large theme is never dissolved into outliers.
fn accept_capped(previous: &ClusterResult, capped: &ClusterResult, dominant: i32) -> bool {
    if capped.n_topics <= previous.n_topics {
        return false;
    }
    let (members, kept) = previous
        .labels
        .iter()
        .zip(&capped.labels)
        .filter(|(&before, _)| before == dominant)
        .fold((0usize, 0usize), |(members, kept), (_, &after)| {
            (members + 1, kept + usize::from(after != OUTLIER_LABEL))
        });
    kept * 2 >= members
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tight, well-separated blobs plus one far-flung outlier should yield
    /// two topics and a noise label. Values are fixed, so this is deterministic
    /// and safe for CI (unlike embedding/quality assertions).
    #[test]
    fn separates_two_blobs_and_marks_outlier() {
        let mut points: Vec<Vec<f32>> = Vec::new();
        for i in 0..10 {
            points.push(vec![0.0 + (i as f32) * 0.01, 0.0]);
        }
        for i in 0..10 {
            points.push(vec![10.0 + (i as f32) * 0.01, 10.0]);
        }
        points.push(vec![100.0, 100.0]); // lone outlier

        let res = cluster(&points, 5, None, None).unwrap();
        assert_eq!(res.n_topics, 2, "labels: {:?}", res.labels);
        assert_eq!(*res.labels.last().unwrap(), OUTLIER_LABEL);
        // Real labels are contiguous from zero.
        assert!(res
            .labels
            .iter()
            .all(|&l| l == OUTLIER_LABEL || (0..2).contains(&l)));
    }

    #[test]
    fn single_point_is_an_outlier_without_a_fabricated_topic() {
        let res = cluster(&[vec![1.0, 2.0]], 10, None, None).unwrap();
        assert_eq!(res.n_topics, 0);
        assert_eq!(res.labels, vec![OUTLIER_LABEL]);
    }

    #[test]
    fn empty_input_is_no_topics() {
        let res = cluster(&[], 10, None, None).unwrap();
        assert_eq!(res.n_topics, 0);
        assert!(res.labels.is_empty());
    }

    /// A large blob made of two nearby tight groups: without a cap HDBSCAN may
    /// select the whole blob, with a cap below its size it must select the
    /// groups inside it.
    #[test]
    fn max_cluster_size_forces_selection_below_a_giant_blob() {
        let mut points: Vec<Vec<f32>> = Vec::new();
        for i in 0..30 {
            points.push(vec![(i as f32) * 0.001, 0.0]);
        }
        for i in 0..30 {
            points.push(vec![0.5 + (i as f32) * 0.001, 0.0]);
        }
        let capped = cluster(&points, 5, Some(40), None).unwrap();
        assert!(capped.n_topics >= 2, "labels: {:?}", capped.labels);
        let largest = (0..capped.n_topics as i32)
            .map(|topic| {
                capped
                    .labels
                    .iter()
                    .filter(|&&label| label == topic)
                    .count()
            })
            .max()
            .unwrap();
        assert!(largest <= 40, "largest cluster {largest}");
    }

    #[test]
    fn auto_keeps_genuine_large_topics_that_are_not_dominant() {
        // Two real clusters of 10 plus an outlier: neither holds more than half
        // of the points, so Auto must not cap them away.
        let mut points: Vec<Vec<f32>> = Vec::new();
        for i in 0..10 {
            points.push(vec![(i as f32) * 0.01, 0.0]);
        }
        for i in 0..10 {
            points.push(vec![10.0 + (i as f32) * 0.01, 10.0]);
        }
        points.push(vec![100.0, 100.0]);
        let res = cluster(&points, 5, None, None).unwrap();
        assert_eq!(res.n_topics, 2, "labels: {:?}", res.labels);
        assert_eq!(res.max_cluster_size, None);
        assert_eq!(
            res.auto.map(|auto| auto.decision),
            Some(AutoDecision::NotNeeded)
        );
    }

    fn result(labels: &[i32]) -> ClusterResult {
        ClusterResult {
            n_topics: labels
                .iter()
                .filter(|&&label| label != OUTLIER_LABEL)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            labels: labels.to_vec(),
            max_cluster_size: None,
            auto: None,
        }
    }

    #[test]
    fn accepts_a_split_that_keeps_the_dominant_topic_in_topics() {
        let previous = result(&[0, 0, 0, 0, 0, 0, 1, 1]);
        let capped = result(&[0, 0, 0, 2, 2, -1, 1, 1]);
        assert!(accept_capped(&previous, &capped, 0));
    }

    #[test]
    fn rejects_a_split_that_turns_the_dominant_topic_into_outliers() {
        let previous = result(&[0, 0, 0, 0, 0, 0, 1, 1]);
        let capped = result(&[2, 3, -1, -1, -1, -1, 1, 1]);
        assert!(!accept_capped(&previous, &capped, 0));
    }

    #[test]
    fn rejects_a_cap_that_finds_no_more_topics() {
        let previous = result(&[0, 0, 0, 0, 1, 1]);
        let capped = result(&[0, 0, 0, -1, 1, 1]);
        assert!(!accept_capped(&previous, &capped, 0));
    }

    #[test]
    fn main_topic_counts_documents_by_their_largest_share_of_text() {
        // doc 0 is mostly topic 0, doc 1 topic 1, doc 2 mostly outliers.
        let labels = [0, 0, 1, -1, 1];
        let documents = PointDocuments {
            doc_indices: &[0, 0, 1, 2, 2],
            weights: &[5, 5, 10, 8, 3],
        };
        let (topic, share) = main_topic_by_documents(&labels, documents).unwrap();
        assert_eq!(topic, 0);
        assert!((share - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(main_topic_by_documents(&[], documents), None);
    }

    /// A topic with a third of the segments but most of the documents
    /// triggers Auto by documents; counted by segments it would not.
    #[test]
    fn auto_triggers_on_documents_not_segments() {
        let mut points: Vec<Vec<f32>> = Vec::new();
        let mut doc_indices = Vec::new();
        for i in 0..10 {
            points.push(vec![(i as f32) * 0.01, 0.0]);
            doc_indices.push(i);
        }
        // Two more topics of 10 segments, each inside a single document.
        for i in 0..10 {
            points.push(vec![10.0 + (i as f32) * 0.01, 10.0]);
            doc_indices.push(10);
        }
        for i in 0..10 {
            points.push(vec![20.0 + (i as f32) * 0.01, 0.0]);
            doc_indices.push(11);
        }
        let weights = vec![1; points.len()];
        let documents = PointDocuments {
            doc_indices: &doc_indices,
            weights: &weights,
        };

        let by_segments = cluster(&points, 5, None, None).unwrap();
        assert_eq!(
            by_segments.auto.map(|auto| auto.decision),
            Some(AutoDecision::NotNeeded)
        );

        let by_documents = cluster(&points, 5, None, Some(documents)).unwrap();
        let auto = by_documents.auto.unwrap();
        assert_ne!(auto.decision, AutoDecision::NotNeeded);
        assert!((auto.document_share - 10.0 / 12.0).abs() < 1e-9, "{auto:?}");
    }

    #[test]
    fn largest_topic_size_ignores_outliers() {
        assert_eq!(largest_topic_size(&[-1, -1, -1, 0, 1, 1]), Some(2));
        assert_eq!(largest_topic_size(&[-1, -1]), None);
    }

    #[test]
    fn a_fixed_max_topic_size_reports_no_auto_decision() {
        let points: Vec<Vec<f32>> = (0..20).map(|i| vec![(i as f32) * 0.01, 0.0]).collect();
        assert!(cluster(&points, 5, Some(15), None).unwrap().auto.is_none());
    }
}
