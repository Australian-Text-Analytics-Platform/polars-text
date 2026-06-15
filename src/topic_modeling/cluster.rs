//! HDBSCAN clustering of reduced chunk embeddings into topics.
//!
//! Why this exists: after PaCMAP reduction, chunks that talk about the same
//! thing sit close together; HDBSCAN turns those density peaks into topics and,
//! crucially, leaves genuinely off-topic chunks as noise (label `-1`) instead of
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
//! Called by: `topic_modeling::run` after `reduce`, on the reduced chunk points.

use anyhow::Result;
use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams};

/// Outlier/noise label emitted by HDBSCAN for chunks that belong to no topic.
/// Mirrors BERTopic's `-1` outlier topic so the rest of the pipeline (rollup,
/// payload, frontend) can treat it the same way.
pub const OUTLIER_LABEL: i32 = -1;

/// Clustering knobs. `min_cluster_size` is the smallest group of chunks that
/// counts as a topic — the backend maps its public `min_topic_size` option onto
/// this, and the topic count is whatever HDBSCAN yields for it. `min_samples`
/// controls how conservative the noise classification is; `None` lets HDBSCAN
/// default it to `min_cluster_size`.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub min_cluster_size: usize,
    pub min_samples: Option<usize>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 10,
            min_samples: None,
        }
    }
}

/// Result of clustering: one label per input point. Labels are contiguous
/// `0..n_topics` for real topics, or `OUTLIER_LABEL` for noise. `n_topics` is
/// the count of distinct non-outlier labels, precomputed for the orchestrator.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub labels: Vec<i32>,
    pub n_topics: usize,
}

/// Cluster `points` into topics.
///
/// Flow: build HDBSCAN hyper-parameters from `cfg` (clamping `min_cluster_size`
/// to the valid `>= 2` range and never exceeding the point count), run the
/// clusterer, and count distinct non-outlier labels. The crate's labels are
/// already contiguous from zero, which `rollup`/`coords` rely on for indexing.
pub fn cluster(points: &[Vec<f32>], cfg: &ClusterConfig) -> Result<ClusterResult> {
    let n = points.len();
    if n < 2 {
        // One or zero points cannot form a density cluster; treat all as a
        // single trivial topic so callers still get a usable labeling.
        return Ok(ClusterResult {
            labels: vec![0; n],
            n_topics: if n == 0 { 0 } else { 1 },
        });
    }

    let min_cluster_size = cfg.min_cluster_size.clamp(2, n);
    let mut builder = HdbscanHyperParams::builder()
        .min_cluster_size(min_cluster_size)
        .dist_metric(DistanceMetric::Euclidean);
    if let Some(ms) = cfg.min_samples {
        builder = builder.min_samples(ms.clamp(1, n));
    }
    let params = builder.build();

    let clusterer = Hdbscan::new(points, params);
    let labels = clusterer
        .cluster()
        .map_err(|e| anyhow::anyhow!("HDBSCAN clustering failed: {e}"))?;

    let n_topics = labels
        .iter()
        .filter(|&&l| l != OUTLIER_LABEL)
        .collect::<std::collections::HashSet<_>>()
        .len();

    Ok(ClusterResult { labels, n_topics })
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

        let cfg = ClusterConfig {
            min_cluster_size: 5,
            min_samples: None,
        };
        let res = cluster(&points, &cfg).unwrap();
        assert_eq!(res.n_topics, 2, "labels: {:?}", res.labels);
        assert_eq!(*res.labels.last().unwrap(), OUTLIER_LABEL);
        // Real labels are contiguous from zero.
        assert!(res
            .labels
            .iter()
            .all(|&l| l == OUTLIER_LABEL || (0..2).contains(&l)));
    }

    #[test]
    fn single_point_is_one_trivial_topic() {
        let res = cluster(&[vec![1.0, 2.0]], &ClusterConfig::default()).unwrap();
        assert_eq!(res.n_topics, 1);
        assert_eq!(res.labels, vec![0]);
    }

    #[test]
    fn empty_input_is_no_topics() {
        let res = cluster(&[], &ClusterConfig::default()).unwrap();
        assert_eq!(res.n_topics, 0);
        assert!(res.labels.is_empty());
    }
}
