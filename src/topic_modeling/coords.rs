//! 2D topic coordinates for the bubble chart.
//!
//! Why this exists: the bubble chart places each topic at an `(x, y)` that should
//! reflect inter-topic similarity — topics about related things should sit near
//! each other. BERTopic gets this from a 2D UMAP of the topic embeddings.
//!
//! Why averaging rather than projecting centroids: PaCMAP needs more than a
//! handful of points to fit (it builds a neighbor graph), and a run typically has
//! only a few topics, so projecting the topic centroids directly is not viable.
//! Instead the orchestrator reduces *all chunk embeddings* to 2D once (a second,
//! visualization-only PaCMAP pass), and each topic's coordinate is the centroid
//! of its member chunks in that 2D space. This keeps us on PaCMAP-only (no PCA),
//! is robust for any topic count, and yields a clean, well-separated layout.
//!
//! Determinism: given the 2D points and labels, the centroid is a pure mean and
//! fully unit-tested. The upstream 2D reduction carries PaCMAP's usual seeded,
//! close-but-not-bit-exact behavior, same as the clustering reduction.
//!
//! Called by: `topic_modeling::run`, after clustering, with the 2D-reduced chunk
//! points and the chunk labels.

use crate::topic_modeling::cluster::OUTLIER_LABEL;

/// Compute one `(x, y)` per topic as the centroid of its non-outlier chunks in
/// the supplied 2D space.
///
/// `points_2d[i]` is chunk `i`'s 2D coordinate (each inner vec length 2);
/// `labels[i]` is its topic. `n_topics` bounds the output (topics are contiguous
/// `0..n_topics`). A topic with no member chunks — which should not occur given
/// clustering produced it — defaults to the origin so indexing stays aligned.
pub fn topic_coords_2d(
    points_2d: &[Vec<f32>],
    labels: &[i32],
    n_topics: usize,
) -> Vec<(f32, f32)> {
    debug_assert_eq!(points_2d.len(), labels.len());

    let mut sums = vec![(0.0f32, 0.0f32); n_topics];
    let mut counts = vec![0usize; n_topics];

    for (pt, &label) in points_2d.iter().zip(labels) {
        if label == OUTLIER_LABEL {
            continue;
        }
        let t = label as usize;
        if t < n_topics && pt.len() >= 2 {
            sums[t].0 += pt[0];
            sums[t].1 += pt[1];
            counts[t] += 1;
        }
    }

    sums.into_iter()
        .zip(counts)
        .map(|((sx, sy), c)| {
            if c == 0 {
                (0.0, 0.0)
            } else {
                (sx / c as f32, sy / c as f32)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_is_mean_of_member_chunks() {
        // Topic 0: (0,0) and (2,2) -> (1,1). Topic 1: (10,10) -> (10,10).
        // One outlier chunk must be ignored.
        let points = vec![
            vec![0.0, 0.0],
            vec![2.0, 2.0],
            vec![10.0, 10.0],
            vec![999.0, 999.0],
        ];
        let labels = vec![0, 0, 1, OUTLIER_LABEL];
        let coords = topic_coords_2d(&points, &labels, 2);
        assert_eq!(coords, vec![(1.0, 1.0), (10.0, 10.0)]);
    }

    #[test]
    fn topic_without_members_defaults_to_origin() {
        let coords = topic_coords_2d(&[vec![1.0, 1.0]], &[0], 2);
        assert_eq!(coords[1], (0.0, 0.0));
    }
}
