//! Cluster a sample of Topic Segments on large corpora (Wordflow issue 330).
//!
//! Why this exists: HDBSCAN compares every pair of points. A news corpus of
//! 26k articles splits into about 550k paragraph segments, roughly 3x10^11
//! pairs per pass, which runs for hours even on every core. When a sample size
//! is given and there are more segments, a seeded random sample of that size
//! goes through PaCMAP and HDBSCAN; every other segment then takes the topic
//! (or the outlier label) of its nearest sampled segment by cosine similarity
//! of the embeddings. Topics are found from the sample; every segment still
//! gets a topic, and outliers keep the sample's share.

use std::sync::Arc;

use rayon::prelude::*;

/// A seeded, platform-independent sample of `limit` indices from `0..n`, in
/// ascending order. SplitMix64 drives a partial Fisher-Yates shuffle, so the
/// same seed always picks the same segments.
pub fn sample_indices(n: usize, limit: usize, seed: u64) -> Vec<usize> {
    if limit >= n {
        return (0..n).collect();
    }
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..limit {
        let span = (n - i) as u64;
        let j = i + (next() % span) as usize;
        indices.swap(i, j);
    }
    indices.truncate(limit);
    indices.sort_unstable();
    indices
}

fn norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

/// Labels for every segment: sampled ones keep their clustered label, every
/// other one copies the label of its nearest sampled segment by cosine
/// similarity (ties go to the earlier sampled segment).
pub fn assign_to_nearest_sampled(
    embeddings: &[Arc<Vec<f32>>],
    sampled: &[usize],
    sampled_labels: &[i32],
) -> Vec<i32> {
    debug_assert_eq!(sampled.len(), sampled_labels.len());
    let sample_vectors: Vec<(&[f32], f32)> = sampled
        .iter()
        .map(|&index| {
            let vector = embeddings[index].as_slice();
            (vector, norm(vector).max(f32::MIN_POSITIVE))
        })
        .collect();
    let mut labels: Vec<i32> = (0..embeddings.len())
        .into_par_iter()
        .map(|index| {
            let vector = embeddings[index].as_slice();
            let own_norm = norm(vector).max(f32::MIN_POSITIVE);
            let mut best = (f32::NEG_INFINITY, 0usize);
            for (position, (sample, sample_norm)) in sample_vectors.iter().enumerate() {
                let dot: f32 = vector.iter().zip(sample.iter()).map(|(a, b)| a * b).sum();
                let similarity = dot / (own_norm * sample_norm);
                if similarity > best.0 {
                    best = (similarity, position);
                }
            }
            sampled_labels[best.1]
        })
        .collect();
    // Sampled segments keep exactly the label HDBSCAN gave them.
    for (position, &index) in sampled.iter().enumerate() {
        labels[index] = sampled_labels[position];
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_is_seeded_sorted_and_of_the_requested_size() {
        let first = sample_indices(1_000, 100, 7);
        assert_eq!(first.len(), 100);
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(first, sample_indices(1_000, 100, 7));
        assert_ne!(first, sample_indices(1_000, 100, 8));
        assert_eq!(sample_indices(5, 10, 1), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn unsampled_segments_take_the_nearest_sampled_label_including_outliers() {
        let vectors = [
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.1, 0.9],
            vec![-0.9, -0.1],
        ];
        let embeddings: Vec<Arc<Vec<f32>>> = vectors.into_iter().map(Arc::new).collect();
        let labels = assign_to_nearest_sampled(&embeddings, &[0, 1, 2], &[3, 5, -1]);
        assert_eq!(labels, vec![3, 5, -1, 3, 5, -1]);
    }
}
