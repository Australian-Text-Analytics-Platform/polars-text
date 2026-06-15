//! PaCMAP dimensionality reduction for topic-modeling chunk embeddings.
//!
//! Why this exists: density-based clustering (HDBSCAN) degrades in the raw
//! 384-dim embedding space (the curse of dimensionality flattens density
//! contrast), so BERTopic reduces to a handful of dimensions first. This module
//! is that step. Per the design decision it uses **PaCMAP only** — there is no
//! PCA-to-50-dims escape hatch; PaCMAP's internal PCA *initialization* is part
//! of the algorithm, not a fallback for it.
//!
//! PaCMAP (JMLR 2021) preserves both local and global structure and is a
//! faithful Rust port, making it the closest available stand-in for UMAP. Output
//! is seeded for reproducibility.
//!
//! Called by: `topic_modeling::run` between embedding and clustering. The
//! orchestrator only calls this when there are enough chunks for neighbor-based
//! reduction; tiny corpora are handled upstream as a single trivial topic rather
//! than reduced here.

use anyhow::Result;
use ndarray::Array2;
use pacmap::{fit_transform, Configuration, Initialization, PairConfiguration};

/// Reduction knobs. `output_dims` is the clustering target dimensionality
/// (~5-15; BERTopic defaults to 5, far below the 2D visualization default).
/// `seed` makes the embedding reproducible across runs given identical input.
#[derive(Debug, Clone)]
pub struct ReduceConfig {
    pub output_dims: usize,
    pub seed: u64,
}

impl Default for ReduceConfig {
    fn default() -> Self {
        Self {
            output_dims: 5,
            seed: 42,
        }
    }
}

/// Minimum number of points PaCMAP can reduce. Its default neighbor count is 10,
/// and it needs strictly more points than neighbors plus headroom for mid-near
/// and far pairs; below this the orchestrator must not call `reduce`.
pub const MIN_POINTS_FOR_REDUCTION: usize = 12;

/// Reduce `points` (each a same-length embedding row) to `cfg.output_dims`.
///
/// Flow: pack the rows into an `ndarray` matrix, run PaCMAP with PCA
/// initialization and a fixed seed, then unpack the reduced matrix back into row
/// vectors for the clusterer. Errors if called with too few points — the
/// orchestrator guards against that, but the check keeps the failure explicit.
pub fn reduce(points: &[Vec<f32>], cfg: &ReduceConfig) -> Result<Vec<Vec<f32>>> {
    let n = points.len();
    if n < MIN_POINTS_FOR_REDUCTION {
        anyhow::bail!("reduce called with {n} points; need at least {MIN_POINTS_FOR_REDUCTION}");
    }
    let dim = points[0].len();
    if dim == 0 {
        anyhow::bail!("reduce called with zero-dimensional points");
    }
    if points.iter().any(|p| p.len() != dim) {
        anyhow::bail!("reduce called with ragged embedding rows");
    }

    let flat: Vec<f32> = points.iter().flatten().copied().collect();
    let matrix = Array2::from_shape_vec((n, dim), flat)
        .map_err(|e| anyhow::anyhow!("failed to build embedding matrix: {e}"))?;

    let config = Configuration {
        embedding_dimensions: cfg.output_dims,
        // PCA init is PaCMAP's standard, deterministic starting point — this is
        // the algorithm's own initialization, not the rejected PCA fallback.
        initialization: Initialization::Pca,
        mid_near_ratio: 0.5,
        far_pair_ratio: 2.0,
        override_neighbors: None,
        seed: Some(cfg.seed),
        pair_configuration: PairConfiguration::Generate,
        learning_rate: 1.0,
        num_iters: (100, 100, 250),
        snapshots: None,
        approx_threshold: 8_000,
    };

    let (embedding, _snapshots) = fit_transform(matrix.view(), config)
        .map_err(|e| anyhow::anyhow!("PaCMAP fit_transform failed: {e}"))?;

    let out_dim = embedding.ncols();
    let reduced = embedding
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    debug_assert!(reduced.iter().all(|r| r.len() == out_dim));
    Ok(reduced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_rejects_too_few_points() {
        let pts = vec![vec![0.0f32, 1.0]; 3];
        let err = reduce(&pts, &ReduceConfig::default()).unwrap_err();
        assert!(err.to_string().contains("at least"), "{err}");
    }

    #[test]
    fn reduce_outputs_requested_dimensionality() {
        // Two well-separated Gaussians-ish blobs in 8-dim, enough points for
        // PaCMAP. We only assert shape + determinism here, never exact values.
        let mut pts = Vec::new();
        for i in 0..40 {
            let base = if i % 2 == 0 { 0.0 } else { 5.0 };
            pts.push((0..8).map(|j| base + (i * j % 3) as f32 * 0.01).collect());
        }
        let cfg = ReduceConfig {
            output_dims: 3,
            seed: 7,
        };
        let a = reduce(&pts, &cfg).unwrap();
        assert_eq!(a.len(), pts.len());
        assert!(a.iter().all(|r| r.len() == 3));
        // Same seed + input => stable embedding. PaCMAP's parallel float
        // reductions make it close-but-not-bit-exact across runs, so we assert
        // approximate (not exact) reproducibility; this is enough for stable
        // downstream clustering.
        let b = reduce(&pts, &cfg).unwrap();
        for (ra, rb) in a.iter().zip(&b) {
            for (x, y) in ra.iter().zip(rb) {
                assert!((x - y).abs() < 1e-2, "{x} vs {y}");
            }
        }
    }
}
