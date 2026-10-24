# Changes made in polars-text

This is pacmap 0.2.6 (https://github.com/beamform/pacmap-rs, Apache-2.0, see LICENSE),
copied unchanged except for the two changes below, both for reproducible results: the
same input and seed give the same embedding on every run and with any thread count (Wordflow issue 437).

- `src/knn.rs`, `find_k_nearest_neighbors_approx`: the usearch index is built as
  `INDEX_SHARDS` (4) fixed shards of consecutive rows, each filled in row order, the shards
  in parallel; every point searches all shards and keeps the k nearest, ties by row. One
  index filled from several threads made the graph, and so the neighbours found, depend on
  thread timing. The shard count is fixed rather than the thread count, so every machine
  finds the same neighbours. Searches are still parallel.
- `src/gradient.rs`, `pacmap_grad`: chunk gradients are computed in parallel in batches of
  the thread pool's size and added in chunk order, instead of `par_bridge().reduce(...)`,
  whose summation order (and float rounding) changed from run to run.

The bench targets and the GitHub workflow files were left out.
