//! Immutable, additive Topic projection context.
//!
//! HDBSCAN's real Topics are retained as the maximum-resolution leaves. A
//! deterministic weighted Ward tree records only how those leaves merge. Cuts
//! aggregate sufficient statistics, so Result queries never need source text,
//! embeddings, PaCMAP, or HDBSCAN.

use std::collections::{BTreeSet, HashMap};
use std::io::Cursor;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::cluster::OUTLIER_LABEL;
use super::ctfidf;
use super::rollup;
use super::{DocumentResult, TopicInfo, TopicModelingResult};

const CONTEXT_VERSION: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Leaf {
    id: usize,
    segment_count: usize,
    centroid_5d: Vec<f64>,
    coordinate_sum: [f64; 2],
    coordinate_count: usize,
    term_counts: Vec<(u32, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SegmentFact {
    document_index: usize,
    leaf_id: i32,
    retained_character_weight: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Merge {
    left: usize,
    right: usize,
    minimum_leaf_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicClusteringContext {
    version: u8,
    document_count: usize,
    natural_cluster_count: usize,
    vocabulary: Vec<String>,
    leaves: Vec<Leaf>,
    merges: Vec<Merge>,
    segments: Vec<SegmentFact>,
    n_chunks: usize,
    truncated_segment_count: usize,
}

#[derive(Debug, Clone)]
struct WardNode {
    node_id: usize,
    minimum_leaf_id: usize,
    weight: usize,
    centroid: Vec<f64>,
}

pub fn prepare_context(
    document_count: usize,
    labels: &[i32],
    reduced_5d: &[Vec<f32>],
    reduced_2d: &[Vec<f32>],
    document_indices: &[usize],
    retained_character_weights: &[usize],
    per_leaf_term_counts: &[HashMap<String, usize>],
    n_chunks: usize,
    truncated_segment_count: usize,
) -> Result<TopicClusteringContext> {
    let segment_count = labels.len();
    if reduced_5d.len() != segment_count
        || reduced_2d.len() != segment_count
        || document_indices.len() != segment_count
        || retained_character_weights.len() != segment_count
    {
        bail!("Topic projection inputs must align by Topic Segment");
    }
    let natural_cluster_count = labels
        .iter()
        .filter(|&&label| label >= 0)
        .map(|&label| label as usize)
        .max()
        .map_or(0, |maximum| maximum + 1);
    if per_leaf_term_counts.len() != natural_cluster_count {
        bail!("Topic projection term counts must align with real Topic leaves");
    }

    let vocabulary = per_leaf_term_counts
        .iter()
        .flat_map(|counts| counts.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let vocabulary_indices = vocabulary
        .iter()
        .enumerate()
        .map(|(index, term)| (term.as_str(), index as u32))
        .collect::<HashMap<_, _>>();

    let dimensions = reduced_5d.first().map_or(0, Vec::len);
    let mut centroid_sums = vec![vec![0.0f64; dimensions]; natural_cluster_count];
    let mut coordinate_sums = vec![[0.0f64; 2]; natural_cluster_count];
    let mut counts = vec![0usize; natural_cluster_count];
    for ((point_5d, point_2d), &label) in reduced_5d.iter().zip(reduced_2d).zip(labels) {
        if label == OUTLIER_LABEL {
            continue;
        }
        let leaf = label as usize;
        if leaf >= natural_cluster_count || point_5d.len() != dimensions || point_2d.len() < 2 {
            bail!("Topic projection received an invalid real Topic label or coordinate");
        }
        counts[leaf] += 1;
        for (sum, &value) in centroid_sums[leaf].iter_mut().zip(point_5d) {
            *sum += value as f64;
        }
        coordinate_sums[leaf][0] += point_2d[0] as f64;
        coordinate_sums[leaf][1] += point_2d[1] as f64;
    }

    let leaves = (0..natural_cluster_count)
        .map(|id| {
            if counts[id] == 0 {
                bail!("Topic projection real Topic leaf has no Topic Segments");
            }
            let count = counts[id] as f64;
            Ok(Leaf {
                id,
                segment_count: counts[id],
                centroid_5d: centroid_sums[id].iter().map(|sum| sum / count).collect(),
                coordinate_sum: coordinate_sums[id],
                coordinate_count: counts[id],
                term_counts: per_leaf_term_counts[id]
                    .iter()
                    .map(|(term, &occurrences)| {
                        Ok((
                            *vocabulary_indices
                                .get(term.as_str())
                                .context("Topic projection vocabulary term is missing")?,
                            occurrences,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let segments = labels
        .iter()
        .zip(document_indices)
        .zip(retained_character_weights)
        .map(
            |((&leaf_id, &document_index), &retained_character_weight)| SegmentFact {
                document_index,
                leaf_id,
                retained_character_weight,
            },
        )
        .collect();
    let merges = build_ward_tree(&leaves)?;

    Ok(TopicClusteringContext {
        version: CONTEXT_VERSION,
        document_count,
        natural_cluster_count,
        vocabulary,
        leaves,
        merges,
        segments,
        n_chunks,
        truncated_segment_count,
    })
}

fn ward_cost(left: &WardNode, right: &WardNode) -> f64 {
    let squared_distance = left
        .centroid
        .iter()
        .zip(&right.centroid)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    (left.weight as f64 * right.weight as f64 / (left.weight + right.weight) as f64)
        * squared_distance
}

fn build_ward_tree(leaves: &[Leaf]) -> Result<Vec<Merge>> {
    let mut active = leaves
        .iter()
        .map(|leaf| WardNode {
            node_id: leaf.id,
            minimum_leaf_id: leaf.id,
            weight: leaf.segment_count,
            centroid: leaf.centroid_5d.clone(),
        })
        .collect::<Vec<_>>();
    let mut merges = Vec::with_capacity(leaves.len().saturating_sub(1));
    while active.len() > 1 {
        let mut best: Option<(f64, usize, usize, usize, usize)> = None;
        for left_index in 0..active.len() - 1 {
            for right_index in left_index + 1..active.len() {
                let left = &active[left_index];
                let right = &active[right_index];
                let ids = if left.minimum_leaf_id <= right.minimum_leaf_id {
                    (left.minimum_leaf_id, right.minimum_leaf_id)
                } else {
                    (right.minimum_leaf_id, left.minimum_leaf_id)
                };
                let candidate = (
                    ward_cost(left, right),
                    ids.0,
                    ids.1,
                    left_index,
                    right_index,
                );
                if best.as_ref().is_none_or(|current| {
                    candidate.0.total_cmp(&current.0).is_lt()
                        || (candidate.0.total_cmp(&current.0).is_eq()
                            && (candidate.1, candidate.2) < (current.1, current.2))
                }) {
                    best = Some(candidate);
                }
            }
        }
        let (_, _, _, left_index, right_index) = best.context("Ward tree has no merge pair")?;
        let right = active.remove(right_index);
        let left = active.remove(left_index);
        if left.centroid.len() != right.centroid.len() {
            bail!("Ward tree leaf centroid dimensions differ");
        }
        let weight = left.weight + right.weight;
        let centroid = left
            .centroid
            .iter()
            .zip(&right.centroid)
            .map(|(a, b)| (a * left.weight as f64 + b * right.weight as f64) / weight as f64)
            .collect();
        let node_id = leaves.len() + merges.len();
        let minimum_leaf_id = left.minimum_leaf_id.min(right.minimum_leaf_id);
        merges.push(Merge {
            left: left.node_id,
            right: right.node_id,
            minimum_leaf_id,
        });
        active.push(WardNode {
            node_id,
            minimum_leaf_id,
            weight,
            centroid,
        });
    }
    Ok(merges)
}

fn leaf_projection_ids(context: &TopicClusteringContext, cluster_count: usize) -> Result<Vec<i32>> {
    let natural = context.natural_cluster_count;
    if cluster_count > natural
        || (natural > 1 && cluster_count < 2)
        || (natural <= 1 && cluster_count != natural)
    {
        bail!("cluster_count {cluster_count} is outside the supported range");
    }
    let mut members = (0..natural).map(|leaf| vec![leaf]).collect::<Vec<_>>();
    for (merge_index, merge) in context
        .merges
        .iter()
        .take(natural.saturating_sub(cluster_count))
        .enumerate()
    {
        let left = std::mem::take(
            members
                .get_mut(merge.left)
                .context("Topic projection merge references an unknown left node")?,
        );
        let right = std::mem::take(
            members
                .get_mut(merge.right)
                .context("Topic projection merge references an unknown right node")?,
        );
        let mut combined = left;
        combined.extend(right);
        let expected_id = natural + merge_index;
        if members.len() != expected_id {
            bail!("Topic projection merge ordering is invalid");
        }
        members.push(combined);
    }
    let active_start = natural.saturating_sub(cluster_count);
    let active_nodes = if cluster_count == natural {
        (0..natural).collect::<Vec<_>>()
    } else {
        // A node is active if it has members and is not consumed by a later applied merge.
        let consumed = context
            .merges
            .iter()
            .take(active_start)
            .flat_map(|merge| [merge.left, merge.right])
            .collect::<BTreeSet<_>>();
        members
            .iter()
            .enumerate()
            .filter(|(node, member)| !member.is_empty() && !consumed.contains(node))
            .map(|(node, _)| node)
            .collect::<Vec<_>>()
    };
    if active_nodes.len() != cluster_count {
        bail!("Topic projection cut did not produce the requested real Topic count");
    }
    let mut ordered = active_nodes
        .into_iter()
        .map(|node| {
            let minimum = *members[node]
                .iter()
                .min()
                .context("Topic projection cluster has no leaves")?;
            Ok((minimum, node))
        })
        .collect::<Result<Vec<_>>>()?;
    ordered.sort_unstable();
    let mut projected = vec![OUTLIER_LABEL; natural];
    for (projected_id, (_, node)) in ordered.into_iter().enumerate() {
        for &leaf in &members[node] {
            projected[leaf] = projected_id as i32;
        }
    }
    Ok(projected)
}

pub fn project(
    context: &TopicClusteringContext,
    cluster_count: usize,
) -> Result<TopicModelingResult> {
    validate_context(context)?;
    let projection_ids = leaf_projection_ids(context, cluster_count)?;
    let mut term_counts = vec![HashMap::<String, usize>::new(); cluster_count];
    let mut coordinate_sums = vec![[0.0f64; 2]; cluster_count];
    let mut coordinate_counts = vec![0usize; cluster_count];
    for leaf in &context.leaves {
        let projected_id = projection_ids[leaf.id] as usize;
        coordinate_sums[projected_id][0] += leaf.coordinate_sum[0];
        coordinate_sums[projected_id][1] += leaf.coordinate_sum[1];
        coordinate_counts[projected_id] += leaf.coordinate_count;
        for &(term_index, occurrences) in &leaf.term_counts {
            let term = context
                .vocabulary
                .get(term_index as usize)
                .context("Topic projection term index is outside the vocabulary")?;
            *term_counts[projected_id].entry(term.clone()).or_insert(0) += occurrences;
        }
    }
    let representative_words = ctfidf::representative_words(&term_counts);
    let topics = (0..cluster_count)
        .map(|id| TopicInfo {
            id: id as i32,
            representative_words: representative_words[id].clone(),
            x: if coordinate_counts[id] == 0 {
                0.0
            } else {
                (coordinate_sums[id][0] / coordinate_counts[id] as f64) as f32
            },
            y: if coordinate_counts[id] == 0 {
                0.0
            } else {
                (coordinate_sums[id][1] / coordinate_counts[id] as f64) as f32
            },
        })
        .collect();
    let document_indices = context
        .segments
        .iter()
        .map(|fact| fact.document_index)
        .collect::<Vec<_>>();
    let labels = context
        .segments
        .iter()
        .map(|fact| {
            if fact.leaf_id == OUTLIER_LABEL {
                OUTLIER_LABEL
            } else {
                projection_ids[fact.leaf_id as usize]
            }
        })
        .collect::<Vec<_>>();
    let weights = context
        .segments
        .iter()
        .map(|fact| fact.retained_character_weight)
        .collect::<Vec<_>>();
    let documents = rollup::rollup(context.document_count, &document_indices, &labels, &weights)
        .into_iter()
        .enumerate()
        .map(|(doc_index, topics)| DocumentResult {
            doc_index,
            dominant_topic: topics.dominant_topic,
            topic_distribution: topics
                .topic_distribution
                .into_iter()
                .map(|entry| (entry.topic_id, entry.proportion))
                .collect(),
        })
        .collect();
    Ok(TopicModelingResult {
        topics,
        documents,
        n_chunks: context.n_chunks,
        truncated_segment_count: context.truncated_segment_count,
        stage_timings_ms: Vec::new(),
        clustering_context: Vec::new(),
    })
}

fn validate_context(context: &TopicClusteringContext) -> Result<()> {
    if context.version != CONTEXT_VERSION {
        bail!(
            "unsupported Topic clustering context version {}",
            context.version
        );
    }
    if context.leaves.len() != context.natural_cluster_count
        || context.merges.len() != context.natural_cluster_count.saturating_sub(1)
        || context
            .leaves
            .iter()
            .enumerate()
            .any(|(id, leaf)| leaf.id != id)
    {
        bail!("Topic clustering context structure is invalid");
    }
    Ok(())
}

pub fn serialize_context(context: &TopicClusteringContext) -> Result<Vec<u8>> {
    validate_context(context)?;
    let message_pack =
        rmp_serde::to_vec_named(context).context("encode Topic clustering context")?;
    zstd::stream::encode_all(Cursor::new(message_pack), 9)
        .context("compress Topic clustering context")
}

pub fn deserialize_context(bytes: &[u8]) -> Result<TopicClusteringContext> {
    let message_pack = zstd::stream::decode_all(Cursor::new(bytes))
        .context("decompress Topic clustering context")?;
    let context: TopicClusteringContext =
        rmp_serde::from_slice(&message_pack).context("decode Topic clustering context")?;
    validate_context(&context)?;
    Ok(context)
}

pub fn project_serialized_context(
    bytes: &[u8],
    cluster_count: usize,
) -> Result<TopicModelingResult> {
    project(&deserialize_context(bytes)?, cluster_count)
}

pub fn natural_cluster_count(context: &TopicClusteringContext) -> usize {
    context.natural_cluster_count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> TopicClusteringContext {
        let labels = vec![0, 0, 1, 1, 2, 2, OUTLIER_LABEL];
        let points_5d = vec![
            vec![0.0],
            vec![0.2],
            vec![10.0],
            vec![10.2],
            vec![30.0],
            vec![30.2],
            vec![99.0],
        ];
        let points_2d = points_5d
            .iter()
            .map(|point| vec![point[0], point[0]])
            .collect::<Vec<_>>();
        let counts = vec![
            HashMap::from([("alpha".to_string(), 4)]),
            HashMap::from([("beta".to_string(), 3)]),
            HashMap::from([("gamma".to_string(), 2)]),
        ];
        prepare_context(
            2,
            &labels,
            &points_5d,
            &points_2d,
            &[0, 0, 0, 0, 1, 1, 1],
            &[1, 1, 1, 1, 2, 2, 4],
            &counts,
            7,
            0,
        )
        .unwrap()
    }

    #[test]
    fn deterministic_cuts_are_exact_and_canonical() {
        let context = fixture();
        let natural = project(&context, 3).unwrap();
        assert_eq!(
            natural
                .topics
                .iter()
                .map(|topic| topic.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let merged = project(&context, 2).unwrap();
        assert_eq!(merged.topics.len(), 2);
        assert_eq!(merged.documents[0].dominant_topic, 0);
        assert_eq!(merged.documents[1].dominant_topic, 1);
    }

    #[test]
    fn serialization_round_trip_and_corruption() {
        let context = fixture();
        let bytes = serialize_context(&context).unwrap();
        assert_eq!(
            project_serialized_context(&bytes, 2).unwrap().topics.len(),
            2
        );
        assert!(project_serialized_context(b"not zstd", 2).is_err());
    }

    #[test]
    fn projection_keeps_outlier_weight_in_distribution() {
        let projected = project(&fixture(), 2).unwrap();
        let document = &projected.documents[1];
        let outlier = document
            .topic_distribution
            .iter()
            .find(|(id, _)| *id == OUTLIER_LABEL)
            .unwrap();
        assert!((outlier.1 - 0.5).abs() < 1e-6);
    }
}
