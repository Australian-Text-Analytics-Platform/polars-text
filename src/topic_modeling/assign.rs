//! Give outlier Topic Segments the topic they are most similar to.
//!
//! Why this exists: HDBSCAN leaves a segment out when it sits in a sparse
//! area at the moment a topic separates from its neighbours. On varied text
//! that is often more than half of the segments, and most of them lie between
//! or beside topics rather than far from all of them. Researchers who read
//! topic sizes as shares of a corpus cannot use a result that leaves half of
//! it out (Monika, Chao, Wordflow 2026-11). After clustering, each outlier
//! segment takes the topic whose centre it is most similar to, when that
//! similarity reaches a threshold; the rest stay outliers. Topics are still
//! found by density, so this adds no topics and allows no smaller ones.
//! BERTopic calls this outlier reduction with the embeddings strategy.
//!
//! A topic's centre is the mean of its members' embeddings, in the original
//! embedding space, before any segment is assigned, so assignments do not
//! pull the centres towards themselves and the result does not depend on the
//! order segments are visited in.
//!
//! Called by: `topic_modeling::run` after clustering (and after topic
//! sampling's nearest-neighbour step), before coverage is rolled up.

use std::sync::Arc;

use rayon::prelude::*;

use super::cluster::OUTLIER_LABEL;

fn norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

/// Unit-length mean embedding of each topic's members; `None` for a topic
/// with no member or a zero mean.
fn topic_centres(
    embeddings: &[Arc<Vec<f32>>],
    labels: &[i32],
    n_topics: usize,
) -> Vec<Option<Vec<f32>>> {
    let width = embeddings.first().map_or(0, |embedding| embedding.len());
    let mut sums = vec![vec![0f64; width]; n_topics];
    let mut counts = vec![0usize; n_topics];
    for (embedding, &label) in embeddings.iter().zip(labels) {
        let Ok(topic) = usize::try_from(label) else {
            continue;
        };
        if topic >= n_topics {
            continue;
        }
        // Each member counts equally, whatever its length.
        let length = f64::from(norm(embedding).max(f32::MIN_POSITIVE));
        for (sum, value) in sums[topic].iter_mut().zip(embedding.iter()) {
            *sum += f64::from(*value) / length;
        }
        counts[topic] += 1;
    }
    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            if count == 0 {
                return None;
            }
            let centre = sum.iter().map(|value| *value as f32).collect::<Vec<_>>();
            let length = norm(&centre);
            (length > 0.0).then(|| centre.iter().map(|value| value / length).collect())
        })
        .collect()
}

/// How similar an outlier must be to its nearest topic's centre to join it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JoinRule {
    /// At least this cosine similarity, the same for every topic.
    Similarity(f32),
    /// At least as similar as this share of the topic's own members are: 0.1
    /// admits an outlier that is as close to the centre as the topic's
    /// least similar tenth. The bar follows each topic's own spread, so it
    /// means the same on a varied corpus and on a uniform one.
    MemberQuantile(f32),
}

/// Cosine similarity of `embedding` to a unit-length `centre`.
fn similarity_to(embedding: &[f32], centre: &[f32]) -> Option<f32> {
    let length = norm(embedding);
    (length > 0.0).then(|| {
        embedding
            .iter()
            .zip(centre)
            .map(|(left, right)| left * right)
            .sum::<f32>()
            / length
    })
}

/// The similarity an outlier needs to join each topic under `rule`.
fn topic_thresholds(
    embeddings: &[Arc<Vec<f32>>],
    labels: &[i32],
    centres: &[Option<Vec<f32>>],
    rule: JoinRule,
) -> Vec<f32> {
    let quantile = match rule {
        JoinRule::Similarity(min_similarity) => return vec![min_similarity; centres.len()],
        JoinRule::MemberQuantile(quantile) => quantile.clamp(0.0, 1.0),
    };
    let mut similarities = vec![Vec::<f32>::new(); centres.len()];
    for (embedding, &label) in embeddings.iter().zip(labels) {
        let Ok(topic) = usize::try_from(label) else {
            continue;
        };
        let Some(Some(centre)) = centres.get(topic) else {
            continue;
        };
        if let Some(similarity) = similarity_to(embedding, centre) {
            similarities[topic].push(similarity);
        }
    }
    similarities
        .into_iter()
        .map(|mut values| {
            if values.is_empty() {
                return f32::INFINITY;
            }
            values.sort_by(f32::total_cmp);
            // Nearest rank: at least `quantile` of the members are at or below it.
            let rank = ((quantile * values.len() as f32).ceil() as usize).clamp(1, values.len());
            values[rank - 1]
        })
        .collect()
}

/// Relabels outlier segments in place and returns how many were given a topic.
///
/// An outlier takes the topic with the most similar centre (ties go to the
/// lower topic id) when its similarity meets `rule` for that topic.
pub fn assign_outliers_to_nearest_topic(
    embeddings: &[Arc<Vec<f32>>],
    labels: &mut [i32],
    n_topics: usize,
    rule: JoinRule,
) -> usize {
    debug_assert_eq!(embeddings.len(), labels.len());
    if n_topics == 0 {
        return 0;
    }
    let centres = topic_centres(embeddings, labels, n_topics);
    let thresholds = topic_thresholds(embeddings, labels, &centres, rule);
    let assignments: Vec<(usize, i32)> = labels
        .par_iter()
        .enumerate()
        .filter(|&(_, &label)| label == OUTLIER_LABEL)
        .filter_map(|(index, _)| {
            let embedding = embeddings[index].as_slice();
            let mut best: Option<(usize, f32)> = None;
            for (topic, centre) in centres.iter().enumerate() {
                let Some(centre) = centre else { continue };
                let similarity = similarity_to(embedding, centre)?;
                if best.is_none_or(|(_, top)| similarity > top) {
                    best = Some((topic, similarity));
                }
            }
            best.filter(|&(topic, similarity)| similarity >= thresholds[topic])
                .map(|(topic, _)| (index, topic as i32))
        })
        .collect();
    for &(index, topic) in &assignments {
        labels[index] = topic;
    }
    assignments.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(rows: &[[f32; 2]]) -> Vec<Arc<Vec<f32>>> {
        rows.iter().map(|row| Arc::new(row.to_vec())).collect()
    }

    #[test]
    fn outliers_join_the_most_similar_topic_above_the_threshold() {
        // Topic 0 points along x, topic 1 along y.
        let embeddings = points(&[
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.8, 0.3],   // near topic 0
            [0.2, 0.7],   // near topic 1
            [-1.0, -1.0], // opposite to both
        ]);
        let mut labels = vec![0, 0, 1, 1, OUTLIER_LABEL, OUTLIER_LABEL, OUTLIER_LABEL];
        let assigned = assign_outliers_to_nearest_topic(
            &embeddings,
            &mut labels,
            2,
            JoinRule::Similarity(0.5),
        );
        assert_eq!(assigned, 2);
        assert_eq!(labels, vec![0, 0, 1, 1, 0, 1, OUTLIER_LABEL]);
    }

    #[test]
    fn a_zero_threshold_still_leaves_dissimilar_segments_out() {
        let embeddings = points(&[[1.0, 0.0], [1.0, 0.1], [-1.0, 0.0]]);
        let mut labels = vec![0, 0, OUTLIER_LABEL];
        // Cosine similarity is negative, below a threshold of zero.
        assert_eq!(
            assign_outliers_to_nearest_topic(
                &embeddings,
                &mut labels,
                1,
                JoinRule::Similarity(0.0)
            ),
            0
        );
        assert_eq!(labels[2], OUTLIER_LABEL);
        // The lowest threshold assigns every one.
        assert_eq!(
            assign_outliers_to_nearest_topic(
                &embeddings,
                &mut labels,
                1,
                JoinRule::Similarity(-1.0)
            ),
            1
        );
        assert_eq!(labels[2], 0);
    }

    #[test]
    fn centres_come_from_members_before_any_assignment() {
        // Two outliers on the far side of topic 0's centre: the first must not
        // drag the centre towards the second.
        let embeddings = points(&[[1.0, 0.0], [1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]);
        let mut labels = vec![0, 0, OUTLIER_LABEL, OUTLIER_LABEL];
        let assigned = assign_outliers_to_nearest_topic(
            &embeddings,
            &mut labels,
            1,
            JoinRule::Similarity(0.5),
        );
        assert_eq!(assigned, 1);
        assert_eq!(labels, vec![0, 0, 0, OUTLIER_LABEL]);
    }

    #[test]
    fn a_member_quantile_sets_each_topic_its_own_bar() {
        // Topic 0 is tight around x; topic 1 is loose around y.
        let embeddings = points(&[
            [1.0, 0.0],
            [1.0, 0.02],
            [1.0, -0.02],
            [0.0, 1.0],
            [0.5, 1.0],
            [-0.5, 1.0],
            [1.0, 0.3], // nearest topic 0, but looser than any of its members
            [0.3, 1.0], // nearest topic 1, within its members' spread
        ]);
        let mut labels = vec![0, 0, 0, 1, 1, 1, OUTLIER_LABEL, OUTLIER_LABEL];
        let assigned = assign_outliers_to_nearest_topic(
            &embeddings,
            &mut labels,
            2,
            JoinRule::MemberQuantile(0.0),
        );
        assert_eq!(assigned, 1);
        assert_eq!(labels[6], OUTLIER_LABEL);
        assert_eq!(labels[7], 1);
    }

    #[test]
    fn nothing_is_assigned_without_topics() {
        let embeddings = points(&[[1.0, 0.0]]);
        let mut labels = vec![OUTLIER_LABEL];
        assert_eq!(
            assign_outliers_to_nearest_topic(
                &embeddings,
                &mut labels,
                0,
                JoinRule::Similarity(0.0)
            ),
            0
        );
    }
}
