//! Hybrid ranking fusion.
//!
//! Pure Rust with no Postgres dependency: the SQL hybrid-search wrapper feeds
//! ranked id lists from the vector index and native full-text search into
//! these functions, keeping the math unit-testable in isolation.
//!
//! Fusion is a pluggable [`FusionStrategy`]: Reciprocal Rank Fusion is the
//! rank-only, zero-tuning default; score-based strategies (RSF / DBSF) can
//! be added behind the same interface later.

use std::collections::HashMap;
use std::fmt;

/// Error returned by fusion for malformed parameters.
///
/// No `PartialEq`: `InvalidK` carries the offending `f64` (possibly NaN, the
/// very value it rejects), and NaN makes derived equality non-reflexive.
/// Assert on it with `matches!` instead.
#[derive(Debug, Clone, Copy)]
pub enum FusionError {
    /// RRF's `k` must be finite and non-negative so every `1 / (k + rank)`
    /// contribution is finite and positive.
    InvalidK { k: f64 },
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionError::InvalidK { k } => {
                write!(f, "invalid RRF k: {k} (must be finite and >= 0)")
            }
        }
    }
}

impl std::error::Error for FusionError {}

/// Robust zero-tuning default for RRF's `k` constant, per the original RRF
/// paper and common practice across search engines.
pub const DEFAULT_RRF_K: f64 = 60.0;

/// A strategy for fusing per-source rankings into one combined ranking.
///
/// Rank-based today; when score-based strategies land, the interface grows
/// to carry per-source scores, with rank-only strategies ignoring them.
pub trait FusionStrategy {
    /// Fuse ranked id lists (best first, rank starting at 1) into
    /// `(id, fused_score)` pairs sorted by score descending, ties broken by
    /// ascending id for determinism. Errs on invalid strategy parameters.
    fn fuse(&self, rankings: &[&[u64]]) -> Result<Vec<(u64, f64)>, FusionError>;
}

/// Reciprocal Rank Fusion: `score(d) = Σ over lists of 1 / (k + rank_d)`.
#[derive(Debug, Clone, Copy)]
pub struct Rrf {
    /// Dampening constant; larger values flatten the gap between ranks.
    pub k: f64,
}

impl Default for Rrf {
    fn default() -> Self {
        Self { k: DEFAULT_RRF_K }
    }
}

impl FusionStrategy for Rrf {
    fn fuse(&self, rankings: &[&[u64]]) -> Result<Vec<(u64, f64)>, FusionError> {
        rrf(rankings, self.k)
    }
}

/// Fuse ranked id lists with Reciprocal Rank Fusion.
///
/// Each list is read best-first with ranks starting at 1; an id accumulates
/// `1 / (k + rank)` from every list it appears in, so items ranked decently in
/// several sources beat items ranked well in just one. Ids may appear in only
/// some lists; a duplicate id within one list counts once, at its best (first)
/// rank. A non-finite or negative `k` is rejected: it would make some
/// contributions infinite or negative and silently invert the ranking.
///
/// Ids are opaque to fusion: every ranking must draw from one shared id
/// space, and the caller owns the encoding (fusing lists keyed by different
/// id schemes silently coalesces unrelated items).
///
/// Allocates only the accumulator map and the output vector.
pub fn rrf(rankings: &[&[u64]], k: f64) -> Result<Vec<(u64, f64)>, FusionError> {
    if !k.is_finite() || k < 0.0 {
        return Err(FusionError::InvalidK { k });
    }
    // Value = (accumulated score, index of the last list that contributed).
    // The second field dedups repeats within a list without per-list sets: a
    // repeat is skipped, and the first occurrence is already the best rank.
    let mut scores: HashMap<u64, (f64, usize)> =
        HashMap::with_capacity(rankings.iter().map(|r| r.len()).sum());
    for (list_idx, ranking) in rankings.iter().enumerate() {
        for (pos, &id) in ranking.iter().enumerate() {
            let contribution = 1.0 / (k + (pos + 1) as f64);
            scores
                .entry(id)
                .and_modify(|(score, last_list)| {
                    if *last_list != list_idx {
                        *score += contribution;
                        *last_list = list_idx;
                    }
                })
                .or_insert((contribution, list_idx));
        }
    }
    let mut fused: Vec<(u64, f64)> = scores
        .into_iter()
        .map(|(id, (score, _))| (id, score))
        .collect();
    fused.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(fused)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn ids(fused: &[(u64, f64)]) -> Vec<u64> {
        fused.iter().map(|&(id, _)| id).collect()
    }

    #[test]
    fn worked_example_matches_hand_computed_scores() {
        // List 1: 10 → rank 1, 20 → 2, 30 → 3.
        // List 2: 20 → rank 1, 40 → 2, 10 → 3.
        let fused = rrf(&[&[10, 20, 30], &[20, 40, 10]], 60.0).unwrap();
        let expected = [
            (20, 1.0 / 61.0 + 1.0 / 62.0),
            (10, 1.0 / 61.0 + 1.0 / 63.0),
            (40, 1.0 / 62.0),
            (30, 1.0 / 63.0),
        ];
        assert_eq!(fused.len(), expected.len());
        for ((id, score), (want_id, want_score)) in fused.iter().zip(expected) {
            assert_eq!(*id, want_id);
            assert!((score - want_score).abs() < EPS, "id {id}: got {score}");
        }
    }

    #[test]
    fn consensus_beats_single_list_number_one() {
        // 2 is ranked #2 in both lists; 1 and 4 are each #1 in one list only.
        let fused = rrf(&[&[1, 2, 3], &[4, 2, 3]], DEFAULT_RRF_K).unwrap();
        assert_eq!(ids(&fused)[0], 2, "consensus item must win: {fused:?}");
    }

    #[test]
    fn exact_ties_break_by_ascending_id() {
        // 1 and 4 score identically (rank 1 in one list each); so do 3 and 9.
        let fused = rrf(&[&[4, 9], &[1, 3]], DEFAULT_RRF_K).unwrap();
        assert_eq!(ids(&fused), vec![1, 4, 3, 9]);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(rrf(&[], DEFAULT_RRF_K).unwrap().is_empty());
        assert!(rrf(&[&[], &[]], DEFAULT_RRF_K).unwrap().is_empty());
    }

    #[test]
    fn single_list_preserves_its_order() {
        let fused = rrf(&[&[7, 3, 9]], DEFAULT_RRF_K).unwrap();
        assert_eq!(ids(&fused), vec![7, 3, 9]);
        assert!(fused.windows(2).all(|w| w[0].1 > w[1].1));
    }

    #[test]
    fn duplicate_within_a_list_counts_once_at_best_rank() {
        let fused = rrf(&[&[5, 6, 5]], 60.0).unwrap();
        assert_eq!(ids(&fused), vec![5, 6]);
        assert!((fused[0].1 - 1.0 / 61.0).abs() < EPS, "got {}", fused[0].1);
        // ...and the duplicate must not block a later list's contribution.
        let fused = rrf(&[&[5, 6, 5], &[5]], 60.0).unwrap();
        assert!((fused[0].1 - 2.0 / 61.0).abs() < EPS, "got {}", fused[0].1);
    }

    #[test]
    fn strategy_default_is_k_60() {
        let via_strategy = Rrf::default().fuse(&[&[10, 20], &[20, 10]]).unwrap();
        let direct = rrf(&[&[10, 20], &[20, 10]], 60.0).unwrap();
        assert_eq!(via_strategy, direct);
    }

    #[test]
    fn invalid_k_is_rejected() {
        assert!(matches!(
            rrf(&[&[1, 2]], -3.0),
            Err(FusionError::InvalidK { k }) if k == -3.0
        ));
        assert!(rrf(&[&[1]], f64::NAN).is_err());
        assert!(rrf(&[&[1]], f64::INFINITY).is_err());
        assert!(Rrf { k: -1.0 }.fuse(&[&[1]]).is_err());
        // k = 0 is a legitimate degenerate choice (score = 1/rank).
        assert!(rrf(&[&[1]], 0.0).is_ok());
    }
}
