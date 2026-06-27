//! In-memory HNSW (Hierarchical Navigable Small World) graph — Brindle's ANN
//! index core. Pure Rust over `&[f32]`; no Postgres dependencies, so it's
//! unit-testable and benchmarkable on its own.
//!
//! Node ids are dense `usize` assigned in insertion order. Neighbor lists live in
//! `links[node][layer]`. Distances come from [`Metric`] (smaller = nearer).
//!
//! Build follows Malkov & Yashunin (<https://arxiv.org/abs/1603.09320>): random
//! level assignment, greedy descent, per-layer beam search, and the diversity
//! heuristic for neighbor selection. Pruning maintains a per-layer degree cap, so
//! edges are bidirectional *at insertion* but a later prune may drop one side —
//! the maintained invariant is the degree cap, not strict symmetry.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use crate::distance::DistanceError;
use crate::vector::Metric;

/// Errors from index operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HnswError {
    /// A vector's length didn't match the index dimensionality.
    DimensionMismatch { expected: usize, got: usize },
    /// An empty vector was supplied.
    EmptyVector,
}

impl std::fmt::Display for HnswError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HnswError::DimensionMismatch { expected, got } => {
                write!(f, "vector dimension mismatch: expected {expected}, got {got}")
            }
            HnswError::EmptyVector => write!(f, "vector must not be empty"),
        }
    }
}

impl std::error::Error for HnswError {}

impl From<DistanceError> for HnswError {
    fn from(e: DistanceError) -> Self {
        match e {
            DistanceError::DimensionMismatch { left, right } => {
                HnswError::DimensionMismatch { expected: left, got: right }
            }
        }
    }
}

/// Build/query parameters.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Neighbors per node on upper layers (layer 0 uses `2*m`).
    pub m: usize,
    /// Candidate-pool size during build (larger = better graph, slower build).
    pub ef_construction: usize,
    /// Distance metric.
    pub metric: Metric,
    /// PRNG seed for reproducible level assignment.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 64,
            metric: Metric::L2,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// A scored candidate `(distance, node id)`. Ordered by distance then id (total
/// order over `f32` via `total_cmp`) for deterministic tie-breaking.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cand {
    dist: f32,
    id: usize,
}

impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist).then(self.id.cmp(&other.id))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform `f64` in `[0, 1)` from the 53 high bits.
#[inline]
fn next_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// In-memory HNSW graph.
pub struct Hnsw {
    metric: Metric,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f64,
    dim: usize,
    vectors: Vec<Vec<f32>>,
    links: Vec<Vec<Vec<usize>>>,
    entry_point: Option<usize>,
    max_layer: usize,
    rng: u64,
}

impl Hnsw {
    /// Create an empty index. `m` is clamped to ≥ 2; dimensionality is fixed by
    /// the first inserted vector.
    pub fn new(params: HnswParams) -> Self {
        let m = params.m.max(2);
        let seed = if params.seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            params.seed
        };
        Self {
            metric: params.metric,
            m,
            m0: m * 2,
            ef_construction: params.ef_construction.max(m),
            ml: 1.0 / (m as f64).ln(),
            dim: 0,
            vectors: Vec::new(),
            links: Vec::new(),
            entry_point: None,
            max_layer: 0,
            rng: seed,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    #[inline]
    fn max_degree(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m0
        } else {
            self.m
        }
    }

    #[inline]
    fn neighbors(&self, id: usize, layer: usize) -> &[usize] {
        self.links[id]
            .get(layer)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    fn random_level(&mut self) -> usize {
        let mut r = next_f64(&mut self.rng);
        if r <= 0.0 {
            r = f64::MIN_POSITIVE;
        }
        (-r.ln() * self.ml).floor() as usize
    }

    /// Greedy beam search within one layer (HNSW SEARCH-LAYER). Returns up to `ef`
    /// nearest candidates to `query`, ascending by distance.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Result<Vec<Cand>, HnswError> {
        let mut visited: HashSet<usize> = HashSet::with_capacity(ef.max(1) * 8);
        let mut frontier: BinaryHeap<Reverse<Cand>> = BinaryHeap::new(); // nearest on top
        let mut results: BinaryHeap<Cand> = BinaryHeap::new(); // farthest on top

        for &ep in entry_points {
            if visited.insert(ep) {
                let d = self.metric.distance(query, &self.vectors[ep])?;
                let c = Cand { dist: d, id: ep };
                frontier.push(Reverse(c));
                results.push(c);
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        while let Some(Reverse(c)) = frontier.pop() {
            let farthest = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest {
                break;
            }
            let degree = self.neighbors(c.id, layer).len();
            for idx in 0..degree {
                let n = self.neighbors(c.id, layer)[idx];
                if visited.insert(n) {
                    let d = self.metric.distance(query, &self.vectors[n])?;
                    let farthest = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                    if results.len() < ef || d < farthest {
                        let nc = Cand { dist: d, id: n };
                        frontier.push(Reverse(nc));
                        results.push(nc);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out = results.into_vec();
        out.sort_unstable();
        Ok(out)
    }

    /// Malkov's neighbor-selection heuristic (Alg. 4): prefer candidates closer to
    /// the base than to any already-selected neighbor (keeps the graph diverse and
    /// navigable), backfilling with the nearest leftovers up to `m`.
    fn select_neighbors_heuristic(
        &self,
        candidates: &[Cand],
        m: usize,
    ) -> Result<Vec<usize>, HnswError> {
        let mut sorted = candidates.to_vec();
        sorted.sort_unstable(); // ascending by distance to the base
        let mut selected: Vec<usize> = Vec::with_capacity(m);

        for cand in &sorted {
            if selected.len() >= m {
                break;
            }
            let mut keep = true;
            for &r in &selected {
                let d = self.metric.distance(&self.vectors[cand.id], &self.vectors[r])?;
                if d < cand.dist {
                    keep = false;
                    break;
                }
            }
            if keep {
                selected.push(cand.id);
            }
        }

        if selected.len() < m {
            for cand in &sorted {
                if selected.len() >= m {
                    break;
                }
                if !selected.contains(&cand.id) {
                    selected.push(cand.id);
                }
            }
        }

        Ok(selected)
    }

    /// Insert a vector, returning its node id. Dimensionality is fixed by the first
    /// insert; later mismatches error rather than panic.
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<usize, HnswError> {
        if vector.is_empty() {
            return Err(HnswError::EmptyVector);
        }
        if self.dim == 0 {
            self.dim = vector.len();
        } else if vector.len() != self.dim {
            return Err(HnswError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let id = self.vectors.len();
        let level = self.random_level();

        self.vectors.push(vector);
        let mut node_links: Vec<Vec<usize>> = Vec::with_capacity(level + 1);
        node_links.resize_with(level + 1, Vec::new);
        self.links.push(node_links);

        let entry = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                self.max_layer = level;
                return Ok(id);
            }
            Some(e) => e,
        };

        let query = self.vectors[id].clone();
        let max_layer = self.max_layer;
        let mut ep_ids = vec![entry];

        // Greedy descent from the top down to just above the new node's level.
        if max_layer > level {
            for lc in ((level + 1)..=max_layer).rev() {
                let w = self.search_layer(&query, &ep_ids, 1, lc)?;
                if let Some(nearest) = w.first() {
                    ep_ids = vec![nearest.id];
                }
            }
        }

        // Connect at each layer from min(level, max_layer) down to 0.
        let start = level.min(max_layer);
        for lc in (0..=start).rev() {
            let w = self.search_layer(&query, &ep_ids, self.ef_construction, lc)?;
            let max_deg = self.max_degree(lc);
            let selected = self.select_neighbors_heuristic(&w, max_deg)?;

            self.links[id][lc] = selected.clone();
            for &n in &selected {
                self.links[n][lc].push(id);
                if self.links[n][lc].len() > max_deg {
                    let nbase = self.vectors[n].clone();
                    let mut cands = Vec::with_capacity(self.links[n][lc].len());
                    for &nn in &self.links[n][lc] {
                        let d = self.metric.distance(&nbase, &self.vectors[nn])?;
                        cands.push(Cand { dist: d, id: nn });
                    }
                    self.links[n][lc] = self.select_neighbors_heuristic(&cands, max_deg)?;
                }
            }

            ep_ids = w.iter().map(|c| c.id).collect();
            if ep_ids.is_empty() {
                ep_ids = vec![entry];
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(id);
        }

        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an index over `n` deterministic random vectors of dimension `dim`.
    fn build(n: usize, dim: usize, seed: u64) -> (Hnsw, Vec<Vec<f32>>) {
        let mut data_rng = seed ^ 0xABCD_EF01;
        let mut h = Hnsw::new(HnswParams {
            m: 8,
            ef_construction: 50,
            metric: Metric::L2,
            seed,
        });
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next_f64(&mut data_rng) as f32).collect();
            h.insert(v.clone()).expect("insert");
            data.push(v);
        }
        (h, data)
    }

    #[test]
    fn len_and_dim() {
        let (h, _) = build(100, 8, 1);
        assert_eq!(h.len(), 100);
        assert_eq!(h.dim(), 8);
        assert!(!h.is_empty());
    }

    #[test]
    fn degree_cap_respected() {
        let (h, _) = build(300, 12, 7);
        for id in 0..h.len() {
            for (lc, nbrs) in h.links[id].iter().enumerate() {
                assert!(
                    nbrs.len() <= h.max_degree(lc),
                    "node {id} layer {lc}: {} neighbors > cap {}",
                    nbrs.len(),
                    h.max_degree(lc)
                );
            }
        }
    }

    #[test]
    fn deterministic_for_seed() {
        let (h1, _) = build(150, 10, 42);
        let (h2, _) = build(150, 10, 42);
        assert_eq!(h1.links, h2.links);
        assert_eq!(h1.entry_point, h2.entry_point);
        assert_eq!(h1.max_layer, h2.max_layer);
    }

    #[test]
    fn empty_index_state() {
        let h = Hnsw::new(HnswParams::default());
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut h = Hnsw::new(HnswParams::default());
        h.insert(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(
            h.insert(vec![1.0, 2.0]).unwrap_err(),
            HnswError::DimensionMismatch { expected: 3, got: 2 }
        );
    }

    #[test]
    fn empty_vector_errors() {
        let mut h = Hnsw::new(HnswParams::default());
        assert_eq!(h.insert(vec![]).unwrap_err(), HnswError::EmptyVector);
    }
}
