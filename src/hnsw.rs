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
//!
//! The degree caps scale with an edge-density multiplier `gamma` (γ) following
//! ACORN (<https://arxiv.org/abs/2403.04871>): a graph built with γ > 1 keeps
//! ~`m·γ` neighbors per node, so the subgraph of nodes surviving a selective
//! predicate stays navigable. γ = 1 is a standard HNSW.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use crate::distance::DistanceError;
use crate::filter::AttrValue;
use crate::vector::Metric;

/// Errors from index operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HnswError {
    /// A vector's length didn't match the index dimensionality.
    DimensionMismatch { expected: usize, got: usize },
    /// An empty vector was supplied.
    EmptyVector,
    /// An operation referenced a node id that doesn't exist.
    UnknownNode { id: usize },
}

impl std::fmt::Display for HnswError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HnswError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "vector dimension mismatch: expected {expected}, got {got}"
                )
            }
            HnswError::EmptyVector => write!(f, "vector must not be empty"),
            HnswError::UnknownNode { id } => write!(f, "unknown node id: {id}"),
        }
    }
}

impl std::error::Error for HnswError {}

impl From<DistanceError> for HnswError {
    fn from(e: DistanceError) -> Self {
        match e {
            DistanceError::DimensionMismatch { left, right } => HnswError::DimensionMismatch {
                expected: left,
                got: right,
            },
        }
    }
}

/// Build/query parameters.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Neighbors per node on upper layers before γ scaling (layer 0 uses `2*m`;
    /// [`HnswParams::gamma`] multiplies both caps).
    pub m: usize,
    /// Candidate-pool size during build (larger = better graph, slower build).
    pub ef_construction: usize,
    /// Edge-density multiplier (ACORN's γ): degree caps scale to `m·γ` (layer 0
    /// to `2m·γ`) so the graph stays navigable for predicates down to selectivity
    /// ~`1/γ`. `1.0` builds a standard HNSW; non-finite values are treated as 1,
    /// finite values clamp to `[1, MAX_GAMMA]`. Density costs ~γ× link memory
    /// and superlinearly more build time (neighbor selection compares a γ-sized
    /// candidate pool against γ-sized neighbor lists).
    pub gamma: f32,
    /// Distance metric.
    pub metric: Metric,
    /// PRNG seed for reproducible level assignment.
    pub seed: u64,
}

impl HnswParams {
    /// Upper bound for [`HnswParams::gamma`]. Selectivity down to 0.1% needs
    /// γ ≈ 1000, so 1024 covers the documented use cases while keeping the
    /// γ-scaled degree caps (and the allocations sized from them) bounded.
    pub const MAX_GAMMA: f32 = 1024.0;
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 64,
            gamma: 1.0,
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
        self.dist
            .total_cmp(&other.dist)
            .then(self.id.cmp(&other.id))
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
    gamma: f32,
    /// γ-scaled degree cap for upper layers (γ = 1 ⇒ `m`).
    m_cap: usize,
    /// γ-scaled degree cap for layer 0 (γ = 1 ⇒ `2m`).
    m0_cap: usize,
    ef_construction: usize,
    ml: f64,
    dim: usize,
    seed: u64,
    // Invariant: `vectors`, `links`, and `deleted` are parallel arrays indexed by
    // node id, and nodes are never removed (delete only tombstones), so every id
    // stored in `links` or `entry_point` remains a valid index for all three.
    // This is what makes the raw `[id]` indexing on the search/insert paths
    // panic-free.
    vectors: Vec<Vec<f32>>,
    links: Vec<Vec<Vec<usize>>>,
    /// Filterable attribute rows parallel to `vectors`, keyed by node id and set
    /// at insert time. Empty for nodes inserted without attributes; consulted by
    /// predicate-aware traversal, never by graph construction.
    attrs: Vec<Vec<AttrValue>>,
    /// Tombstone flags parallel to `vectors`; deleted nodes route but never return.
    deleted: Vec<bool>,
    entry_point: Option<usize>,
    max_layer: usize,
    rng: u64,
}

/// Summarizes the graph instead of dumping vectors and adjacency lists — a
/// derived `Debug` would materialize gigabytes for a production-sized index.
impl std::fmt::Debug for Hnsw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hnsw")
            .field("len", &self.len())
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("m", &self.m)
            .field("max_layer", &self.max_layer)
            .field("entry_point", &self.entry_point)
            .finish_non_exhaustive()
    }
}

impl Hnsw {
    /// γ-scaled degree caps, the level normalizer, and the ef floor, derived
    /// from `m` and a raw `gamma`. `gamma` is clamped to `[1, MAX_GAMMA]`
    /// (non-finite ⇒ 1). One definition shared by construction and decoding so
    /// the two can't drift. Returns `(gamma, m_cap, m0_cap, ml, ef_floor)`.
    fn derived_params(m: usize, gamma: f32) -> (f32, usize, usize, f64, usize) {
        // γ below 1 would only thin the graph (defeating its purpose) and an
        // unbounded γ would blow the degree caps up into absurd allocation
        // sizes, so non-finite values mean "no densification" and finite ones
        // clamp.
        let gamma = if gamma.is_finite() {
            gamma.clamp(1.0, HnswParams::MAX_GAMMA)
        } else {
            1.0
        };
        let m_cap = (m as f32 * gamma).round() as usize;
        let m0_cap = ((m * 2) as f32 * gamma).round() as usize;
        // Level assignment stays tied to the base `m`: γ densifies within layers
        // without reshaping the hierarchy.
        let ml = 1.0 / (m as f64).ln();
        // A γ-dense layer-0 list only fills if the build pool supplies that many
        // candidates; without densification keep the historical `m` floor so
        // existing builds reproduce exactly.
        let ef_floor = if gamma > 1.0 { m0_cap } else { m };
        (gamma, m_cap, m0_cap, ml, ef_floor)
    }

    /// Create an empty index. `m` is clamped to ≥ 2 and `gamma` to ≥ 1;
    /// dimensionality is fixed by the first inserted vector.
    pub fn new(params: HnswParams) -> Self {
        let m = params.m.max(2);
        let (gamma, m_cap, m0_cap, ml, ef_floor) = Self::derived_params(m, params.gamma);
        let seed = if params.seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            params.seed
        };
        Self {
            metric: params.metric,
            m,
            gamma,
            m_cap,
            m0_cap,
            ef_construction: params.ef_construction.max(ef_floor),
            ml,
            dim: 0,
            seed,
            vectors: Vec::new(),
            links: Vec::new(),
            attrs: Vec::new(),
            deleted: Vec::new(),
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

    /// Filterable attribute row stored for `id`, in the order it was inserted.
    /// An unknown id or an attribute-free node yields an empty slice (so a
    /// predicate over a missing column simply doesn't match — never panics).
    #[inline]
    pub fn attrs(&self, id: usize) -> &[AttrValue] {
        self.attrs.get(id).map(Vec::as_slice).unwrap_or(&[])
    }

    #[inline]
    fn max_degree(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m0_cap
        } else {
            self.m_cap
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
                frontier.push(Reverse(c)); // route through any node, incl. tombstoned
                if !self.deleted[ep] {
                    results.push(c);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        while let Some(Reverse(c)) = frontier.pop() {
            let farthest = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest {
                break;
            }
            for &n in self.neighbors(c.id, layer) {
                if visited.insert(n) {
                    let d = self.metric.distance(query, &self.vectors[n])?;
                    let farthest = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                    if results.len() < ef || d < farthest {
                        let nc = Cand { dist: d, id: n };
                        frontier.push(Reverse(nc)); // route through tombstoned nodes
                        if !self.deleted[n] {
                            results.push(nc);
                            if results.len() > ef {
                                results.pop();
                            }
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
                let d = self
                    .metric
                    .distance(&self.vectors[cand.id], &self.vectors[r])?;
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

    /// Insert a vector with no filterable attributes, returning its node id.
    /// Equivalent to [`Hnsw::insert_with_attrs`] with an empty row.
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<usize, HnswError> {
        self.insert_with_attrs(vector, Vec::new())
    }

    /// Insert a vector and its filterable attribute row, returning its node id.
    /// The row is stored verbatim, keyed by the id, for predicate-aware
    /// traversal; it does not influence graph construction, so a graph built with
    /// or without attributes is byte-identical. Dimensionality is fixed by the
    /// first insert; later mismatches error rather than panic.
    pub fn insert_with_attrs(
        &mut self,
        vector: Vec<f32>,
        attrs: Vec<AttrValue>,
    ) -> Result<usize, HnswError> {
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
        self.attrs.push(attrs);
        self.deleted.push(false);
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

    /// Approximate k-nearest-neighbor search. Returns up to `k` `(distance, id)`
    /// pairs, nearest first. `ef_search` is the layer-0 candidate budget (raised to
    /// at least `k`); larger means higher recall and more work. Distances are in
    /// the metric's internal form (squared, for `Metric::L2`).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(f32, usize)>, HnswError> {
        let entry = match self.entry_point {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };
        if k == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dim {
            return Err(HnswError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }

        let mut ep_ids = vec![entry];
        for lc in (1..=self.max_layer).rev() {
            let w = self.search_layer(query, &ep_ids, 1, lc)?;
            if let Some(nearest) = w.first() {
                ep_ids = vec![nearest.id];
            }
        }

        let ef = ef_search.max(k);
        let mut w = self.search_layer(query, &ep_ids, ef, 0)?;
        w.truncate(k);
        Ok(w.into_iter().map(|c| (c.dist, c.id)).collect())
    }

    /// Exact brute-force k-NN over all stored vectors. Used as the recall ceiling
    /// in tests/benchmarks (and as a correctness oracle).
    pub fn brute_force(&self, query: &[f32], k: usize) -> Result<Vec<(f32, usize)>, HnswError> {
        if !self.is_empty() && query.len() != self.dim {
            return Err(HnswError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut all: Vec<Cand> = Vec::with_capacity(self.vectors.len());
        for (id, v) in self.vectors.iter().enumerate() {
            if self.deleted[id] {
                continue;
            }
            all.push(Cand {
                dist: self.metric.distance(query, v)?,
                id,
            });
        }
        all.sort_unstable();
        all.truncate(k);
        Ok(all.into_iter().map(|c| (c.dist, c.id)).collect())
    }

    /// Number of tombstoned (soft-deleted) nodes.
    pub fn deleted_count(&self) -> usize {
        self.deleted.iter().filter(|&&d| d).count()
    }

    /// Number of live (non-deleted) nodes.
    pub fn live_len(&self) -> usize {
        self.len() - self.deleted_count()
    }

    /// Soft-delete a node: it stays in the graph as a routing-only stepping stone
    /// but is never returned by [`Hnsw::search`] / [`Hnsw::brute_force`].
    /// Idempotent. Errors (never panics) on an unknown id. Reclaim space later
    /// with [`Hnsw::compact`].
    pub fn delete(&mut self, id: usize) -> Result<(), HnswError> {
        if id >= self.deleted.len() {
            return Err(HnswError::UnknownNode { id });
        }
        self.deleted[id] = true;
        Ok(())
    }

    /// Rebuild the graph from the live nodes only, dropping all tombstones, and
    /// return the new node count. Node ids are **not** stable across compaction
    /// (they're reassigned densely); a caller holding an external id↔node mapping
    /// must rebuild it afterwards. Each live node's attribute row moves with it.
    pub fn compact(&mut self) -> Result<usize, HnswError> {
        let mut live: Vec<(Vec<f32>, Vec<AttrValue>)> = Vec::with_capacity(self.live_len());
        for (id, &deleted) in self.deleted.iter().enumerate() {
            if !deleted {
                live.push((self.vectors[id].clone(), self.attrs[id].clone()));
            }
        }
        let mut fresh = Hnsw::new(HnswParams {
            m: self.m,
            ef_construction: self.ef_construction,
            gamma: self.gamma,
            metric: self.metric,
            seed: self.seed,
        });
        for (vector, attrs) in live {
            fresh.insert_with_attrs(vector, attrs)?;
        }
        let n = fresh.len();
        *self = fresh;
        Ok(n)
    }
}

/// Errors from decoding a graph serialized by [`Hnsw::to_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HnswDecodeError {
    /// The input ended before the encoding said it would.
    Truncated,
    /// The input doesn't start with the serialized-graph magic.
    BadMagic,
    /// The format version is one this build doesn't understand.
    UnsupportedVersion { got: u16 },
    /// A structurally impossible value (bad metric code, id out of range, …).
    Invalid(&'static str),
}

impl std::fmt::Display for HnswDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HnswDecodeError::Truncated => write!(f, "serialized graph is truncated"),
            HnswDecodeError::BadMagic => write!(f, "not a serialized graph (bad magic)"),
            HnswDecodeError::UnsupportedVersion { got } => {
                write!(f, "unsupported graph format version {got}")
            }
            HnswDecodeError::Invalid(what) => write!(f, "invalid serialized graph: {what}"),
        }
    }
}

impl std::error::Error for HnswDecodeError {}

const CODEC_MAGIC: u32 = 0x4248_4E57; // "BHNW"
const CODEC_VERSION: u16 = 1;

/// Bounds-checked little-endian reader over untrusted bytes. Every accessor
/// errors (never panics) past the end of the buffer.
struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], HnswDecodeError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.buf.len())
            .ok_or(HnswDecodeError::Truncated)?;
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, HnswDecodeError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, HnswDecodeError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    fn u32(&mut self) -> Result<u32, HnswDecodeError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self) -> Result<u64, HnswDecodeError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }

    /// A `u64` that must fit in `usize` (counts, ids, sizes).
    fn len(&mut self) -> Result<usize, HnswDecodeError> {
        usize::try_from(self.u64()?)
            .map_err(|_| HnswDecodeError::Invalid("value exceeds address space"))
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn done(&self) -> bool {
        self.pos == self.buf.len()
    }
}

impl Hnsw {
    /// Upper bound on the byte length of this graph's serialization. Exact for
    /// a freshly built graph (degree caps hold); a safe over-estimate after
    /// pruning only leaves neighbor lists shorter. Use it to size a buffer
    /// before [`Hnsw::to_bytes_into`].
    pub fn serialized_len_hint(&self) -> usize {
        let per_node = 13 + self.dim * 4 + (self.m0_cap + 2) * 8;
        72 + self.vectors.len() * per_node
    }

    /// Serialize the graph to a self-contained, versioned little-endian blob.
    /// The PRNG state is included, so a graph restored with [`Hnsw::from_bytes`]
    /// continues inserting exactly as the original would have.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.serialized_len_hint());
        self.to_bytes_into(&mut out);
        out
    }

    /// Append this graph's serialization to `out`. Lets a caller frame the
    /// graph into a larger buffer (e.g. a length-prefixed on-disk payload)
    /// without allocating a standalone graph blob first.
    pub fn to_bytes_into(&self, out: &mut Vec<u8>) {
        // This interim codec persists graph structure only, not attribute rows:
        // build-produced graphs carry none, and predicate persistence is a
        // later feature. Trip loudly if an attr-bearing graph reaches here
        // before the format is extended.
        debug_assert!(
            self.attrs.iter().all(|a| a.is_empty()),
            "codec v1 does not persist attribute rows"
        );
        out.reserve(self.serialized_len_hint());
        out.extend_from_slice(&CODEC_MAGIC.to_le_bytes());
        out.extend_from_slice(&CODEC_VERSION.to_le_bytes());
        out.push(self.metric.code());
        out.push(0); // reserved
        out.extend_from_slice(&(self.m as u64).to_le_bytes());
        out.extend_from_slice(&self.gamma.to_bits().to_le_bytes());
        out.extend_from_slice(&(self.ef_construction as u64).to_le_bytes());
        out.extend_from_slice(&(self.dim as u64).to_le_bytes());
        out.extend_from_slice(&(self.max_layer as u64).to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&self.rng.to_le_bytes());
        let entry = self.entry_point.map_or(u64::MAX, |e| e as u64);
        out.extend_from_slice(&entry.to_le_bytes());
        out.extend_from_slice(&(self.vectors.len() as u64).to_le_bytes());
        for id in 0..self.vectors.len() {
            out.push(u8::from(self.deleted[id]));
            out.extend_from_slice(&(self.links[id].len() as u64).to_le_bytes());
            for x in &self.vectors[id] {
                out.extend_from_slice(&x.to_le_bytes());
            }
            for layer in &self.links[id] {
                out.extend_from_slice(&(layer.len() as u64).to_le_bytes());
                for &neighbor in layer {
                    out.extend_from_slice(&(neighbor as u64).to_le_bytes());
                }
            }
        }
    }

    /// Reconstruct a graph serialized by [`Hnsw::to_bytes`].
    ///
    /// Validates everything needed for memory safety (magic, version, id
    /// ranges, flag values); the graph's *quality* invariants (degree caps,
    /// neighbor choice) are trusted as-is, since a well-formed-but-worse graph
    /// only degrades recall, never safety.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HnswDecodeError> {
        let mut r = ByteReader::new(bytes);
        if r.u32()? != CODEC_MAGIC {
            return Err(HnswDecodeError::BadMagic);
        }
        let version = r.u16()?;
        if version != CODEC_VERSION {
            return Err(HnswDecodeError::UnsupportedVersion { got: version });
        }
        let metric =
            Metric::from_code(r.u8()?).ok_or(HnswDecodeError::Invalid("unknown metric code"))?;
        let _reserved = r.u8()?;
        let m = r.len()?;
        if m < 2 {
            return Err(HnswDecodeError::Invalid("m out of range"));
        }
        let gamma = f32::from_bits(r.u32()?);
        if !gamma.is_finite() || !(1.0..=HnswParams::MAX_GAMMA).contains(&gamma) {
            return Err(HnswDecodeError::Invalid("gamma out of range"));
        }
        // Same derivation `new()` uses, so a decoded graph's degree caps and ef
        // floor match a freshly built one's.
        let (_, m_cap, m0_cap, ml, ef_floor) = Self::derived_params(m, gamma);
        let ef_construction = r.len()?;
        if ef_construction < ef_floor {
            return Err(HnswDecodeError::Invalid("ef_construction below floor"));
        }
        let dim = r.len()?;
        let max_layer = r.len()?;
        let seed = r.u64()?;
        let rng = r.u64()?;
        // Encoded graphs come from `new()`, which never produces a zero seed
        // or zero PRNG state (xorshift preserves nonzero-ness).
        if seed == 0 || rng == 0 {
            return Err(HnswDecodeError::Invalid("PRNG state must be nonzero"));
        }
        let entry_raw = r.u64()?;
        let n = r.len()?;
        let entry_point = if entry_raw == u64::MAX {
            None
        } else {
            let e = usize::try_from(entry_raw)
                .ok()
                .filter(|&e| e < n)
                .ok_or(HnswDecodeError::Invalid("entry point out of range"))?;
            Some(e)
        };
        if n > 0 && entry_point.is_none() {
            return Err(HnswDecodeError::Invalid(
                "nonempty graph without entry point",
            ));
        }
        if n > 0 && dim == 0 {
            return Err(HnswDecodeError::Invalid(
                "nonempty graph with zero dimension",
            ));
        }

        let vector_bytes = dim
            .checked_mul(4)
            .ok_or(HnswDecodeError::Invalid("dimension out of range"))?;
        // Pre-size from `n` only up to what the remaining bytes could possibly
        // hold, so a corrupt count can't trigger a huge allocation. Same
        // pattern below for per-node level and link counts. `saturating_add`
        // because `vector_bytes` can be near `usize::MAX` on a crafted blob.
        let min_node_bytes = 17usize.saturating_add(vector_bytes);
        let plausible_n = n.min(r.remaining() / min_node_bytes);
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(plausible_n);
        let mut links: Vec<Vec<Vec<usize>>> = Vec::with_capacity(plausible_n);
        let mut deleted: Vec<bool> = Vec::with_capacity(plausible_n);
        for _ in 0..n {
            deleted.push(match r.u8()? {
                0 => false,
                1 => true,
                _ => return Err(HnswDecodeError::Invalid("bad tombstone flag")),
            });
            let levels = r.len()?;
            if levels == 0 {
                return Err(HnswDecodeError::Invalid("node with no layers"));
            }
            if levels - 1 > max_layer {
                return Err(HnswDecodeError::Invalid("node above the top layer"));
            }
            let raw = r.take(vector_bytes)?;
            let mut vector = Vec::with_capacity(dim);
            for c in raw.chunks_exact(4) {
                vector.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            vectors.push(vector);
            if levels > r.remaining() / 8 {
                return Err(HnswDecodeError::Truncated);
            }
            let mut node_links = Vec::with_capacity(levels);
            for _ in 0..levels {
                let count = r.len()?;
                if count > r.remaining() / 8 {
                    return Err(HnswDecodeError::Truncated);
                }
                let mut layer = Vec::with_capacity(count);
                for _ in 0..count {
                    let id = r.len()?;
                    if id >= n {
                        return Err(HnswDecodeError::Invalid("link target out of range"));
                    }
                    layer.push(id);
                }
                node_links.push(layer);
            }
            links.push(node_links);
        }
        if !r.done() {
            return Err(HnswDecodeError::Invalid("trailing bytes"));
        }

        // `insert` maintains "the entry point owns the top layer"; a graph
        // violating it would panic in a later insert's layer indexing, so
        // reject it here. This also bounds max_layer by a real node's layer
        // count, keeping search's top-down descent finite.
        match entry_point {
            Some(e) => {
                // `len() - 1` cannot underflow: every node has ≥ 1 layer.
                if links[e].len() - 1 != max_layer {
                    return Err(HnswDecodeError::Invalid(
                        "entry point does not own the top layer",
                    ));
                }
            }
            None => {
                if max_layer != 0 {
                    return Err(HnswDecodeError::Invalid(
                        "empty graph with nonzero top layer",
                    ));
                }
            }
        }

        Ok(Self {
            metric,
            m,
            gamma,
            m_cap,
            m0_cap,
            ef_construction,
            ml,
            dim,
            seed,
            vectors,
            links,
            // Build-produced graphs carry no attribute rows; reconstruct the
            // parallel table empty so `attrs(id)` stays consistent.
            attrs: vec![Vec::new(); n],
            deleted,
            entry_point,
            max_layer,
            rng,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an index over `n` deterministic random vectors of dimension `dim`.
    fn build(n: usize, dim: usize, seed: u64) -> (Hnsw, Vec<Vec<f32>>) {
        build_gamma(n, dim, seed, 1.0)
    }

    /// Like [`build`], with an explicit edge-density multiplier.
    fn build_gamma(n: usize, dim: usize, seed: u64, gamma: f32) -> (Hnsw, Vec<Vec<f32>>) {
        let mut data_rng = seed ^ 0xABCD_EF01;
        let mut h = Hnsw::new(HnswParams {
            m: 8,
            ef_construction: 50,
            gamma,
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

    /// FNV-1a over the full graph structure (links, entry point, max layer).
    /// Stable across platforms and std versions, unlike `DefaultHasher`.
    fn graph_fingerprint(h: &Hnsw) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut hash = FNV_OFFSET;
        let mut mix = |x: u64| {
            for b in x.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        };
        for layers in &h.links {
            mix(layers.len() as u64);
            for nbrs in layers {
                mix(nbrs.len() as u64);
                for &n in nbrs {
                    mix(n as u64);
                }
            }
        }
        mix(h.entry_point.map(|e| e as u64 + 1).unwrap_or(0));
        mix(h.max_layer as u64);
        hash
    }

    /// Fingerprint of `build(200, 16, 42)` captured from the build code as it was
    /// before `gamma` existed. Guards the promise that γ = 1 reproduces that
    /// graph byte-for-byte; a deliberate algorithm change must update the
    /// constant and say so in the commit message.
    const REFERENCE_FINGERPRINT: u64 = 0xD0A7_2FC5_93B4_C94B;

    #[test]
    fn gamma_one_reproduces_pre_gamma_graph() {
        let (h, _) = build(200, 16, 42);
        assert_eq!(
            graph_fingerprint(&h),
            REFERENCE_FINGERPRINT,
            "graph structure drifted from the pre-gamma build"
        );
    }

    #[test]
    fn invalid_gamma_treated_as_one() {
        for gamma in [0.5, 0.0, -3.0, f32::NAN, f32::INFINITY] {
            let (h, _) = build_gamma(200, 16, 42, gamma);
            assert_eq!(
                graph_fingerprint(&h),
                REFERENCE_FINGERPRINT,
                "gamma = {gamma} did not clamp to 1"
            );
        }
    }

    #[test]
    fn len_and_dim() {
        let (h, _) = build(100, 8, 1);
        assert_eq!(h.len(), 100);
        assert_eq!(h.dim(), 8);
        assert!(!h.is_empty());
    }

    fn assert_degree_caps(h: &Hnsw) {
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
    fn degree_cap_respected() {
        let (h, _) = build(300, 12, 7);
        assert_degree_caps(&h);
    }

    #[test]
    fn deterministic_for_seed() {
        for gamma in [1.0, 2.0] {
            let (h1, _) = build_gamma(150, 10, 42, gamma);
            let (h2, _) = build_gamma(150, 10, 42, gamma);
            assert_eq!(h1.links, h2.links, "gamma = {gamma}");
            assert_eq!(h1.entry_point, h2.entry_point);
            assert_eq!(h1.max_layer, h2.max_layer);
        }
    }

    #[test]
    fn degree_scales_with_gamma() {
        let avg_layer0_degree = |h: &Hnsw| {
            let total: usize = (0..h.len()).map(|id| h.neighbors(id, 0).len()).sum();
            total as f64 / h.len() as f64
        };
        let (h1, _) = build_gamma(400, 12, 21, 1.0);
        let (h2, _) = build_gamma(400, 12, 21, 2.0);

        // Doubling γ doubles the degree caps; realized average degree (and with it
        // link memory) should grow by well over half that headroom.
        let (d1, d2) = (avg_layer0_degree(&h1), avg_layer0_degree(&h2));
        assert!(
            d2 >= 1.5 * d1,
            "γ=2 avg layer-0 degree {d2:.2} not ≥ 1.5× γ=1's {d1:.2}"
        );

        assert_degree_caps(&h2);
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
            HnswError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn empty_vector_errors() {
        let mut h = Hnsw::new(HnswParams::default());
        assert_eq!(h.insert(vec![]).unwrap_err(), HnswError::EmptyVector);
    }

    #[test]
    fn search_on_empty_index_is_empty() {
        let h = Hnsw::new(HnswParams::default());
        assert!(h.search(&[1.0, 2.0], 5, 10).unwrap().is_empty());
    }

    #[test]
    fn search_returns_self_at_rank_zero() {
        let (h, data) = build(200, 16, 3);
        for &probe in &[0usize, 50, 199] {
            let res = h.search(&data[probe], 5, 50).unwrap();
            assert_eq!(
                res[0].1, probe,
                "expected node {probe} at rank 0, got {res:?}"
            );
            assert!(res[0].0 < 1e-6, "self distance not ~0: {}", res[0].0);
        }
    }

    /// Unfiltered recall@10 over 40 queries against an 800-vector build with the
    /// given edge density.
    fn recall_at_10(gamma: f32) -> f64 {
        let (dim, n, k, queries) = (24usize, 800usize, 10usize, 40usize);
        let mut rng = 0x1234_5678u64;
        let mut h = Hnsw::new(HnswParams {
            m: 16,
            ef_construction: 100,
            gamma,
            metric: Metric::L2,
            seed: 99,
        });
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next_f64(&mut rng) as f32).collect();
            h.insert(v).unwrap();
        }

        let mut hits = 0usize;
        for _ in 0..queries {
            let q: Vec<f32> = (0..dim).map(|_| next_f64(&mut rng) as f32).collect();
            let approx = h.search(&q, k, 64).unwrap();
            let exact: HashSet<usize> = h.brute_force(&q, k).unwrap().iter().map(|x| x.1).collect();
            hits += approx.iter().filter(|(_, id)| exact.contains(id)).count();
        }
        hits as f64 / (queries * k) as f64
    }

    #[test]
    fn recall_at_10_beats_threshold() {
        let recall = recall_at_10(1.0);
        assert!(recall >= 0.9, "recall@10 too low: {recall:.3}");
    }

    #[test]
    fn gamma_dense_recall_not_below_baseline() {
        // Densification is for filtered search; it must not cost unfiltered
        // recall. Same 0.9 bar the γ=1 build is held to above.
        let recall = recall_at_10(2.0);
        assert!(recall >= 0.9, "recall@10 at γ=2 too low: {recall:.3}");
    }

    #[test]
    fn huge_gamma_clamps_to_bound() {
        let mut h = Hnsw::new(HnswParams {
            m: 2,
            ef_construction: 4,
            gamma: f32::MAX,
            metric: Metric::L2,
            seed: 1,
        });
        assert_eq!(h.m0_cap, (4.0 * HnswParams::MAX_GAMMA) as usize);
        // Inserting must allocate from the clamped caps, not panic.
        for i in 0..5 {
            h.insert(vec![i as f32, 0.0]).expect("insert");
        }
    }

    #[test]
    fn delete_excludes_from_results() {
        let (mut h, data) = build(200, 16, 5);
        let target = 42usize;
        h.delete(target).unwrap();
        let res = h.search(&data[target], 5, 50).unwrap();
        assert!(
            res.iter().all(|(_, id)| *id != target),
            "deleted node was returned: {res:?}"
        );
    }

    #[test]
    fn delete_unknown_id_errors() {
        let (mut h, _) = build(10, 4, 1);
        assert_eq!(
            h.delete(999).unwrap_err(),
            HnswError::UnknownNode { id: 999 }
        );
    }

    #[test]
    fn delete_all_but_one() {
        let (mut h, data) = build(60, 8, 2);
        for id in 0..h.len() {
            if id != 7 {
                h.delete(id).unwrap();
            }
        }
        let res = h.search(&data[7], 3, 64).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].1, 7);
    }

    #[test]
    fn recall_holds_under_tombstones() {
        let (dim, n, k, queries) = (24usize, 800usize, 10usize, 30usize);
        let mut rng = 0xBEEF_0001u64;
        let mut h = Hnsw::new(HnswParams {
            m: 16,
            ef_construction: 100,
            gamma: 1.0,
            metric: Metric::L2,
            seed: 11,
        });
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next_f64(&mut rng) as f32).collect();
            h.insert(v).unwrap();
        }
        let mut del_rng = 0x0BAD_F00Du64;
        for id in 0..h.len() {
            if next_f64(&mut del_rng) < 0.2 {
                h.delete(id).unwrap();
            }
        }
        assert!(h.deleted_count() > 0);

        let mut hits = 0usize;
        for _ in 0..queries {
            let q: Vec<f32> = (0..dim).map(|_| next_f64(&mut rng) as f32).collect();
            let approx = h.search(&q, k, 96).unwrap();
            assert!(
                approx.iter().all(|(_, id)| !h.deleted[*id]),
                "search returned a tombstoned node"
            );
            let exact: HashSet<usize> = h.brute_force(&q, k).unwrap().iter().map(|x| x.1).collect();
            hits += approx.iter().filter(|(_, id)| exact.contains(id)).count();
        }
        let recall = hits as f64 / (queries * k) as f64;
        assert!(
            recall >= 0.85,
            "recall under ~20% tombstones too low: {recall:.3}"
        );
    }

    #[test]
    fn compact_removes_tombstones_preserving_recall() {
        let (dim, n) = (20usize, 500usize);
        let mut rng = 0x1111_2222u64;
        let mut h = Hnsw::new(HnswParams {
            m: 16,
            ef_construction: 100,
            gamma: 1.0,
            metric: Metric::L2,
            seed: 3,
        });
        let mut data = Vec::new();
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next_f64(&mut rng) as f32).collect();
            h.insert(v.clone()).unwrap();
            data.push(v);
        }
        for id in (0..h.len()).step_by(3) {
            h.delete(id).unwrap();
        }
        let live_before = h.live_len();
        assert!(h.deleted_count() > 0);

        let new_n = h.compact().unwrap();
        assert_eq!(new_n, live_before);
        assert_eq!(h.deleted_count(), 0);
        assert_eq!(h.len(), live_before);

        // data[1] survived (1 % 3 != 0); still findable at rank 0 after remap
        let res = h.search(&data[1], 3, 64).unwrap();
        assert!(
            res[0].0 < 1e-6,
            "survivor self-distance not ~0: {}",
            res[0].0
        );
    }

    #[test]
    fn attributes_stored_and_survive_build() {
        let mut h = Hnsw::new(HnswParams::default());
        let a = h
            .insert_with_attrs(
                vec![1.0, 2.0],
                vec![AttrValue::Int(42), AttrValue::Float(9.5)],
            )
            .unwrap();
        let b = h
            .insert_with_attrs(vec![3.0, 4.0], vec![AttrValue::Null])
            .unwrap();
        let c = h.insert(vec![5.0, 6.0]).unwrap(); // no attributes

        assert_eq!(h.attrs(a), &[AttrValue::Int(42), AttrValue::Float(9.5)]);
        assert_eq!(h.attrs(b), &[AttrValue::Null]);
        assert!(h.attrs(c).is_empty());
        // Unknown id is empty, not a panic.
        assert!(h.attrs(999).is_empty());
    }

    #[test]
    fn attributes_follow_nodes_through_compaction() {
        // Each vector is tagged with an attribute encoding its identity, so we can
        // confirm the row stayed paired with the right vector after the id remap.
        let mut h = Hnsw::new(HnswParams::default());
        let survivors = 0..40usize;
        for i in survivors.clone() {
            h.insert_with_attrs(vec![i as f32, 0.0], vec![AttrValue::Int(i as i64)])
                .unwrap();
        }
        // Tombstone the even-numbered nodes.
        for id in (0..h.len()).step_by(2) {
            h.delete(id).unwrap();
        }
        h.compact().unwrap();

        // For every odd marker that survived, find its new id and check the tag.
        for i in survivors.filter(|i| i % 2 == 1) {
            let res = h.search(&[i as f32, 0.0], 1, 32).unwrap();
            assert_eq!(res.len(), 1);
            let new_id = res[0].1;
            assert_eq!(
                h.attrs(new_id),
                &[AttrValue::Int(i as i64)],
                "vector {i} lost or mismatched its attribute after compaction"
            );
        }
    }

    #[test]
    fn bytes_round_trip_preserves_graph() {
        let (mut h, _) = build(250, 12, 21);
        h.delete(17).unwrap();
        h.delete(42).unwrap();
        let restored = Hnsw::from_bytes(&h.to_bytes()).expect("decode");
        assert_eq!(restored.links, h.links, "neighbor lists must round-trip");
        assert_eq!(restored.entry_point, h.entry_point);
        assert_eq!(restored.vectors, h.vectors);
        assert_eq!(restored.deleted, h.deleted);
        assert_eq!(restored.max_layer, h.max_layer);
        assert_eq!(restored.dim(), h.dim());
        assert_eq!(restored.metric(), h.metric());
        assert_eq!(
            (
                restored.m,
                restored.m_cap,
                restored.m0_cap,
                restored.ef_construction
            ),
            (h.m, h.m_cap, h.m0_cap, h.ef_construction)
        );
        assert_eq!(restored.gamma, h.gamma);
        assert_eq!((restored.seed, restored.rng), (h.seed, h.rng));
    }

    #[test]
    fn bytes_round_trip_preserves_gamma() {
        // A γ-dense graph must come back with the same density (caps derive from
        // the persisted γ) and the same neighbor lists.
        let (h, _) = build_gamma(200, 12, 7, 4.0);
        let restored = Hnsw::from_bytes(&h.to_bytes()).expect("decode");
        assert_eq!(restored.gamma, 4.0);
        assert_eq!(
            (restored.m_cap, restored.m0_cap),
            (h.m_cap, h.m0_cap),
            "degree caps must survive the round-trip"
        );
        assert_eq!(restored.links, h.links);
    }

    #[test]
    fn bytes_round_trip_empty_index() {
        let h = Hnsw::new(HnswParams::default());
        let restored = Hnsw::from_bytes(&h.to_bytes()).expect("decode");
        assert!(restored.is_empty());
        assert_eq!(restored.dim(), 0);
        assert_eq!(restored.entry_point, None);
        assert!(restored.search(&[1.0, 2.0], 5, 10).unwrap().is_empty());
    }

    #[test]
    fn restored_index_inserts_identically() {
        let (mut original, _) = build(80, 8, 33);
        let mut restored = Hnsw::from_bytes(&original.to_bytes()).unwrap();
        let mut rng = 0xFEED_u64;
        for _ in 0..40 {
            let v: Vec<f32> = (0..8).map(|_| next_f64(&mut rng) as f32).collect();
            original.insert(v.clone()).unwrap();
            restored.insert(v).unwrap();
        }
        assert_eq!(original.links, restored.links);
        assert_eq!(original.entry_point, restored.entry_point);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let (h, _) = build(10, 4, 1);
        let mut bytes = h.to_bytes();
        bytes[0] ^= 0xFF;
        assert_eq!(
            Hnsw::from_bytes(&bytes).unwrap_err(),
            HnswDecodeError::BadMagic
        );
    }

    #[test]
    fn decode_rejects_newer_version() {
        let (h, _) = build(10, 4, 1);
        let mut bytes = h.to_bytes();
        bytes[4] = 0xFF; // version lives right after the 4-byte magic
        assert!(matches!(
            Hnsw::from_bytes(&bytes),
            Err(HnswDecodeError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn decode_rejects_inflated_max_layer() {
        // Corrupting max_layer must fail the entry-point/top-layer check
        // instead of decoding into a graph whose search loops over phantom
        // layers (or whose next insert panics).
        let (h, _) = build(10, 4, 1);
        let mut bytes = h.to_bytes();
        // Layout: magic(4) version(2) metric(1) reserved(1) m(8) gamma(4) ef(8)
        // dim(8), then max_layer at offset 36.
        bytes[36..44].copy_from_slice(&(h.max_layer as u64 + 1).to_le_bytes());
        assert_eq!(
            Hnsw::from_bytes(&bytes).unwrap_err(),
            HnswDecodeError::Invalid("entry point does not own the top layer")
        );
    }

    #[test]
    fn decode_rejects_nonzero_max_layer_on_empty_graph() {
        let h = Hnsw::new(HnswParams::default());
        let mut bytes = h.to_bytes();
        bytes[36..44].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(
            Hnsw::from_bytes(&bytes).unwrap_err(),
            HnswDecodeError::Invalid("empty graph with nonzero top layer")
        );
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let (h, _) = build(10, 4, 1);
        let mut bytes = h.to_bytes();
        bytes.push(0);
        assert_eq!(
            Hnsw::from_bytes(&bytes).unwrap_err(),
            HnswDecodeError::Invalid("trailing bytes")
        );
    }

    #[test]
    fn decode_of_any_prefix_errors_not_panics() {
        let (h, _) = build(12, 4, 5);
        let bytes = h.to_bytes();
        for cut in 0..bytes.len() {
            assert!(
                Hnsw::from_bytes(&bytes[..cut]).is_err(),
                "prefix of {cut} bytes decoded successfully"
            );
        }
    }

    #[test]
    fn higher_ef_does_not_reduce_recall() {
        let (dim, n) = (20usize, 500usize);
        let mut rng = 0x0000_CAFEu64;
        let mut h = Hnsw::new(HnswParams {
            m: 12,
            ef_construction: 80,
            gamma: 1.0,
            metric: Metric::L2,
            seed: 7,
        });
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next_f64(&mut rng) as f32).collect();
            h.insert(v).unwrap();
        }
        let q: Vec<f32> = (0..dim).map(|_| next_f64(&mut rng) as f32).collect();
        let exact: HashSet<usize> = h.brute_force(&q, 10).unwrap().iter().map(|x| x.1).collect();
        let recall = |ef: usize| {
            h.search(&q, 10, ef)
                .unwrap()
                .iter()
                .filter(|(_, id)| exact.contains(id))
                .count()
        };
        assert!(
            recall(100) >= recall(10),
            "more ef_search should not reduce recall"
        );
    }
}
