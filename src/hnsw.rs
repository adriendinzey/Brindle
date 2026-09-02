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
//! Search is optionally *predicate-aware* ([`Hnsw::search_filtered`]): a node
//! that fails the predicate is never returned, but is still traversed as a
//! bridge to its own neighbors, so a selective filter cannot strand the matching
//! nodes behind non-matching ones.
//!
//! The degree caps scale with an edge-density multiplier `gamma` (γ) following
//! ACORN (<https://arxiv.org/abs/2403.04871>): a graph built with γ > 1 keeps
//! ~`m·γ` neighbors per node, so the subgraph of nodes surviving a selective
//! predicate stays navigable. γ = 1 is a standard HNSW.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use crate::distance::DistanceError;
use crate::filter::{AttrValue, Predicate};
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
    /// The graph is full: node ids are 32-bit, so it cannot grow further.
    NodeLimitReached { limit: usize },
    /// A vector had more components than a graph can carry.
    DimensionTooLarge { got: usize, max: usize },
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
            HnswError::NodeLimitReached { limit } => {
                write!(
                    f,
                    "graph is full: {limit} nodes is the ceiling for 32-bit ids"
                )
            }
            HnswError::DimensionTooLarge { got, max } => {
                write!(f, "vector has {got} dimensions, the maximum is {max}")
            }
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

    /// Largest `m` a graph may declare. Matches the ceiling enforced when an
    /// index is created, so no index that exists can fail to decode.
    pub const MAX_M: usize = 128;

    /// Ceiling on the layer-0 degree cap `2·m·γ`.
    ///
    /// `m` and `γ` are each bounded, but it is their *product* that sizes link
    /// storage, and on the decode path both arrive inside the payload. Fixed
    /// stride means a node reserves its full cap whether or not it uses it, so
    /// this is the number that turns a small blob into a large allocation. The
    /// same ceiling is enforced when an index is created.
    pub const MAX_LAYER0_DEGREE: usize = 2048;

    /// Largest dimensionality a graph may declare.
    ///
    /// The decode reads each node's vector through a scratch buffer sized from
    /// this field, before anything relates it to how many bytes the payload
    /// actually holds — so without a ceiling an 85-byte blob can ask the
    /// allocator for gigabytes, and a failed allocation aborts the process
    /// rather than raising an error a session can catch. Matches the limit the
    /// SQL vector type enforces on input, so no vector that can be stored can
    /// fail to decode.
    pub const MAX_DIM: usize = 16_000;

    /// Ceiling on a graph's top layer. Levels are drawn geometrically, so even
    /// at the smallest permitted `m` the chance of reaching this is ~2⁻⁶⁴: a
    /// payload claiming more is corrupt, and each extra layer a node claims
    /// reserves another stride of link slots.
    pub const MAX_LAYER: usize = 64;
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

/// The mutable state of one layer's beam search: every node already considered,
/// the frontier still to expand (nearest first), and the result set (farthest
/// first, capped at `ef`).
///
/// Frontier membership and result membership are deliberately separate: a
/// tombstoned or non-matching node may route the search without ever being
/// eligible to come back as an answer.
struct Beam {
    visited: HashSet<usize>,
    frontier: BinaryHeap<Reverse<Cand>>,
    results: BinaryHeap<Cand>,
    ef: usize,
    /// Remaining allowance for routing *through* non-matching nodes when a
    /// filtered expansion turns up no matching neighbor at all. Spending the
    /// caller's own budget bounds that fallback, so a predicate nothing
    /// satisfies still finishes promptly instead of sweeping the graph.
    detours: usize,
}

impl Beam {
    fn new(ef: usize) -> Self {
        Self {
            visited: HashSet::with_capacity(ef.max(1) * 8),
            frontier: BinaryHeap::new(),
            results: BinaryHeap::new(),
            ef,
            detours: ef,
        }
    }

    /// Distance of the farthest result held, or infinity while none are held.
    #[inline]
    fn farthest(&self) -> f32 {
        self.results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY)
    }

    /// Queue `cand` for expansion, recording it as a result only when
    /// `admissible`.
    #[inline]
    fn push(&mut self, cand: Cand, admissible: bool) {
        self.frontier.push(Reverse(cand));
        if admissible {
            self.results.push(cand);
            if self.results.len() > self.ef {
                self.results.pop();
            }
        }
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
    /// Node count. Kept explicitly because `vectors` is now flat, so its length
    /// no longer identifies the node count when `dim` is 0.
    n: usize,
    // Invariant: node ids index `vectors`, `layer_base`, `attrs` and `deleted` in
    // parallel, and nodes are never removed (delete only tombstones), so every id
    // stored in `links` or `entry_point` stays valid for all of them. This is what
    // makes the raw indexing on the search/insert paths panic-free.
    //
    // Storage is flat: `vectors` holds every node's components end to end, strided
    // by `dim`, and neighbor lists live in fixed-width slots rather than in a
    // `Vec` per node per layer. Rebuilding a nested shape cost one allocation per
    // node per layer on every load, which dominated the decode.
    vectors: Vec<f32>,
    /// Prefix sum of per-node layer counts, `n + 1` entries: node `id` owns
    /// layers `layer_base[id] .. layer_base[id + 1]`. Both a node's layer count
    /// and its slot offset derive from this, so neither needs its own array.
    layer_base: Vec<u32>,
    /// Live neighbor count per (node, layer), indexed by `layer_base[id] + layer`.
    link_counts: Vec<u32>,
    /// Neighbor ids in fixed-width slots. A layer's width is its degree cap plus
    /// one: `insert` appends before pruning, so a list is transiently one over.
    ///
    /// Slots past a layer's live count hold **unspecified** bytes — pruning
    /// shortens a list without clearing what it left behind, and a decoded graph
    /// zero-fills instead. Only `neighbors` defines the graph's value; comparing
    /// this array raw would see differences that carry no meaning.
    links: Vec<u32>,
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
            n: 0,
            vectors: Vec::new(),
            layer_base: vec![0],
            link_counts: Vec::new(),
            links: Vec::new(),
            attrs: Vec::new(),
            deleted: Vec::new(),
            entry_point: None,
            max_layer: 0,
            rng: seed,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Neighbors per node on upper layers, after the `>= 2` clamp
    /// [`Hnsw::new`] applies. Layer 0 and γ scaling are derived from it.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Build-time candidate pool, after [`Hnsw::new`] raised it to the graph's
    /// degree-cap floor — so it can exceed the requested value.
    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// Edge-density multiplier in effect, after clamping to
    /// `[1, HnswParams::MAX_GAMMA]`.
    pub fn gamma(&self) -> f32 {
        self.gamma
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

    /// Width of one neighbor slot run. One over the degree cap, because `insert`
    /// appends a neighbor and only then prunes back to the cap.
    /// Largest node count the `u32` ids and layer offsets can address.
    pub const MAX_NODES: usize = u32::MAX as usize;

    #[inline]
    fn slot_width(&self, layer: usize) -> usize {
        self.max_degree(layer) + 1
    }

    /// Layers node `id` participates in.
    #[inline]
    fn levels(&self, id: usize) -> usize {
        (self.layer_base[id + 1] - self.layer_base[id]) as usize
    }

    /// Start of node `id`'s slot run in `links`. Every earlier node contributed
    /// one layer-0 run plus one upper-layer run per level above 0, and
    /// `layer_base[id] - id` is exactly that count of upper layers.
    #[inline]
    fn link_base(&self, id: usize) -> usize {
        let upper = self.layer_base[id] as usize - id;
        id * self.slot_width(0) + upper * self.slot_width(1)
    }

    /// Start of node `id`'s slots for `layer`.
    #[inline]
    fn layer_slot(&self, id: usize, layer: usize) -> usize {
        let within = if layer == 0 {
            0
        } else {
            self.slot_width(0) + (layer - 1) * self.slot_width(1)
        };
        self.link_base(id) + within
    }

    #[inline]
    fn vector(&self, id: usize) -> &[f32] {
        let start = id * self.dim;
        &self.vectors[start..start + self.dim]
    }

    #[inline]
    fn neighbors(&self, id: usize, layer: usize) -> &[u32] {
        if id >= self.n || layer >= self.levels(id) {
            return &[];
        }
        let start = self.layer_slot(id, layer);
        let count = self.link_counts[self.layer_base[id] as usize + layer] as usize;
        &self.links[start..start + count]
    }

    /// Replace node `id`'s neighbor list at `layer`. `ids` must fit the slot
    /// width, which holds because callers pass a list the degree cap bounded.
    fn set_neighbors(&mut self, id: usize, layer: usize, ids: &[u32]) {
        let start = self.layer_slot(id, layer);
        self.links[start..start + ids.len()].copy_from_slice(ids);
        self.link_counts[self.layer_base[id] as usize + layer] = ids.len() as u32;
    }

    /// Append one neighbor to node `id`'s list at `layer`, returning the new
    /// length. The slot run is one wider than the degree cap, so an append onto
    /// a full list still fits and the caller prunes immediately after.
    fn push_neighbor(&mut self, id: usize, layer: usize, neighbor: u32) -> usize {
        let idx = self.layer_base[id] as usize + layer;
        let count = self.link_counts[idx] as usize;
        let start = self.layer_slot(id, layer);
        self.links[start + count] = neighbor;
        self.link_counts[idx] = (count + 1) as u32;
        count + 1
    }

    /// Append a node's storage across the flat arrays and return its id.
    ///
    /// Node ids and layer offsets are stored as `u32`, which caps a graph at
    /// [`Hnsw::MAX_NODES`] nodes. That is far beyond what the interim
    /// whole-image format can hold — every insert rewrites the entire index —
    /// but the ceiling is the representation's, not the format's, so it is
    /// checked here rather than assumed.
    fn push_node(
        &mut self,
        vector: &[f32],
        attrs: Vec<AttrValue>,
        levels: usize,
    ) -> Result<usize, HnswError> {
        if self.n >= Self::MAX_NODES {
            return Err(HnswError::NodeLimitReached {
                limit: Self::MAX_NODES,
            });
        }
        let id = self.n;
        self.vectors.extend_from_slice(vector);
        self.attrs.push(attrs);
        self.deleted.push(false);
        let end = self.layer_base[id] + levels as u32;
        self.layer_base.push(end);
        self.link_counts.resize(self.link_counts.len() + levels, 0);
        let slots = self.slot_width(0) + (levels - 1) * self.slot_width(1);
        self.links.resize(self.links.len() + slots, 0);
        self.n += 1;
        Ok(id)
    }

    fn random_level(&mut self) -> usize {
        let mut r = next_f64(&mut self.rng);
        if r <= 0.0 {
            r = f64::MIN_POSITIVE;
        }
        (-r.ln() * self.ml).floor() as usize
    }

    /// Whether `predicate` requires the predicate-aware traversal path. Both
    /// `None` and the match-all predicate take the plain HNSW path, so adding
    /// filtering support left unfiltered search bit-for-bit unchanged.
    #[inline]
    fn is_filtered(predicate: Option<&Predicate>) -> bool {
        !matches!(predicate, None | Some(Predicate::All))
    }

    /// Whether node `id` satisfies `predicate` — vacuously true without one.
    #[inline]
    fn node_matches(&self, id: usize, predicate: Option<&Predicate>) -> bool {
        match predicate {
            None => true,
            Some(p) => p.matches(self.attrs(id)),
        }
    }

    /// Whether `id` may be *returned*: live, and satisfying the predicate.
    /// Routing through a node is a separate, laxer question.
    #[inline]
    fn admissible(&self, id: usize, predicate: Option<&Predicate>) -> bool {
        !self.deleted[id] && self.node_matches(id, predicate)
    }

    /// Score `id` and fold it into `beam`, unless it was already considered or
    /// is too far to improve on the results already held. Reports whether it
    /// actually joined the frontier.
    fn visit(
        &self,
        query: &[f32],
        id: usize,
        predicate: Option<&Predicate>,
        beam: &mut Beam,
    ) -> Result<bool, HnswError> {
        if !beam.visited.insert(id) {
            return Ok(false);
        }
        let d = self.metric.distance(query, self.vector(id))?;
        if beam.results.len() < beam.ef || d < beam.farthest() {
            beam.push(Cand { dist: d, id }, self.admissible(id, predicate));
            return Ok(true);
        }
        Ok(false)
    }

    /// Fold `id`'s neighbors into `beam`.
    ///
    /// Unfiltered this is plain HNSW: every neighbor is a candidate. Under a
    /// predicate only matching neighbors can be answers, and a selective
    /// predicate can leave a node with none at all — the matching subgraph
    /// fragments and greedy search dead-ends. ACORN's fix, implemented here:
    /// when the neighbor list yields fewer than `m` matching nodes, hop *over*
    /// the non-matching neighbors and take *their* neighbors instead, so
    /// traversal crosses filtered-out regions rather than stopping at them.
    ///
    /// Two bounds keep that from degenerating into a breadth-first sweep: one
    /// expansion reaches at most two hops (a node arrived at across a bridge is
    /// not itself bridged over *within that expansion*, though it expands
    /// normally once popped), and at most `m` bridges are taken per node,
    /// stopping early as soon as `m` matching neighbors have been produced. Each
    /// expansion is therefore bounded work; what terminates the search itself is
    /// `visited` — a node joins the frontier only on first sight.
    ///
    /// Two hops is not always far enough — under a very selective predicate a
    /// node can have no match anywhere in its two-hop neighborhood. Rather than
    /// return nothing at all, such a node routes on through its non-matching
    /// neighbors (which still can never be returned), limited by the beam's
    /// detour allowance.
    ///
    /// Bridging trades predicate evaluations for distance computations: a
    /// two-hop node is scored only if it matches, and inline attributes are
    /// tested without touching a vector. The detour is the exception — it scores
    /// non-matching neighbors precisely in order to walk through them.
    fn expand(
        &self,
        query: &[f32],
        id: usize,
        layer: usize,
        predicate: Option<&Predicate>,
        beam: &mut Beam,
        bridges: &mut Vec<usize>,
    ) -> Result<(), HnswError> {
        let neighbors = self.neighbors(id, layer);
        if !Self::is_filtered(predicate) {
            for &n in neighbors {
                self.visit(query, n as usize, predicate, beam)?;
            }
            return Ok(());
        }

        // Both bounds scale with the graph's own base degree, so bridging costs
        // a small multiple of the work an unfiltered expansion already does.
        let target = self.m;
        let max_bridges = self.m;

        // The non-matching neighbors are collected as they are classified rather
        // than re-tested on a second pass; `bridges` is owned by the enclosing
        // layer search, so it is reused across expansions instead of allocating.
        bridges.clear();
        let mut matching = 0usize;
        for &n in neighbors {
            let n = n as usize;
            if self.node_matches(n, predicate) {
                matching += 1;
                self.visit(query, n, predicate, beam)?;
            } else if bridges.len() < max_bridges {
                bridges.push(n);
            }
        }

        for &bridge in bridges.iter() {
            if matching >= target {
                break;
            }
            for &nn in self.neighbors(bridge, layer) {
                if matching >= target {
                    break;
                }
                let nn = nn as usize;
                if self.node_matches(nn, predicate) {
                    matching += 1;
                    self.visit(query, nn, predicate, beam)?;
                }
            }
        }

        if matching == 0 {
            // Stranded: nothing matches within two hops. Keep walking through
            // the non-matching nodes themselves — greedily, so the walk still
            // heads toward the query.
            //
            // The allowance is charged per node *enqueued*, not per stranded
            // node: it is popping one of these that costs a two-hop scan, so
            // charging once per expansion would let a single unit queue a whole
            // neighbor list — γ² work per unit, and the denser the graph the
            // worse the bill.
            for &n in neighbors {
                if beam.detours == 0 {
                    break;
                }
                if self.visit(query, n as usize, predicate, beam)? {
                    beam.detours -= 1;
                }
            }
        }
        Ok(())
    }

    /// Greedy beam search within one layer (HNSW SEARCH-LAYER). Returns up to `ef`
    /// nearest candidates to `query`, ascending by distance.
    ///
    /// With a predicate, only matching nodes are returned and the `ef` budget is
    /// spent on those alone; [`Hnsw::expand`] covers how traversal still reaches
    /// them across non-matching regions.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        predicate: Option<&Predicate>,
    ) -> Result<Vec<Cand>, HnswError> {
        let filtered = Self::is_filtered(predicate);
        let mut beam = Beam::new(ef);
        // Scratch for `expand`'s bridge list, hoisted here so the filtered path
        // allocates once per layer search rather than once per expansion.
        let mut bridges: Vec<usize> = Vec::new();

        for &ep in entry_points {
            if beam.visited.insert(ep) {
                let d = self.metric.distance(query, self.vector(ep))?;
                // A seed routes even when tombstoned or non-matching: it may be
                // the only way into the region that does match.
                beam.push(Cand { dist: d, id: ep }, self.admissible(ep, predicate));
            }
        }

        while let Some(Reverse(c)) = beam.frontier.pop() {
            // A filtered search admits only matching nodes, so a result set that
            // is not yet full means the budget is still unspent and its farthest
            // entry is no cutoff. Unfiltered, the original rule stands and
            // existing graphs and query results reproduce exactly.
            if c.dist > beam.farthest() && (!filtered || beam.results.len() >= ef) {
                break;
            }
            self.expand(query, c.id, layer, predicate, &mut beam, &mut bridges)?;
        }

        let mut out = beam.results.into_vec();
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
                let d = self.metric.distance(self.vector(cand.id), self.vector(r))?;
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
        // Refused on the way in, not only on the way back. The decoder caps the
        // dimension it will accept, because it sizes a buffer from that field
        // before anything relates it to the payload's length — so a graph that
        // took a wider vector would serialize to something it could never read
        // back, and the index built from it would be unreadable rather than
        // merely wrong. The two limits are the same number for that reason.
        if vector.len() > HnswParams::MAX_DIM {
            return Err(HnswError::DimensionTooLarge {
                got: vector.len(),
                max: HnswParams::MAX_DIM,
            });
        }
        if self.dim == 0 {
            self.dim = vector.len();
        } else if vector.len() != self.dim {
            return Err(HnswError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let level = self.random_level();
        let id = self.push_node(&vector, attrs, level + 1)?;

        let entry = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                self.max_layer = level;
                return Ok(id);
            }
            Some(e) => e,
        };

        let query = self.vector(id).to_vec();
        let max_layer = self.max_layer;
        let mut ep_ids = vec![entry];

        // Greedy descent from the top down to just above the new node's level.
        if max_layer > level {
            for lc in ((level + 1)..=max_layer).rev() {
                let w = self.search_layer(&query, &ep_ids, 1, lc, None)?;
                if let Some(nearest) = w.first() {
                    ep_ids = vec![nearest.id];
                }
            }
        }

        // Connect at each layer from min(level, max_layer) down to 0.
        let start = level.min(max_layer);
        for lc in (0..=start).rev() {
            let w = self.search_layer(&query, &ep_ids, self.ef_construction, lc, None)?;
            let max_deg = self.max_degree(lc);
            let selected = self.select_neighbors_heuristic(&w, max_deg)?;

            let chosen: Vec<u32> = selected.iter().map(|&n| n as u32).collect();
            self.set_neighbors(id, lc, &chosen);
            for &n in &selected {
                // Append then prune, which is why a slot run is one wider than
                // the cap: the pruned set is chosen from the same candidates the
                // nested layout considered, so the resulting graph is identical.
                let len = self.push_neighbor(n, lc, id as u32);
                if len > max_deg {
                    let nbase = self.vector(n).to_vec();
                    let mut cands = Vec::with_capacity(len);
                    for &nn in self.neighbors(n, lc) {
                        let nn = nn as usize;
                        let d = self.metric.distance(&nbase, self.vector(nn))?;
                        cands.push(Cand { dist: d, id: nn });
                    }
                    let kept = self.select_neighbors_heuristic(&cands, max_deg)?;
                    let kept: Vec<u32> = kept.into_iter().map(|x| x as u32).collect();
                    self.set_neighbors(n, lc, &kept);
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
        self.search_inner(query, k, ef_search, None)
    }

    /// Approximate k-nearest-neighbor search restricted to nodes whose stored
    /// attributes satisfy `predicate`. Same contract as [`Hnsw::search`], with
    /// one guarantee added: a node that fails the predicate is never returned.
    ///
    /// The `ef_search` budget is spent on matching nodes alone, and traversal
    /// bridges over non-matching ones to reach them, which is what holds recall
    /// up where post-filtering — searching blind, then discarding — collapses.
    /// [`Predicate::All`] takes the unfiltered fast path.
    ///
    /// Recall under a *very* selective predicate is what the graph's `gamma` is
    /// for: bridging can reconnect a thinned graph, but building dense enough to
    /// not need it is cheaper at query time.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        predicate: &Predicate,
    ) -> Result<Vec<(f32, usize)>, HnswError> {
        self.search_inner(query, k, ef_search, Some(predicate))
    }

    fn search_inner(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        predicate: Option<&Predicate>,
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
            // The upper layers exist to navigate, not to answer: descend through
            // the graph as built and apply the predicate only on layer 0, where
            // results are actually collected.
            let w = self.search_layer(query, &ep_ids, 1, lc, None)?;
            if let Some(nearest) = w.first() {
                ep_ids = vec![nearest.id];
            }
        }

        let ef = ef_search.max(k);
        let mut w = self.search_layer(query, &ep_ids, ef, 0, predicate)?;
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
        let mut all: Vec<Cand> = Vec::with_capacity(self.n);
        for id in 0..self.n {
            if self.deleted[id] {
                continue;
            }
            all.push(Cand {
                dist: self.metric.distance(query, self.vector(id))?,
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
                live.push((self.vector(id).to_vec(), self.attrs[id].clone()));
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
/// Bumped to 2 when attribute rows joined the payload. Version 1 blobs are
/// rejected rather than read as attribute-free: a filtered scan over silently
/// missing attributes returns *zero rows* instead of failing, so a loud
/// `UnsupportedVersion` (rebuild the index) is the safer incompatibility. No
/// release ever wrote a v1 blob.
const CODEC_VERSION: u16 = 2;

// Attribute value tags. `Null` carries no payload; the numeric variants carry
// 8 little-endian bytes, floats as raw bits so a `NaN` survives the round trip.
const ATTR_TAG_NULL: u8 = 0;
const ATTR_TAG_INT: u8 = 1;
const ATTR_TAG_FLOAT: u8 = 2;

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
}

impl GraphBytes for ByteReader<'_> {
    fn read_exact(&mut self, out: &mut [u8]) -> Result<(), HnswDecodeError> {
        out.copy_from_slice(self.take(out.len())?);
        Ok(())
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
}

/// Reads from a slice, so callers already holding the bytes — and the framing
/// tests — drive exactly the same code as a page walk.
impl GraphBytes for &[u8] {
    fn read_exact(&mut self, out: &mut [u8]) -> Result<(), HnswDecodeError> {
        if out.len() > self.len() {
            return Err(HnswDecodeError::Truncated);
        }
        let (head, tail) = self.split_at(out.len());
        out.copy_from_slice(head);
        *self = tail;
        Ok(())
    }

    fn remaining(&self) -> usize {
        self.len()
    }
}

/// The serialized bytes of a graph, consumed strictly front to back.
///
/// A decoder cannot simply take a slice over the stored form: the storage layer
/// keeps the blob across index pages, and a graph large enough to matter spans
/// more pages than a backend may pin at once — so the bytes are delivered a
/// buffer at a time rather than borrowed whole. Reading through this trait lets
/// the same decoder serve a plain slice and a page walk, and lets the page walk
/// avoid materializing the entire blob first.
pub trait GraphBytes {
    /// Fill `out` completely, or fail if fewer bytes remain.
    fn read_exact(&mut self, out: &mut [u8]) -> Result<(), HnswDecodeError>;

    /// Bytes not yet consumed. The decoder bounds its allocations by this, so an
    /// implementation must never report more than it can actually supply.
    fn remaining(&self) -> usize;

    /// Read one little-endian value of the named width, advancing the source.
    fn u8(&mut self) -> Result<u8, HnswDecodeError> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }

    fn u16(&mut self) -> Result<u16, HnswDecodeError> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn u32(&mut self) -> Result<u32, HnswDecodeError> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, HnswDecodeError> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// A `u64` that must fit in `usize` (counts, ids, sizes).
    fn read_len(&mut self) -> Result<usize, HnswDecodeError> {
        usize::try_from(self.u64()?)
            .map_err(|_| HnswDecodeError::Invalid("value exceeds address space"))
    }

    /// Whether every byte has been consumed.
    fn done(&self) -> bool {
        self.remaining() == 0
    }
}

impl Hnsw {
    /// Upper bound on the byte length of this graph's serialization, walking
    /// the real per-node layer and attribute counts rather than assuming the
    /// degree caps — a node above layer 0 carries one neighbor list per level,
    /// which a per-node constant understates. Exact unless a row holds `Null`s,
    /// which encode to 1 byte where this budgets 9. Use it to size a buffer
    /// before [`Hnsw::to_bytes_into`].
    pub fn serialized_len_hint(&self) -> usize {
        // Header, then per node: tombstone flag, level count, vector, attribute
        // count, each layer's own count, and the ids and values themselves.
        let mut total = 76 + self.n * (17 + self.dim * 4);
        for id in 0..self.n {
            for layer in 0..self.levels(id) {
                total += 8 + self.neighbors(id, layer).len() * 8;
            }
            total += self.attrs[id].len() * 9;
        }
        total
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
        out.extend_from_slice(&(self.n as u64).to_le_bytes());
        for id in 0..self.n {
            out.push(u8::from(self.deleted[id]));
            out.extend_from_slice(&(self.levels(id) as u64).to_le_bytes());
            for x in self.vector(id) {
                out.extend_from_slice(&x.to_le_bytes());
            }
            for layer in 0..self.levels(id) {
                let ids = self.neighbors(id, layer);
                out.extend_from_slice(&(ids.len() as u64).to_le_bytes());
                for &neighbor in ids {
                    out.extend_from_slice(&(neighbor as u64).to_le_bytes());
                }
            }
            out.extend_from_slice(&(self.attrs[id].len() as u64).to_le_bytes());
            for value in &self.attrs[id] {
                match value {
                    AttrValue::Null => out.push(ATTR_TAG_NULL),
                    AttrValue::Int(v) => {
                        out.push(ATTR_TAG_INT);
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    AttrValue::Float(v) => {
                        out.push(ATTR_TAG_FLOAT);
                        out.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
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
        Self::from_graph_bytes(&mut ByteReader::new(bytes))
    }

    /// Decode from any [`GraphBytes`] source. The storage layer uses this to
    /// decode straight out of index pages, which avoids assembling the whole
    /// serialized blob in memory only to read it once and drop it.
    pub fn from_graph_bytes<S: GraphBytes + ?Sized>(r: &mut S) -> Result<Self, HnswDecodeError> {
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
        let m = r.read_len()?;
        // Bounded on both sides. The upper bound is not cosmetic: `m` and
        // `gamma` below size the link storage, and that is the one allocation
        // here the payload's own length does not constrain. A Rust allocation
        // that fails aborts the process instead of raising an error a session
        // can catch, so an unbounded one turns a corrupt index into a crash of
        // every backend.
        if !(2..=HnswParams::MAX_M).contains(&m) {
            return Err(HnswDecodeError::Invalid("m out of range"));
        }
        let gamma = f32::from_bits(r.u32()?);
        if !gamma.is_finite() || !(1.0..=HnswParams::MAX_GAMMA).contains(&gamma) {
            return Err(HnswDecodeError::Invalid("gamma out of range"));
        }
        // Same derivation `new()` uses, so a decoded graph's degree caps and ef
        // floor match a freshly built one's.
        let (_, m_cap, m0_cap, ml, ef_floor) = Self::derived_params(m, gamma);
        if m0_cap > HnswParams::MAX_LAYER0_DEGREE {
            return Err(HnswDecodeError::Invalid("degree cap out of range"));
        }
        let ef_construction = r.read_len()?;
        if ef_construction < ef_floor {
            return Err(HnswDecodeError::Invalid("ef_construction below floor"));
        }
        let dim = r.read_len()?;
        if dim > HnswParams::MAX_DIM {
            return Err(HnswDecodeError::Invalid("dimension out of range"));
        }
        let max_layer = r.read_len()?;
        if max_layer > HnswParams::MAX_LAYER {
            return Err(HnswDecodeError::Invalid("top layer out of range"));
        }
        let seed = r.u64()?;
        let rng = r.u64()?;
        // Encoded graphs come from `new()`, which never produces a zero seed
        // or zero PRNG state (xorshift preserves nonzero-ness).
        if seed == 0 || rng == 0 {
            return Err(HnswDecodeError::Invalid("PRNG state must be nonzero"));
        }
        let entry_raw = r.u64()?;
        let n = r.read_len()?;
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
        let min_node_bytes = 25usize.saturating_add(vector_bytes);
        let plausible_n = n.min(r.remaining() / min_node_bytes);
        // Ceiling on the link slots the whole graph may reserve.
        //
        // The header bounds alone do not constrain this. Fixed stride reserves a
        // node's full degree cap on every layer it claims, so one 8-byte layer
        // count can reserve `cap * 4` bytes of padding the payload never has to
        // fill — a node declaring the maximum layers with no neighbours at all
        // costs a few hundred bytes and reserves a few hundred kilobytes. At the
        // permitted extremes that is a thousandfold, and a Vec reservation that
        // fails aborts the process rather than raising an error a session can
        // catch, which for an extension means every backend dies.
        //
        // A graph a build actually produces reserves roughly one slot per
        // neighbour it stores, and stores each neighbour as 8 payload bytes, so
        // slots land near payload/8. Allowing one slot per payload byte leaves
        // eight times that headroom — ample for a sparse graph, far under what
        // the padding trick needs. The floor keeps tiny graphs, where the
        // per-node cap dominates a short payload, from tripping it.
        // The floor covers small graphs, where a node's full stride dwarfs a
        // short payload. It has to clear what a *build* actually produces at the
        // densest legal settings, not just what looks generous: at m = 2 with
        // gamma = 512 the layer-0 cap lands exactly on its ceiling, half the
        // nodes draw a second layer, and a few hundred nodes then reserve past
        // 256K slots — rejecting one of those would make an index that exists
        // permanently unreadable, which is worse than the allocation it guards
        // against. Sized to clear the worst observed by a wide margin; at any
        // size where an attacker gains from the padding, `remaining` dominates
        // and the floor is irrelevant.
        let max_slots = r.remaining().saturating_add(1 << 22);
        // Sized from `plausible_n` so the decode does not reallocate, and flat so
        // it does not allocate per node at all. `links` is sized for the common
        // case of a single layer; nodes above layer 0 grow it, which is rare.
        let slot0 = m0_cap + 1;
        let slot_up = m_cap + 1;
        let mut vectors: Vec<f32> = Vec::with_capacity(plausible_n * dim);
        let mut layer_base: Vec<u32> = Vec::with_capacity(plausible_n + 1);
        layer_base.push(0);
        let mut link_counts: Vec<u32> = Vec::with_capacity(plausible_n);
        // Clamped by the same ceiling as the per-node growth below: the
        // reservation is itself an allocation, and sizing it from the header
        // alone is how a small payload asks for gigabytes up front.
        let mut links: Vec<u32> = Vec::with_capacity((plausible_n * slot0).min(max_slots));
        let mut attrs: Vec<Vec<AttrValue>> = Vec::with_capacity(plausible_n);
        let mut scratch: Vec<u8> = Vec::new();
        let mut deleted: Vec<bool> = Vec::with_capacity(plausible_n);
        for _ in 0..n {
            deleted.push(match r.u8()? {
                0 => false,
                1 => true,
                _ => return Err(HnswDecodeError::Invalid("bad tombstone flag")),
            });
            let levels = r.read_len()?;
            if levels == 0 {
                return Err(HnswDecodeError::Invalid("node with no layers"));
            }
            if levels - 1 > max_layer {
                return Err(HnswDecodeError::Invalid("node above the top layer"));
            }
            // Read through a reusable scratch buffer rather than borrowing: a
            // page source has no contiguous slice to lend. The buffer is one
            // vector wide and stays hot in cache, so this costs far less than
            // assembling the whole blob to borrow from.
            scratch.resize(vector_bytes, 0);
            r.read_exact(&mut scratch)?;
            vectors.extend(
                scratch
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c)),
            );
            if levels > r.remaining() / 8 {
                return Err(HnswDecodeError::Truncated);
            }
            layer_base.push(
                u32::try_from(link_counts.len() + levels)
                    .map_err(|_| HnswDecodeError::Invalid("too many layers"))?,
            );
            let node_slots = slot0 + (levels - 1) * slot_up;
            let node_start = links.len();
            if node_start.saturating_add(node_slots) > max_slots {
                return Err(HnswDecodeError::Invalid(
                    "link storage exceeds what the payload could describe",
                ));
            }
            links.resize(node_start + node_slots, 0);
            for layer in 0..levels {
                let count = r.read_len()?;
                let cap = if layer == 0 { m0_cap } else { m_cap };
                if count > cap {
                    return Err(HnswDecodeError::Invalid("neighbor list over degree cap"));
                }
                if count > r.remaining() / 8 {
                    return Err(HnswDecodeError::Truncated);
                }
                let within = if layer == 0 {
                    0
                } else {
                    slot0 + (layer - 1) * slot_up
                };
                let start = node_start + within;
                for slot in 0..count {
                    let id = r.read_len()?;
                    if id >= n {
                        return Err(HnswDecodeError::Invalid("link target out of range"));
                    }
                    links[start + slot] = id as u32;
                }
                link_counts.push(count as u32);
            }

            let attr_count = r.read_len()?;
            // Every value costs at least a tag byte, so a count exceeding what
            // is left is corrupt — checked before reserving, like the counts above.
            if attr_count > r.remaining() {
                return Err(HnswDecodeError::Truncated);
            }
            // The reservation is clamped separately, because that check alone is
            // weaker here than for the fixed-width counts: an all-`Null` row is
            // 1 byte per value but 16 bytes per value in memory, so the count on
            // its own would let a crafted blob reserve 16x the bytes it supplies.
            // A `Null`-heavy row simply grows the vector instead.
            let mut row = Vec::with_capacity(attr_count.min(r.remaining() / 9));
            for _ in 0..attr_count {
                row.push(match r.u8()? {
                    ATTR_TAG_NULL => AttrValue::Null,
                    ATTR_TAG_INT => AttrValue::Int(r.u64()? as i64),
                    ATTR_TAG_FLOAT => AttrValue::Float(f64::from_bits(r.u64()?)),
                    _ => return Err(HnswDecodeError::Invalid("bad attribute tag")),
                });
            }
            attrs.push(row);
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
                // `- 1` cannot underflow: every node has ≥ 1 layer.
                let entry_levels = (layer_base[e + 1] - layer_base[e]) as usize;
                if entry_levels - 1 != max_layer {
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
            n,
            vectors,
            layer_base,
            link_counts,
            links,
            attrs,
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
    use crate::filter::Atom;
    use std::ops::Bound;

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
        for id in 0..h.len() {
            let levels = h.levels(id);
            mix(levels as u64);
            for lc in 0..levels {
                let nbrs = h.neighbors(id, lc);
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

    /// Live neighbor lists, which is what the graph's value *is*. The backing
    /// slot array cannot be compared directly: its padding is unspecified, so a
    /// raw comparison both fails on meaningless differences and could pass on a
    /// wrong count that happened to leave matching bytes behind.
    fn adjacency(h: &Hnsw) -> Vec<Vec<Vec<u32>>> {
        (0..h.len())
            .map(|id| {
                (0..h.levels(id))
                    .map(|lc| h.neighbors(id, lc).to_vec())
                    .collect()
            })
            .collect()
    }

    /// A byte source that hands out at most `chunk` bytes per call, so every
    /// multi-byte value in the encoding straddles a boundary at some chunk size.
    /// The storage layer's page walk has exactly this shape; this exercises the
    /// decoder's half of it without a running server.
    struct ChokedSource<'a> {
        buf: &'a [u8],
        pos: usize,
        chunk: usize,
    }

    impl GraphBytes for ChokedSource<'_> {
        fn read_exact(&mut self, out: &mut [u8]) -> Result<(), HnswDecodeError> {
            if out.len() > self.buf.len() - self.pos {
                return Err(HnswDecodeError::Truncated);
            }
            let mut filled = 0;
            while filled < out.len() {
                let take = (out.len() - filled).min(self.chunk);
                out[filled..filled + take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
                self.pos += take;
                filled += take;
            }
            Ok(())
        }

        fn remaining(&self) -> usize {
            self.buf.len() - self.pos
        }
    }

    #[test]
    fn decodes_identically_from_a_fragmented_source() {
        let (h, _) = build(200, 12, 9);
        let bytes = h.to_bytes();
        let whole = Hnsw::from_bytes(&bytes).expect("decode");

        // 1 and 3 split every scalar; 7 lands mid-vector; a prime near the
        // vector width keeps the seams from lining up with node boundaries.
        for chunk in [1usize, 3, 7, 53] {
            let mut src = ChokedSource {
                buf: &bytes,
                pos: 0,
                chunk,
            };
            let piecewise = Hnsw::from_graph_bytes(&mut src).expect("decode");
            assert_eq!(
                adjacency(&piecewise),
                adjacency(&whole),
                "chunk size {chunk} changed the graph"
            );
            assert_eq!(piecewise.vectors, whole.vectors, "chunk size {chunk}");
            assert_eq!(piecewise.entry_point, whole.entry_point);
            assert_eq!(src.remaining(), 0, "chunk size {chunk} left bytes unread");
        }
    }

    /// Header for a blob that is well-formed up to the fields under test, so a
    /// case fails on the bound it targets rather than on framing.
    fn header(m: u64, gamma: f32, max_layer: u64, n: u64) -> Vec<u8> {
        header_ef(m, gamma, max_layer, n, 64)
    }

    /// As [`header`], with an explicit `ef_construction`. A densified graph
    /// floors it at the layer-0 cap, so a fixture using a large gamma has to
    /// supply one or it is rejected before reaching what it meant to test.
    fn header_ef(m: u64, gamma: f32, max_layer: u64, n: u64, ef: u64) -> Vec<u8> {
        header_dim(m, gamma, max_layer, n, ef, 2)
    }

    /// As [`header_ef`], with an explicit `dim`.
    fn header_dim(m: u64, gamma: f32, max_layer: u64, n: u64, ef: u64, dim: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&CODEC_MAGIC.to_le_bytes());
        b.extend_from_slice(&CODEC_VERSION.to_le_bytes());
        b.push(Metric::L2.code());
        b.push(0);
        b.extend_from_slice(&m.to_le_bytes());
        b.extend_from_slice(&gamma.to_bits().to_le_bytes());
        b.extend_from_slice(&ef.to_le_bytes()); // ef_construction
        b.extend_from_slice(&dim.to_le_bytes()); // dim
        b.extend_from_slice(&max_layer.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // seed
        b.extend_from_slice(&1u64.to_le_bytes()); // rng
        b.extend_from_slice(&0u64.to_le_bytes()); // entry point
        b.extend_from_slice(&n.to_le_bytes());
        b
    }

    /// Reports how close a graph a build actually produces comes to the
    /// decode's slot ceiling. Ignored by default — it is a margin probe rather
    /// than an assertion, and it takes a couple of minutes.
    ///
    /// Run it after changing the floor, `MAX_LAYER0_DEGREE`, or the stride:
    ///
    /// ```text
    /// cargo test --release --features pg17 --lib slot_ceiling_margin -- --ignored --nocapture
    /// ```
    ///
    /// The floor was raised to 1 << 22 because 1 << 18 refused real graphs. At
    /// the time of writing the worst legitimate case uses 19.5% of the ceiling
    /// (m = 2, gamma = 512, n = 300, dim = 1), against 105% before.
    #[test]
    #[ignore]
    fn slot_ceiling_margin() {
        let mut worst = 0.0f64;
        let mut worst_desc = String::new();
        for &(m, gamma) in &[
            (2usize, 512.0f32),
            (2, 256.0),
            (3, 341.0),
            (4, 256.0),
            (16, 4.0),
            (128, 8.0),
        ] {
            for seed in 1u64..=40 {
                for n in [1usize, 2, 64, 130, 154, 200, 300] {
                    for dim in [1usize, 8] {
                        let mut h = Hnsw::new(HnswParams {
                            m,
                            ef_construction: 4,
                            gamma,
                            metric: Metric::L2,
                            seed,
                        });
                        let mut st = seed | 1;
                        for _ in 0..n {
                            let v: Vec<f32> = (0..dim)
                                .map(|_| {
                                    st ^= st << 13;
                                    st ^= st >> 7;
                                    st ^= st << 17;
                                    (st >> 11) as f32 / (1u64 << 53) as f32
                                })
                                .collect();
                            if h.insert(v).is_err() {
                                break;
                            }
                        }
                        let bytes = h.to_bytes();
                        let budget = (bytes.len() - 76) + (1 << 22);
                        let slots: usize = (0..h.len())
                            .map(|id| (h.m0_cap + 1) + (h.levels(id) - 1) * (h.m_cap + 1))
                            .sum();
                        let ratio = slots as f64 / budget as f64;
                        if ratio > worst {
                            worst = ratio;
                            worst_desc = format!(
                                "m={m} gamma={gamma} n={n} dim={dim} seed={seed} \
                                 slots={slots} budget={budget}"
                            );
                        }
                        assert!(
                            Hnsw::from_bytes(&bytes).is_ok(),
                            "the ceiling refused a graph this build produced: {worst_desc}"
                        );
                    }
                }
            }
        }
        println!(
            "worst legitimate use of the slot ceiling: {:.1}% -- {worst_desc}",
            worst * 100.0
        );
    }

    #[test]
    fn decodes_graphs_built_at_the_densest_legal_settings() {
        // The slot ceiling guards against a payload reserving far more than it
        // describes, and the cost of getting it wrong is asymmetric: too loose
        // risks an allocation abort, too tight makes an index that already
        // exists permanently unreadable. This walks the side the attack-shaped
        // test cannot see.
        //
        // m = 2 with gamma = 512 is the worst legal case, not an arbitrary one:
        // it puts the layer-0 cap exactly on its ceiling while ml = 1/ln 2 sends
        // half the nodes to a second layer, so the wide upper stride is paid on
        // a payload that stays short. Small n is where the per-node stride most
        // outweighs the bytes, so the sizes sweep the region that matters rather
        // than a large graph where `remaining` dominates.
        for &(m, gamma) in &[(2usize, 512.0f32), (16, 4.0), (128, 8.0), (16, 1.0)] {
            for &n in &[1usize, 2, 64, 130, 154, 200, 300] {
                for &dim in &[1usize, 8] {
                    for &seed in &[1u64, 27, 0x9E37_79B9_7F4A_7C15] {
                        let mut h = Hnsw::new(HnswParams {
                            m,
                            ef_construction: 4,
                            gamma,
                            metric: Metric::L2,
                            seed,
                        });
                        let mut state = seed | 1;
                        for _ in 0..n {
                            let v: Vec<f32> = (0..dim)
                                .map(|_| {
                                    state ^= state << 13;
                                    state ^= state >> 7;
                                    state ^= state << 17;
                                    (state >> 11) as f32 / (1u64 << 53) as f32
                                })
                                .collect();
                            h.insert(v).expect("insert");
                        }
                        let bytes = h.to_bytes();
                        Hnsw::from_bytes(&bytes).unwrap_or_else(|e| {
                            panic!(
                                "a graph this build produced will not decode: \
                                 m={m} gamma={gamma} n={n} dim={dim} seed={seed:#x} \
                                 blob={} bytes: {e}",
                                bytes.len()
                            )
                        });
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_a_payload_that_reserves_far_more_than_it_describes() {
        // The header ceilings bound m, the degree cap and the top layer, and a
        // payload sitting exactly *at* all three still gets through them. Fixed
        // stride is what makes that dangerous: a node claiming the maximum
        // layers with no neighbours on any of them costs a few hundred bytes and
        // reserves a few hundred kilobytes, so a megabyte of payload asks for
        // gigabytes. The reservation failing aborts the process rather than
        // raising a catchable error, so this must be refused, not merely slow.
        //
        // m = 2 with gamma = 512 lands the layer-0 cap on its ceiling exactly.
        let m0_cap = 2048usize;
        let m_cap = 1024usize;
        let max_layer = HnswParams::MAX_LAYER;
        let levels = max_layer + 1;
        let nodes = 2000usize;

        // ef_construction is floored at the layer-0 cap for a densified graph,
        // so it has to clear 2048 or the header is refused before the payload
        // this case is actually about is ever read.
        let mut blob = header_ef(2, 512.0, max_layer as u64, nodes as u64, m0_cap as u64);
        for _ in 0..nodes {
            blob.push(0); // not tombstoned
            blob.extend_from_slice(&(levels as u64).to_le_bytes());
            // The header helper declares dim = 2, so supply both components.
            blob.extend_from_slice(&1.0f32.to_le_bytes());
            blob.extend_from_slice(&2.0f32.to_le_bytes());
            for _ in 0..levels {
                blob.extend_from_slice(&0u64.to_le_bytes()); // no neighbours
            }
            blob.extend_from_slice(&0u64.to_le_bytes()); // no attributes
        }

        let reserved_slots = nodes * (m0_cap + 1 + max_layer * (m_cap + 1));
        assert!(
            reserved_slots * 4 > blob.len() * 100,
            "fixture is not amplifying: {} bytes would reserve {} slots",
            blob.len(),
            reserved_slots
        );

        let err = Hnsw::from_bytes(&blob)
            .expect_err("a payload reserving 100x what it describes must be refused");
        assert!(
            matches!(
                err,
                HnswDecodeError::Invalid("link storage exceeds what the payload could describe")
            ),
            "must be refused by the slot ceiling specifically, got {err:?}"
        );
    }

    #[test]
    fn rejects_headers_that_would_size_a_huge_allocation() {
        // Link storage is sized from `m` and `gamma`, which arrive in the blob,
        // and fixed stride reserves a node's whole cap whether or not it uses
        // it — so these fields, unlike the rest, are not bounded by how many
        // bytes the payload actually supplies. A Rust allocation that fails
        // aborts the process rather than raising a catchable error, which for
        // an extension means every backend dies and the server enters crash
        // recovery. A handful of bytes must not be able to ask for terabytes.
        for (label, blob) in [
            ("m beyond the ceiling", header(1 << 40, 1024.0, 0, 1)),
            (
                "m times gamma beyond the degree cap",
                header(128, 1024.0, 0, 1),
            ),
            ("top layer beyond the ceiling", header(16, 1.0, 1 << 40, 1)),
            // The vector read reserves a scratch buffer from `dim` before
            // anything compares it to the payload's length, so an 85-byte blob
            // could ask the allocator for gigabytes — and a failed reservation
            // aborts the process rather than raising a catchable error.
            (
                "dimension beyond the ceiling",
                header_dim(16, 1.0, 0, 1, 64, 1 << 30),
            ),
        ] {
            let err = Hnsw::from_bytes(&blob)
                .expect_err(&format!("{label}: should be rejected, not decoded"));
            assert!(
                matches!(err, HnswDecodeError::Invalid(_)),
                "{label}: expected a validation error, got {err:?}"
            );
        }
    }

    fn assert_degree_caps(h: &Hnsw) {
        for id in 0..h.len() {
            for lc in 0..h.levels(id) {
                let nbrs = h.neighbors(id, lc);
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
        assert_eq!(
            adjacency(&restored),
            adjacency(&h),
            "neighbor lists must round-trip"
        );
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
        assert_eq!(adjacency(&restored), adjacency(&h));
    }

    #[test]
    fn serialized_len_hint_bounds_the_real_encoding() {
        // It sizes the buffer callers reserve, so an under-estimate is a silent
        // realloc; the multi-layer and attribute terms are the easy ones to get
        // wrong, so cover graphs deep enough to have upper layers.
        for &(n, dim, gamma) in &[(1usize, 4usize, 1.0f32), (200, 16, 1.0), (200, 12, 4.0)] {
            let (h, _) = build_gamma(n, dim, 7, gamma);
            assert!(
                h.to_bytes().len() <= h.serialized_len_hint(),
                "hint {} under-estimated {} bytes (n={n}, dim={dim}, gamma={gamma})",
                h.serialized_len_hint(),
                h.to_bytes().len()
            );
        }
        let empty = Hnsw::new(HnswParams::default());
        assert!(empty.to_bytes().len() <= empty.serialized_len_hint());

        let mut attrs = Hnsw::new(HnswParams::default());
        attrs
            .insert_with_attrs(
                vec![1.0, 2.0],
                vec![AttrValue::Int(1), AttrValue::Null, AttrValue::Float(2.0)],
            )
            .expect("insert");
        assert!(attrs.to_bytes().len() <= attrs.serialized_len_hint());
    }

    #[test]
    fn bytes_round_trip_preserves_attributes() {
        let mut h = Hnsw::new(HnswParams::default());
        let rows = [
            vec![AttrValue::Int(42), AttrValue::Float(9.5)],
            vec![AttrValue::Null],
            Vec::new(), // a node inserted without attributes
            vec![
                AttrValue::Int(i64::MIN),
                AttrValue::Int(i64::MAX),
                AttrValue::Float(f64::NAN),
                AttrValue::Float(f64::NEG_INFINITY),
            ],
        ];

        for (i, row) in rows.iter().enumerate() {
            h.insert_with_attrs(vec![i as f32, 1.0], row.clone())
                .expect("insert");
        }

        let restored = Hnsw::from_bytes(&h.to_bytes()).expect("decode");
        for id in 0..rows.len() {
            let (before, after) = (h.attrs(id), restored.attrs(id));
            assert_eq!(before.len(), after.len(), "row {id} changed length");
            for (b, a) in before.iter().zip(after) {
                match (b, a) {
                    // NaN is never equal to itself, so compare the bits the
                    // codec actually promises to preserve.
                    (AttrValue::Float(x), AttrValue::Float(y)) => {
                        assert_eq!(x.to_bits(), y.to_bits(), "row {id} float changed")
                    }
                    _ => assert_eq!(b, a, "row {id} value changed"),
                }
            }
        }
    }

    #[test]
    fn round_tripped_index_answers_filtered_search() {
        // The failure this guards is silent: without persisted attributes every
        // node decodes attribute-free, so a predicate matches nothing and a
        // filtered scan returns zero rows rather than erroring.
        let (h, mut rng) = build_labeled(400, 16, 1.0);
        let restored = Hnsw::from_bytes(&h.to_bytes()).expect("decode");
        let pred = selectivity(10);
        for _ in 0..10 {
            let q: Vec<f32> = (0..16).map(|_| next_f64(&mut rng) as f32).collect();
            let after = restored
                .search_filtered(&q, 10, 64, &pred)
                .expect("filtered search");
            assert!(!after.is_empty(), "round-tripped index matched nothing");
            assert_eq!(
                after,
                h.search_filtered(&q, 10, 64, &pred)
                    .expect("filtered search"),
                "filtered results changed across a round-trip"
            );
        }
    }

    #[test]
    fn decode_rejects_bad_attribute_tag() {
        let mut h = Hnsw::new(HnswParams::default());
        h.insert_with_attrs(vec![1.0, 2.0], vec![AttrValue::Int(7)])
            .expect("insert");
        let mut bytes = h.to_bytes();
        // The lone attribute's tag is the last 9 bytes' first byte.
        let tag = bytes.len() - 9;
        assert_eq!(bytes[tag], ATTR_TAG_INT);
        bytes[tag] = 0xEE;
        assert!(matches!(
            Hnsw::from_bytes(&bytes),
            Err(HnswDecodeError::Invalid("bad attribute tag"))
        ));
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

    // ---- predicate-aware (filtered) search -------------------------------

    /// A labelled index: `n` random vectors, node `i` tagged `Int(i % 100)` in
    /// column 0, so `col0 < b` selects a `b%` share that is uncorrelated with
    /// vector position. Returns the index and the PRNG state, so queries drawn
    /// afterwards stay deterministic and disjoint from the data.
    fn build_labeled(n: usize, dim: usize, gamma: f32) -> (Hnsw, u64) {
        let mut rng = 0x51EC_71F1u64;
        let mut h = Hnsw::new(HnswParams {
            m: 16,
            ef_construction: 100,
            gamma,
            metric: Metric::L2,
            seed: 4242,
        });
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| next_f64(&mut rng) as f32).collect();
            h.insert_with_attrs(v, vec![AttrValue::Int((i % 100) as i64)])
                .expect("insert");
        }
        (h, rng)
    }

    /// `col0 < percent` — matches `percent`% of a [`build_labeled`] index.
    fn selectivity(percent: i64) -> Predicate {
        Predicate::And(vec![Atom::Range {
            col: 0,
            lo: Bound::Unbounded,
            hi: Bound::Excluded(AttrValue::Int(percent)),
        }])
    }

    /// Filtered recall@k for predicate-aware search vs naive post-filtering, both
    /// given the same `ef`. The ceiling is the exact top-k *among matching nodes*.
    fn filtered_recall(h: &Hnsw, rng: &mut u64, pred: &Predicate, ef: usize) -> (f64, f64) {
        let (dim, k, queries) = (h.dim(), 10usize, 20usize);
        let (mut aware_hits, mut post_hits, mut total) = (0usize, 0usize, 0usize);

        for _ in 0..queries {
            let q: Vec<f32> = (0..dim).map(|_| next_f64(rng) as f32).collect();
            let truth: HashSet<usize> = h
                .brute_force(&q, h.len())
                .expect("brute force")
                .into_iter()
                .filter(|&(_, id)| pred.matches(h.attrs(id)))
                .map(|(_, id)| id)
                .take(k)
                .collect();
            total += truth.len();

            let aware = h.search_filtered(&q, k, ef, pred).expect("filtered search");
            assert!(
                aware.iter().all(|&(_, id)| pred.matches(h.attrs(id))),
                "filtered search returned a non-matching node"
            );
            aware_hits += aware.iter().filter(|(_, id)| truth.contains(id)).count();

            // Naive post-filter: spend the same budget blind, then discard the
            // non-matches from what came back.
            post_hits += h
                .search(&q, ef, ef)
                .expect("search")
                .iter()
                .filter(|&&(_, id)| pred.matches(h.attrs(id)))
                .take(k)
                .filter(|(_, id)| truth.contains(id))
                .count();
        }
        (
            aware_hits as f64 / total as f64,
            post_hits as f64 / total as f64,
        )
    }

    #[test]
    fn filtered_search_returns_only_matching_nodes() {
        let (h, mut rng) = build_labeled(600, 16, 1.0);
        let pred = selectivity(10);
        for _ in 0..10 {
            let q: Vec<f32> = (0..16).map(|_| next_f64(&mut rng) as f32).collect();
            let res = h
                .search_filtered(&q, 10, 64, &pred)
                .expect("filtered search");
            assert!(!res.is_empty(), "filtered search found nothing");
            for &(_, id) in &res {
                assert!(
                    pred.matches(h.attrs(id)),
                    "node {id} was returned but does not match: {:?}",
                    h.attrs(id)
                );
            }
        }
    }

    /// FNV-1a over the ids `search` returns for a fixed graph and query set.
    /// Guards the *query* path the way [`graph_fingerprint`] guards the build
    /// path: a change after `search_layer` — truncation, ordering, tie-breaks —
    /// moves this without moving the graph. Ids only, not distances, so it does
    /// not turn into a float-formatting test.
    fn search_fingerprint(h: &Hnsw) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut hash = FNV_OFFSET;
        let mut mix = |x: u64| {
            for b in x.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        };
        let mut rng = 0x00C0_FFEEu64;
        for _ in 0..20 {
            let q: Vec<f32> = (0..16).map(|_| next_f64(&mut rng) as f32).collect();
            for &(k, ef) in &[(1usize, 1usize), (10, 64), (25, 200)] {
                let res = h.search(&q, k, ef).expect("search");
                mix(res.len() as u64);
                for (rank, &(_, id)) in res.iter().enumerate() {
                    mix(rank as u64);
                    mix(id as u64);
                }
            }
        }
        hash
    }

    /// Captured by running [`search_fingerprint`] against the search code as it
    /// stood before predicates existed. Unlike the match-all equality test
    /// below — where both sides run the same branch by construction — this pins
    /// unfiltered results to what they were *before* filtering was added. A
    /// deliberate change to unfiltered search must update the constant and say
    /// so in the commit message.
    const SEARCH_REFERENCE_FINGERPRINT: u64 = 0x4746_07C6_BEFA_E8F9;

    #[test]
    fn unfiltered_search_reproduces_pre_filter_results() {
        let (mut h, _) = build(200, 16, 42);
        // Tombstones included: they are the one case where the filtered and
        // unfiltered termination rules could diverge without the graph changing.
        for id in (0..h.len()).step_by(7) {
            h.delete(id).expect("delete");
        }
        assert_eq!(
            search_fingerprint(&h),
            SEARCH_REFERENCE_FINGERPRINT,
            "unfiltered search results drifted from their pre-filter behavior"
        );
    }

    #[test]
    fn match_all_predicate_reproduces_unfiltered_search() {
        let (mut h, mut rng) = build_labeled(400, 16, 1.0);
        // Tombstones are where the filtered and unfiltered termination rules
        // could most easily diverge, so hold the fast path to equality with them
        // present.
        for id in (0..h.len()).step_by(7) {
            h.delete(id).expect("delete");
        }
        for _ in 0..10 {
            let q: Vec<f32> = (0..16).map(|_| next_f64(&mut rng) as f32).collect();
            assert_eq!(
                h.search_filtered(&q, 10, 64, &Predicate::All)
                    .expect("filtered search"),
                h.search(&q, 10, 64).expect("search"),
                "the match-all predicate diverged from unfiltered search"
            );
        }
    }

    #[test]
    fn filtered_search_never_returns_tombstoned_matches() {
        let (mut h, mut rng) = build_labeled(400, 16, 1.0);
        let pred = selectivity(50);
        let buried: Vec<usize> = (0..h.len())
            .filter(|&id| pred.matches(h.attrs(id)) && id % 3 == 0)
            .collect();
        for &id in &buried {
            h.delete(id).expect("delete");
        }
        let dead: HashSet<usize> = buried.into_iter().collect();
        for _ in 0..10 {
            let q: Vec<f32> = (0..16).map(|_| next_f64(&mut rng) as f32).collect();
            for (_, id) in h
                .search_filtered(&q, 10, 64, &pred)
                .expect("filtered search")
            {
                assert!(!dead.contains(&id), "returned tombstoned node {id}");
            }
        }
    }

    #[test]
    fn unsatisfiable_predicate_terminates_and_returns_nothing() {
        let (h, mut rng) = build_labeled(600, 16, 1.0);
        // No node carries this label, so every neighbor is a bridge and nothing
        // can ever enter the result set — the case where unbounded bridging
        // would sweep the whole graph, or not terminate at all.
        let pred = Predicate::And(vec![Atom::Eq {
            col: 0,
            value: AttrValue::Int(4242),
        }]);
        for _ in 0..5 {
            let q: Vec<f32> = (0..16).map(|_| next_f64(&mut rng) as f32).collect();
            assert!(h
                .search_filtered(&q, 10, 64, &pred)
                .expect("filtered search")
                .is_empty());
        }
    }

    #[test]
    fn bridging_reaches_a_lone_match() {
        // One node in 200 qualifies. Its neighbors are, almost surely, all
        // filtered out — so it is reachable only by hopping over them.
        let (mut h, mut rng) = (
            Hnsw::new(HnswParams {
                m: 16,
                ef_construction: 100,
                gamma: 1.0,
                metric: Metric::L2,
                seed: 31,
            }),
            0x0BAD_5EEDu64,
        );
        let n = 200usize;
        let needle = 137usize;
        for i in 0..n {
            let v: Vec<f32> = (0..16).map(|_| next_f64(&mut rng) as f32).collect();
            h.insert_with_attrs(v, vec![AttrValue::Int((i == needle) as i64)])
                .expect("insert");
        }
        let pred = Predicate::And(vec![Atom::Eq {
            col: 0,
            value: AttrValue::Int(1),
        }]);
        for _ in 0..10 {
            let q: Vec<f32> = (0..16).map(|_| next_f64(&mut rng) as f32).collect();
            let res = h
                .search_filtered(&q, 10, 64, &pred)
                .expect("filtered search");
            assert_eq!(
                res.iter().map(|&(_, id)| id).collect::<Vec<_>>(),
                vec![needle],
                "the only matching node was not reached"
            );
        }
    }

    #[test]
    fn empty_conjunction_matches_everything() {
        // Vacuously true, so it takes the *filtered* path while admitting every
        // node — the combination most likely to expose a bookkeeping bug.
        let (h, mut rng) = build_labeled(400, 16, 1.0);
        let pred = Predicate::And(Vec::new());
        for _ in 0..5 {
            let q: Vec<f32> = (0..16).map(|_| next_f64(&mut rng) as f32).collect();
            let res = h
                .search_filtered(&q, 10, 64, &pred)
                .expect("filtered search");
            assert_eq!(res.len(), 10);
            let exact = h.brute_force(&q, 1).expect("brute force");
            assert_eq!(res[0], exact[0], "nearest neighbor missed");
        }
    }

    #[test]
    fn selective_filter_keeps_recall_over_tombstones() {
        // The case that pays for the filtered search continuing past a
        // partly-filled result heap. A tombstoned *matching* node satisfies the
        // predicate, so it never triggers a detour, yet it can never fill the
        // budget either — stopping at the first candidate beyond the farthest
        // live match therefore truncates the search exactly where matches are
        // scarcest. Dropping that clause measures ~0.85 here, below this bar.
        let (mut h, mut rng) = build_labeled(2000, 16, 1.0);
        // Every third node, and 100 is not a multiple of 3, so a matching row
        // survives to be found.
        for id in (0..h.len()).step_by(3) {
            h.delete(id).expect("delete");
        }
        let (aware, _) = filtered_recall(&h, &mut rng, &selectivity(1), 64);
        assert!(
            aware >= 0.9,
            "recall@10 at 1% selectivity over tombstones: {aware:.3}"
        );
    }

    /// The differentiator, in numbers. As the filter tightens, naive
    /// post-filtering spends its budget on rows it then throws away; predicate-
    /// aware traversal keeps spending it on rows that qualify.
    ///
    /// Measured on a γ = 1 graph — the *unfavourable* case. Densification is the
    /// other half of ACORN's answer, so a graph with no spare edges is where
    /// bridging has to carry recall on its own. Numbers observed at the time of
    /// writing are quoted per case; the bars sit below them with margin.
    ///
    /// The 1% bar is what pins the two-hop bridge specifically: delete the bridge
    /// loop and this case falls to ~0.61, well under it. Keep that in mind before
    /// relaxing it — a softer bound there would let the bridge rot untested.
    #[test]
    fn filtered_recall_beats_post_filtering() {
        let (h, mut rng) = build_labeled(2000, 16, 1.0);
        let ef = 64;

        // Half the index qualifies: post-filtering copes fine here, so the bar is
        // that predicate-awareness costs nothing to get its wins elsewhere.
        // Observed: aware 1.000, post 1.000.
        let (aware, post) = filtered_recall(&h, &mut rng, &selectivity(50), ef);
        assert!(aware >= 0.95, "recall@10 at 50% selectivity: {aware:.3}");
        assert!(
            aware >= post - 0.02,
            "50% selectivity: aware {aware:.3} regressed against post-filter {post:.3}"
        );

        // One row in ten: post-filtering is already losing a third of the answers.
        // Observed: aware 0.995, post 0.625.
        let (aware, post) = filtered_recall(&h, &mut rng, &selectivity(10), ef);
        assert!(aware >= 0.9, "recall@10 at 10% selectivity: {aware:.3}");
        assert!(
            aware >= post + 0.25,
            "10% selectivity: aware {aware:.3} vs post-filter {post:.3} at ef={ef}"
        );

        // One row in a hundred: post-filtering has collapsed.
        // Observed: aware 0.970, post 0.060.
        let (aware, post) = filtered_recall(&h, &mut rng, &selectivity(1), ef);
        assert!(aware >= 0.9, "recall@10 at 1% selectivity: {aware:.3}");
        assert!(
            aware >= post + 0.5,
            "1% selectivity: aware {aware:.3} vs post-filter {post:.3} at ef={ef}"
        );
    }
}
