//! Vector quantization: compact codes that rank candidates cheaply, and an
//! exact re-rank that restores the final order.
//!
//! Pure Rust, no Postgres dependency. A [`Quantizer`] is fitted to a sample,
//! encodes each vector into a fixed number of bytes, and estimates the distance
//! from a query to an encoded vector without decoding it. Estimates are only
//! good enough to *shortlist*: [`QuantizedVectors::search`] keeps the `r·k`
//! best estimates and re-scores them against the original `f32` vectors with
//! [`rerank`], which is what holds recall near the exact search's.
//!
//! Two quantizers:
//!
//! - [`ScalarQuantizer`] — 8 bits per component, an affine map per dimension
//!   fitted to the sample's range. 4× smaller than `f32`.
//! - [`Rabitq`] — RaBitQ (Gao & Long, SIGMOD 2024,
//!   <https://dl.acm.org/doi/10.1145/3654970>): one bit per component after a
//!   random orthogonal rotation, with an unbiased distance estimator whose
//!   error has a per-vector bound. The sign bits are 32× smaller than `f32`.
//!
//! Codes are plain bytes so they can be stored as they are; every multi-byte
//! field is little-endian.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use crate::distance::DistanceError;
use crate::vector::Metric;

/// Errors from fitting, encoding, or searching quantized vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeError {
    /// A vector's length didn't match the quantizer's dimensionality.
    DimensionMismatch { expected: usize, got: usize },
    /// A quantizer was asked for zero dimensions.
    EmptyVector,
    /// There was nothing to fit to.
    EmptySample,
    /// A component was NaN or infinite. Rejected rather than encoded because a
    /// single one would poison the fitted range or centroid for every code.
    NonFinite { index: usize },
    /// A code buffer was not [`Quantizer::code_len`] bytes long.
    CodeLength { expected: usize, got: usize },
    /// A re-rank candidate had no vector to re-score against.
    UnknownId { id: usize },
    /// A prepared query was handed to a quantizer of a different shape than
    /// the one that prepared it.
    QueryMismatch,
}

impl fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantizeError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "vector dimension mismatch: expected {expected}, got {got}"
                )
            }
            QuantizeError::EmptyVector => write!(f, "vector must not be empty"),
            QuantizeError::EmptySample => write!(f, "cannot fit a quantizer to no vectors"),
            QuantizeError::NonFinite { index } => {
                write!(f, "vector component {index} is not finite")
            }
            QuantizeError::CodeLength { expected, got } => {
                write!(f, "code is {got} bytes, expected {expected}")
            }
            QuantizeError::UnknownId { id } => write!(f, "no vector for candidate id {id}"),
            QuantizeError::QueryMismatch => {
                write!(f, "query was prepared by a different quantizer")
            }
        }
    }
}

impl std::error::Error for QuantizeError {}

impl From<DistanceError> for QuantizeError {
    fn from(e: DistanceError) -> Self {
        match e {
            DistanceError::DimensionMismatch { left, right } => QuantizeError::DimensionMismatch {
                expected: left,
                got: right,
            },
        }
    }
}

/// A vector compression scheme with an asymmetric distance estimator: data
/// vectors are encoded once, queries keep (nearly) full precision, and the
/// distance between the two is estimated straight from the code.
///
/// Estimates follow [`Metric::distance`]'s convention — smaller is nearer,
/// squared for L2 — so they rank candidates the way an exact search would, up
/// to the estimator's error.
pub trait Quantizer {
    /// What the estimator needs from a query, prepared once and reused for
    /// every code the query is compared against.
    type Query;

    /// Fit a quantizer for `dim`-dimensional vectors under `metric` to a
    /// sample representative of the data. Errs on `dim == 0`, an empty sample,
    /// or a sample vector of the wrong length or with a non-finite component.
    fn fit<'a, I>(metric: Metric, dim: usize, sample: I) -> Result<Self, QuantizeError>
    where
        Self: Sized,
        I: IntoIterator<Item = &'a [f32]>;

    /// The metric estimates are expressed in.
    fn metric(&self) -> Metric;

    /// Dimensionality of the vectors this quantizer encodes.
    fn dim(&self) -> usize;

    /// Bytes in one code.
    fn code_len(&self) -> usize;

    /// Encode `vector` into `code`, which must be exactly
    /// [`code_len`](Quantizer::code_len) bytes.
    fn encode(&self, vector: &[f32], code: &mut [u8]) -> Result<(), QuantizeError>;

    /// Prepare `query` for estimating its distance to many codes.
    fn prepare_query(&self, query: &[f32]) -> Result<Self::Query, QuantizeError>;

    /// Estimated distance from a prepared query to an encoded vector. This is
    /// the per-candidate hot loop, and does not allocate.
    ///
    /// `query` must come from this quantizer's
    /// [`prepare_query`](Quantizer::prepare_query). One prepared for another
    /// dimensionality is an error; one from a same-sized quantizer with other
    /// parameters cannot be told apart, and its estimates mean nothing.
    fn estimate_distance(&self, query: &Self::Query, code: &[u8]) -> Result<f32, QuantizeError>;
}

/// Validate `v` against `dim` and return the factor that maps it into the
/// space the metric compares in: unit length for cosine (a zero vector stays
/// zero), unchanged otherwise. `f64`, because a subnormal norm's reciprocal
/// overflows `f32`.
fn metric_scale(metric: Metric, v: &[f32], dim: usize) -> Result<f64, QuantizeError> {
    if v.len() != dim {
        return Err(QuantizeError::DimensionMismatch {
            expected: dim,
            got: v.len(),
        });
    }
    if let Some(index) = v.iter().position(|x| !x.is_finite()) {
        return Err(QuantizeError::NonFinite { index });
    }
    if metric != Metric::Cosine {
        return Ok(1.0);
    }
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    Ok(if norm > 0.0 { 1.0 / norm } else { 0.0 })
}

fn check_code_len(code: &[u8], expected: usize) -> Result<(), QuantizeError> {
    if code.len() != expected {
        return Err(QuantizeError::CodeLength {
            expected,
            got: code.len(),
        });
    }
    Ok(())
}

/// 8-bit scalar quantization: each component maps affinely onto `0..=255`
/// across the range the fitted sample spans in that dimension.
///
/// Values outside the fitted range clamp to its ends, so fit on a sample that
/// covers the data. This is the simple baseline; [`Rabitq`] is the quantizer
/// to use, at an eighth of the size.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    metric: Metric,
    lower: Vec<f32>,
    /// Width of one code step per component; zero where the sample was
    /// constant, which encodes every value there as the lower bound.
    step: Vec<f32>,
}

/// A query prepared for [`ScalarQuantizer`].
#[derive(Debug, Clone)]
pub struct ScalarQuery {
    /// Per-component terms: `q - lower` under L2, `q · step` otherwise.
    terms: Vec<f32>,
    /// The part of the estimate that does not depend on the code.
    bias: f32,
}

impl ScalarQuantizer {
    const LEVELS: f32 = 255.0;
}

impl Quantizer for ScalarQuantizer {
    type Query = ScalarQuery;

    fn fit<'a, I>(metric: Metric, dim: usize, sample: I) -> Result<Self, QuantizeError>
    where
        I: IntoIterator<Item = &'a [f32]>,
    {
        if dim == 0 {
            return Err(QuantizeError::EmptyVector);
        }
        let mut lower = vec![f32::INFINITY; dim];
        let mut upper = vec![f32::NEG_INFINITY; dim];
        let mut fitted = 0usize;
        for v in sample {
            let scale = metric_scale(metric, v, dim)?;
            for ((lo, hi), &x) in lower.iter_mut().zip(upper.iter_mut()).zip(v) {
                let x = (f64::from(x) * scale) as f32;
                *lo = lo.min(x);
                *hi = hi.max(x);
            }
            fitted += 1;
        }
        if fitted == 0 {
            return Err(QuantizeError::EmptySample);
        }
        let step = lower
            .iter()
            .zip(&upper)
            .map(|(&lo, &hi)| (hi - lo) / Self::LEVELS)
            .collect();
        Ok(Self {
            metric,
            lower,
            step,
        })
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn dim(&self) -> usize {
        self.lower.len()
    }

    fn code_len(&self) -> usize {
        self.lower.len()
    }

    fn encode(&self, vector: &[f32], code: &mut [u8]) -> Result<(), QuantizeError> {
        check_code_len(code, self.code_len())?;
        let scale = metric_scale(self.metric, vector, self.dim())?;
        for ((c, &x), (&lo, &step)) in code
            .iter_mut()
            .zip(vector)
            .zip(self.lower.iter().zip(&self.step))
        {
            let x = (f64::from(x) * scale) as f32;
            *c = if step > 0.0 {
                ((x - lo) / step).round().clamp(0.0, Self::LEVELS) as u8
            } else {
                0
            };
        }
        Ok(())
    }

    fn prepare_query(&self, query: &[f32]) -> Result<ScalarQuery, QuantizeError> {
        let scale = metric_scale(self.metric, query, self.dim())?;
        let mut terms = Vec::with_capacity(query.len());
        let mut q_dot_lower = 0.0f32;
        for ((&x, &lo), &step) in query.iter().zip(&self.lower).zip(&self.step) {
            let x = (f64::from(x) * scale) as f32;
            q_dot_lower += x * lo;
            terms.push(match self.metric {
                Metric::L2 => x - lo,
                Metric::Cosine | Metric::InnerProduct => x * step,
            });
        }
        let bias = match self.metric {
            Metric::L2 => 0.0,
            Metric::InnerProduct => -q_dot_lower,
            Metric::Cosine => 1.0 - q_dot_lower,
        };
        Ok(ScalarQuery { terms, bias })
    }

    fn estimate_distance(&self, query: &ScalarQuery, code: &[u8]) -> Result<f32, QuantizeError> {
        check_code_len(code, self.code_len())?;
        if query.terms.len() != self.dim() {
            return Err(QuantizeError::QueryMismatch);
        }
        Ok(match self.metric {
            Metric::L2 => query
                .terms
                .iter()
                .zip(&self.step)
                .zip(code)
                .map(|((&t, &step), &c)| {
                    let d = t - f32::from(c) * step;
                    d * d
                })
                .sum(),
            Metric::Cosine | Metric::InnerProduct => {
                let dot: f32 = query
                    .terms
                    .iter()
                    .zip(code)
                    .map(|(&t, &c)| t * f32::from(c))
                    .sum();
                query.bias - dot
            }
        })
    }
}

/// SplitMix64 (Steele, Lea & Flood 2014). The level generator in the graph is
/// a plain xorshift, which is fine for coin flips but weak in its low bits;
/// here every bit becomes a sign in the rotation, so every bit has to be good.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// In-place Walsh–Hadamard transform, unnormalized. `x.len()` must be a power
/// of two.
fn walsh_hadamard(x: &mut [f32]) {
    let mut half = 1;
    while half < x.len() {
        for pair in x.chunks_exact_mut(2 * half) {
            let (a, b) = pair.split_at_mut(half);
            for (u, v) in a.iter_mut().zip(b.iter_mut()) {
                let (s, d) = (*u + *v, *u - *v);
                *u = s;
                *v = d;
            }
        }
        half *= 2;
    }
}

/// A random orthogonal transform of `R^len`, `len` a multiple of 64, applied in
/// `O(len · log len)` and described entirely by `len` and a seed.
///
/// RaBitQ's analysis assumes a uniformly random rotation; storing one as a
/// dense matrix costs `len²` floats per index (9 MB at 1 536 dimensions) and
/// `len²` multiplies per vector. This is a structured stand-in of the kind
/// RaBitQ implementations use: each round flips signs at random, then applies
/// a normalized Hadamard transform to the largest power-of-two block — the
/// leading one on even rounds, the trailing one on odd. Both steps are
/// orthogonal, so the whole is too.
///
/// When `len` is not a power of two the two blocks overlap (by at least 64
/// components, since the block is more than half of `len`), and the overlap is
/// what mixes them: whatever one round spreads into it, the next spreads across
/// the other block. A butterfly between the halves, as some implementations
/// add, measured worse here — it hands half of a component's weight to a
/// partner the next round's block does not cover, so a basis vector came out
/// with a component near 10/√len instead of 5/√len.
#[derive(Debug, Clone)]
struct Rotation {
    len: usize,
    /// Largest power of two `<= len`.
    block: usize,
    /// One sign bit per component per round, `len / 64` words per round.
    signs: Vec<u64>,
}

impl Rotation {
    /// Dense data hides a weak rotation; sparse data does not. Over 90 000
    /// estimates on 768-d vectors with four non-zero components, two rounds
    /// leave ±3σ covering 98.8% with the worst estimate 22σ out, three 99.65%
    /// and 9.8σ, four 99.65% and 4.6σ — what a Gaussian predicts (99.73%, and
    /// ~4.4σ for the largest of that many draws).
    const ROUNDS: usize = 4;

    fn new(len: usize, seed: u64) -> Self {
        let mut state = seed;
        let signs = (0..Self::ROUNDS * len / 64)
            .map(|_| splitmix64(&mut state))
            .collect();
        let block = len.checked_ilog2().map_or(0, |bits| 1usize << bits);
        Self { len, block, signs }
    }

    fn apply(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.len);
        let n = x.len();
        let block = self.block.min(n);
        let scale = (block as f32).sqrt().recip();
        for (round, signs) in self.signs.chunks_exact((n / 64).max(1)).enumerate() {
            for (chunk, &word) in x.chunks_exact_mut(64).zip(signs) {
                for (j, v) in chunk.iter_mut().enumerate() {
                    let flip = ((word >> j) & 1) as u32;
                    *v = f32::from_bits(v.to_bits() ^ (flip << 31));
                }
            }
            let part = if round % 2 == 0 {
                &mut x[..block]
            } else {
                &mut x[n - block..]
            };
            walsh_hadamard(part);
            for v in part.iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// RaBitQ with one bit per dimension (Gao & Long, SIGMOD 2024).
///
/// Encoding centres a vector on the fitted centroid, rotates it by a random
/// orthogonal transform, and keeps the sign of each rotated component. The
/// rotation is what lets one bit per component carry the vector's direction:
/// it spreads the vector's energy evenly across components however the data's
/// variance was distributed beforehand, so no component's sign is wasted.
///
/// The estimator is RaBitQ's, not a Hamming distance. Writing `o` for the
/// centred, normalized data vector, `ō` for its sign code scaled to unit
/// length, and `q` for the query, it estimates `⟨o, q⟩` as `⟨ō, q⟩ / ⟨ō, o⟩`.
/// Dividing by the code's alignment with its own vector is what makes the
/// estimate unbiased, and the alignment also fixes the error's spread:
/// [`Rabitq::error_bound`] reports it per pair. The query is itself quantized
/// to 4 bits per component, which turns `⟨ō, q⟩` into five popcounts per
/// 64 dimensions.
///
/// One formula covers every metric. L2 and inner product both follow from
/// `⟨x − c, y − c⟩` plus terms that depend on only one side, which are folded
/// into a per-vector and a per-query constant; cosine is inner product over
/// unit vectors.
///
/// A code is [`Quantizer::code_len`] bytes: three `f32` correction factors,
/// then one sign bit per rotated component, packed least-significant first
/// into `u64` words. The rotation runs in `dim` rounded up to a multiple of 64,
/// so the bits are exactly 1/32 of the `f32` vector's size when `dim` is a
/// multiple of 64; the factors add 12 bytes on top.
///
/// The persisted state is `(metric, centroid, seed)` — the rotation is
/// regenerated from the seed — and [`Rabitq::new`] rebuilds an identical
/// quantizer from exactly that.
#[derive(Debug, Clone)]
pub struct Rabitq {
    metric: Metric,
    centroid: Vec<f32>,
    centroid_norm_sq: f32,
    seed: u64,
    rotation: Rotation,
}

/// A query prepared for [`Rabitq`]: the rotated, centred query quantized to
/// 4 bits per component, stored as bit planes aligned with the code's words.
#[derive(Debug, Clone)]
pub struct RabitqQuery {
    /// `planes[w][b]` holds bit `b` of the 4-bit level of every component in
    /// code word `w`.
    planes: Vec<[u64; Rabitq::QUERY_BITS]>,
    /// Value of level 0.
    lower: f32,
    /// Value of one level step.
    step: f32,
    /// Sum of the quantized components.
    sum: f32,
    /// The part of the estimate that depends on the query alone.
    add: f32,
    /// Norm of the centred query, which scales the error bound.
    norm: f32,
}

/// Per-vector constants a code carries ahead of its bits.
#[derive(Debug, Clone, Copy)]
struct Factors {
    /// The part of the estimate that depends on the vector alone.
    add: f32,
    /// Multiplies the code–query signed dot product.
    rescale: f32,
    /// One standard deviation of the estimate, per unit of query norm.
    sigma: f32,
}

impl Factors {
    const BYTES: usize = 12;

    fn to_bytes(self) -> [u8; Self::BYTES] {
        let [a0, a1, a2, a3] = self.add.to_le_bytes();
        let [r0, r1, r2, r3] = self.rescale.to_le_bytes();
        let [s0, s1, s2, s3] = self.sigma.to_le_bytes();
        [a0, a1, a2, a3, r0, r1, r2, r3, s0, s1, s2, s3]
    }

    fn from_bytes(bytes: [u8; Self::BYTES]) -> Self {
        let [a0, a1, a2, a3, r0, r1, r2, r3, s0, s1, s2, s3] = bytes;
        Self {
            add: f32::from_le_bytes([a0, a1, a2, a3]),
            rescale: f32::from_le_bytes([r0, r1, r2, r3]),
            sigma: f32::from_le_bytes([s0, s1, s2, s3]),
        }
    }
}

impl Rabitq {
    /// Bits per component of a prepared query, the paper's choice. Measured,
    /// the rounding moves an estimate by 0.13–0.16σ on average, against the
    /// ~0.8σ the 1-bit code itself contributes.
    const QUERY_BITS: usize = 4;

    /// How many standard deviations [`Rabitq::error_bound`] spans: the
    /// paper's `ε₀`, at the value RaBitQ uses by default.
    pub const BOUND_SIGMAS: f32 = 1.9;

    /// Rotation seed [`Quantizer::fit`] uses.
    pub const DEFAULT_SEED: u64 = 0x5EED_5EED_5EED_5EED;

    /// A quantizer with the given centroid and rotation seed — what a stored
    /// index rebuilds its quantizer from. `dim` is the centroid's length. Errs
    /// on an empty or non-finite centroid.
    pub fn new(metric: Metric, centroid: Vec<f32>, seed: u64) -> Result<Self, QuantizeError> {
        if centroid.is_empty() {
            return Err(QuantizeError::EmptyVector);
        }
        if let Some(index) = centroid.iter().position(|x| !x.is_finite()) {
            return Err(QuantizeError::NonFinite { index });
        }
        let centroid_norm_sq = centroid
            .iter()
            .map(|&c| f64::from(c) * f64::from(c))
            .sum::<f64>() as f32;
        let padded = centroid.len().div_ceil(64) * 64;
        Ok(Self {
            metric,
            centroid_norm_sq,
            seed,
            rotation: Rotation::new(padded, seed),
            centroid,
        })
    }

    /// The centroid codes are centred on, in the metric's space (unit-length
    /// inputs for cosine).
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// The rotation's seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Dimensionality the rotation runs in: `dim` rounded up to a multiple of
    /// 64, which is also the number of sign bits per code.
    pub fn padded_dim(&self) -> usize {
        self.rotation.len
    }

    /// Half-width of the confidence interval around
    /// [`estimate_distance`](Quantizer::estimate_distance) for this pair:
    /// [`Rabitq::BOUND_SIGMAS`] standard deviations of the estimator's error,
    /// in the metric's own units.
    ///
    /// The bound is probabilistic over the random rotation: the paper proves
    /// it holds with probability at least `1 − 2·exp(−c₀·ε₀²)`. In practice the
    /// error is close to Gaussian, so ±1.9σ covers ~94% of pairs and ±3σ
    /// (scale this by `3.0 / BOUND_SIGMAS`) ~99.7% — measured at 93.8–94.2%
    /// and 99.7% across the test fixtures, query rounding included. σ itself
    /// models only the 1-bit code; the query's 4-bit rounding adds a further
    /// 0.13–0.16σ on average.
    pub fn error_bound(&self, query: &RabitqQuery, code: &[u8]) -> Result<f32, QuantizeError> {
        let (factors, _) = self.split_code(code)?;
        Ok(Self::BOUND_SIGMAS * factors.sigma * query.norm)
    }

    fn dim(&self) -> usize {
        self.centroid.len()
    }

    /// Centre `v` (scaled by `scale`) on the centroid, zero-pad it to the
    /// rotation's length, and rotate it. Also returns `⟨v·scale, centroid⟩`,
    /// which the inner-product metrics need and which is cheapest taken here.
    fn rotate_centred(&self, v: &[f32], scale: f64) -> (Vec<f32>, f64) {
        let mut rotated = vec![0.0f32; self.rotation.len];
        let mut dot_centroid = 0.0f64;
        for ((r, &x), &c) in rotated.iter_mut().zip(v).zip(&self.centroid) {
            let u = f64::from(x) * scale;
            dot_centroid += u * f64::from(c);
            *r = (u - f64::from(c)) as f32;
        }
        self.rotation.apply(&mut rotated);
        (rotated, dot_centroid)
    }

    /// The factors for a vector whose rotated, centred form has squared norm
    /// `norm_sq` and L1 norm `l1`.
    ///
    /// The estimate of `⟨x − c, q − c⟩` is `‖x − c‖² · s / l1`, where `s` is
    /// the signed sum `Σ ±qᵢ` the code selects: `⟨ō, q⟩ / ⟨ō, o⟩` with both
    /// `√dim` factors and one norm cancelled. L2 needs it twice
    /// (`‖x‖² + ‖q‖² − 2⟨x, q⟩`, centred); inner product and cosine once,
    /// negated, which is what `scale` carries.
    fn factors(&self, norm_sq: f64, l1: f64, dot_centroid: f64) -> Factors {
        let scale = match self.metric {
            Metric::L2 => 2.0,
            Metric::Cosine | Metric::InnerProduct => 1.0,
        };
        let add = match self.metric {
            Metric::L2 => norm_sq,
            Metric::Cosine | Metric::InnerProduct => -dot_centroid,
        };
        if l1 <= 0.0 {
            // The vector sits on the centroid: its centred part is zero, so
            // the estimate is exact with no code term at all.
            return Factors {
                add: add as f32,
                rescale: 0.0,
                sigma: 0.0,
            };
        }
        let n = self.rotation.len as f64;
        let norm = norm_sq.sqrt();
        // ⟨ō, o⟩: at least 1/√n, about 0.8 for a well-rotated vector.
        let alignment = l1 / (n.sqrt() * norm);
        let spread = (1.0 - alignment * alignment).max(0.0).sqrt();
        Factors {
            add: add as f32,
            rescale: (-scale * norm_sq / l1) as f32,
            sigma: (scale * norm * spread / (alignment * (n - 1.0).sqrt())) as f32,
        }
    }

    /// Validate a code's length and split it into its factors and its sign
    /// words.
    fn split_code<'c>(&self, code: &'c [u8]) -> Result<(Factors, &'c [[u8; 8]]), QuantizeError> {
        check_code_len(code, self.code_len())?;
        let Some((head, bits)) = code.split_first_chunk::<{ Factors::BYTES }>() else {
            return Err(QuantizeError::CodeLength {
                expected: self.code_len(),
                got: code.len(),
            });
        };
        Ok((Factors::from_bytes(*head), bits.as_chunks::<8>().0))
    }
}

impl Quantizer for Rabitq {
    type Query = RabitqQuery;

    /// Fits the centroid — the sample mean, in the metric's space — and uses
    /// [`Rabitq::DEFAULT_SEED`] for the rotation.
    fn fit<'a, I>(metric: Metric, dim: usize, sample: I) -> Result<Self, QuantizeError>
    where
        I: IntoIterator<Item = &'a [f32]>,
    {
        if dim == 0 {
            return Err(QuantizeError::EmptyVector);
        }
        let mut sum = vec![0.0f64; dim];
        let mut fitted = 0usize;
        for v in sample {
            let scale = metric_scale(metric, v, dim)?;
            for (s, &x) in sum.iter_mut().zip(v) {
                *s += f64::from(x) * scale;
            }
            fitted += 1;
        }
        if fitted == 0 {
            return Err(QuantizeError::EmptySample);
        }
        let centroid = sum.iter().map(|&s| (s / fitted as f64) as f32).collect();
        Self::new(metric, centroid, Self::DEFAULT_SEED)
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn dim(&self) -> usize {
        Rabitq::dim(self)
    }

    fn code_len(&self) -> usize {
        Factors::BYTES + self.rotation.len / 8
    }

    fn encode(&self, vector: &[f32], code: &mut [u8]) -> Result<(), QuantizeError> {
        let (expected, got) = (self.code_len(), code.len());
        check_code_len(code, expected)?;
        let scale = metric_scale(self.metric, vector, self.dim())?;
        let (rotated, dot_centroid) = self.rotate_centred(vector, scale);
        let (norm_sq, l1) = rotated.iter().fold((0.0f64, 0.0f64), |(sq, l1), &x| {
            let x = f64::from(x);
            (sq + x * x, l1 + x.abs())
        });
        let factors = self.factors(norm_sq, l1, dot_centroid);

        let Some((head, bits)) = code.split_first_chunk_mut::<{ Factors::BYTES }>() else {
            return Err(QuantizeError::CodeLength { expected, got });
        };
        *head = factors.to_bytes();
        let (words, _) = bits.as_chunks_mut::<8>();
        for (word, chunk) in words.iter_mut().zip(rotated.chunks_exact(64)) {
            let signs = chunk
                .iter()
                .enumerate()
                .fold(0u64, |w, (j, &x)| w | (u64::from(x > 0.0) << j));
            *word = signs.to_le_bytes();
        }
        Ok(())
    }

    fn prepare_query(&self, query: &[f32]) -> Result<RabitqQuery, QuantizeError> {
        let scale = metric_scale(self.metric, query, self.dim())?;
        let (rotated, dot_centroid) = self.rotate_centred(query, scale);

        let (lower, upper, norm_sq) = rotated.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY, 0.0f64),
            |(lo, hi, sq), &x| (lo.min(x), hi.max(x), sq + f64::from(x) * f64::from(x)),
        );
        let max_level = ((1u32 << Self::QUERY_BITS) - 1) as f32;
        let step = (upper - lower) / max_level;

        let mut planes = vec![[0u64; Self::QUERY_BITS]; rotated.len() / 64];
        let mut level_sum = 0u64;
        for (word_planes, chunk) in planes.iter_mut().zip(rotated.chunks_exact(64)) {
            for (j, &x) in chunk.iter().enumerate() {
                let level = if step > 0.0 {
                    ((x - lower) / step).round().clamp(0.0, max_level) as u64
                } else {
                    0
                };
                level_sum += level;
                for (b, plane) in word_planes.iter_mut().enumerate() {
                    *plane |= ((level >> b) & 1) << j;
                }
            }
        }

        let add = match self.metric {
            Metric::L2 => norm_sq,
            Metric::InnerProduct => f64::from(self.centroid_norm_sq) - dot_centroid,
            Metric::Cosine => 1.0 + f64::from(self.centroid_norm_sq) - dot_centroid,
        };
        Ok(RabitqQuery {
            planes,
            lower,
            step,
            sum: rotated.len() as f32 * lower + step * level_sum as f32,
            add: add as f32,
            norm: norm_sq.sqrt() as f32,
        })
    }

    fn estimate_distance(&self, query: &RabitqQuery, code: &[u8]) -> Result<f32, QuantizeError> {
        let (factors, words) = self.split_code(code)?;
        if query.planes.len() != words.len() {
            return Err(QuantizeError::QueryMismatch);
        }
        let mut ones = 0u64;
        let mut weighted = 0u64;
        for (word, planes) in words.iter().zip(&query.planes) {
            let word = u64::from_le_bytes(*word);
            ones += u64::from(word.count_ones());
            for (b, plane) in planes.iter().enumerate() {
                weighted += u64::from((word & plane).count_ones()) << b;
            }
        }
        // Σ over set bits of the quantized query component, then Σ ±qᵢ.
        let selected = query.lower * ones as f32 + query.step * weighted as f32;
        let signed = 2.0 * selected - query.sum;
        Ok(factors.add + query.add + factors.rescale * signed)
    }
}

/// A `(distance, id)` pair ordered nearest first, ties broken by id. NaN sorts
/// after every number, as PostgreSQL orders floats, so an undefined distance
/// can never displace a real one.
#[derive(Debug, Clone, Copy)]
struct Scored {
    dist: f32,
    id: usize,
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .is_nan()
            .cmp(&other.dist.is_nan())
            .then(self.dist.total_cmp(&other.dist))
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Scored {}

/// How many candidates per requested result [`QuantizedVectors::search`]
/// re-ranks when the caller has no better figure: the `r` in `r·k`.
///
/// Measured with RaBitQ over 10 000 vectors, recall@10 against an exact scan:
///
/// | data (128-d)                 | r = 1 | 5    | 10   | 20   | 50   |
/// |------------------------------|-------|------|------|------|------|
/// | 64 Gaussian clusters, L2     | 0.26  | 0.73 | 0.94 | 1.00 | 1.00 |
/// | rank-16 subspace + noise, L2 | 0.60  | 0.98 | 1.00 | 1.00 | 1.00 |
/// | isotropic Gaussian, L2       | 0.28  | 0.60 | 0.77 | 0.88 | 0.98 |
///
/// 20 is the smallest factor that recovers the structured fixtures. The
/// isotropic row is the worst case, not a typical one: every distance there is
/// nearly the same, which is also the data graph search itself does worst on
/// (`docs/BENCHMARKS.md`).
pub const DEFAULT_RERANK_FACTOR: usize = 20;

/// Re-score `candidates` exactly under `metric` and return the `k` nearest as
/// `(distance, id)` pairs, nearest first.
///
/// `vector` maps a candidate id to its full-precision vector; an id it has no
/// vector for is an error rather than a silent drop, since it means the
/// candidate source and the vector store disagree. Candidates are scored as
/// given, so a duplicated id comes back twice. Distances are exact, in the
/// metric's internal form (squared for L2).
pub fn rerank<'v, I, F>(
    metric: Metric,
    query: &[f32],
    candidates: I,
    k: usize,
    vector: F,
) -> Result<Vec<(f32, usize)>, QuantizeError>
where
    I: IntoIterator<Item = usize>,
    F: Fn(usize) -> Option<&'v [f32]>,
{
    let candidates = candidates.into_iter();
    let mut scored = Vec::with_capacity(candidates.size_hint().0);
    for id in candidates {
        let v = vector(id).ok_or(QuantizeError::UnknownId { id })?;
        scored.push(Scored {
            dist: metric.distance(query, v)?,
            id,
        });
    }
    if k < scored.len() {
        scored.select_nth_unstable(k);
        scored.truncate(k);
    }
    scored.sort_unstable();
    Ok(scored.into_iter().map(|s| (s.dist, s.id)).collect())
}

/// A set of vectors encoded by one quantizer, codes stored end to end. Ids are
/// dense and assigned in push order.
#[derive(Debug, Clone)]
pub struct QuantizedVectors<Q> {
    quantizer: Q,
    codes: Vec<u8>,
}

impl<Q: Quantizer> QuantizedVectors<Q> {
    /// An empty set encoded by `quantizer`.
    pub fn new(quantizer: Q) -> Self {
        Self {
            quantizer,
            codes: Vec::new(),
        }
    }

    /// Encode `vector` and return its id. On error nothing is added.
    pub fn push(&mut self, vector: &[f32]) -> Result<usize, QuantizeError> {
        let id = self.len();
        let start = self.codes.len();
        self.codes.resize(start + self.quantizer.code_len(), 0);
        if let Err(e) = self.quantizer.encode(vector, &mut self.codes[start..]) {
            self.codes.truncate(start);
            return Err(e);
        }
        Ok(id)
    }

    /// Number of encoded vectors.
    pub fn len(&self) -> usize {
        self.codes
            .len()
            .checked_div(self.quantizer.code_len())
            .unwrap_or(0)
    }

    /// True when nothing has been encoded.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// The quantizer the codes were encoded with.
    pub fn quantizer(&self) -> &Q {
        &self.quantizer
    }

    /// The code for `id`, or `None` past the end.
    pub fn code(&self, id: usize) -> Option<&[u8]> {
        let width = self.quantizer.code_len();
        let start = id.checked_mul(width)?;
        self.codes.get(start..start.checked_add(width)?)
    }

    /// Total bytes the codes occupy.
    pub fn code_bytes(&self) -> usize {
        self.codes.len()
    }

    /// The `n` codes nearest `query` by estimated distance, as
    /// `(estimate, id)` pairs nearest first. Scans every code.
    pub fn candidates(&self, query: &[f32], n: usize) -> Result<Vec<(f32, usize)>, QuantizeError> {
        Ok(self
            .shortlist(query, n)?
            .into_sorted_vec()
            .into_iter()
            .map(|s| (s.dist, s.id))
            .collect())
    }

    /// The `n` nearest codes by estimate, in heap order.
    fn shortlist(&self, query: &[f32], n: usize) -> Result<BinaryHeap<Scored>, QuantizeError> {
        let prepared = self.quantizer.prepare_query(query)?;
        if n == 0 {
            return Ok(BinaryHeap::new());
        }
        let width = self.quantizer.code_len().max(1);
        let mut nearest = BinaryHeap::with_capacity(n.min(self.len()) + 1);
        for (id, code) in self.codes.chunks_exact(width).enumerate() {
            let cand = Scored {
                dist: self.quantizer.estimate_distance(&prepared, code)?,
                id,
            };
            if nearest.len() < n {
                nearest.push(cand);
            } else if let Some(mut farthest) = nearest.peek_mut() {
                if cand < *farthest {
                    *farthest = cand;
                }
            }
        }
        Ok(nearest)
    }

    /// Approximate k-nearest-neighbor search: shortlist the
    /// `k · rerank_factor` nearest codes by estimate, then re-score the
    /// shortlist exactly with [`rerank`]. Returns up to `k` `(distance, id)`
    /// pairs, nearest first, with exact distances.
    ///
    /// `rerank_factor` is the recall/cost knob — see
    /// [`DEFAULT_RERANK_FACTOR`]; `1` re-ranks only the `k` codes the
    /// estimates already chose, which fixes their order but not their
    /// membership. `vector` supplies the full-precision vector for an id.
    pub fn search<'v, F>(
        &self,
        query: &[f32],
        k: usize,
        rerank_factor: usize,
        vector: F,
    ) -> Result<Vec<(f32, usize)>, QuantizeError>
    where
        F: Fn(usize) -> Option<&'v [f32]>,
    {
        let shortlist = self.shortlist(query, k.saturating_mul(rerank_factor.max(1)))?;
        rerank(
            self.quantizer.metric(),
            query,
            shortlist.into_vec().into_iter().map(|s| s.id),
            k,
            vector,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Deterministic test data: SplitMix64 uniforms, Box–Muller Gaussians.
    struct Rng(u64);

    impl Rng {
        fn uniform(&mut self) -> f64 {
            (splitmix64(&mut self.0) >> 11) as f64 / (1u64 << 53) as f64
        }

        fn gaussian(&mut self) -> f32 {
            let u1 = 1.0 - self.uniform();
            let u2 = self.uniform();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        }

        fn gaussian_vec(&mut self, dim: usize) -> Vec<f32> {
            (0..dim).map(|_| self.gaussian()).collect()
        }
    }

    /// `n` points from a mixture of `clusters` Gaussians: centres ~ N(0, 1),
    /// points around them with standard deviation `spread`.
    fn clustered(
        rng: &mut Rng,
        n: usize,
        dim: usize,
        clusters: usize,
        spread: f32,
    ) -> Vec<Vec<f32>> {
        let centres: Vec<Vec<f32>> = (0..clusters).map(|_| rng.gaussian_vec(dim)).collect();
        (0..n)
            .map(|i| {
                centres[i % clusters]
                    .iter()
                    .map(|&c| c + spread * rng.gaussian())
                    .collect()
            })
            .collect()
    }

    /// Vectors with `nonzero` random components set and the rest zero: the
    /// input a weak rotation fails on, since one barely-mixed axis shows.
    fn sparse(rng: &mut Rng, n: usize, dim: usize, nonzero: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                let mut v = vec![0.0f32; dim];
                for _ in 0..nonzero {
                    let i = (rng.uniform() * dim as f64) as usize;
                    v[i] = rng.gaussian();
                }
                v
            })
            .collect()
    }

    fn fit<Q: Quantizer>(metric: Metric, data: &[Vec<f32>]) -> Q {
        Q::fit(metric, data[0].len(), data.iter().map(Vec::as_slice)).unwrap()
    }

    fn encode_all<Q: Quantizer>(q: Q, data: &[Vec<f32>]) -> QuantizedVectors<Q> {
        let mut set = QuantizedVectors::new(q);
        for v in data {
            set.push(v).unwrap();
        }
        set
    }

    fn lookup<'a>(data: &'a [Vec<f32>]) -> impl Fn(usize) -> Option<&'a [f32]> {
        |id| data.get(id).map(Vec::as_slice)
    }

    /// Recall@k of quantized search at `rerank_factor` against an exact scan
    /// of the same vectors — whose own recall is 1 by definition, so this *is*
    /// the ratio to the f32 recall.
    fn recall<Q: Quantizer>(
        set: &QuantizedVectors<Q>,
        data: &[Vec<f32>],
        queries: &[Vec<f32>],
        k: usize,
        rerank_factor: usize,
    ) -> f64 {
        let metric = set.quantizer().metric();
        let mut hits = 0usize;
        for q in queries {
            let truth: HashSet<usize> = rerank(metric, q, 0..data.len(), k, lookup(data))
                .unwrap()
                .into_iter()
                .map(|(_, id)| id)
                .collect();
            let got = set.search(q, k, rerank_factor, lookup(data)).unwrap();
            assert_eq!(got.len(), k);
            hits += got.iter().filter(|(_, id)| truth.contains(id)).count();
        }
        hits as f64 / (queries.len() * k) as f64
    }

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    // --- rotation -----------------------------------------------------------

    #[test]
    fn rotation_preserves_norms_and_inner_products() {
        let mut rng = Rng(1);
        for len in [64, 128, 192, 768, 1536] {
            let rotation = Rotation::new(len, 42);
            for _ in 0..20 {
                let (x, y) = (rng.gaussian_vec(len), rng.gaussian_vec(len));
                let (mut rx, mut ry) = (x.clone(), y.clone());
                rotation.apply(&mut rx);
                rotation.apply(&mut ry);
                let scale = (dot(&x, &x) * dot(&y, &y)).sqrt();
                assert!(
                    (dot(&rx, &rx) - dot(&x, &x)).abs() <= 1e-5 * dot(&x, &x),
                    "len {len}: norm not preserved"
                );
                assert!(
                    (dot(&rx, &ry) - dot(&x, &y)).abs() <= 1e-5 * scale,
                    "len {len}: inner product not preserved"
                );
            }
        }
    }

    #[test]
    fn rotation_spreads_every_axis() {
        // A basis vector is the least spread input there is. A random rotation
        // leaves its largest component near 3.5/√len; a block some round never
        // reaches, or weight stranded outside the next round's block, shows up
        // as one far larger.
        for len in [64, 192, 768, 960] {
            let rotation = Rotation::new(len, 42);
            for axis in 0..len {
                let mut e = vec![0.0f32; len];
                e[axis] = 1.0;
                rotation.apply(&mut e);
                let largest = e.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                assert!(
                    largest <= 6.0 / (len as f32).sqrt(),
                    "len {len}, axis {axis}: a component kept {largest}"
                );
            }
        }
    }

    #[test]
    fn rotation_is_fixed_by_its_seed() {
        let x = Rng(3).gaussian_vec(192);
        let rotated = |seed| {
            let mut v = x.clone();
            Rotation::new(192, seed).apply(&mut v);
            v
        };
        assert_eq!(rotated(7), rotated(7));
        assert_ne!(rotated(7), rotated(8));
    }

    // --- estimator ----------------------------------------------------------

    /// How RaBitQ's estimates sit against the truth, in units of the
    /// per-pair standard deviation the code reports.
    struct ErrorStats {
        mean_z: f64,
        within_bound: f64,
        within_3_sigma: f64,
        max_z: f64,
        /// Mean shift the 4-bit query rounding causes, in σ.
        query_rounding: f64,
    }

    fn error_stats(metric: Metric, data: &[Vec<f32>], queries: &[Vec<f32>]) -> ErrorStats {
        let set = encode_all(fit::<Rabitq>(metric, data), data);
        let rq = set.quantizer();
        let (mut z_sum, mut bound_hits, mut sigma3_hits, mut max_z, mut rounding) =
            (0.0, 0usize, 0usize, 0.0f64, 0.0);
        let mut pairs = 0usize;
        for q in queries {
            let prepared = rq.prepare_query(q).unwrap();
            let scale = metric_scale(metric, q, rq.dim()).unwrap();
            let (rotated_q, _) = rq.rotate_centred(q, scale);
            for (id, v) in data.iter().enumerate() {
                let code = set.code(id).unwrap();
                let estimate = f64::from(rq.estimate_distance(&prepared, code).unwrap());
                let truth = f64::from(metric.distance(q, v).unwrap());
                let bound = f64::from(rq.error_bound(&prepared, code).unwrap());
                let sigma = bound / f64::from(Rabitq::BOUND_SIGMAS);
                let z = (estimate - truth) / sigma;
                z_sum += z;
                bound_hits += usize::from((estimate - truth).abs() <= bound);
                sigma3_hits += usize::from(z.abs() <= 3.0);
                max_z = max_z.max(z.abs());

                // The same estimate from the unrounded query.
                let (factors, words) = rq.split_code(code).unwrap();
                let signed: f64 = words
                    .iter()
                    .zip(rotated_q.chunks_exact(64))
                    .map(|(w, chunk)| {
                        let w = u64::from_le_bytes(*w);
                        chunk
                            .iter()
                            .enumerate()
                            .map(|(j, &x)| if (w >> j) & 1 == 1 { x } else { -x })
                            .map(f64::from)
                            .sum::<f64>()
                    })
                    .sum();
                let unrounded = f64::from(factors.add)
                    + f64::from(prepared.add)
                    + f64::from(factors.rescale) * signed;
                rounding += ((estimate - unrounded) / sigma).abs();
                pairs += 1;
            }
        }
        let pairs = pairs as f64;
        ErrorStats {
            mean_z: z_sum / pairs,
            within_bound: bound_hits as f64 / pairs,
            within_3_sigma: sigma3_hits as f64 / pairs,
            max_z,
            query_rounding: rounding / pairs,
        }
    }

    #[test]
    fn rabitq_estimate_is_unbiased_and_within_its_error_bound() {
        let mut rng = Rng(5);
        let fixtures = [
            ("clustered, 128-d", clustered(&mut rng, 2_020, 128, 32, 0.5)),
            // Not a multiple of 64: the zero padding takes part in the rotation.
            ("clustered, 100-d", clustered(&mut rng, 2_020, 100, 32, 0.5)),
            // Not a power of two: the two Hadamard blocks overlap, not coincide.
            ("sparse, 768-d", sparse(&mut rng, 2_020, 768, 4)),
        ];
        for (name, all) in &fixtures {
            let (data, queries) = all.split_at(2_000);
            for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
                let s = error_stats(metric, data, queries);
                let label = format!("{name}, {metric:?}");
                // Unbiased: 40 000 pairs put the mean's own noise near 0.005σ.
                assert!(
                    s.mean_z.abs() < 0.03,
                    "{label}: mean error {:+.3}σ",
                    s.mean_z
                );
                // ±1.9σ is ~94.3% of a Gaussian; ±3σ is 99.73%.
                assert!(
                    (0.92..=0.97).contains(&s.within_bound),
                    "{label}: bound covers {:.3}",
                    s.within_bound
                );
                assert!(
                    s.within_3_sigma >= 0.99,
                    "{label}: ±3σ covers {:.4}",
                    s.within_3_sigma
                );
                assert!(
                    s.max_z < 6.0,
                    "{label}: an estimate was off by {:.1}σ",
                    s.max_z
                );
                assert!(
                    s.query_rounding < 0.25,
                    "{label}: query rounding moved estimates {:.3}σ",
                    s.query_rounding
                );
            }
        }
    }

    #[test]
    fn rabitq_is_exact_where_the_code_has_nothing_to_estimate() {
        // A vector on the centroid has no centred part, and a query on the
        // centroid has none either: in both cases the code's term vanishes and
        // only the exactly stored constants remain.
        let mut rng = Rng(9);
        let centroid = rng.gaussian_vec(100);
        for metric in [Metric::L2, Metric::InnerProduct] {
            let rq = Rabitq::new(metric, centroid.clone(), 1).unwrap();
            let other = rng.gaussian_vec(100);
            for (data, query) in [(&centroid, &other), (&other, &centroid)] {
                let mut code = vec![0u8; rq.code_len()];
                rq.encode(data, &mut code).unwrap();
                let prepared = rq.prepare_query(query).unwrap();
                let estimate = rq.estimate_distance(&prepared, &code).unwrap();
                let truth = metric.distance(query, data).unwrap();
                assert!(
                    (estimate - truth).abs() <= 1e-4 * truth.abs().max(1.0),
                    "{metric:?}: estimated {estimate}, true {truth}"
                );
            }
        }
    }

    // --- size ---------------------------------------------------------------

    #[test]
    fn one_bit_codes_are_32x_smaller_than_f32() {
        for (dim, total_ratio) in [(64, 12.8), (128, 18.2), (768, 28.4), (1536, 30.1)] {
            let rq = Rabitq::new(Metric::L2, vec![0.0; dim], 1).unwrap();
            let f32_bytes = dim * 4;
            let sign_bytes = rq.code_len() - Factors::BYTES;
            assert_eq!(sign_bytes * 32, f32_bytes, "{dim}-d: sign bits");
            let ratio = f32_bytes as f64 / rq.code_len() as f64;
            assert!(ratio >= total_ratio, "{dim}-d: {ratio:.1}× with factors");

            let mut set = QuantizedVectors::new(rq);
            let mut rng = Rng(2);
            for _ in 0..100 {
                set.push(&rng.gaussian_vec(dim)).unwrap();
            }
            assert_eq!(set.code_bytes(), 100 * (dim / 8 + Factors::BYTES));
        }
        // Padding to whole words is the price of a dimension that is not a
        // multiple of 64.
        let rq = Rabitq::new(Metric::L2, vec![0.0; 100], 1).unwrap();
        assert_eq!((rq.padded_dim(), rq.code_len()), (128, 16 + Factors::BYTES));
    }

    #[test]
    fn scalar_codes_are_one_byte_per_component() {
        let data = clustered(&mut Rng(4), 10, 96, 2, 0.5);
        let sq: ScalarQuantizer = fit(Metric::L2, &data);
        assert_eq!(sq.code_len(), 96);
    }

    // --- recall -------------------------------------------------------------

    #[test]
    fn rabitq_with_rerank_keeps_95_percent_of_exact_recall() {
        // The fixture behind DEFAULT_RERANK_FACTOR's clustered row.
        let mut rng = Rng(7);
        let all = clustered(&mut rng, 10_050, 128, 64, 0.5);
        let (data, queries) = all.split_at(10_000);
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let set = encode_all(fit::<Rabitq>(metric, data), data);
            let reranked = recall(&set, data, queries, 10, DEFAULT_RERANK_FACTOR);
            assert!(
                reranked >= 0.95,
                "{metric:?}: recall@10 {reranked:.3} with re-rank"
            );
            // Re-rank is what makes the codes usable: the estimates alone pick
            // a fraction of the true top 10.
            let estimates_only = recall(&set, data, queries, 10, 1);
            assert!(
                estimates_only < reranked,
                "{metric:?}: {estimates_only:.3} without re-rank vs {reranked:.3} with"
            );
        }
    }

    #[test]
    fn rabitq_recall_holds_when_dim_is_not_a_power_of_two() {
        let mut rng = Rng(8);
        for dim in [100, 192] {
            let all = clustered(&mut rng, 5_030, dim, 32, 0.5);
            let (data, queries) = all.split_at(5_000);
            let set = encode_all(fit::<Rabitq>(Metric::L2, data), data);
            let r = recall(&set, data, queries, 10, DEFAULT_RERANK_FACTOR);
            assert!(r >= 0.95, "{dim}-d: recall@10 {r:.3}");
        }
    }

    #[test]
    fn scalar_with_rerank_is_near_exact() {
        let mut rng = Rng(6);
        let all = clustered(&mut rng, 5_030, 128, 32, 0.5);
        let (data, queries) = all.split_at(5_000);
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let set = encode_all(fit::<ScalarQuantizer>(metric, data), data);
            let r = recall(&set, data, queries, 10, 2);
            assert!(r >= 0.99, "{metric:?}: recall@10 {r:.3}");
        }
    }

    // --- persisted state ----------------------------------------------------

    /// Inputs built from exact decimal fractions, so neither the data nor its
    /// codes depend on a platform's `ln`/`cos`.
    fn exact_inputs(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut state = 0xC0DE;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| (splitmix64(&mut state) % 1000) as f32 / 100.0 - 5.0)
                    .collect()
            })
            .collect()
    }

    fn codes_of(rq: &Rabitq, data: &[Vec<f32>]) -> Vec<u8> {
        encode_all(rq.clone(), data).codes
    }

    #[test]
    fn rebuilding_from_centroid_and_seed_reproduces_every_code() {
        let data = exact_inputs(50, 100);
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let fitted: Rabitq = fit(metric, &data);
            let rebuilt = Rabitq::new(metric, fitted.centroid().to_vec(), fitted.seed()).unwrap();
            assert_eq!(codes_of(&fitted, &data), codes_of(&rebuilt, &data));
        }
    }

    /// FNV-1a of the codes [`exact_inputs`] encode to under the default seed.
    /// Stored indexes will hold codes, and the rotation is rebuilt from its
    /// seed rather than stored, so any change to the rotation, the seed
    /// expansion, or the code layout silently invalidates them. A deliberate
    /// change must update this constant and say so.
    const REFERENCE_CODES: u64 = 0x72E2_C2F9_894A_224D;

    #[test]
    fn code_format_is_stable() {
        // 150 pads to 192, which is not a power of two: every step of the
        // rotation and the padding are pinned.
        let data = exact_inputs(50, 150);
        let rq: Rabitq = fit(Metric::L2, &data);
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for b in codes_of(&rq, &data) {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        assert_eq!(hash, REFERENCE_CODES, "codes changed: {hash:#018X}");
    }

    // --- errors and edges ---------------------------------------------------

    fn rejects_bad_fits<Q: Quantizer + std::fmt::Debug>() {
        let ok = [1.0f32, 2.0];
        let fit = |dim, sample: &[&[f32]]| Q::fit(Metric::L2, dim, sample.iter().copied());
        assert_eq!(fit(0, &[]).unwrap_err(), QuantizeError::EmptyVector);
        assert_eq!(fit(2, &[]).unwrap_err(), QuantizeError::EmptySample);
        assert_eq!(
            fit(2, &[&ok, &[1.0]]).unwrap_err(),
            QuantizeError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        );
        assert_eq!(
            fit(2, &[&ok, &[1.0, f32::NAN]]).unwrap_err(),
            QuantizeError::NonFinite { index: 1 }
        );
        assert_eq!(
            fit(2, &[&[f32::INFINITY, 0.0]]).unwrap_err(),
            QuantizeError::NonFinite { index: 0 }
        );
    }

    #[test]
    fn fit_rejects_what_it_cannot_learn_from() {
        rejects_bad_fits::<Rabitq>();
        rejects_bad_fits::<ScalarQuantizer>();
    }

    #[test]
    fn new_rejects_an_unusable_centroid() {
        assert_eq!(
            Rabitq::new(Metric::L2, Vec::new(), 1).unwrap_err(),
            QuantizeError::EmptyVector
        );
        assert_eq!(
            Rabitq::new(Metric::L2, vec![0.0, f32::NAN], 1).unwrap_err(),
            QuantizeError::NonFinite { index: 1 }
        );
    }

    fn rejects_bad_inputs<Q: Quantizer>(q: &Q) {
        let dim = q.dim();
        let v = vec![0.5f32; dim];
        let len = q.code_len();
        let mut short = vec![0u8; len - 1];
        let wrong_len = QuantizeError::CodeLength {
            expected: len,
            got: len - 1,
        };
        assert_eq!(q.encode(&v, &mut short).unwrap_err(), wrong_len);

        let mut code = vec![0u8; len];
        assert_eq!(
            q.encode(&v[1..], &mut code).unwrap_err(),
            QuantizeError::DimensionMismatch {
                expected: dim,
                got: dim - 1
            }
        );
        let mut nan = v.clone();
        nan[3] = f32::NAN;
        assert_eq!(
            q.encode(&nan, &mut code).unwrap_err(),
            QuantizeError::NonFinite { index: 3 }
        );
        assert!(matches!(
            q.prepare_query(&nan),
            Err(QuantizeError::NonFinite { index: 3 })
        ));
        assert!(matches!(
            q.prepare_query(&v[1..]),
            Err(QuantizeError::DimensionMismatch { .. })
        ));

        q.encode(&v, &mut code).unwrap();
        let prepared = q.prepare_query(&v).unwrap();
        assert_eq!(
            q.estimate_distance(&prepared, &code[1..]).unwrap_err(),
            wrong_len
        );
    }

    #[test]
    fn malformed_inputs_are_errors_not_panics() {
        let data = clustered(&mut Rng(10), 20, 70, 2, 0.5);
        rejects_bad_inputs(&fit::<Rabitq>(Metric::L2, &data));
        rejects_bad_inputs(&fit::<ScalarQuantizer>(Metric::L2, &data));
    }

    fn rejects_foreign_queries<Q: Quantizer>(narrow: &Q, wide: &Q) {
        let mut code = vec![0u8; narrow.code_len()];
        narrow.encode(&vec![0.5; narrow.dim()], &mut code).unwrap();
        let foreign = wide.prepare_query(&vec![0.5; wide.dim()]).unwrap();
        assert_eq!(
            narrow.estimate_distance(&foreign, &code).unwrap_err(),
            QuantizeError::QueryMismatch
        );
    }

    #[test]
    fn a_query_prepared_for_another_dimensionality_is_refused() {
        // 64 against 256 dimensions: a mismatch the code-length check alone
        // cannot see, since it never looks at the query.
        let narrow = clustered(&mut Rng(13), 10, 64, 2, 0.5);
        let wide = clustered(&mut Rng(14), 10, 256, 2, 0.5);
        rejects_foreign_queries::<Rabitq>(&fit(Metric::L2, &narrow), &fit(Metric::L2, &wide));
        rejects_foreign_queries::<ScalarQuantizer>(
            &fit(Metric::L2, &narrow),
            &fit(Metric::L2, &wide),
        );
    }

    #[test]
    fn failed_push_adds_nothing() {
        let data = clustered(&mut Rng(11), 20, 64, 2, 0.5);
        let mut set = encode_all(fit::<Rabitq>(Metric::L2, &data), &data);
        let before = set.code_bytes();
        assert!(set.push(&[1.0; 63]).is_err());
        assert_eq!((set.len(), set.code_bytes()), (20, before));
        assert_eq!(set.push(&data[0]).unwrap(), 20);
        assert!(set.code(20).is_some() && set.code(21).is_none());
    }

    #[test]
    fn rerank_errs_on_a_candidate_with_no_vector() {
        let data = vec![vec![0.0f32, 1.0]];
        assert_eq!(
            rerank(Metric::L2, &[0.0, 0.0], [0, 1], 2, lookup(&data)).unwrap_err(),
            QuantizeError::UnknownId { id: 1 }
        );
    }

    #[test]
    fn nan_distances_rank_last() {
        // x86 produces NaN with the sign bit set, which total_cmp alone would
        // rank ahead of every number.
        let mut scored: Vec<Scored> = [-f32::NAN, f32::NAN, f32::INFINITY, 1.0, -2.0]
            .iter()
            .enumerate()
            .map(|(id, &dist)| Scored { dist, id })
            .collect();
        scored.sort_unstable();
        let order: Vec<usize> = scored.iter().map(|s| s.id).collect();
        assert_eq!(&order[..3], &[4, 3, 2]);

        // A zero vector has no cosine distance to anything.
        let data = vec![vec![0.0f32, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let set = encode_all(fit::<Rabitq>(Metric::Cosine, &data), &data);
        let got = set.search(&[1.0, 0.1], 3, 1, lookup(&data)).unwrap();
        let ids: Vec<usize> = got.iter().map(|&(_, id)| id).collect();
        assert_eq!(ids, [1, 2, 0]);
        assert!(got[2].0.is_nan());
    }

    #[test]
    fn search_returns_what_exists_in_order() {
        let data = clustered(&mut Rng(12), 30, 64, 3, 0.5);
        let set = encode_all(fit::<Rabitq>(Metric::L2, &data), &data);
        assert!(set
            .search(&data[0], 0, 5, lookup(&data))
            .unwrap()
            .is_empty());

        let everything = set.search(&data[0], 100, 5, lookup(&data)).unwrap();
        assert_eq!(everything.len(), 30);
        assert!(everything.windows(2).all(|w| w[0].0 <= w[1].0));
        assert_eq!(everything[0], (0.0, 0));

        let shortlist = set.candidates(&data[0], 100).unwrap();
        assert_eq!(shortlist.len(), 30);
        assert!(shortlist.windows(2).all(|w| w[0].0 <= w[1].0));

        let empty = QuantizedVectors::new(fit::<Rabitq>(Metric::L2, &data));
        assert!(empty
            .search(&data[0], 10, 5, lookup(&data))
            .unwrap()
            .is_empty());
        assert!(matches!(
            empty.search(&data[0][1..], 10, 5, lookup(&data)),
            Err(QuantizeError::DimensionMismatch { .. })
        ));
    }
}
