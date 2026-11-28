//! Metric selection and vector validation — the bridge from the pure distance
//! kernels ([`crate::distance`]) to the index. No Postgres dependencies, so this
//! is unit-testable with plain `cargo test`.
//!
//! Every metric is normalized so that **smaller is nearer**, which lets the HNSW
//! candidate heaps treat the value uniformly regardless of metric.

use crate::distance::{self, DistanceError};

/// Distance metric an index is built/queried with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean distance. Internally uses **squared** L2 (monotonic with true L2
    /// but cheaper — no `sqrt`). Take the square root yourself only if you need a
    /// real distance magnitude; ordering is identical.
    L2,
    /// Cosine distance, `1 - cosine_similarity`, in `[0, 2]`.
    Cosine,
    /// Negative inner product, so larger similarity ⇒ smaller value ("nearer").
    InnerProduct,
}

impl Metric {
    /// Distance between two equal-length vectors under this metric. Smaller is
    /// always nearer. Propagates [`DistanceError`] on a dimension mismatch.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> Result<f32, DistanceError> {
        match self {
            Metric::L2 => distance::l2_squared(a, b),
            Metric::Cosine => distance::cosine_distance(a, b),
            Metric::InnerProduct => distance::negative_inner_product(a, b),
        }
    }

    /// Stable on-disk discriminant for serialized indexes. Existing codes must
    /// never be renumbered — persisted graphs decode through them.
    pub fn code(self) -> u8 {
        match self {
            Metric::L2 => 0,
            Metric::Cosine => 1,
            Metric::InnerProduct => 2,
        }
    }

    /// Inverse of [`Metric::code`]; `None` for an unknown discriminant.
    pub fn from_code(code: u8) -> Option<Metric> {
        match code {
            0 => Some(Metric::L2),
            1 => Some(Metric::Cosine),
            2 => Some(Metric::InnerProduct),
            _ => None,
        }
    }
}

/// Validate that `v` matches the expected dimensionality.
#[inline]
pub fn validate_dim(v: &[f32], expected: usize) -> Result<(), DistanceError> {
    if v.len() != expected {
        return Err(DistanceError::DimensionMismatch {
            left: expected,
            right: v.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn l2_uses_squared_distance() {
        // squared L2 of [1,2,3] vs [4,5,6] = 9+9+9 = 27 (no sqrt)
        let d = Metric::L2
            .distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .unwrap();
        assert!((d - 27.0).abs() < EPS, "got {d}");
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        let d = Metric::Cosine.distance(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!((d - 1.0).abs() < EPS, "got {d}");
    }

    #[test]
    fn inner_product_smaller_is_nearer() {
        // identical vectors (largest dot) must be "nearer" than orthogonal ones
        let near = Metric::InnerProduct
            .distance(&[1.0, 1.0], &[1.0, 1.0])
            .unwrap(); // -2
        let far = Metric::InnerProduct
            .distance(&[1.0, 1.0], &[0.0, 0.0])
            .unwrap(); //  0
        assert!(near < far, "near={near} should be < far={far}");
    }

    #[test]
    fn dimension_mismatch_is_error_not_panic() {
        assert!(Metric::L2.distance(&[1.0, 2.0], &[1.0]).is_err());
        assert!(validate_dim(&[1.0, 2.0], 3).is_err());
        assert!(validate_dim(&[1.0, 2.0], 2).is_ok());
    }
}
