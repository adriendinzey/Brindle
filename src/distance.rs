//! Distance kernels.
//!
//! These are **pure Rust over `&[f32]` slices** with no Postgres dependency, so
//! they can be unit-tested with `cargo test` and benchmarked in isolation. The
//! pgrx binding layer (`lib.rs`) wraps them and converts errors into Postgres
//! `ERROR`s at the boundary.
//!
//! Loops are written to be autovectorization-friendly (simple `zip` folds) and
//! allocation-free — the hot path never allocates. Explicit SIMD
//! (`std::arch` / `wide`) is a Phase 5 optimization, gated behind benchmarks.

use std::fmt;

/// Error returned by distance kernels for malformed input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceError {
    /// Operands have different dimensionality.
    DimensionMismatch { left: usize, right: usize },
}

impl fmt::Display for DistanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceError::DimensionMismatch { left, right } => {
                write!(f, "vector dimension mismatch: {left} != {right}")
            }
        }
    }
}

impl std::error::Error for DistanceError {}

#[inline]
fn check_dims(a: &[f32], b: &[f32]) -> Result<(), DistanceError> {
    if a.len() != b.len() {
        return Err(DistanceError::DimensionMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(())
}

/// Dot product. Assumes equal length (callers check). Allocation-free.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Squared Euclidean (L2²) distance. Avoids the `sqrt` for ranking, where
/// monotonicity is all that matters.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> Result<f32, DistanceError> {
    check_dims(a, b)?;
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        acc += d * d;
    }
    Ok(acc)
}

/// Euclidean (L2) distance.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> Result<f32, DistanceError> {
    Ok(l2_squared(a, b)?.sqrt())
}

/// Cosine *distance* = `1 - cosine_similarity`, in `[0, 2]`.
///
/// If either operand has zero magnitude the result is `NaN` (cosine is
/// undefined there), matching pgvector's behavior rather than erroring on
/// legitimately-stored zero vectors. Computed in a single pass.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32, DistanceError> {
    check_dims(a, b)?;
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return Ok(f32::NAN);
    }
    Ok(1.0 - dot / denom)
}

/// Inner product (raw dot). Larger means more similar.
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> Result<f32, DistanceError> {
    check_dims(a, b)?;
    Ok(dot(a, b))
}

/// Negative inner product — pgvector's `<#>` operator semantics, so that
/// "smaller is nearer" holds uniformly across distance functions.
#[inline]
pub fn negative_inner_product(a: &[f32], b: &[f32]) -> Result<f32, DistanceError> {
    Ok(-inner_product(a, b)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn l2_known_value() {
        // sqrt(3^2 + 3^2 + 3^2) = sqrt(27) = 5.196152...
        let d = l2_distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert!((d - 5.196_152).abs() < EPS, "got {d}");
    }

    #[test]
    fn l2_squared_skips_sqrt() {
        let d = l2_squared(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert!((d - 27.0).abs() < EPS, "got {d}");
    }

    #[test]
    fn distance_to_self_is_zero() {
        let v = [0.3, -1.2, 4.0, 9.5];
        assert!(l2_distance(&v, &v).unwrap().abs() < EPS);
        assert!(cosine_distance(&v, &v).unwrap().abs() < EPS);
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        let d = cosine_distance(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!((d - 1.0).abs() < EPS, "got {d}");
    }

    #[test]
    fn cosine_opposite_is_two() {
        let d = cosine_distance(&[1.0, 0.0], &[-1.0, 0.0]).unwrap();
        assert!((d - 2.0).abs() < EPS, "got {d}");
    }

    #[test]
    fn cosine_zero_vector_is_nan() {
        let d = cosine_distance(&[0.0, 0.0], &[1.0, 2.0]).unwrap();
        assert!(d.is_nan());
    }

    #[test]
    fn inner_product_basic() {
        assert!((inner_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap() - 32.0).abs() < EPS);
        assert!(
            (negative_inner_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap() + 32.0).abs()
                < EPS
        );
    }

    #[test]
    fn dimension_mismatch_errors() {
        let err = l2_distance(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert_eq!(err, DistanceError::DimensionMismatch { left: 2, right: 1 });
    }
}
