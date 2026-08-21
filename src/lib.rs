//! Brindle — filter-aware, hybrid vector search for PostgreSQL.
//!
//! Current surface: pure distance functions over `real[]` plus the `brindle`
//! index access method, which builds an HNSW graph and answers `ORDER BY
//! embedding <-> $1 LIMIT k` from it. Incremental inserts and the remaining
//! work — ACORN-style filtering, hybrid RRF — land in later phases, see
//! `docs/ROADMAP.md`.
//!
//! Layering: all algorithmic logic lives in dependency-free modules (e.g.
//! [`distance`], [`hnsw`]); this file and [`index_am`] are the thin pgrx
//! boundary that adapts Postgres types and turns `Result::Err` into a
//! Postgres `ERROR`.

use pgrx::prelude::*;

pub mod distance;
pub mod filter;
pub mod fusion;
pub mod hnsw;
pub mod index_am;
pub mod vector;

::pgrx::pg_module_magic!();

/// Adapt a kernel `Result` into a Postgres value, raising a clean `ERROR` on
/// failure instead of panicking. Keeps `unwrap()` out of the SQL-facing path.
#[inline]
fn or_error(result: Result<f32, distance::DistanceError>) -> f32 {
    result.unwrap_or_else(|e| error!("brindle: {e}"))
}

/// Euclidean (L2) distance between two `real[]` vectors.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_l2_distance(a: Vec<f32>, b: Vec<f32>) -> f32 {
    or_error(distance::l2_distance(&a, &b))
}

/// Cosine distance (`1 - cosine_similarity`) between two `real[]` vectors.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_cosine_distance(a: Vec<f32>, b: Vec<f32>) -> f32 {
    or_error(distance::cosine_distance(&a, &b))
}

/// Inner product (raw dot product) between two `real[]` vectors.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_inner_product(a: Vec<f32>, b: Vec<f32>) -> f32 {
    or_error(distance::inner_product(&a, &b))
}

/// Negative inner product (`<#>` semantics: smaller is nearer).
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_negative_inner_product(a: Vec<f32>, b: Vec<f32>) -> f32 {
    or_error(distance::negative_inner_product(&a, &b))
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn l2_distance_via_spi() {
        let d = Spi::get_one::<f32>(
            "SELECT brindle_l2_distance(ARRAY[1,2,3]::real[], ARRAY[4,5,6]::real[])",
        )
        .expect("SPI failed")
        .expect("null result");
        assert!((d - 5.196_152).abs() < 1e-4, "got {d}");
    }

    #[pg_test]
    fn cosine_distance_via_spi() {
        let d = Spi::get_one::<f32>(
            "SELECT brindle_cosine_distance(ARRAY[1,0]::real[], ARRAY[0,1]::real[])",
        )
        .expect("SPI failed")
        .expect("null result");
        assert!((d - 1.0).abs() < 1e-4, "got {d}");
    }

    #[pg_test]
    fn inner_product_via_spi() {
        let d = Spi::get_one::<f32>(
            "SELECT brindle_inner_product(ARRAY[1,2,3]::real[], ARRAY[4,5,6]::real[])",
        )
        .expect("SPI failed")
        .expect("null result");
        assert!((d - 32.0).abs() < 1e-4, "got {d}");
    }

    #[pg_test(error = "brindle: vector dimension mismatch: 2 != 1")]
    fn dimension_mismatch_raises_error() {
        Spi::get_one::<f32>("SELECT brindle_l2_distance(ARRAY[1,2]::real[], ARRAY[1]::real[])")
            .expect("SPI failed");
    }
}

/// pgrx test harness configuration (used by `cargo pgrx test`).
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
