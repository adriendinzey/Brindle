//! Brindle's algorithms, with no Postgres dependency.
//!
//! Everything here is plain Rust over slices: [`distance`] kernels, [`vector`]
//! metric selection, the [`hnsw`] graph and its predicate-aware search, the
//! [`filter`] predicate model, and [`fusion`] of ranked lists. The `brindle`
//! extension crate adapts Postgres types to and from these and turns their
//! `Result::Err`s into Postgres `ERROR`s.
//!
//! Keeping this crate free of any Postgres dependency is what lets its tests
//! run as a plain `cargo test -p brindle-core` — no `pg_config`, no running
//! database.

pub mod distance;
pub mod filter;
pub mod fusion;
pub mod hnsw;
pub mod vector;
