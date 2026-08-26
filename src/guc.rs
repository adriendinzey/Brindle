//! Session-level configuration (GUCs) for brindle.
//!
//! Boundary layer: these are the runtime knobs a user reaches through `SET` /
//! `postgresql.conf`. Build-time knobs are per-index instead and live in
//! [`crate::index_am::options`].

use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};

/// Backing storage for `brindle.ef_search`. The boot value matches
/// [`crate::hnsw::HnswParams::default`]'s `ef_construction`, so an untuned
/// session searches with the same budget the graph was built at.
static EF_SEARCH: GucSetting<i32> = GucSetting::<i32>::new(DEFAULT_EF_SEARCH);

const DEFAULT_EF_SEARCH: i32 = 64;

/// A candidate pool below the requested `k` cannot return `k` neighbors, so the
/// search widens it anyway; 1 is simply the smallest value that means anything.
const MIN_EF_SEARCH: i32 = 1;

/// `ef_search` sizes a scan's candidate heap and visited set, so it is the one
/// knob a user can point at memory. The cap keeps a single scan's transient
/// allocation bounded while still allowing a near-exhaustive walk of a large
/// index.
const MAX_EF_SEARCH: i32 = 10_000;

/// Register brindle's GUCs. Call once from `_PG_init`.
pub fn init() {
    GucRegistry::define_int_guc(
        "brindle.ef_search",
        "Candidate pool size for brindle index scans.",
        "Larger values raise recall and cost. This is also the ceiling on how \
         many rows a scan can return: a LIMIT above it comes back short, so \
         raise this to see further. Applies per query, and needs no rebuild.",
        &EF_SEARCH,
        MIN_EF_SEARCH,
        MAX_EF_SEARCH,
        GucContext::Userset,
        GucFlags::default(),
    );
}

/// The current session's search budget, for use by an index scan.
pub fn ef_search() -> usize {
    // Postgres clamps assignments to the registered range, so this only guards
    // against a read before `init()` ran.
    EF_SEARCH.get().max(MIN_EF_SEARCH) as usize
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    use crate::guc;

    #[pg_test]
    fn ef_search_defaults_to_boot_value() {
        let shown = Spi::get_one::<String>("SHOW brindle.ef_search")
            .expect("SPI failed")
            .expect("null result");
        assert_eq!(shown, "64");
        assert_eq!(guc::ef_search(), 64);
    }

    #[pg_test]
    fn set_ef_search_is_visible_to_the_extension() {
        Spi::run("SET brindle.ef_search = 200").expect("set");
        let shown = Spi::get_one::<String>("SHOW brindle.ef_search")
            .expect("SPI failed")
            .expect("null result");
        assert_eq!(shown, "200");
        // The Rust-side read is what a scan actually consults.
        assert_eq!(guc::ef_search(), 200);

        Spi::run("RESET brindle.ef_search").expect("reset");
        assert_eq!(guc::ef_search(), 64);
    }

    #[pg_test(
        error = "200000 is outside the valid range for parameter \"brindle.ef_search\" (1 .. 10000)"
    )]
    fn ef_search_rejects_out_of_range() {
        Spi::run("SET brindle.ef_search = 200000").expect("set");
    }

    #[pg_test(
        error = "0 is outside the valid range for parameter \"brindle.ef_search\" (1 .. 10000)"
    )]
    fn ef_search_rejects_zero() {
        Spi::run("SET brindle.ef_search = 0").expect("set");
    }
}
