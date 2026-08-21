//! Per-index build options: `CREATE INDEX ... USING brindle (...) WITH (...)`.
//!
//! Boundary layer. Postgres stores the `WITH (...)` clause as text on
//! `pg_class`, hands it to `amoptions` whenever the index is opened, and
//! caches the parsed struct on `Relation.rd_options`. `build_params` reads it
//! back at build time.
//!
//! The parsed values only ever reach a *build* — the graph serializes the
//! parameters it was built with, so an `ALTER INDEX ... SET (...)` changes
//! nothing until the index is rebuilt.

use core::ffi::{c_char, CStr};
use core::sync::atomic::{AtomicU32, Ordering};

use pgrx::pg_guard;
use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::hnsw::HnswParams;

/// The parsed `WITH (...)` clause, cached on `Relation.rd_options`.
///
/// `#[repr(C)]` with a leading varlena header is the layout `build_reloptions`
/// fills in: it allocates this struct and writes each parsed value at the byte
/// offset named in [`parse_table`].
#[repr(C)]
struct BrindleOptions {
    /// varlena length header, written by `build_reloptions`.
    vl_len_: i32,
    m: i32,
    ef_construction: i32,
    gamma: f64,
}

const M_OPT: &CStr = c"m";
const EF_CONSTRUCTION_OPT: &CStr = c"ef_construction";
const GAMMA_OPT: &CStr = c"gamma";

/// `m` below 2 cannot form a navigable graph. The upper bound is well past the
/// useful range (the HNSW paper's sweet spot is 5–48); the bound that actually
/// caps allocation is [`MAX_LAYER0_DEGREE`], since γ multiplies `m`.
const MIN_M: i32 = 2;
const MAX_M: i32 = 128;

/// Ceiling on the layer-0 degree cap `2·m·γ` that the core derives from these
/// options.
///
/// `m` and `gamma` are each individually bounded, but their *product* is what
/// sizes every per-node link list — and, because the core floors
/// `ef_construction` at the layer-0 cap for a densified graph, the build's
/// candidate pool and visited set too. Left unchecked, `m = 128, gamma = 1024`
/// asks for 262144 neighbors per node, which no build survives. Rejecting the
/// combination up front turns an out-of-memory CREATE INDEX into an error
/// naming the two knobs responsible.
const MAX_LAYER0_DEGREE: f64 = 2048.0;

/// The build pool must hold at least a handful of candidates to choose from.
/// `Hnsw::new` raises a value below the graph's degree cap to that floor, so a
/// small setting is corrected rather than rejected.
const MIN_EF_CONSTRUCTION: i32 = 4;
const MAX_EF_CONSTRUCTION: i32 = 1000;

/// Postgres assigns extension reloption kinds at load time; 0 is only a
/// placeholder for the window before [`init`] runs, which is before any
/// relation can be opened.
static RELOPT_KIND: AtomicU32 = AtomicU32::new(0);

/// Register brindle's index reloptions. Call once from `_PG_init`.
pub fn init() {
    let defaults = HnswParams::default();
    // SAFETY: these are the documented registration calls for a custom
    // reloption kind, valid only from _PG_init; the name/description pointers
    // are 'static and Postgres copies what it keeps.
    unsafe {
        let kind = pg_sys::add_reloption_kind();
        RELOPT_KIND.store(kind, Ordering::Relaxed);

        // Changing a build option only takes effect on a rebuild, so hold the
        // strongest lock: it is what REINDEX will need anyway.
        let lockmode = pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE;
        pg_sys::add_int_reloption(
            kind,
            M_OPT.as_ptr(),
            c"Neighbors per graph node.".as_ptr(),
            defaults.m as i32,
            MIN_M,
            MAX_M,
            lockmode,
        );
        pg_sys::add_int_reloption(
            kind,
            EF_CONSTRUCTION_OPT.as_ptr(),
            c"Candidate pool size during index build.".as_ptr(),
            defaults.ef_construction as i32,
            MIN_EF_CONSTRUCTION,
            MAX_EF_CONSTRUCTION,
            lockmode,
        );
        pg_sys::add_real_reloption(
            kind,
            GAMMA_OPT.as_ptr(),
            c"Edge-density multiplier keeping the graph navigable under selective filters."
                .as_ptr(),
            defaults.gamma as f64,
            1.0,
            HnswParams::MAX_GAMMA as f64,
            lockmode,
        );
    }
}

/// Where `build_reloptions` writes each parsed option.
fn parse_table() -> [pg_sys::relopt_parse_elt; 3] {
    [
        parse_elt(
            M_OPT,
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            core::mem::offset_of!(BrindleOptions, m),
        ),
        parse_elt(
            EF_CONSTRUCTION_OPT,
            pg_sys::relopt_type::RELOPT_TYPE_INT,
            core::mem::offset_of!(BrindleOptions, ef_construction),
        ),
        parse_elt(
            GAMMA_OPT,
            pg_sys::relopt_type::RELOPT_TYPE_REAL,
            core::mem::offset_of!(BrindleOptions, gamma),
        ),
    ]
}

fn parse_elt(
    name: &'static CStr,
    opttype: pg_sys::relopt_type::Type,
    offset: usize,
) -> pg_sys::relopt_parse_elt {
    pg_sys::relopt_parse_elt {
        optname: name.as_ptr() as *const c_char,
        opttype,
        offset: offset as core::ffi::c_int,
    }
}

/// `amoptions`: parse and validate a `WITH (...)` clause into [`BrindleOptions`].
#[pg_guard]
pub(super) unsafe extern "C" fn amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    let elems = parse_table();
    // SAFETY: build_reloptions is the standard AM entry point for this; it
    // rejects unknown names and out-of-range values itself (raising an ERROR
    // when `validate`), and returns NULL when no options are set.
    let parsed = pg_sys::build_reloptions(
        reloptions,
        validate,
        RELOPT_KIND.load(Ordering::Relaxed),
        core::mem::size_of::<BrindleOptions>(),
        elems.as_ptr(),
        elems.len() as core::ffi::c_int,
    )
    .cast::<BrindleOptions>();

    // build_reloptions checks each option against its own range; the limit on
    // what they derive together is ours to enforce, and only when Postgres
    // asked us to validate (an index already on disk was accepted once).
    if validate {
        // SAFETY: non-NULL means build_reloptions allocated our struct.
        if let Some(options) = parsed.as_ref() {
            reject_excessive_degree(options);
        }
    }

    parsed.cast()
}

/// Raise a clear `ERROR` when `m` and `gamma` together ask for more neighbors
/// per node than [`MAX_LAYER0_DEGREE`] allows.
fn reject_excessive_degree(options: &BrindleOptions) {
    let layer0_degree = 2.0 * f64::from(options.m) * options.gamma;
    if layer0_degree > MAX_LAYER0_DEGREE {
        // The literal `ERROR` arm diverges, which is the truth: reporting at
        // ERROR longjmps out of this frame.
        ereport!(
            ERROR,
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "brindle: m = {} with gamma = {} would give each node up to {} neighbors",
                options.m, options.gamma, layer0_degree as u64
            ),
            format!(
                "The layer-0 degree cap 2 * m * gamma must not exceed {}.",
                MAX_LAYER0_DEGREE as u64
            )
        );
    }
}

/// Build parameters for an index: the `WITH (...)` clause where one was given,
/// defaults otherwise. One definition shared by `ambuild` and `ambuildempty`,
/// so an unlogged index's init fork can never disagree with its main fork.
///
/// # Safety
/// `index` must be a valid, open index relation.
pub(super) unsafe fn build_params(index: pg_sys::Relation) -> HnswParams {
    // The metric is the operator class's to choose, not a reloption: it decides
    // what the graph means, and changing it would invalidate the graph rather
    // than retune it. Both returns below build on these defaults, so neither
    // can drop it.
    let defaults = HnswParams {
        metric: super::opclass::index_metric(index),
        ..HnswParams::default()
    };
    // SAFETY: caller guarantees `index` is a live relation. Postgres populates
    // rd_options by calling our own `amoptions`, so when it is non-NULL it
    // points at a BrindleOptions, and it lives as long as the relcache entry.
    let options = (*index).rd_options.cast::<BrindleOptions>();
    let Some(options) = options.as_ref() else {
        return defaults;
    };
    HnswParams {
        m: options.m as usize,
        ef_construction: options.ef_construction as usize,
        gamma: options.gamma as f32,
        ..defaults
    }
}
