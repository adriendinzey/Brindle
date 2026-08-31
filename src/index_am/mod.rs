//! The brindle Index Access Method: `CREATE INDEX ... USING brindle`.
//!
//! Boundary layer only — it adapts the Postgres AM callback ABI to the pure
//! [`crate::hnsw`] core. `ambuild` scans the heap into an in-memory graph and
//! persists it through [`storage`] using the build parameters [`options`]
//! parses from the index's `WITH (...)` clause, `aminsert` extends that stored
//! graph a tuple at a time, and `ambulkdelete` tombstones what VACUUM reclaims;
//! [`scan`] answers `ORDER BY` distance queries from it. The metric a build
//! ranks by and the type its column holds both come from the index's operator
//! class — see [`opclass`].

pub mod opclass;
pub mod options;
pub mod scan;
pub mod storage;

use core::ffi::c_void;

use pgrx::itemptr::{item_pointer_get_both, item_pointer_set_all};
use pgrx::prelude::*;
use pgrx::{pg_guard, pg_sys, Array, FromDatum, PgBox};

use crate::hnsw::{Hnsw, HnswParams};
use crate::pg_vector;
use opclass::VectorKind;
use options::build_params;
use storage::TidPair;

/// Handler returning the filled `IndexAmRoutine`; referenced by
/// `CREATE ACCESS METHOD brindle`.
#[pg_extern(sql = "
CREATE FUNCTION brindle_amhandler(internal) RETURNS index_am_handler
PARALLEL SAFE IMMUTABLE STRICT COST 0.0001
LANGUAGE c AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';
CREATE ACCESS METHOD brindle TYPE INDEX HANDLER brindle_amhandler;
")]
fn brindle_amhandler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    // SAFETY: alloc_node pallocs a zeroed IndexAmRoutine with the right tag.
    let mut routine =
        unsafe { PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine) };

    routine.amstrategies = 0; // ordering operators only, so no fixed strategy set
    routine.amsupport = opclass::SUPPORT_PROCS; // distance, metric, indexed type
    routine.amoptsprocnum = 0;
    routine.amcanorder = false;
    routine.amcanorderbyop = true; // distance ORDER BY is the point of this AM
    routine.amcanbackward = false;
    routine.amcanunique = false;
    routine.amcanmulticol = false;
    routine.amoptionalkey = true; // ORDER BY-only scans carry no key clause
    routine.amsearcharray = false;
    routine.amsearchnulls = false;
    routine.amstorage = false;
    routine.amclusterable = false;
    routine.ampredlocks = false;
    routine.amcanparallel = false;
    routine.amcaninclude = false;
    routine.amusemaintenanceworkmem = false;
    routine.amparallelvacuumoptions = pg_sys::VACUUM_OPTION_NO_PARALLEL as u8;
    routine.amkeytype = pg_sys::InvalidOid;
    // Fields added by newer majors (amsummarizing, amcanbuildparallel, ...)
    // keep their zeroed defaults, which is the behavior we want everywhere.

    routine.ambuild = Some(ambuild);
    routine.ambuildempty = Some(ambuildempty);
    routine.aminsert = Some(aminsert);
    routine.ambulkdelete = Some(ambulkdelete);
    routine.amvacuumcleanup = Some(amvacuumcleanup);
    routine.amcostestimate = Some(amcostestimate);
    routine.amoptions = Some(options::amoptions);
    routine.amvalidate = Some(amvalidate);
    routine.ambeginscan = Some(scan::ambeginscan);
    routine.amrescan = Some(scan::amrescan);
    routine.amgettuple = Some(scan::amgettuple);
    routine.amendscan = Some(scan::amendscan);

    routine.into_pg_boxed()
}

// `real[]` is the on-ramp for data that already lives in arrays; brindle_vector
// is the compact form. Both index through the same access method. Strategy
// numbers are per metric — see [`opclass`].
extension_sql!(
    r#"
CREATE OPERATOR <-> (
    LEFTARG = real[], RIGHTARG = real[],
    FUNCTION = brindle_l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR CLASS real_array_l2_ops
    DEFAULT FOR TYPE real[] USING brindle AS
    OPERATOR 1 <-> (real[], real[]) FOR ORDER BY float_ops,
    FUNCTION 1 brindle_l2_squared_distance(real[], real[]),
    FUNCTION 2 (real[], real[]) brindle_l2_metric(),
    FUNCTION 3 (real[], real[]) brindle_real_array_kind();
"#,
    name = "brindle_real_array_l2_ops",
    requires = [
        brindle_amhandler,
        brindle_l2_distance,
        brindle_l2_squared_distance,
        brindle_l2_metric,
        brindle_real_array_kind,
    ],
);

/// Accumulates the graph and the node-id → heap-TID table during a build.
/// Node ids are dense insertion order, so `heap_tids[i]` addresses node `i`.
struct BuildState {
    hnsw: Hnsw,
    heap_tids: Vec<TidPair>,
    kind: VectorKind,
}

/// Cost that takes an index path out of contention. Finite on purpose: the
/// planner derives an index path's run cost from `total - startup`, so two
/// infinities would leave it `NaN`, and a `NaN` cost compares equal to
/// everything — the path would then be dropped by luck of path ordering rather
/// than by being expensive. Matches Postgres' own `disable_cost`.
const DISABLED_COST: pg_sys::Cost = 1.0e10;

/// Copy one indexed datum into an owned `Vec<f32>`, whichever type the column
/// holds.
///
/// # Safety
/// `datum` must be a non-null value of `kind`'s type.
unsafe fn f32_vec_from_datum(kind: VectorKind, datum: pg_sys::Datum) -> Vec<f32> {
    match kind {
        VectorKind::RealArray => f32_vec_from_array_datum(datum),
        // Reading the components in place keeps this to the one copy the graph
        // needs, with no intermediate array.
        VectorKind::Vector => pg_vector::components_from_datum(datum),
    }
}

/// Copy one `real[]` datum into an owned `Vec<f32>`.
///
/// # Safety
/// `datum` must be a non-null, valid `real[]` value.
unsafe fn f32_vec_from_array_datum(datum: pg_sys::Datum) -> Vec<f32> {
    let array = Array::<f32>::from_polymorphic_datum(datum, false, pg_sys::FLOAT4ARRAYOID)
        .unwrap_or_else(|| error!("brindle: could not read real[] value"));
    // as_slice fails exactly when the array carries a null bitmap; this is the
    // per-tuple hot path of the build, so take the single-memcpy route and
    // keep the element-wise walk for the error message only.
    match array.as_slice() {
        Ok(values) => values.to_vec(),
        Err(_) => error!("brindle: vector must not contain NULL elements"),
    }
}

/// Per-tuple callback for `table_index_build_scan`.
#[pg_guard]
unsafe extern "C" fn build_callback(
    _index: pg_sys::Relation,
    tid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut c_void,
) {
    // SAFETY: `state` is the BuildState that ambuild passed to
    // table_index_build_scan; Postgres hands it back unchanged, and the scan
    // does not outlive ambuild's stack frame.
    let state = &mut *state.cast::<BuildState>();
    if *isnull {
        return; // NULL vectors are not indexed; a distance scan can't rank them
    }
    // SAFETY: single-column AM (amcanmulticol=false), so values[0]/isnull[0]
    // is the only entry, and we just checked it is not null. `tid` points at
    // the scanned tuple's live ItemPointerData for the duration of this call.
    let vector = f32_vec_from_datum(state.kind, *values);
    match state.hnsw.insert(vector) {
        Ok(_) => state.heap_tids.push(item_pointer_get_both(*tid)),
        Err(e) => error!("brindle: {e}"),
    }
}

#[pg_guard]
unsafe extern "C" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let mut state = BuildState {
        hnsw: Hnsw::new(build_params(index)),
        heap_tids: Vec::new(),
        kind: opclass::index_kind(index),
    };

    // SAFETY: heap/index/index_info are the live, locked relations Postgres
    // handed us; build_callback only touches `state`, which outlives the scan.
    let heap_tuples = pg_sys::table_index_build_scan(
        heap,
        index,
        index_info,
        true,
        true,
        Some(build_callback),
        (&mut state as *mut BuildState).cast(),
        core::ptr::null_mut(),
    );

    let index_tuples = state.heap_tids.len() as f64;
    let blob = storage::encode_index(&state.hnsw, &state.heap_tids);
    // SAFETY: CREATE INDEX/REINDEX hands ambuild a freshly created, exclusively
    // locked relfilenode, so the main fork is empty as write_index_blob requires.
    storage::write_index_blob(index, &blob, pg_sys::ForkNumber::MAIN_FORKNUM);

    // SAFETY: alloc0 pallocs a zeroed IndexBuildResult in the caller's memory
    // context, which owns it from here on.
    let mut result = PgBox::<pg_sys::IndexBuildResult>::alloc0();
    result.heap_tuples = heap_tuples;
    result.index_tuples = index_tuples;
    result.into_pg()
}

#[pg_guard]
unsafe extern "C" fn ambuildempty(index: pg_sys::Relation) {
    // The init fork of an unlogged index: an empty graph, so the index is
    // valid (and empty) after a crash resets the main fork from it.
    let hnsw = Hnsw::new(build_params(index));
    let blob = storage::encode_index(&hnsw, &[]);
    // SAFETY: Postgres calls ambuildempty right after creating the (empty)
    // init fork of the exclusively locked new index, as write_index_blob
    // requires.
    storage::write_index_blob(index, &blob, pg_sys::ForkNumber::INIT_FORKNUM);
}

/// Add one heap tuple to the persisted graph, returning whether it was indexed.
///
/// The strategy is load-modify-store: read the whole index back, insert into
/// the in-memory graph, write the whole image out again. Graph quality is
/// therefore exactly a rebuild's — the point of doing it this way — but the
/// cost is O(index) per row, which suits a trickle of inserts and not a bulk
/// load. Loading a large table is still far faster as `COPY` followed by
/// `CREATE INDEX`. That cost is O(index) *WAL* too — the whole image is logged
/// as full page images per row, so one insert into a 4 MB index writes ~4 MB of
/// WAL, which replicas and archiving pay for as well.
///
/// The whole sequence runs under an exclusive [`storage::IMAGE_LOCK_BLOCK`]
/// lock, so that two inserts cannot each load the image the other is replacing
/// and lose a row, and so that a concurrent scan reads one image rather than
/// halves of two.
///
/// TODO: append the new node and its edges to the stored graph in place rather
/// than rewriting the whole image.
// Signature fixed by the index AM ABI.
#[allow(clippy::too_many_arguments)]
#[pg_guard]
unsafe extern "C" fn aminsert(
    index: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    // SAFETY: single-column AM (amcanmulticol=false), so values[0]/isnull[0] is
    // the only entry Postgres filled in.
    if *isnull {
        return false; // NULL vectors are not indexed; a distance scan can't rank them
    }
    // The column's type decides how the datum is read, exactly as it does for a
    // build — reading a vector as an array would reinterpret its header.
    let vector = f32_vec_from_datum(opclass::index_kind(index), *values);

    // SAFETY: `index` is the open index relation Postgres locked for this
    // insert. An error below unwinds to abort, which releases the page lock
    // along with every other lock the transaction holds.
    pg_sys::LockPage(
        index,
        storage::IMAGE_LOCK_BLOCK,
        pg_sys::ExclusiveLock as i32,
    );

    let (mut hnsw, mut tids) = storage::load_index(index);
    let id = match hnsw.insert(vector) {
        Ok(id) => id,
        Err(e) => error!("brindle: {e}"),
    };
    // Ids are dense insertion order and the load checked that the graph and the
    // table have the same length, so the new id addresses the slot being filled.
    if id != tids.len() {
        error!("brindle: index graph and heap-pointer table are out of step");
    }
    // SAFETY: `heap_tid` points at the ItemPointerData of the tuple being
    // inserted, valid for this call.
    tids.push(item_pointer_get_both(*heap_tid));

    storage::rewrite_index_blob(index, &storage::encode_index(&hnsw, &tids));
    pg_sys::UnlockPage(
        index,
        storage::IMAGE_LOCK_BLOCK,
        pg_sys::ExclusiveLock as i32,
    );
    true
}

/// Tombstone every node whose heap tuple VACUUM is about to remove.
///
/// This is not an optimization. VACUUM recycles a dead tuple's line pointer as
/// soon as every index has confirmed it dropped its references, so a node left
/// pointing at a recycled slot would later resolve to whatever new row landed
/// there — a wrong answer rather than a miss. Tombstoned nodes still route
/// traffic through the graph but are never returned, which is exactly the
/// guarantee needed here.
///
/// Cost is the same load-modify-store as `aminsert`, once per batch of dead
/// tuples rather than once per tuple.
///
/// TODO: reclaim the space too — tombstones keep their slot in the graph and
/// their heap pointer in the table forever, so a table churned enough grows the
/// index without bound. Compaction has to renumber node ids and the pointer
/// table with them.
#[pg_guard]
unsafe extern "C" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let Some(is_dead) = callback else {
        return stats;
    };
    // SAFETY: VACUUM passes a live IndexVacuumInfo naming the open index.
    let index = (*info).index;

    pg_sys::LockPage(
        index,
        storage::IMAGE_LOCK_BLOCK,
        pg_sys::ExclusiveLock as i32,
    );
    let (mut hnsw, tids) = storage::load_index(index);

    let tombstoned_before = hnsw.deleted_count();
    for (id, &(block, offset)) in tids.iter().enumerate() {
        let mut tid = pg_sys::ItemPointerData::default();
        item_pointer_set_all(&mut tid, block, offset);
        // SAFETY: VACUUM's own callback, given its own state and a tuple id
        // that lives for the call.
        if is_dead(&mut tid, callback_state) {
            if let Err(e) = hnsw.delete(id) {
                error!("brindle: {e}");
            }
        }
    }

    let removed = hnsw.deleted_count() - tombstoned_before;
    if removed > 0 {
        storage::rewrite_index_blob(index, &storage::encode_index(&hnsw, &tids));
    }
    pg_sys::UnlockPage(
        index,
        storage::IMAGE_LOCK_BLOCK,
        pg_sys::ExclusiveLock as i32,
    );

    // VACUUM either hands us the result of an earlier pass to accumulate into,
    // or NULL on the first pass, in which case allocating it is the AM's job.
    // SAFETY: a non-NULL `stats` is Postgres' own, palloc'd in the vacuum
    // context that outlives this call.
    let mut result = PgBox::from_pg(stats);
    if result.is_null() {
        result = PgBox::<pg_sys::IndexBulkDeleteResult>::alloc0().into_pg_boxed();
    }
    result.num_pages =
        pg_sys::RelationGetNumberOfBlocksInFork(index, pg_sys::ForkNumber::MAIN_FORKNUM);
    result.num_index_tuples = hnsw.live_len() as f64;
    result.tuples_removed += removed as f64;
    result.into_pg()
}

#[pg_guard]
unsafe extern "C" fn amvacuumcleanup(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    // Postgres refreshes the index's pg_class.relpages/reltuples from whatever
    // this returns, and skips the refresh entirely on NULL. Returning the input
    // unchanged therefore left those stale for the common case: a VACUUM that
    // finds no dead tuples never calls ambulkdelete, so nothing ever reported
    // how far the index had grown since it was built. The planner was costing
    // scans against the build-time size.
    //
    // SAFETY: `info` is Postgres' own vacuum state for this call. Postgres never
    // passes NULL here, and the code below relies on that too.
    if (*info).analyze_only {
        // An analyze-only pass must not touch the index; it is only here to let
        // an AM update its own statistics, and brindle keeps none of its own.
        return stats;
    }

    // SAFETY: a non-NULL `stats` is Postgres' own, palloc'd in the vacuum
    // context that outlives this call; otherwise allocating it is the AM's job.
    let mut result = PgBox::from_pg(stats);
    let fresh = result.is_null();
    if fresh {
        result = PgBox::<pg_sys::IndexBulkDeleteResult>::alloc0().into_pg_boxed();
    }

    // SAFETY: `info.index` is the open index relation Postgres locked for this
    // vacuum.
    let index = (*info).index;
    result.num_pages =
        pg_sys::RelationGetNumberOfBlocksInFork(index, pg_sys::ForkNumber::MAIN_FORKNUM);
    if fresh {
        // Only when ambulkdelete did not already count them. Every live heap
        // tuple has an index entry and tombstones are the rows the heap has
        // dropped, so the count Postgres just took over the heap is the same
        // number the graph would report — and reading the graph to ask it would
        // turn a callback that costs microseconds into one that reads the whole
        // index on every vacuum that finds nothing to do.
        result.num_index_tuples = (*info).num_heap_tuples;
        result.estimated_count = (*info).estimated_count;
    }
    result.into_pg()
}

// Signature fixed by the index AM ABI.
#[allow(clippy::too_many_arguments)]
#[pg_guard]
unsafe extern "C" fn amcostestimate(
    root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    // SAFETY: Postgres always passes valid, writable out-pointers to
    // amcostestimate, and `path` is the candidate IndexPath being costed.

    // The AM has no search operators, so without an ORDER BY distance clause it
    // has nothing to contribute: price it out of the plan. A partial index gets
    // here, because a matching predicate alone is enough for the planner to
    // build a candidate path.
    if (*path).indexorderbys.is_null() {
        *index_startup_cost = DISABLED_COST;
        *index_total_cost = DISABLED_COST;
        *index_selectivity = 0.0;
        *index_correlation = 0.0;
        *index_pages = 0.0;
        return;
    }

    // Cost the graph walk, not a full index read: descent visits about one
    // neighbor list per layer above 0, and the expected top layer for `n` nodes
    // under the level distribution (1/ln m) is log_m(n). Presetting this stops
    // genericcostestimate from assuming every index tuple is visited, which is
    // what lets a `LIMIT k` plan prefer this index over sorting the whole table.
    // TODO: fold in `brindle.ef_search` on both axes. It sizes the scan's work,
    // so a tuned-up session looks cheaper here than it is — and it is now also a
    // ceiling on the *rows* a scan yields, which this estimate does not model at
    // all: selectivity stays 1.0, so a scan returning `ef_search` rows is costed
    // as returning every matching one, and plans above it are sized off that
    // number. The graph load, still rebuilt once per scan, is missing too.
    let m = HnswParams::default().m.max(2) as f64;
    let tuples = (*(*path).indexinfo).tuples.max(1.0);
    let entry_level = (tuples.ln() / m.ln()).floor().max(0.0);

    let mut costs = pg_sys::GenericCosts {
        numIndexTuples: (entry_level + 2.0) * m,
        ..Default::default()
    };
    pg_sys::genericcostestimate(root, path, loop_count, &mut costs);

    // The search runs to completion before the first tuple comes back, so the
    // startup cost is the whole index cost.
    *index_startup_cost = costs.indexTotalCost;
    *index_total_cost = costs.indexTotalCost;
    *index_selectivity = costs.indexSelectivity;
    *index_correlation = costs.indexCorrelation;
    *index_pages = costs.numIndexPages;
}

#[pg_guard]
unsafe extern "C" fn amvalidate(_opclass_oid: pg_sys::Oid) -> bool {
    // TODO: check the support functions an operator class must supply — proc 1's
    // signature, and that procs 2 and 3 return codes this build understands — so
    // a malformed class is rejected here rather than at the first build.
    true
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use core::ffi::c_void;

    use pgrx::itemptr::item_pointer_get_both;
    use pgrx::prelude::*;
    use pgrx::{pg_guard, pg_sys, PgRelation};

    use crate::hnsw::{Hnsw, HnswParams};
    use crate::index_am::storage::{self, TidPair};
    use crate::vector::Metric;

    /// The deterministic 4-dim vector the SQL fixtures build for row `i`.
    fn fixture_vector(i: i64) -> Vec<f32> {
        vec![
            (i % 17) as f32,
            ((i * 7) % 13) as f32,
            ((i * 3) % 11) as f32,
            (i % 5) as f32,
        ]
    }

    /// Insert rows `from..=to` of the fixture, each carrying
    /// [`fixture_vector`] of its own id.
    fn insert_fixture_rows(table: &str, from: i64, to: i64) {
        Spi::run(&format!(
            "INSERT INTO {table}
             SELECT i, ARRAY[(i % 17)::real, ((i * 7) % 13)::real,
                             ((i * 3) % 11)::real, (i % 5)::real]
             FROM generate_series({from}, {to}) i"
        ))
        .expect("insert");
    }

    fn create_fixture(table: &str, rows: i64) {
        Spi::run(&format!("CREATE TABLE {table} (id int, embedding real[])")).expect("create");
        insert_fixture_rows(table, 1, rows);
    }

    /// [`fixture_vector`] as a SQL literal.
    fn fixture_literal(row: i64) -> String {
        let elements: Vec<String> = fixture_vector(row).iter().map(|v| v.to_string()).collect();
        format!("ARRAY[{}]::real[]", elements.join(","))
    }

    fn plan_of(sql: &str) -> String {
        Spi::connect(|client| {
            client
                .select(&format!("EXPLAIN (COSTS OFF) {sql}"), None, None)
                .expect("explain")
                .filter_map(|row| row.get::<String>(1).expect("plan line"))
                .collect::<Vec<String>>()
                .join("\n")
        })
    }

    /// The graph's own nearest neighbor for `query`, with a candidate budget
    /// wide enough that a miss means a broken graph rather than an unlucky walk.
    fn nearest(hnsw: &Hnsw, query: &[f32]) -> (f32, usize) {
        hnsw.search(query, 1, 128)
            .expect("search")
            .first()
            .copied()
            .expect("a non-empty graph returns a neighbor")
    }

    /// The persisted graph and heap-pointer table of an index, read back the
    /// way every future reader will read them.
    fn load_persisted(index: &str) -> (Hnsw, Vec<TidPair>) {
        // SAFETY: the index exists; PgRelation holds AccessShare on it for the
        // duration of the read.
        let relation = unsafe { PgRelation::open_with_name(index) }.expect("open index");
        unsafe { storage::load_index(relation.as_ptr()) }
    }

    /// The `id` of the row a stored heap pointer addresses.
    fn heap_row_at(tid: &TidPair, table: &str) -> i64 {
        let (block, offset) = tid;
        Spi::get_one::<i64>(&format!(
            "SELECT id::bigint FROM {table} WHERE ctid = '({block},{offset})'::tid"
        ))
        .expect("spi")
        .expect("the stored heap pointer addresses a live row")
    }

    fn index_pages(index: &str) -> i64 {
        Spi::get_one::<i64>(&format!(
            "SELECT pg_relation_size('{index}') / current_setting('block_size')::bigint"
        ))
        .expect("spi")
        .expect("non-null")
    }

    #[pg_test]
    fn create_index_builds_and_persists() {
        create_fixture("t_build", 200);
        Spi::run("CREATE INDEX t_build_idx ON t_build USING brindle (embedding)")
            .expect("create index");
        let pages = index_pages("t_build_idx");
        assert!(pages >= 2, "expected metapage + data pages, got {pages}");
    }

    #[pg_test]
    fn reads_a_shrunken_image_without_running_into_the_old_tail() {
        // The writer re-initializes leftover pages instead of truncating, so a
        // rewrite that shrinks the payload leaves the relation the same size
        // with stale pages past the new end. This checks the reader stops where
        // the metapage says the payload does, rather than reading on into them.
        //
        // It does *not* cover the reader's skip-an-empty-page branch: `left`
        // halts the read before the tail, so that branch stays unreachable
        // while the metapage length is trusted. Verified by breaking the branch
        // and watching this test still pass — it is a bound behind a bound, not
        // a tested path.
        //
        // Nothing reaches a shrunken image from SQL yet — tombstones keep their
        // slot, so bulk delete does not shrink it, and reclaiming them is not
        // wired to a vacuum entry point. Hence the direct rewrite.
        create_fixture("t_tail", 400);
        Spi::run("CREATE INDEX t_tail_idx ON t_tail USING brindle (embedding)")
            .expect("create index");
        let before = index_pages("t_tail_idx");
        assert!(before > 3, "fixture must span several pages, got {before}");

        // SAFETY: the index exists and PgRelation keeps it open across both
        // calls; the blob is a well-formed encoding of the graph beside it.
        let (restored, tids) = unsafe {
            let relation = PgRelation::open_with_name("t_tail_idx").expect("open index");
            let mut small = Hnsw::new(HnswParams::default());
            for i in 0..5u32 {
                small
                    .insert(vec![i as f32, (i + 1) as f32])
                    .expect("insert");
            }
            let small_tids: Vec<TidPair> = (0..5).map(|i| (0u32, i as u16 + 1)).collect();
            storage::rewrite_index_blob(
                relation.as_ptr(),
                &storage::encode_index(&small, &small_tids),
            );
            storage::load_index(relation.as_ptr())
        };

        assert_eq!(
            index_pages("t_tail_idx"),
            before,
            "the relation keeps its pages; only the payload shrinks"
        );
        assert_eq!(restored.len(), 5, "read back across the emptied tail");
        assert_eq!(tids.len(), 5);
    }

    #[pg_test]
    fn create_index_on_empty_table() {
        Spi::run("CREATE TABLE t_empty (id int, embedding real[])").expect("create");
        Spi::run("CREATE INDEX t_empty_idx ON t_empty USING brindle (embedding)")
            .expect("create index");
        assert!(index_pages("t_empty_idx") >= 2);
    }

    #[pg_test]
    fn create_index_on_unlogged_table() {
        // Unlogged tables exercise ambuildempty (the init fork) too.
        Spi::run("CREATE UNLOGGED TABLE t_unlogged (id int, embedding real[])").expect("create");
        Spi::run("INSERT INTO t_unlogged SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 50) i")
            .expect("insert");
        Spi::run("CREATE INDEX t_unlogged_idx ON t_unlogged USING brindle (embedding)")
            .expect("create index");
        assert!(index_pages("t_unlogged_idx") >= 2);
    }

    #[pg_test]
    fn built_graph_round_trips_through_pages() {
        create_fixture("t_rt", 100);
        // NULL vectors must be skipped, not break the build.
        Spi::run("INSERT INTO t_rt SELECT i, NULL FROM generate_series(101, 105) i")
            .expect("insert nulls");
        Spi::run("CREATE INDEX t_rt_idx ON t_rt USING brindle (embedding)").expect("create index");

        // SAFETY: freshly created index; AccessShare via PgRelation keeps it open.
        let index = unsafe { PgRelation::open_with_name("t_rt_idx") }.expect("open index");
        let (restored, tids) = unsafe { storage::load_index(index.as_ptr()) };

        assert_eq!(
            restored.len(),
            100,
            "only the 100 non-NULL rows are indexed"
        );
        assert_eq!(tids.len(), 100);
        assert!(tids.iter().all(|&(_, off)| off >= 1));

        // The persisted graph must behave exactly like an in-memory build over
        // the same vectors in heap order.
        let mut local = Hnsw::new(HnswParams::default());
        for i in 1..=100 {
            local.insert(fixture_vector(i)).expect("insert");
        }
        for query in [
            vec![1.0_f32, 2.0, 3.0, 4.0],
            vec![16.0, 0.0, 10.0, 0.5],
            vec![8.0, 6.5, 5.0, 2.0],
        ] {
            assert_eq!(
                restored.search(&query, 10, 64).expect("search"),
                local.search(&query, 10, 64).expect("search"),
                "persisted graph diverged from in-memory build for {query:?}"
            );
        }
    }

    #[pg_test(error = "brindle: vector dimension mismatch: expected 4, got 2")]
    fn create_index_rejects_mixed_dimensions() {
        create_fixture("t_mixed", 10);
        Spi::run("INSERT INTO t_mixed VALUES (11, ARRAY[1,2]::real[])").expect("insert");
        Spi::run("CREATE INDEX ON t_mixed USING brindle (embedding)").expect("create index");
    }

    #[pg_test(error = "brindle: vector must not contain NULL elements")]
    fn create_index_rejects_null_elements() {
        Spi::run("CREATE TABLE t_nullelem (id int, embedding real[])").expect("create");
        Spi::run("INSERT INTO t_nullelem VALUES (1, ARRAY[1, NULL, 3]::real[])").expect("insert");
        Spi::run("CREATE INDEX ON t_nullelem USING brindle (embedding)").expect("create index");
    }

    #[pg_test]
    fn rows_inserted_into_a_vector_column_are_indexed() {
        // The insert path reads the datum the column's operator class describes,
        // so the vector type has to work here as well as `real[]` does.
        Spi::run("CREATE TABLE t_ins_vec (id int, embedding brindle_vector)").expect("create");
        Spi::run(
            "INSERT INTO t_ins_vec
             SELECT i, ARRAY[(i % 17)::real, ((i * 7) % 13)::real,
                             ((i * 3) % 11)::real, (i % 5)::real]::brindle_vector
             FROM generate_series(1, 50) i",
        )
        .expect("insert");
        Spi::run("CREATE INDEX t_ins_vec_idx ON t_ins_vec USING brindle (embedding)")
            .expect("create index");

        Spi::run("INSERT INTO t_ins_vec VALUES (51, '[3,9,6,1]'::brindle_vector)")
            .expect("insert after build");

        let (hnsw, tids) = load_persisted("t_ins_vec_idx");
        assert_eq!(hnsw.len(), 51, "the new row joined the graph");
        let (dist, id) = nearest(&hnsw, &[3.0, 9.0, 6.0, 1.0]);
        assert_eq!(dist, 0.0, "the inserted vector is its own nearest neighbor");
        assert_eq!(heap_row_at(&tids[id], "t_ins_vec"), 51);
    }

    #[pg_test]
    fn rows_inserted_after_build_are_indexed() {
        create_fixture("t_ins", 100);
        Spi::run("CREATE INDEX t_ins_idx ON t_ins USING brindle (embedding)")
            .expect("create index");
        insert_fixture_rows("t_ins", 101, 105);

        let (hnsw, tids) = load_persisted("t_ins_idx");
        assert_eq!(hnsw.len(), 105, "the five new rows joined the graph");
        assert_eq!(tids.len(), 105);

        for row in 101..=105 {
            let (dist, id) = nearest(&hnsw, &fixture_vector(row));
            assert_eq!(dist, 0.0, "row {row}'s own vector is its nearest neighbor");
            assert_eq!(
                heap_row_at(&tids[id], "t_ins"),
                row,
                "node {id} points at the wrong heap tuple"
            );
        }
    }

    #[pg_test]
    fn rows_inserted_after_build_come_back_from_an_index_scan() {
        // The end-to-end version of the check above: not just present in the
        // persisted graph, but actually returned by `ORDER BY <->`.
        create_fixture("t_ins_scan", 100);
        Spi::run("CREATE INDEX t_ins_scan_idx ON t_ins_scan USING brindle (embedding)")
            .expect("create index");
        insert_fixture_rows("t_ins_scan", 101, 105);
        Spi::run("SET LOCAL enable_seqscan = off").expect("set");

        for row in 101..=105 {
            let sql = format!(
                "SELECT id FROM t_ins_scan ORDER BY embedding <-> {} LIMIT 1",
                fixture_literal(row)
            );
            // Without this the assertion below would pass on a sequential scan
            // and prove nothing about the index.
            let plan = plan_of(&sql);
            assert!(
                plan.contains("Index Scan using t_ins_scan_idx"),
                "expected an index scan, got:\n{plan}"
            );
            assert_eq!(
                Spi::get_one::<i32>(&sql).expect("spi").expect("a row"),
                row as i32,
                "row {row} was inserted after the build but the index scan did not return it"
            );
        }
    }

    #[pg_test]
    fn insert_skips_null_vectors() {
        create_fixture("t_ins_null", 20);
        Spi::run("CREATE INDEX t_ins_null_idx ON t_ins_null USING brindle (embedding)")
            .expect("create index");
        Spi::run("INSERT INTO t_ins_null VALUES (21, NULL)").expect("insert");

        let (hnsw, tids) = load_persisted("t_ins_null_idx");
        assert_eq!(hnsw.len(), 20, "a NULL vector must not become a node");
        assert_eq!(tids.len(), 20);
    }

    #[pg_test]
    fn insert_into_index_built_on_empty_table() {
        Spi::run("CREATE TABLE t_ins_empty (id int, embedding real[])").expect("create");
        Spi::run("CREATE INDEX t_ins_empty_idx ON t_ins_empty USING brindle (embedding)")
            .expect("create index");
        Spi::run("INSERT INTO t_ins_empty VALUES (1, ARRAY[1,2,3,4]::real[])").expect("insert");

        // An index built over no rows has no dimensionality yet; the first
        // insert is what fixes it.
        let (hnsw, tids) = load_persisted("t_ins_empty_idx");
        assert_eq!(hnsw.dim(), 4);
        assert_eq!(hnsw.len(), 1);
        assert_eq!(tids.len(), 1);
    }

    #[pg_test]
    fn inserts_extend_the_index_onto_new_pages() {
        // Wide vectors so the appended nodes are certain to outgrow the pages
        // the build allocated, exercising the extend path of the rewrite.
        Spi::run("CREATE TABLE t_ins_grow (id int, embedding real[])").expect("create");
        let load = |from: i64, to: i64| {
            Spi::run(&format!(
                "INSERT INTO t_ins_grow
                 SELECT i, ARRAY(SELECT ((i * d) % 97)::real
                                 FROM generate_series(1, 64) d)
                 FROM generate_series({from}, {to}) i"
            ))
            .expect("insert");
        };
        load(1, 30);
        Spi::run("CREATE INDEX t_ins_grow_idx ON t_ins_grow USING brindle (embedding)")
            .expect("create index");
        let pages_after_build = index_pages("t_ins_grow_idx");

        load(31, 90);
        assert!(
            index_pages("t_ins_grow_idx") > pages_after_build,
            "60 more 64-dim vectors must have needed more pages than {pages_after_build}"
        );

        let (hnsw, tids) = load_persisted("t_ins_grow_idx");
        assert_eq!(hnsw.len(), 90);
        assert_eq!(tids.len(), 90);
        // Every heap pointer still addresses the row whose vector its node holds
        // after the graph has been re-laid-out 60 times.
        for row in 1..=90 {
            let query: Vec<f32> = (1..=64).map(|d| ((row * d) % 97) as f32).collect();
            let (_, id) = nearest(&hnsw, &query);
            assert_eq!(heap_row_at(&tids[id], "t_ins_grow"), row);
        }
    }

    #[pg_test(error = "brindle: vector dimension mismatch: expected 4, got 2")]
    fn insert_rejects_mismatched_dimension() {
        create_fixture("t_ins_dim", 10);
        Spi::run("CREATE INDEX ON t_ins_dim USING brindle (embedding)").expect("create index");
        Spi::run("INSERT INTO t_ins_dim VALUES (11, ARRAY[1,2]::real[])").expect("insert");
    }

    /// Build an index over the shared fixture and read back the parameters the
    /// persisted graph was actually built with.
    fn build_with(table: &str, index: &str, with_clause: &str) -> Hnsw {
        create_fixture(table, 60);
        Spi::run(&format!(
            "CREATE INDEX {index} ON {table} USING brindle (embedding) {with_clause}"
        ))
        .expect("create index");
        // SAFETY: freshly created index; AccessShare via PgRelation keeps it open.
        let rel = unsafe { PgRelation::open_with_name(index) }.expect("open index");
        unsafe { storage::load_index(rel.as_ptr()) }.0
    }

    #[pg_test]
    fn reloptions_are_honored_at_build() {
        let small = build_with("t_opt_m8", "t_opt_m8_idx", "WITH (m = 8)");
        let large = build_with("t_opt_m32", "t_opt_m32_idx", "WITH (m = 32)");

        assert_eq!(small.m(), 8);
        assert_eq!(large.m(), 32);
        // A denser graph is a bigger graph: the degree caps scale with m, so
        // the serialized link table must grow with it.
        assert!(
            large.to_bytes().len() > small.to_bytes().len(),
            "m = 32 should produce more links than m = 8"
        );
    }

    #[pg_test]
    fn reloptions_cover_every_build_parameter() {
        let hnsw = build_with(
            "t_opt_all",
            "t_opt_all_idx",
            "WITH (m = 6, ef_construction = 90, gamma = 2.5)",
        );
        assert_eq!(hnsw.m(), 6);
        assert_eq!(hnsw.ef_construction(), 90);
        assert!((hnsw.gamma() - 2.5).abs() < 1e-6, "got {}", hnsw.gamma());
    }

    #[pg_test]
    fn omitted_reloptions_fall_back_to_defaults() {
        let defaults = HnswParams::default();
        // ef_construction alone; m and gamma must keep their defaults.
        let hnsw = build_with(
            "t_opt_part",
            "t_opt_part_idx",
            "WITH (ef_construction = 128)",
        );
        assert_eq!(hnsw.ef_construction(), 128);
        assert_eq!(hnsw.m(), defaults.m);
        assert_eq!(hnsw.gamma(), defaults.gamma);

        let bare = build_with("t_opt_none", "t_opt_none_idx", "");
        assert_eq!(bare.m(), defaults.m);
        assert_eq!(bare.ef_construction(), defaults.ef_construction);
        assert_eq!(bare.gamma(), defaults.gamma);
    }

    #[pg_test(error = "unrecognized parameter \"nonesuch\"")]
    fn unknown_reloption_is_rejected() {
        create_fixture("t_opt_bad", 5);
        Spi::run("CREATE INDEX ON t_opt_bad USING brindle (embedding) WITH (nonesuch = 1)")
            .expect("create index");
    }

    #[pg_test]
    fn builds_over_out_of_line_vectors() {
        // Wide enough to be stored out of line, which is the path where reading
        // a vector has to detoast a copy and then release it.
        const DIMS: i64 = 2000;
        const ROWS: i64 = 20;
        Spi::run("CREATE TABLE t_toast_build (id int, embedding brindle_vector)").expect("create");
        Spi::run(&format!(
            "INSERT INTO t_toast_build
             SELECT i, ('[' || (SELECT string_agg(((j * i) % 97)::text, ',')
                                FROM generate_series(1, {DIMS}) j) || ']')::brindle_vector
             FROM generate_series(1, {ROWS}) i"
        ))
        .expect("insert");
        Spi::run("CREATE INDEX t_toast_build_idx ON t_toast_build USING brindle (embedding)")
            .expect("create index");

        // SAFETY: freshly created index; AccessShare via PgRelation keeps it open.
        let index = unsafe { PgRelation::open_with_name("t_toast_build_idx") }.expect("open");
        let (graph, tids) = unsafe { storage::load_index(index.as_ptr()) };
        assert_eq!(graph.len() as i64, ROWS);
        assert_eq!(tids.len() as i64, ROWS);

        // Detoasting produced the stored components, not a copy of some other
        // row: searching for row 3's own vector finds row 3's node first.
        let row = 3;
        let query: Vec<f32> = (1..=DIMS).map(|j| ((j * row) % 97) as f32).collect();
        let nearest = graph.search(&query, 1, 64).expect("search");
        assert_eq!(
            nearest.first().map(|&(_, node)| node),
            Some(row as usize - 1)
        );
    }

    #[pg_test(error = "brindle: operator class does not match the type of the indexed column")]
    fn an_operator_class_that_misreports_its_type_is_rejected() {
        create_fixture("t_mismatch", 5);
        Spi::run(
            "CREATE OPERATOR CLASS mismatched_ops FOR TYPE real[] USING brindle AS
                 FUNCTION 1 brindle_l2_squared_distance(real[], real[]),
                 FUNCTION 2 (real[], real[]) brindle_l2_metric(),
                 FUNCTION 3 (real[], real[]) brindle_vector_kind()",
        )
        .expect("create operator class");
        Spi::run("CREATE INDEX ON t_mismatch USING brindle (embedding mismatched_ops)")
            .expect("create index");
    }

    const METRIC_FIXTURE_ROWS: i64 = 200;

    /// A vector whose direction and magnitude vary independently, so ranking by
    /// angle (cosine) and by distance (L2) disagree — which is what makes the
    /// metric an index was built with observable in its results.
    fn metric_fixture_vector(i: i64) -> Vec<f32> {
        let scale = (1 + i % 5) as f32;
        vec![
            (i % 13) as f32 * scale,
            (i % 7) as f32 * scale,
            scale,
            (i % 3) as f32 * scale,
        ]
    }

    /// Build an index over the metric fixture and return the graph Postgres
    /// persisted for it. `opclass` is empty to take the column's default.
    fn build_metric_fixture(table: &str, opclass: &str) -> Hnsw {
        Spi::run(&format!(
            "CREATE TABLE {table} (id int, embedding brindle_vector)"
        ))
        .expect("create");
        Spi::run(&format!(
            "INSERT INTO {table}
             SELECT i, ARRAY[((i % 13) * (1 + i % 5))::real, ((i % 7) * (1 + i % 5))::real,
                             (1 + i % 5)::real, ((i % 3) * (1 + i % 5))::real]::brindle_vector
             FROM generate_series(1, {METRIC_FIXTURE_ROWS}) i"
        ))
        .expect("insert");
        Spi::run(&format!(
            "CREATE INDEX {table}_idx ON {table} USING brindle (embedding {opclass})"
        ))
        .expect("create index");

        // SAFETY: freshly created index; AccessShare via PgRelation keeps it open.
        let index = unsafe { PgRelation::open_with_name(&format!("{table}_idx")) }.expect("open");
        let (graph, _tids) = unsafe { storage::load_index(index.as_ptr()) };
        graph
    }

    /// The same graph built in process, so any difference in results is the
    /// metric and not the build.
    fn in_memory_fixture(metric: Metric) -> Hnsw {
        let mut hnsw = Hnsw::new(HnswParams {
            metric,
            ..HnswParams::default()
        });
        for i in 1..=METRIC_FIXTURE_ROWS {
            hnsw.insert(metric_fixture_vector(i)).expect("insert");
        }
        hnsw
    }

    #[pg_test]
    fn each_opclass_builds_its_own_metric() {
        let query = vec![3.0_f32, 2.0, 1.0, 1.0];
        for (opclass, metric, other) in [
            ("brindle_vector_l2_ops", Metric::L2, Metric::Cosine),
            ("brindle_vector_cosine_ops", Metric::Cosine, Metric::L2),
            ("brindle_vector_ip_ops", Metric::InnerProduct, Metric::L2),
        ] {
            let indexed = build_metric_fixture(&format!("t_{opclass}"), opclass);
            assert_eq!(
                indexed.metric(),
                metric,
                "{opclass} recorded the wrong metric"
            );

            let expected = in_memory_fixture(metric)
                .search(&query, 10, 64)
                .expect("search");
            let under_other = in_memory_fixture(other)
                .search(&query, 10, 64)
                .expect("search");
            assert_ne!(
                expected, under_other,
                "fixture cannot tell {metric:?} from {other:?} apart"
            );
            assert_eq!(
                indexed.search(&query, 10, 64).expect("search"),
                expected,
                "{opclass} ranked by the wrong metric"
            );
        }
    }

    /// Stands in for VACUUM's "has this heap tuple gone away?" callback.
    ///
    /// # Safety
    /// `state` must point at the `Vec<TidPair>` of doomed tuples.
    #[pg_guard]
    unsafe extern "C" fn tid_is_doomed(tid: pg_sys::ItemPointer, state: *mut c_void) -> bool {
        let doomed = &*state.cast::<Vec<TidPair>>();
        doomed.contains(&item_pointer_get_both(*tid))
    }

    /// Run the index's bulk-delete pass over `doomed`, returning the number of
    /// tuples it reported removing. VACUUM itself cannot run inside the
    /// transaction wrapped around a test, so this drives the callback VACUUM
    /// would drive.
    ///
    /// That makes this a unit test of the callback, not of vacuuming: which rows
    /// Postgres decides are dead, whether it calls the AM at all, and what it
    /// does with the result are all outside it. `tests/sql/` covers the path
    /// end to end against a committed database — see `scripts/sql_test.sh`.
    fn bulk_delete(index: &str, mut doomed: Vec<TidPair>) -> f64 {
        // SAFETY: the index exists; PgRelation keeps it open across the call,
        // and `doomed` outlives the callback that reads it.
        unsafe {
            let relation = PgRelation::open_with_name(index).expect("open index");
            let mut info = pg_sys::IndexVacuumInfo {
                index: relation.as_ptr(),
                message_level: pg_sys::DEBUG2 as i32,
                ..Default::default()
            };
            let stats = pg_sys::index_bulk_delete(
                &mut info,
                core::ptr::null_mut(),
                Some(tid_is_doomed),
                (&mut doomed as *mut Vec<TidPair>).cast(),
            );
            (*stats).tuples_removed
        }
    }

    #[pg_test]
    fn the_default_opclass_is_l2() {
        assert_eq!(
            build_metric_fixture("t_default_vector", "").metric(),
            Metric::L2
        );

        create_fixture("t_default_array", 50);
        Spi::run("CREATE INDEX t_default_array_idx ON t_default_array USING brindle (embedding)")
            .expect("create index");
        // SAFETY: freshly created index; AccessShare via PgRelation keeps it open.
        let index = unsafe { PgRelation::open_with_name("t_default_array_idx") }.expect("open");
        let (graph, _tids) = unsafe { storage::load_index(index.as_ptr()) };
        assert_eq!(graph.metric(), Metric::L2);
    }

    #[pg_test]
    fn vacuumed_rows_stop_being_returned() {
        // A node left pointing at a vacuumed tuple is worse than a missing one:
        // VACUUM recycles that line pointer, so the node would later resolve to
        // whichever row lands in the slot.
        create_fixture("t_vac", 60);
        Spi::run("CREATE INDEX t_vac_idx ON t_vac USING brindle (embedding)")
            .expect("create index");

        let (_, tids) = load_persisted("t_vac_idx");
        let doomed: Vec<TidPair> = tids
            .iter()
            .copied()
            .filter(|tid| (1..=3).contains(&heap_row_at(tid, "t_vac")))
            .collect();
        assert_eq!(doomed.len(), 3);

        Spi::run("DELETE FROM t_vac WHERE id <= 3").expect("delete");
        assert_eq!(bulk_delete("t_vac_idx", doomed.clone()), 3.0);

        let (hnsw, tids) = load_persisted("t_vac_idx");
        assert_eq!(hnsw.len(), 60, "tombstones keep their slot in the graph");
        assert_eq!(hnsw.live_len(), 57);
        for row in 1..=3 {
            let hits = hnsw.search(&fixture_vector(row), 5, 128).expect("search");
            assert!(
                hits.iter().all(|&(_, id)| !doomed.contains(&tids[id])),
                "a vacuumed tuple was still reachable through the graph"
            );
        }

        // A row inserted afterwards is indexed as usual, alongside the tombstones.
        Spi::run("INSERT INTO t_vac VALUES (61, ARRAY[2,9,6,1]::real[])").expect("insert");
        let (hnsw, tids) = load_persisted("t_vac_idx");
        assert_eq!(hnsw.len(), 61);
        assert_eq!(hnsw.live_len(), 58);
        let (dist, id) = nearest(&hnsw, &[2.0, 9.0, 6.0, 1.0]);
        assert_eq!(dist, 0.0);
        assert_eq!(heap_row_at(&tids[id], "t_vac"), 61);
    }

    #[pg_test]
    fn bulk_delete_over_a_live_table_removes_nothing() {
        create_fixture("t_vac_live", 20);
        Spi::run("CREATE INDEX t_vac_live_idx ON t_vac_live USING brindle (embedding)")
            .expect("create index");

        assert_eq!(bulk_delete("t_vac_live_idx", Vec::new()), 0.0);
        let (hnsw, _) = load_persisted("t_vac_live_idx");
        assert_eq!(hnsw.live_len(), 20);
        assert_eq!(hnsw.deleted_count(), 0);
    }

    #[pg_test(error = "value 1 out of bounds for option \"m\"")]
    fn out_of_range_reloption_is_rejected() {
        create_fixture("t_opt_range", 5);
        Spi::run("CREATE INDEX ON t_opt_range USING brindle (embedding) WITH (m = 1)")
            .expect("create index");
    }

    #[pg_test(error = "invalid value for integer option \"m\": abc")]
    fn non_numeric_reloption_is_rejected() {
        create_fixture("t_opt_type", 5);
        Spi::run("CREATE INDEX ON t_opt_type USING brindle (embedding) WITH (m = 'abc')")
            .expect("create index");
    }

    /// `m` and `gamma` are each in range, but together they would size every
    /// node's link list — and the build's candidate pool — far past what a
    /// build can hold.
    #[pg_test(
        error = "brindle: m = 128 with gamma = 1024 would give each node up to 262144 neighbors"
    )]
    fn excessive_degree_combination_is_rejected() {
        create_fixture("t_opt_degree", 5);
        Spi::run(
            "CREATE INDEX ON t_opt_degree USING brindle (embedding) WITH (m = 128, gamma = 1024)",
        )
        .expect("create index");
    }

    #[pg_test]
    fn dense_but_bounded_combination_is_accepted() {
        // 2 * 16 * 64 = 2048, exactly the layer-0 degree ceiling.
        let hnsw = build_with("t_opt_edge", "t_opt_edge_idx", "WITH (m = 16, gamma = 64)");
        assert_eq!(hnsw.m(), 16);
        assert!((hnsw.gamma() - 64.0).abs() < 1e-6, "got {}", hnsw.gamma());
    }
}
