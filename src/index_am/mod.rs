//! The brindle Index Access Method: `CREATE INDEX ... USING brindle`.
//!
//! Boundary layer only — it adapts the Postgres AM callback ABI to the pure
//! [`crate::hnsw`] core. `ambuild` scans the heap into an in-memory graph and
//! persists it through [`storage`]; [`scan`] answers `ORDER BY` distance
//! queries from it. Incremental inserts are still stubbed with a clear error.
//! The metric a build ranks by and the type its column holds both come from the
//! index's operator class — see [`opclass`].

pub mod opclass;
pub mod scan;
pub mod storage;

use core::ffi::c_void;

use pgrx::itemptr::item_pointer_get_both;
use pgrx::prelude::*;
use pgrx::{pg_guard, pg_sys, Array, FromDatum, PgBox};

use crate::hnsw::{Hnsw, HnswParams};
use crate::pg_vector;
use opclass::VectorKind;
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
    routine.amoptions = Some(amoptions);
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

/// Build parameters for an index. One definition shared by `ambuild` and
/// `ambuildempty`, so an unlogged index's init fork can never disagree with
/// its main fork.
///
/// # Safety
/// `index` must be an open index relation of the brindle access method.
unsafe fn build_params(index: pg_sys::Relation) -> HnswParams {
    // TODO: take m/ef_construction from reloptions; they are fixed to defaults.
    HnswParams {
        metric: opclass::index_metric(index),
        ..HnswParams::default()
    }
}

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

// Signature fixed by the index AM ABI.
#[allow(clippy::too_many_arguments)]
#[pg_guard]
unsafe extern "C" fn aminsert(
    _index: pg_sys::Relation,
    _values: *mut pg_sys::Datum,
    _isnull: *mut bool,
    _heap_tid: pg_sys::ItemPointer,
    _heap: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    // TODO: incremental insert — load the graph, insert, rewrite the blob (or
    // better, land page-structured storage first).
    error!("brindle: inserting into an existing brindle index is not supported yet; REINDEX after loading data");
}

#[pg_guard]
unsafe extern "C" fn ambulkdelete(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    _callback: pg_sys::IndexBulkDeleteCallback,
    _callback_state: *mut c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    // TODO: propagate dead TIDs into graph tombstones. Until then dead heap
    // tuples linger in the graph, costing space and recall.
    //
    // Skipping the callback is only safe because `aminsert` rejects every
    // INSERT and UPDATE. VACUUM takes our silence to mean the index holds no
    // reference to the dead TIDs and recycles their line pointers, so a table
    // that could still gain tuples would eventually hand one of those slots to
    // an unrelated row — which a scan would then return as if it were the
    // indexed one. Whoever lifts the `aminsert` restriction has to honor this
    // callback in the same change.
    //
    // Passing `stats` through (possibly NULL) tells VACUUM we have nothing to
    // report, which is the truth.
    stats
}

#[pg_guard]
unsafe extern "C" fn amvacuumcleanup(
    _info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    stats
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
    // TODO: fold in ef_search once it is configurable, and the cost of loading
    // the graph while storage still rebuilds it once per scan.
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
unsafe extern "C" fn amoptions(reloptions: pg_sys::Datum, validate: bool) -> *mut pg_sys::bytea {
    // TODO: real reloptions (m, ef_construction, metric).
    if validate && reloptions.value() != 0 {
        error!("brindle: this index type has no options yet");
    }
    core::ptr::null_mut()
}

#[pg_guard]
unsafe extern "C" fn amvalidate(_opclass_oid: pg_sys::Oid) -> bool {
    // TODO: verify the opclass shape (support proc 1 signature) once the
    // vector type and multiple metrics exist.
    true
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;
    use pgrx::PgRelation;

    use crate::hnsw::{Hnsw, HnswParams};
    use crate::index_am::storage;
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

    fn create_fixture(table: &str, rows: i64) {
        Spi::run(&format!("CREATE TABLE {table} (id int, embedding real[])")).expect("create");
        Spi::run(&format!(
            "INSERT INTO {table}
             SELECT i, ARRAY[(i % 17)::real, ((i * 7) % 13)::real,
                             ((i * 3) % 11)::real, (i % 5)::real]
             FROM generate_series(1, {rows}) i"
        ))
        .expect("insert");
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

    #[pg_test(
        error = "brindle: inserting into an existing brindle index is not supported yet; REINDEX after loading data"
    )]
    fn insert_after_build_is_rejected() {
        create_fixture("t_ins", 10);
        Spi::run("CREATE INDEX ON t_ins USING brindle (embedding)").expect("create index");
        Spi::run("INSERT INTO t_ins VALUES (11, ARRAY[1,2,3,4]::real[])").expect("insert");
    }

    #[pg_test(error = "brindle: this index type has no options yet")]
    fn reloptions_are_rejected() {
        create_fixture("t_opts", 5);
        Spi::run("CREATE INDEX ON t_opts USING brindle (embedding) WITH (m = 16)")
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
}
