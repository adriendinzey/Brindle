//! Index scan: answers `ORDER BY <indexed column> <-> <query vector>` from the
//! persisted graph.
//!
//! This is an *ordering* scan (`amcanorderbyop`), not a matching one: there is
//! no `WHERE` clause for the AM to satisfy, only an order-by operator whose
//! right-hand argument is the query vector. Results stream nearest-first and the
//! executor stops whenever it has seen enough.
//!
//! # How many rows a scan can produce
//!
//! One search, at a candidate budget of `brindle.ef_search`, read at the start of
//! each scan so a session can retune accuracy against latency without rebuilding
//! anything. Its results are returned in distance order and then the scan ends —
//! so `ORDER BY ... LIMIT n` yields **at most `ef_search` rows** — fewer when
//! the walk converges early or routes through tombstoned nodes — and a caller
//! who wants more raises the budget.
//!
//! That ceiling is deliberate, and it is the whole reason this module is shaped
//! this way. `amcanorderbyop` promises the planner that rows arrive in operator
//! order, and on that promise the planner deletes its own Sort node — so nothing
//! downstream can repair an order the access method got wrong. Producing more
//! rows than one search holds means widening the search, and a wider HNSW search
//! is not a superset of a narrower one: it can turn up a row nearer than one
//! already handed over, which is an inversion no `LIMIT` will ever notice and no
//! user can tune away. Postgres' reorder queue (`xs_recheckorderby`) cannot
//! rescue it either — it needs a non-decreasing *lower bound* per row, and a
//! graph walk cannot bound what it has not visited.
//!
//! Between an approximate result set and an approximate ordering, this scan takes
//! the first: recall is the trade the caller chose by reaching for an ANN index,
//! ordering is a contract they cannot see break.
//!
//! TODO: an opt-in mode that keeps searching past the budget for callers who need
//! completeness more than ordering, the way pgvector's iterative scans do.

use core::ffi::{c_int, c_void};

use pgrx::itemptr::item_pointer_set_all;
use pgrx::prelude::*;
use pgrx::{pg_guard, pg_sys};

use super::opclass;
use super::storage::{self, TidPair};
use crate::guc;
use crate::hnsw::{Hnsw, HnswError};

/// Errors raised while producing scan results.
#[derive(Debug)]
enum ScanError {
    Search(HnswError),
    /// The graph returned a node the node-id → TID table doesn't cover, which
    /// only a corrupted index can produce.
    UnmappedNode(usize),
}

impl std::fmt::Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScanError::Search(e) => write!(f, "{e}"),
            ScanError::UnmappedNode(id) => {
                write!(f, "corrupted index: node {id} has no heap pointer")
            }
        }
    }
}

impl std::error::Error for ScanError {}

impl From<HnswError> for ScanError {
    fn from(e: HnswError) -> Self {
        ScanError::Search(e)
    }
}

/// The search behind one scan: the graph it loaded, the node-id → TID table, and
/// the result of the one search this scan runs, with a cursor into it.
struct ScanSearch {
    hnsw: Hnsw,
    /// `tids[i]` is the heap address of graph node `i`.
    tids: Vec<TidPair>,
    query: Vec<f32>,
    /// The search's results as heap addresses, nearest first.
    results: Vec<TidPair>,
    cursor: usize,
}

impl ScanSearch {
    fn new(hnsw: Hnsw, tids: Vec<TidPair>) -> Self {
        Self {
            hnsw,
            tids,
            query: Vec::new(),
            results: Vec::new(),
            cursor: 0,
        }
    }

    /// Begin (or restart) the scan for `query`, running its one search.
    fn start(&mut self, query: Vec<f32>) -> Result<(), ScanError> {
        self.query = query;
        self.cursor = 0;
        // Cleared before the search rather than after: a search that fails must
        // not leave the previous scan's results sitting under the new query.
        self.results.clear();
        // Read per scan: a session that raised ef_search expects the next query
        // to use it, not the value in force when the scan was opened.
        let budget = guc::ef_search().max(1);

        let found = self.hnsw.search(&self.query, budget, budget)?;
        self.results.reserve(found.len());
        for (_, id) in found {
            match self.tids.get(id) {
                Some(&tid) => self.results.push(tid),
                None => return Err(ScanError::UnmappedNode(id)),
            }
        }
        Ok(())
    }

    /// End the scan without returning anything.
    fn stop(&mut self) {
        self.query = Vec::new();
        self.results = Vec::new();
        self.cursor = 0;
    }

    /// The next heap TID, nearest first, or `None` once the search's results are
    /// spent — which is also where the scan ends, even if the caller wanted more.
    ///
    /// Infallible today, since handing out an already-resolved TID cannot fail;
    /// the `Result` is kept because a scan that resolves TIDs lazily, or resumes
    /// a search, would need it back.
    fn next(&mut self) -> Result<Option<TidPair>, ScanError> {
        let tid = self.results.get(self.cursor).copied();
        if tid.is_some() {
            self.cursor += 1;
        }
        Ok(tid)
    }
}

/// Scan state as Postgres sees it, reachable through `IndexScanDesc::opaque`.
///
/// [`ScanSearch`] lives on the Rust heap, which no memory context can reclaim,
/// and `amendscan` is not guaranteed to run — an error unwinds straight past it.
/// So the allocation is also tied to the memory context that owns the scan
/// descriptor: whichever path runs first frees it, and the other sees the null.
#[repr(C)]
struct ScanState {
    /// Must stay first: the registered callback is this field's address.
    cleanup: pg_sys::MemoryContextCallback,
    search: *mut ScanSearch,
}

/// Drop the Rust-heap search state, at most once.
///
/// Runs as a memory-context reset callback, i.e. possibly while an error is
/// already unwinding, where raising another error would be fatal. Dropping the
/// graph's `Vec`s cannot fail, so there is nothing to guard against.
///
/// # Safety
/// `arg` must point at a live [`ScanState`].
unsafe extern "C" fn free_scan_state(arg: *mut c_void) {
    let state = arg.cast::<ScanState>();
    if !(*state).search.is_null() {
        drop(Box::from_raw((*state).search));
        (*state).search = core::ptr::null_mut();
    }
}

/// Park `search` in a fresh [`ScanState`] owned by the current memory context —
/// the same context `RelationGetIndexScan` allocates the scan descriptor in, so
/// the state lives exactly as long as the scan it belongs to.
///
/// # Safety
/// Must be called with the memory context that owns the scan descriptor current.
unsafe fn new_scan_state(search: ScanSearch) -> *mut ScanState {
    let state = pg_sys::palloc0(core::mem::size_of::<ScanState>()).cast::<ScanState>();
    (*state).search = Box::into_raw(Box::new(search));
    (*state).cleanup.func = Some(free_scan_state);
    (*state).cleanup.arg = state.cast();
    pg_sys::MemoryContextRegisterResetCallback(
        pg_sys::CurrentMemoryContext,
        core::ptr::addr_of_mut!((*state).cleanup),
    );
    state
}

/// The search state behind an active scan.
///
/// # Safety
/// `scan` must be a live brindle scan descriptor.
unsafe fn scan_search<'a>(scan: pg_sys::IndexScanDesc) -> &'a mut ScanSearch {
    let state = (*scan).opaque.cast::<ScanState>();
    if state.is_null() || (*state).search.is_null() {
        error!("brindle: index scan has no state");
    }
    &mut *(*state).search
}

#[pg_guard]
pub(super) unsafe extern "C" fn ambeginscan(
    index: pg_sys::Relation,
    nkeys: c_int,
    norderbys: c_int,
) -> pg_sys::IndexScanDesc {
    let scan = pg_sys::RelationGetIndexScan(index, nkeys, norderbys);
    // TODO: read the graph through the buffer manager instead of deserializing
    // the whole index once per scan.
    // SAFETY: Postgres opened and locked `index` for this scan.
    let (hnsw, tids) = storage::load_index(index);
    (*scan).opaque = new_scan_state(ScanSearch::new(hnsw, tids)).cast();
    scan
}

#[pg_guard]
pub(super) unsafe extern "C" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    keys: pg_sys::ScanKey,
    _nkeys: c_int,
    orderbys: pg_sys::ScanKey,
    _norderbys: c_int,
) {
    // Postgres re-evaluates the keys between scans and expects the AM to keep
    // its own copies in the scan descriptor.
    // SAFETY: both arrays were sized for numberOfKeys/numberOfOrderBys entries
    // by RelationGetIndexScan, and the caller's arrays are at least as long.
    if !keys.is_null() && (*scan).numberOfKeys > 0 {
        core::ptr::copy(keys, (*scan).keyData, (*scan).numberOfKeys as usize);
    }
    if !orderbys.is_null() && (*scan).numberOfOrderBys > 0 {
        core::ptr::copy(
            orderbys,
            (*scan).orderByData,
            (*scan).numberOfOrderBys as usize,
        );
    }

    if (*scan).numberOfOrderBys < 1 || (*scan).orderByData.is_null() {
        error!("brindle: an index scan needs an ORDER BY <distance operator> clause");
    }

    let search = scan_search(scan);
    // SAFETY: orderByData holds numberOfOrderBys initialized keys, checked above.
    let key = &*(*scan).orderByData;
    if key.sk_flags & pg_sys::SK_ISNULL as i32 != 0 {
        search.stop(); // nothing ranks against a NULL query vector
        return;
    }
    // SAFETY: an ordering operator's arguments have the type its operator class
    // indexes, so the key argument is a non-null value of that type — which is
    // what the class reports here. Resolving it per rescan costs one support
    // call against a scan that re-searches the graph anyway.
    let query =
        super::f32_vec_from_datum(opclass::index_kind((*scan).indexRelation), key.sk_argument);
    if let Err(e) = search.start(query) {
        error!("brindle: {e}");
    }
}

#[pg_guard]
pub(super) unsafe extern "C" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    direction: pg_sys::ScanDirection::Type,
) -> bool {
    // amcanbackward is false, so Postgres only ever drives this forward.
    if direction != pg_sys::ScanDirection::ForwardScanDirection {
        error!("brindle: index scans only run forward");
    }

    // `scan.kill_prior_tuple` asks the AM to mark the previously returned
    // entry dead. Recording that needs the same machinery as vacuum
    // integration; ignoring the hint is always legal, just less efficient.
    match scan_search(scan).next() {
        Ok(Some((block, offset))) => {
            item_pointer_set_all(&mut (*scan).xs_heaptid, block, offset);
            // The graph ranks whole rows against the query, so neither the row
            // nor its position needs re-checking above the AM.
            (*scan).xs_recheck = false;
            (*scan).xs_recheckorderby = false;
            true
        }
        Ok(None) => false,
        Err(e) => error!("brindle: {e}"),
    }
}

#[pg_guard]
pub(super) unsafe extern "C" fn amendscan(scan: pg_sys::IndexScanDesc) {
    let state = (*scan).opaque.cast::<ScanState>();
    if !state.is_null() {
        // The state itself stays allocated: its reset callback is still
        // registered against the memory context and must find valid memory.
        free_scan_state(state.cast());
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use std::collections::HashSet;

    use pgrx::prelude::*;
    use pgrx::PgRelation;

    use super::{storage, ScanSearch};

    const DIM: usize = 8;
    /// Big enough that a graph search (rather than the exact tail) answers a
    /// `LIMIT k` query, and that the planner prefers the index to a sort.
    const ROWS: i64 = 1000;
    const K: usize = 10;
    /// An arbitrary point in the fixture's `[0,1]^DIM` cube.
    const QUERY: [f32; DIM] = [0.13, 0.77, 0.41, 0.92, 0.05, 0.63, 0.28, 0.55];

    /// The module's own [`QUERY`] as a `real[]` literal.
    fn query_literal() -> String {
        array_literal(&QUERY)
    }

    fn array_literal(query: &[f32]) -> String {
        let elements: Vec<String> = query.iter().map(|v| v.to_string()).collect();
        format!("ARRAY[{}]::real[]", elements.join(","))
    }

    fn vector_literal(query: &[f32]) -> String {
        let elements: Vec<String> = query.iter().map(|v| v.to_string()).collect();
        format!("'[{}]'::brindle_vector", elements.join(","))
    }

    /// A table of `rows` random `DIM`-dimensional vectors with a brindle index.
    /// `setseed` keeps a failing run reproducible.
    fn create_indexed_fixture(table: &str, rows: i64) {
        Spi::run(&format!("CREATE TABLE {table} (id int, embedding real[])")).expect("create");
        Spi::run("SELECT setseed(0.42)").expect("seed");
        Spi::run(&format!(
            "INSERT INTO {table}
             SELECT i, array_agg(random()::real)
             FROM generate_series(1, {rows}) i, generate_series(1, {DIM}) d
             GROUP BY i"
        ))
        .expect("insert");
        Spi::run(&format!(
            "CREATE INDEX {table}_idx ON {table} USING brindle (embedding)"
        ))
        .expect("create index");
    }

    fn plan_of(sql: &str) -> String {
        let explain = format!("EXPLAIN (COSTS OFF) {sql}");
        Spi::connect(|client| {
            client
                .select(explain.as_str(), None, None)
                .expect("explain")
                .filter_map(|row| row.get::<String>(1).expect("plan line"))
                .collect::<Vec<String>>()
                .join("\n")
        })
    }

    fn assert_uses_index(sql: &str, index: &str) {
        let plan = plan_of(sql);
        assert!(
            plan.contains(&format!("Index Scan using {index}")),
            "expected an index scan on {index}, got:\n{plan}"
        );
    }

    /// Ids from `sql`, in the order the query returned them.
    fn ordered_ids(sql: &str) -> Vec<i32> {
        Spi::get_one::<Vec<i32>>(&format!(
            "SELECT array_agg(id ORDER BY ord)
             FROM (SELECT id, row_number() OVER () AS ord FROM ({sql}) s) w"
        ))
        .expect("spi")
        .expect("non-null")
    }

    fn approximate(table: &str) -> String {
        format!(
            "SELECT id FROM {table} ORDER BY embedding <-> {} LIMIT {K}",
            query_literal()
        )
    }

    /// The same top-`K`, but through a plain function call the index can't answer.
    fn exact(table: &str) -> String {
        format!(
            "SELECT id FROM {table} ORDER BY brindle_l2_distance(embedding, {}) LIMIT {K}",
            query_literal()
        )
    }

    /// The same fixture in the `brindle_vector` type, indexed with the cosine
    /// operator class.
    fn create_cosine_fixture(table: &str, rows: i64) {
        Spi::run(&format!(
            "CREATE TABLE {table} (id int, embedding brindle_vector)"
        ))
        .expect("create");
        Spi::run("SELECT setseed(0.42)").expect("seed");
        Spi::run(&format!(
            "INSERT INTO {table}
             SELECT i, array_agg(random()::real)::brindle_vector
             FROM generate_series(1, {rows}) i, generate_series(1, {DIM}) d
             GROUP BY i"
        ))
        .expect("insert");
        Spi::run(&format!(
            "CREATE INDEX {table}_idx ON {table}
             USING brindle (embedding brindle_vector_cosine_ops)"
        ))
        .expect("create index");
    }

    /// The module's own [`QUERY`] as a `brindle_vector` literal.
    fn vector_query_literal() -> String {
        vector_literal(&QUERY)
    }

    /// The operator class an index is created with, not a constant, decides how
    /// the graph ranks — visible in the rows a query gets back.
    #[pg_test]
    fn a_cosine_operator_class_ranks_by_cosine() {
        create_cosine_fixture("t_cosine", ROWS);
        let query = vector_query_literal();

        let sql = format!("SELECT id FROM t_cosine ORDER BY embedding <=> {query} LIMIT {K}");
        assert_uses_index(&sql, "t_cosine_idx");

        let by_cosine = ordered_ids(&format!(
            "SELECT id FROM t_cosine
             ORDER BY brindle_vector_cosine_distance(embedding, {query}) LIMIT {K}"
        ));
        let by_l2 = ordered_ids(&format!(
            "SELECT id FROM t_cosine
             ORDER BY brindle_vector_l2_distance(embedding, {query}) LIMIT {K}"
        ));
        assert_ne!(
            by_cosine, by_l2,
            "fixture does not distinguish cosine from L2"
        );

        let approx = ordered_ids(&sql);
        let hits = approx.iter().filter(|id| by_cosine.contains(id)).count();
        assert!(
            hits * 10 >= K * 9,
            "recall {hits}/{K} against the exact cosine ordering\n  index: {approx:?}\n  cosine: {by_cosine:?}\n  l2: {by_l2:?}"
        );
    }

    #[pg_test]
    fn order_by_distance_plans_an_index_scan() {
        create_indexed_fixture("t_plan", ROWS);
        assert_uses_index(&approximate("t_plan"), "t_plan_idx");
    }

    #[pg_test]
    fn index_scan_matches_exact_ordering() {
        create_indexed_fixture("t_recall", ROWS);
        let sql = approximate("t_recall");
        assert_uses_index(&sql, "t_recall_idx");

        let approx = ordered_ids(&sql);
        let exact = ordered_ids(&exact("t_recall"));
        assert_eq!(approx.len(), K);
        assert_eq!(exact.len(), K);

        // A flat 0.9 on purpose, unlike the calibrated sweep below: this fixture
        // is 8-dimensional, where the graph finds the exact answer whatever
        // state it is in, so the number here cannot discriminate quality and is
        // only asserting that the narrow `real[]` path returns sane rows at all.
        // Calibrating it would be calibrating noise.
        let hits = approx.iter().filter(|id| exact.contains(id)).count();
        assert!(
            hits * 10 >= K * 9,
            "recall {hits}/{K} below 0.9\n  index: {approx:?}\n  exact: {exact:?}"
        );
    }

    // --- recall sweep ----------------------------------------------------
    //
    // The gate for "this index returns the right rows": for each metric, the
    // ids an index scan returns against the ids an exact ordering returns, at
    // several `k`, averaged over a set of queries. Recall is id-set overlap@k,
    // the ann-benchmarks measure.

    /// Queries per metric per `k`. Enough that one unlucky query cannot pass or
    /// fail the gate on its own, while keeping the sweep to a few seconds.
    const RECALL_QUERIES: usize = 15;

    /// The sweep's own fixture is wider and deeper than the rest of this
    /// module's: at 8 dimensions a graph search finds the exact answer whatever
    /// state the graph is in, so a gate built there would pass a degraded index
    /// as readily as a good one. 32 dimensions is where neighbor quality starts
    /// to matter.
    const RECALL_DIM: usize = 32;
    const RECALL_ROWS: i64 = 2000;

    /// `k` values swept. 1 checks the nearest neighbor itself, 10 is the
    /// headline number, and 25 and 50 are deep pages where a graph walk has had
    /// room to go wrong. All of them sit under [`SWEEP_EF_SEARCH`], because a
    /// scan stops at its budget: a deeper `k` would measure that ceiling rather
    /// than the graph's quality.
    const RECALL_K: [usize; 4] = [1, 10, 25, 50];

    /// The candidate budget the sweep measures at, pinned rather than inherited:
    /// [`RECALL_FLOORS`] is calibrated against this value, and a session or
    /// cluster setting left elsewhere would quietly turn the gate into a
    /// measurement of something else — a high budget hiding a real regression, a
    /// low one failing for an unrelated reason.
    ///
    /// It must be at least the deepest `k`: a scan returns what one search found
    /// and then stops, so measuring recall@100 on a 64-candidate budget would
    /// measure that ceiling rather than the graph's quality.
    const SWEEP_EF_SEARCH: usize = 64;

    /// Mean overlap@k the index must reach against the exact ordering, one floor
    /// per swept depth, at [`SWEEP_EF_SEARCH`] over [`RECALL_ROWS`] rows of
    /// [`RECALL_DIM`] dimensions.
    ///
    /// A single flat bar cannot do this job. Recall falls with depth — the walk
    /// has more chances to miss — so one number is either too loose at `k = 50`
    /// or impossible at `k = 1`. These floors are calibrated per depth against
    /// what the graph actually achieves, and against what a *degraded* graph
    /// achieves, by rebuilding the fixture with a smaller `m` (`WITH (m = N)`,
    /// i.e. fewer neighbors kept per node) and re-running:
    ///
    /// | build          | L2 @1/@10/@25/@50             | cosine                        |
    /// |----------------|-------------------------------|-------------------------------|
    /// | `m = 16` ships | 1.000 / 1.000 / 0.995 / 0.993 | 1.000 / 1.000 / 1.000 / 0.999 |
    /// | `m = 12`       | 1.000 / 1.000 / 0.981 / 0.975 | 1.000 / 0.993 / 0.997 / 0.987 |
    /// | `m = 8`        | 1.000 / 0.967 / 0.952 / 0.944 | 1.000 / 0.987 / 0.976 / 0.952 |
    /// | `m = 4`        | 0.867 / 0.860 / 0.805 / 0.745 | 0.933 / 0.893 / 0.816 / 0.760 |
    ///
    /// Inner product is held to the same floors: 1.000 / 0.993 / 0.997 / 0.996
    /// shipped, 1.000 / 0.987 / 0.987 / 0.963 halved. Note how narrowly it
    /// catches that halving: at `k = 50` it lands 0.007 *below* the floor,
    /// against 0.026 for L2 and 0.018 for cosine. It detects the same regression
    /// on about a third of the margin — the weakest of the three signals, not a
    /// redundant one.
    ///
    /// Every figure here is identical on PostgreSQL 16 and 17, the two majors
    /// CI builds; the fixture comes from `setseed`, whose generator did not
    /// change between them. It did change in 16, so these floors are calibrated
    /// for 16 and 17 and a run on an older supported major would be measuring a
    /// different fixture against them.
    ///
    /// Two things fall out of that table. Depth is where the signal is: at
    /// `k = 1` even a quarter-connectivity graph scores 0.933 under cosine, so
    /// no floor there can discriminate — 0.90 stays the conventional bar for a
    /// usable index rather than a regression detector. And the useful floor sits
    /// between what a healthy graph scores and what a halved one does — at
    /// `k = 50`, between 0.993 and 0.944.
    ///
    /// The deep floors sit at 0.97, roughly the midpoint of that gap, and that
    /// one number does the work at every depth below the top: a halved graph is
    /// caught on all three metrics — L2 at `k = 10`, `25` and `50`, cosine and
    /// inner product at `k = 50`. Raising `k = 10` to 0.98 was tried and
    /// reverted: it caught nothing extra, because halved cosine and halved inner
    /// product both score 0.987 there and clear either bar, while it cut the
    /// shipped inner product's headroom to 0.013 — two missed rows out of 150.
    /// Detection here comes from depth, not from a tighter bar at a shallow `k`.
    ///
    /// The shipped index clears these by 2 to 3 points. A tighter bar would also
    /// catch `m = 12`, at the price of a gate that trips on ordinary variation;
    /// that trade is the reason these are floors and not targets — they answer
    /// "is the graph still sound", not "is it exactly as good as it was".
    const RECALL_FLOORS: [(usize, f64); RECALL_K.len()] =
        [(1, 0.90), (10, 0.97), (25, 0.97), (50, 0.97)];

    /// The floor for `k` — looked up in [`RECALL_FLOORS`] rather than defaulted,
    /// so adding a depth to the sweep without calibrating one fails loudly here
    /// instead of quietly inheriting a neighbor's number.
    fn mean_floor(k: usize) -> f64 {
        RECALL_FLOORS
            .iter()
            .find(|(depth, _)| *depth == k)
            .map(|(_, floor)| *floor)
            .unwrap_or_else(|| panic!("recall@{k} has no calibrated floor"))
    }

    /// No single query may fall below this, whatever the mean says.
    ///
    /// Be precise about what that buys, because the obvious justification is
    /// wrong: a *fully* collapsed query does not need this floor. One query at
    /// zero drags the mean to 0.933 at `k = 10`, 0.931 at `k = 25`, 0.929 at
    /// `k = 50` — under every floor above, so the mean catches it alone. What
    /// this adds is the band the mean tolerates: a single query between roughly
    /// 0.62 and 0.80 at `k = 50`, or 0.585 and 0.80 at `k = 25`, passes on the
    /// mean and fails here. That is "one query lost a fifth to two-fifths of its
    /// neighbours while the other fourteen stayed healthy" — a lopsided graph
    /// rather than a uniformly worse one, which is a shape the mean is built to
    /// hide.
    ///
    /// The shipped index's worst single query is 0.900 — inner product at
    /// `k = 10`. That is one missed row of headroom, not ten comfortable points:
    /// a single query's recall is quantised to `1/k`, so at `k = 10` the next
    /// step down is exactly this floor.
    ///
    /// Not applied at `k = 1`, where the quantum is the entire measurement:
    /// `overlap/k` there is 0.0 or 1.0, so any floor in between demands a
    /// perfect nearest neighbour from every query — stricter than the 0.90 mean
    /// this depth is deliberately held to, and a red build for the single miss
    /// that mean exists to tolerate.
    const MIN_QUERY_RECALL: f64 = 0.80;

    /// Deterministic query points, from a small xorshift rather than Postgres'
    /// PRNG: the fixture's own values come from `setseed`, and a query set that
    /// depends on nothing but this test keeps a failure reproducible.
    fn recall_queries() -> Vec<Vec<f32>> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut unit = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Top 24 bits, so the value is exact in an f32.
            (state >> 40) as f32 / 16_777_216.0
        };
        (0..RECALL_QUERIES)
            .map(|_| (0..RECALL_DIM).map(|_| unit() - 0.5).collect())
            .collect()
    }

    /// A table of centered random vectors with a brindle index over `opclass`.
    ///
    /// Centered on the origin so directions spread over the whole sphere:
    /// cosine ranks by angle alone, and data confined to one orthant would make
    /// every pair look alike, turning a cosine recall number into noise.
    fn create_recall_fixture(table: &str, column_type: &str, cast: &str, opclass: &str) {
        Spi::run(&format!(
            "CREATE TABLE {table} (id int, embedding {column_type})"
        ))
        .expect("create");
        Spi::run("SELECT setseed(0.17)").expect("seed");
        Spi::run(&format!(
            "INSERT INTO {table}
             SELECT i, array_agg((random() - 0.5)::real){cast}
             FROM generate_series(1, {RECALL_ROWS}) i, generate_series(1, {RECALL_DIM}) d
             GROUP BY i"
        ))
        .expect("insert");
        Spi::run(&format!(
            "CREATE INDEX {table}_idx ON {table} USING brindle (embedding {opclass})"
        ))
        .expect("create index");
    }

    /// Sweep one metric: assert mean recall@k clears its floor in
    /// [`RECALL_FLOORS`] for every `k`, that no single query falls below
    /// [`MIN_QUERY_RECALL`], and that the two sides of the comparison are what
    /// they claim to be — the approximate query answered by the index, the exact
    /// one not.
    fn assert_recall_sweep(
        table: &str,
        operator: &str,
        distance_fn: &str,
        literal: fn(&[f32]) -> String,
    ) {
        let index = format!("{table}_idx");
        let queries = recall_queries();
        let approximate = |query: &str, k: usize| {
            format!("SELECT id FROM {table} ORDER BY embedding {operator} {query} LIMIT {k}")
        };
        // Ordering by a plain function call leaves the planner nothing the index
        // can answer, so the baseline is computed independently of what it is
        // measuring. Asserted below rather than assumed.
        let exact = |query: &str, k: usize| {
            format!("SELECT id FROM {table} ORDER BY {distance_fn}(embedding, {query}) LIMIT {k}")
        };

        // Scoped to the test's transaction, so the sweep neither inherits a
        // stray setting nor leaves one behind.
        Spi::run(&format!("SET LOCAL brindle.ef_search = {SWEEP_EF_SEARCH}"))
            .expect("pin ef_search");

        let widest = RECALL_K
            .iter()
            .copied()
            .max()
            .expect("RECALL_K is not empty");
        let sample = literal(&queries[0]);
        assert_uses_index(&approximate(&sample, widest), &index);
        let baseline_plan = plan_of(&exact(&sample, widest));
        assert!(
            !baseline_plan.contains(&index),
            "the exact baseline was answered by the index it is meant to check:\n{baseline_plan}"
        );

        // One scan per query point answers every `k`. `LIMIT` is invisible to an
        // access method: *within a scan* the order rows come out in depends only
        // on the query, `ef_search`, and the graph, and a smaller `LIMIT` merely
        // stops that scan sooner — so a shorter list is the longer one's prefix
        // at any `k` the budget can serve.
        //
        // What `LIMIT` does reach is the planner: a different `k` could be
        // costed onto a different path, and two lists from two different plans
        // need not relate at all. That is the assumption the assertion below
        // guards, and the reason this shortcut is checked rather than trusted.
        let widest_ids = ordered_ids(&approximate(&sample, widest));
        for k in RECALL_K {
            // Check the premise, not just its consequence: at a `k` where every
            // id is correct anyway, a flip to a sort would return the same list
            // and slip past the comparison below.
            assert_uses_index(&approximate(&sample, k), &index);
            assert_eq!(
                ordered_ids(&approximate(&sample, k)),
                widest_ids[..k],
                "LIMIT {k} is not the prefix of LIMIT {widest}, so one scan \
                 cannot stand in for the others"
            );
        }

        let mut hits = [0.0; RECALL_K.len()];
        let mut worst = [1.0_f64; RECALL_K.len()];
        for query in &queries {
            let query = literal(query);
            let approx = ordered_ids(&approximate(&query, widest));
            let exact = ordered_ids(&exact(&query, widest));
            assert_eq!(approx.len(), widest, "index returned {} rows", approx.len());
            assert_eq!(
                exact.len(),
                widest,
                "baseline returned {} rows",
                exact.len()
            );

            for (slot, k) in RECALL_K.iter().enumerate() {
                // Overlap of *distinct* ids. Counting matches instead would let
                // a row returned twice count twice, inflating recall in the one
                // test whose job is to notice the scan misbehaving.
                let truth: HashSet<i32> = exact[..*k].iter().copied().collect();
                let found: HashSet<i32> = approx[..*k].iter().copied().collect();
                let overlap = found.intersection(&truth).count() as f64;
                hits[slot] += overlap;
                // Tracked alongside the mean because a mean over 15 queries can
                // absorb one collapsed query, which is what a region of the
                // graph falling out of reach looks like from here.
                worst[slot] = worst[slot].min(overlap / *k as f64);
            }
        }

        let recalls: Vec<(usize, f64, f64)> = RECALL_K
            .iter()
            .enumerate()
            .map(|(slot, k)| (*k, hits[slot] / (k * queries.len()) as f64, worst[slot]))
            .collect();
        // Report the whole sweep whichever `k` fails: one weak `k` next to the
        // others is the difference between "the graph is off" and "this depth
        // is where it thins out".
        let summary: Vec<String> = recalls
            .iter()
            .map(|(k, mean, worst)| format!("recall@{k} {mean:.3} (worst query {worst:.3})"))
            .collect();

        for (k, recall, worst) in &recalls {
            let floor = mean_floor(*k);
            assert!(
                *recall >= floor,
                "{table}: mean recall@{k} below {floor} over {} queries [{}]",
                queries.len(),
                summary.join(", ")
            );
            // Skipped at the shallowest depth, where this floor would assert a
            // perfect nearest neighbour rather than the absence of a collapse.
            if *k > 1 {
                assert!(
                    *worst >= MIN_QUERY_RECALL,
                    "{table}: one query's recall@{k} fell to {worst:.3}, below \
                     {MIN_QUERY_RECALL} — the mean can hide a single collapsed query [{}]",
                    summary.join(", ")
                );
            }
        }
    }

    #[pg_test]
    fn l2_index_recall_clears_the_threshold() {
        create_recall_fixture("t_sweep_l2", "real[]", "", "");
        assert_recall_sweep("t_sweep_l2", "<->", "brindle_l2_distance", array_literal);
    }

    #[pg_test]
    fn cosine_index_recall_clears_the_threshold() {
        create_recall_fixture(
            "t_sweep_cos",
            "brindle_vector",
            "::brindle_vector",
            "brindle_vector_cosine_ops",
        );
        assert_recall_sweep(
            "t_sweep_cos",
            "<=>",
            "brindle_vector_cosine_distance",
            vector_literal,
        );
    }

    /// Inner product held to the same floors as the other two, which is worth a
    /// caveat rather than a shrug: `<#>` is not a metric — no triangle
    /// inequality — and HNSW's greedy walk is built on the assumption of one, so
    /// a graph ordered by inner product can behave much worse than the same data
    /// under L2. On this fixture it does not, because the vectors are centred on
    /// the origin with similar magnitudes, which keeps inner product close to
    /// cosine. A dataset whose magnitudes vary wildly is where the two diverge,
    /// and this gate would not see it: what this asserts is that the opclass
    /// wiring and the graph are sound, not that inner product is safe in
    /// general.
    #[pg_test]
    fn inner_product_index_recall_clears_the_threshold() {
        create_recall_fixture(
            "t_sweep_ip",
            "brindle_vector",
            "::brindle_vector",
            "brindle_vector_ip_ops",
        );
        assert_recall_sweep(
            "t_sweep_ip",
            "<#>",
            "brindle_vector_negative_inner_product",
            vector_literal,
        );
    }

    #[pg_test]
    fn index_scan_returns_rows_nearest_first() {
        create_indexed_fixture("t_order", ROWS);
        let query = query_literal();
        let ordered = format!("SELECT id FROM t_order ORDER BY embedding <-> {query} LIMIT 50");
        assert_uses_index(&ordered, "t_order_idx");

        let monotone = Spi::get_one::<bool>(&format!(
            "SELECT coalesce(bool_and(d >= prev), true)
             FROM (SELECT d, lag(d) OVER () AS prev
                   FROM (SELECT brindle_l2_distance(embedding, {query}) AS d
                         FROM t_order ORDER BY embedding <-> {query} LIMIT 50) s) w
             WHERE prev IS NOT NULL"
        ))
        .expect("spi")
        .expect("non-null");
        assert!(monotone, "distances were not non-decreasing");
    }

    /// Count rows in `sql` that came back nearer than the row before them —
    /// the inversions an `ORDER BY` promises never to produce. `distance` is the
    /// operator's own distance function, computed from the heap.
    fn order_inversions(table: &str, operator: &str, distance: &str, literal: &str) -> i64 {
        let rows = format!(
            "SELECT {distance}(embedding, {literal}) AS d
             FROM {table} ORDER BY embedding {operator} {literal} LIMIT 200"
        );
        Spi::get_one::<i64>(&format!(
            "SELECT count(*)
             FROM (SELECT d, lag(d) OVER () AS prev FROM ({rows}) s) w
             WHERE prev IS NOT NULL AND d < prev"
        ))
        .expect("spi")
        .expect("non-null")
    }

    /// The same promise as [`index_scan_returns_rows_nearest_first`], but with a
    /// `LIMIT` far past the budget — the shape that used to make the scan widen
    /// mid-stream and hand back a row nearer than one already returned.
    ///
    /// Both metrics, because each operator has its own distance function and the
    /// units differ: `<->` is true L2 while the graph ranks by its square.
    #[pg_test]
    fn index_scan_stays_ordered_past_the_budget() {
        // The recall fixture, not the 8-dimensional one: at 8 dimensions the
        // graph finds the exact answer whatever budget it is given, so nothing
        // ever turned up late to be out of order.
        create_recall_fixture("t_order_l2", "real[]", "::real[]", "");
        create_recall_fixture(
            "t_order_cos",
            "brindle_vector",
            "::brindle_vector",
            "brindle_vector_cosine_ops",
        );
        Spi::run("SET LOCAL brindle.ef_search = 16").expect("set");

        // Three query points, not the full sweep: ordering is a property of the
        // scan rather than of the data, so this checks the contract cheaply and
        // leaves measuring quality to the recall sweep.
        for query in recall_queries().into_iter().take(3) {
            let as_array = array_literal(&query);
            let as_vector = vector_literal(&query);
            for (table, operator, distance, literal) in [
                ("t_order_l2", "<->", "brindle_l2_distance", &as_array),
                (
                    "t_order_cos",
                    "<=>",
                    "brindle_vector_cosine_distance",
                    &as_vector,
                ),
            ] {
                let sql = format!(
                    "SELECT id FROM {table} ORDER BY embedding {operator} {literal} LIMIT 200"
                );
                assert_uses_index(&sql, &format!("{table}_idx"));

                let inversions = order_inversions(table, operator, distance, literal);
                assert_eq!(
                    inversions, 0,
                    "{table}: {inversions} rows came back nearer than the row \
                     before them — ORDER BY promises distance order"
                );
            }
        }
    }

    #[pg_test]
    fn a_scan_stops_at_its_budget_and_repeats_nothing() {
        create_indexed_fixture("t_drain", ROWS);
        // Ask for far more rows than the budget holds. The scan returns what one
        // search found — no more, since producing more would mean widening, and
        // a wider search can beat a row already handed over.
        Spi::run("SET LOCAL enable_seqscan = off").expect("set");
        Spi::run("SET LOCAL brindle.ef_search = 40").expect("set");
        let sql = format!(
            "SELECT id FROM t_drain ORDER BY embedding <-> {} LIMIT {}",
            query_literal(),
            ROWS * 5
        );
        assert_uses_index(&sql, "t_drain_idx");

        let counts = Spi::get_one::<Vec<i64>>(&format!(
            "SELECT ARRAY[count(*), count(DISTINCT id)] FROM ({sql}) s"
        ))
        .expect("spi")
        .expect("non-null");
        assert_eq!(
            counts,
            vec![40, 40],
            "a scan yields at most ef_search rows, each once"
        );
    }

    #[pg_test]
    fn scan_of_an_empty_index_returns_nothing() {
        Spi::run("CREATE TABLE t_none (id int, embedding real[])").expect("create");
        Spi::run("CREATE INDEX t_none_idx ON t_none USING brindle (embedding)").expect("index");
        Spi::run("SET LOCAL enable_seqscan = off").expect("set");
        let sql = format!(
            "SELECT id FROM t_none ORDER BY embedding <-> {} LIMIT 10",
            query_literal()
        );
        assert_uses_index(&sql, "t_none_idx");
        let rows = Spi::get_one::<i64>(&format!("SELECT count(*) FROM ({sql}) s"))
            .expect("spi")
            .expect("non-null");
        assert_eq!(rows, 0);
    }

    #[pg_test]
    fn null_query_vector_ranks_nothing() {
        create_indexed_fixture("t_null", 100);
        // A custom plan folds the strict operator against NULL and never reaches
        // the index; a generic plan keeps the parameter, which is the case the
        // scan itself has to handle.
        Spi::run("SET LOCAL plan_cache_mode = force_generic_plan").expect("set");
        Spi::run("SET LOCAL enable_seqscan = off").expect("set");
        Spi::run(&format!(
            "PREPARE nearest(real[]) AS
             SELECT id FROM t_null ORDER BY embedding <-> $1 LIMIT {K}"
        ))
        .expect("prepare");
        assert_uses_index("EXECUTE nearest(NULL)", "t_null_idx");

        let rows = Spi::connect(|client| {
            client
                .select("EXECUTE nearest(NULL)", None, None)
                .expect("execute")
                .count()
        });
        assert_eq!(rows, 0, "a NULL query vector must rank nothing");
    }

    #[pg_test]
    fn partial_index_is_only_used_for_ordered_queries() {
        Spi::run("CREATE TABLE t_part (id int, embedding real[])").expect("create");
        Spi::run("SELECT setseed(0.42)").expect("seed");
        Spi::run(&format!(
            "INSERT INTO t_part
             SELECT i, array_agg(random()::real)
             FROM generate_series(1, {ROWS}) i, generate_series(1, {DIM}) d
             GROUP BY i"
        ))
        .expect("insert");
        Spi::run("CREATE INDEX t_part_idx ON t_part USING brindle (embedding) WHERE id > 500")
            .expect("create index");

        // A matching predicate alone makes the planner cost this index, even
        // though the AM can answer nothing without an ORDER BY. It has to come
        // out priced out, not merely unattractive.
        let plan = plan_of("SELECT id FROM t_part WHERE id > 500");
        assert!(
            plan.contains("Seq Scan on t_part"),
            "expected the index to be priced out, got:\n{plan}"
        );

        let sql = format!(
            "SELECT id FROM t_part WHERE id > 500 ORDER BY embedding <-> {} LIMIT {K}",
            query_literal()
        );
        assert_uses_index(&sql, "t_part_idx");
        let ids = ordered_ids(&sql);
        assert_eq!(ids.len(), K);
        assert!(
            ids.iter().all(|&id| id > 500),
            "partial index returned rows outside its predicate: {ids:?}"
        );
    }

    /// The session's `brindle.ef_search` is what sizes a scan's one search, and
    /// so how many rows it can return. Asserted on the scan state rather than
    /// through SQL, to read the count the search itself produced.
    #[pg_test]
    fn ef_search_sizes_the_scans_candidate_budget() {
        create_indexed_fixture("t_ef", 200);
        let (hnsw, tids) = {
            // SAFETY: the index exists; PgRelation holds AccessShare on it.
            let relation = unsafe { PgRelation::open_with_name("t_ef_idx") }.expect("open index");
            unsafe { storage::load_index(relation.as_ptr()) }
        };

        // The budget is observable as how many rows one search yields, since the
        // scan returns exactly what it found and then ends.
        let mut search = ScanSearch::new(hnsw, tids);

        Spi::run("SET brindle.ef_search = 137").expect("set");
        search.start(QUERY.to_vec()).expect("start");
        assert_eq!(
            search.results.len(),
            137,
            "a scan searches at the session's value"
        );

        Spi::run("SET brindle.ef_search = 41").expect("set");
        search.start(QUERY.to_vec()).expect("start");
        assert_eq!(
            search.results.len(),
            41,
            "restarting picks up a value set since the scan was opened"
        );

        Spi::run("RESET brindle.ef_search").expect("reset");
        search.start(QUERY.to_vec()).expect("start");
        assert_eq!(
            search.results.len(),
            64,
            "reset returns the scan to the default"
        );
    }

    #[pg_test(error = "brindle: vector dimension mismatch: expected 8, got 2")]
    fn query_vector_dimension_must_match() {
        create_indexed_fixture("t_dim", 50);
        Spi::run("SET LOCAL enable_seqscan = off").expect("set");
        Spi::run("SELECT id FROM t_dim ORDER BY embedding <-> ARRAY[1,2]::real[] LIMIT 1")
            .expect("scan");
    }
}
