//! Index scan: answers `ORDER BY <indexed column> <-> <query vector>` from the
//! persisted graph.
//!
//! This is an *ordering* scan (`amcanorderbyop`), not a matching one: there is
//! no `WHERE` clause for the AM to satisfy, only an order-by operator whose
//! right-hand argument is the query vector. Results stream nearest-first and the
//! executor stops whenever it has seen enough.
//!
//! # How many neighbors to fetch
//!
//! `LIMIT` is not visible to an access method, so the scan can't size its search
//! to the caller. It fetches a batch and grows on demand:
//!
//! 1. The first batch searches the graph with a candidate budget of
//!    [`DEFAULT_EF_SEARCH`].
//! 2. Draining a batch doubles the budget and re-runs the search from scratch.
//!    A wider search is not a strict superset of a narrower one, so results
//!    already handed to the executor are filtered out by node id — a row is
//!    never returned twice.
//! 3. Once the budget reaches the number of live nodes the scan switches to an
//!    exact pass. A graph walk can leave nodes unreachable, and an unfiltered
//!    `ORDER BY` that quietly dropped rows would be a wrong answer rather than a
//!    recall trade-off; the exact tail also costs no more than the sort the
//!    planner would otherwise have chosen.
//!
//! Because the budget doubles, a scan that reads `n` rows does about `2n` rows'
//! worth of search in total. Bookkeeping is one node id per row returned.
//!
//! TODO: stream directly out of the graph's candidate heap instead of re-running
//! the search, so a deep scan resumes rather than restarts.

use core::ffi::{c_int, c_void};
use std::collections::HashSet;

use pgrx::itemptr::item_pointer_set_all;
use pgrx::prelude::*;
use pgrx::{pg_guard, pg_sys};

use super::storage::{self, TidPair};
use crate::hnsw::{Hnsw, HnswError};

/// Layer-0 candidate budget for a scan's first batch. Larger means better recall
/// and more work per query.
// TODO: read this from a `brindle.ef_search` GUC and per-index reloptions.
const DEFAULT_EF_SEARCH: usize = 64;

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
/// the current batch of results with a cursor into it.
struct ScanSearch {
    hnsw: Hnsw,
    /// `tids[i]` is the heap address of graph node `i`.
    tids: Vec<TidPair>,
    /// Live node count, fixed for the life of the scan (the graph is a private
    /// copy that nothing mutates).
    live: usize,
    query: Vec<f32>,
    /// Current batch as `(node id, heap tid)`, nearest first.
    batch: Vec<(usize, TidPair)>,
    cursor: usize,
    /// Candidate budget for the next batch; doubles until it covers the graph.
    budget: usize,
    /// Node ids already handed to the executor, so a wider re-search never
    /// repeats a row.
    emitted: HashSet<usize>,
    /// Set once the graph can produce nothing new.
    drained: bool,
}

impl ScanSearch {
    fn new(hnsw: Hnsw, tids: Vec<TidPair>) -> Self {
        let live = hnsw.live_len();
        Self {
            hnsw,
            tids,
            live,
            query: Vec::new(),
            batch: Vec::new(),
            cursor: 0,
            budget: DEFAULT_EF_SEARCH,
            emitted: HashSet::new(),
            drained: true,
        }
    }

    /// Begin (or restart) the scan for `query`, filling the first batch.
    fn start(&mut self, query: Vec<f32>) -> Result<(), ScanError> {
        self.query = query;
        self.batch.clear();
        self.cursor = 0;
        self.budget = DEFAULT_EF_SEARCH;
        self.emitted.clear();
        self.drained = false;
        self.refill()
    }

    /// End the scan without returning anything.
    fn stop(&mut self) {
        self.query = Vec::new();
        self.batch.clear();
        self.cursor = 0;
        self.emitted = HashSet::new();
        self.drained = true;
    }

    /// The next heap TID, nearest first, or `None` once the scan is exhausted.
    fn next(&mut self) -> Result<Option<TidPair>, ScanError> {
        while self.cursor >= self.batch.len() {
            if self.drained {
                return Ok(None);
            }
            self.refill()?;
        }
        let (id, tid) = self.batch[self.cursor];
        self.cursor += 1;
        self.emitted.insert(id);
        Ok(Some(tid))
    }

    /// Search again with the current budget and keep whatever hasn't been
    /// returned yet. See the module docs for why the widest batch is exact.
    fn refill(&mut self) -> Result<(), ScanError> {
        let budget = self.budget.max(1);
        let found = if budget >= self.live {
            self.drained = true;
            self.hnsw.brute_force(&self.query, budget)?
        } else {
            self.budget = budget.saturating_mul(2);
            self.hnsw.search(&self.query, budget, budget)?
        };

        self.batch.clear();
        self.cursor = 0;
        for (_, id) in found {
            if self.emitted.contains(&id) {
                continue;
            }
            match self.tids.get(id) {
                Some(&tid) => self.batch.push((id, tid)),
                None => return Err(ScanError::UnmappedNode(id)),
            }
        }
        Ok(())
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
    // SAFETY: the operator class restricts the ordering operator's arguments to
    // real[], so the key argument is a non-null real[] datum.
    let query = super::f32_vec_from_array_datum(key.sk_argument);
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
    use pgrx::prelude::*;

    const DIM: usize = 8;
    /// Big enough that a graph search (rather than the exact tail) answers a
    /// `LIMIT k` query, and that the planner prefers the index to a sort.
    const ROWS: i64 = 1000;
    const K: usize = 10;
    /// An arbitrary point in the fixture's `[0,1]^DIM` cube.
    const QUERY: [f32; DIM] = [0.13, 0.77, 0.41, 0.92, 0.05, 0.63, 0.28, 0.55];

    fn query_literal() -> String {
        let elements: Vec<String> = QUERY.iter().map(|v| v.to_string()).collect();
        format!("ARRAY[{}]::real[]", elements.join(","))
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

        let hits = approx.iter().filter(|id| exact.contains(id)).count();
        assert!(
            hits * 10 >= K * 9,
            "recall {hits}/{K} below 0.9\n  index: {approx:?}\n  exact: {exact:?}"
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

    #[pg_test]
    fn scan_past_the_first_batch_returns_every_row_once() {
        create_indexed_fixture("t_drain", ROWS);
        // Asking for far more rows than one search budget holds forces the scan
        // to widen repeatedly; it must terminate having returned each row once.
        Spi::run("SET LOCAL enable_seqscan = off").expect("set");
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
        assert_eq!(counts, vec![ROWS, ROWS], "expected {ROWS} distinct rows");
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

    #[pg_test(error = "brindle: vector dimension mismatch: expected 8, got 2")]
    fn query_vector_dimension_must_match() {
        create_indexed_fixture("t_dim", 50);
        Spi::run("SET LOCAL enable_seqscan = off").expect("set");
        Spi::run("SELECT id FROM t_dim ORDER BY embedding <-> ARRAY[1,2]::real[] LIMIT 1")
            .expect("scan");
    }
}
