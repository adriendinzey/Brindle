# Changelog

All notable, user-visible changes to Brindle are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/) (pre-1.0, so minor
versions may break).

## [Unreleased]

### Added

- **Filtered vector search from SQL.** A `WHERE` clause on an indexed attribute
  column is pushed into the graph traversal, so the search returns *k* rows that
  satisfy the predicate rather than *k* nearest rows that are then filtered down
  to fewer:

  ```sql
  CREATE INDEX ON docs USING brindle (embedding, tenant_id, price);
  SELECT id FROM docs
   WHERE tenant_id = 7 AND price < 50
   ORDER BY embedding <-> $1 LIMIT 10;
  ```

  Filterable columns are **key columns after the vector**, not `INCLUDE`
  columns: Postgres only matches a qual to a column in the search key, so a
  predicate on an `INCLUDE` column never reaches the index and the executor
  filters afterwards. Measured at 1% selectivity on 20 000 rows, pushdown
  returns the full 10 requested rows at recall 0.9 against an exact scan; the
  same query filtered afterwards comes up short, because `ef_search` bounds the
  candidates before the filter is applied.

  Equality and range comparisons (`=`, `<`, `<=`, `>`, `>=`) are pushed for
  `bool`, `int2`, `int4`, `int8`, `float4` and `float8` columns. Comparisons
  work across widths within a family — `bigint_col = 42` pushes without the
  literal needing a cast — because the integer types share one operator family
  and the float types another. Integers and floats do not mix, since the stored
  value and the bound have to compare as one type. A `NULL`
  attribute satisfies no comparison, as in SQL.

  A qual the planner never offers the index — a `<>`, an expression, an
  unsupported column type — stays with the executor as it always did. A scan key
  the index *does* receive and cannot express, such as one whose value turns out
  to be NULL at run time, is refused rather than dropped: the scan reports a
  recheck and the executor re-tests every row it returns.

  Float comparisons follow PostgreSQL's total order, in which `'NaN' = 'NaN'` is
  true and `NaN` sorts above every other value — so a row storing `NaN`, and a
  `NaN` bound, return what a sequential scan returns.

  Because the attributes are search keys, the planner may also choose this index
  for a plain `WHERE attr = v` with no `ORDER BY` at all. That now works — it
  reads every node and tests the predicate — but is priced so the planner
  reaches for it only when nothing else can serve.
- PostgreSQL extension scaffold (`CREATE EXTENSION brindle`), built with pgrx.
- Distance kernels — squared L2, cosine, and (negative) inner product — exposed
  to SQL as `brindle_l2_distance`, `brindle_cosine_distance`,
  `brindle_inner_product`, and `brindle_negative_inner_product` over `real[]`.
- Pure-Rust in-memory HNSW graph: construction, layered search, brute-force
  reference search, and soft-delete with compaction.
- The `brindle` index access method: `CREATE INDEX ... USING brindle (embedding)`
  builds an HNSW graph over a `real[]` column and stores it in the index
  relation.
- Nearest-neighbor index scans — `SELECT ... ORDER BY embedding <-> $1 LIMIT k`
  is answered from the index, nearest first, through the new `<->` L2 distance
  operator over `real[]` and the default `real_array_l2_ops` operator class.
  A scan runs one search at `brindle.ef_search` and returns its rows in distance
  order, so it yields **at most `ef_search` rows**: a `LIMIT` larger than the
  budget — or an `ORDER BY` with no `LIMIT` at all — comes back short, and
  raising `brindle.ef_search` is how you see further. The ceiling is what buys
  the ordering guarantee, since producing more rows means widening the search,
  and a wider graph search can turn up a row nearer than one already returned.
- Rows inserted after `CREATE INDEX` are picked up automatically — `INSERT` and
  `UPDATE` no longer error, and the new vectors are findable through an index
  scan without a `REINDEX`. Each insert rewrites the stored graph, so the cost
  is proportional to index size: bulk loads are still far faster as `COPY`
  followed by `CREATE INDEX`.
- `VACUUM` integration: entries for deleted rows are tombstoned so the index
  never returns a heap slot that has been recycled for a different row.
- Criterion micro-benchmarks for the distance kernels.
- CI: rustfmt, clippy, and pgrx integration tests on PostgreSQL 16 and 17.
- Repository hygiene: PR/issue templates, Dependabot (cargo + GitHub Actions),
  contributor guide.

- **A filtered-search benchmark against pgvector**, sweeping predicate
  selectivity for both an uncorrelated and a correlated label:
  `SELECTIVITY=1 PGVECTOR=1 scripts/bench_index.sh`. Results, method and caveats
  in `docs/BENCHMARKS.md`, with the chart in the README.

  On 100 000 rows × 128 dimensions with a correlated filter at 1% selectivity,
  Brindle answers at recall@10 of 0.940 in 3.2 ms. The exact sequential scan --
  perfect by definition -- takes 25.9 ms, and pgvector's iterative scan reaches
  0.117 in 60 ms, so at the tight end brute force beats the ANN index on both
  axes and Brindle's real competition is the scan. Given its best scan budget
  pgvector reaches 0.773 at 140 ms.

  It does not win everywhere, and the write-up says where. With an
  *uncorrelated* filter the two indexes are hard to separate -- against
  pgvector's iterative scan Brindle wins four of eight cells and loses four,
  and iterative scan is ahead at 1% selectivity with the default beam.
  Post-filtering is the cheapest arm at every tight point while answering a
  different question at a recall near zero. pgvector's build is randomised, so
  its figures are reported as a range across rebuilds rather than as a number.

### Changed

- **Fixed: a selective filter could strand the search in one fragment of the
  matching rows — and `brindle.ef_search` stopped helping.** At around one row in
  twenty matching, the matching rows stop forming one connected graph: a matching
  row usually has one or two matching neighbours and they are often each other's.
  Traversal only bridged out of a row with *no* matching neighbour, so it
  explored one fragment and stopped, and because the frontier was empty rather
  than the budget spent, raising `ef_search` bought nothing. Measured on 10 000
  rows at 1% selectivity, recall@10 was 0.356 / 0.455 / 0.526 / 0.530 at
  `ef_search` 64 / 128 / 256 / 512 — a ceiling, not a curve.

  A search that ends with fewer than `ef_search` matching rows now goes back to
  the rows it walked past and bridges out of them, nearest first. Recall@10 on
  that fixture becomes 0.500 / 0.709 / 0.922 / 0.985, and on a 20 000-row
  128-dimensional index at 1% selectivity the old ceiling of 0.928 becomes 1.000.

  It costs nothing where it is not needed: a query whose result heap fills never
  reaches it — its counters and recall are identical to not having it — and a
  predicate nothing satisfies is unchanged. A filter *correlated* with vector
  position benefits too, at 1% selectivity going 0.872 → 0.905.

  Where it does fire it is not cheap, and the trade is the point. Whenever fewer
  rows match than `brindle.ef_search`, the heap cannot fill and the query spends
  its detour allowance in full: on 20 000 rows at 128 dimensions and 1%
  selectivity, `ef_search` 256 goes from 0.79 ms to 5.73 ms per query for recall
  0.933 → 0.997, and `ef_search` 1024 from 0.77 ms to 20.87 ms for 0.933 →
  1.000. That is recall which was previously unreachable at any setting. Lower
  `brindle.ef_search` if you would rather have the old cost than the extra
  recall.
- **Fixed: a float comparison could return the wrong rows.** The index ordered
  floats by IEEE 754, where `NaN` compares with nothing, while PostgreSQL gives
  floats a total order in which `'NaN' = 'NaN'` holds and `NaN` sorts above
  everything. A row storing `NaN` was therefore dropped from a comparison SQL
  would satisfy: on 400 rows plus two storing `NaN`, `WHERE score >= 1` returned
  398 rows by sequential scan and 396 through the index.

  This was documented as costing rows and never wrong ones, which does not
  survive negation — under `NOT EXISTS` the dropped rows came back as *extra*
  ones, 4 by sequential scan against 6 through the index. Float comparisons now
  follow PostgreSQL's rule, so both agree. A `NaN` bound is no longer refused at
  the boundary either, since it is now answerable.
- **A filter that correlates with vector position is now answered.** When the
  matching rows sit *away* from the query — a tenant whose documents cluster, a
  price band, a date range — the search previously could not reach them at all:
  on a 10 000-row fixture whose filter selects regions 46 units from the query,
  `WHERE price < 5 ORDER BY embedding <-> $1 LIMIT 10` returned **zero rows** at
  every `brindle.ef_search` below 5000, while an uncorrelated filter of the same
  5% selectivity returned all ten. Raising `gamma` did not help.

  The layer descent now probes each layer above 0 for a node that matches the
  predicate and hands one down, so the bottom layer starts inside the matching
  region rather than tens of hops away. The same query returns 10 of 10 rows at
  the default `ef_search`, at recall 1.00 against an exact scan (0.90 at 1%
  selectivity). Navigation itself is unchanged, and an unfiltered search is
  unaffected in both results and cost.

  Uncorrelated filters — the case that already worked — are unchanged in recall
  and cost roughly 10% more per query in vector distances. See
  `docs/FILTERING.md` § 2(c).
- **The filtered walk now has a cost ceiling that does not grow with the index.**
  A tombstoned row still satisfies a predicate, so a graph whose matching rows
  have all been deleted gave the traversal nothing to stop on and it walked the
  whole index — 156 ms at 100 000 rows. Filtered traversal is now bounded by a
  total expansion allowance (16 × `ef_search`) as well as by the existing detour
  allowance, which holds that case flat: measured 206 → 2007 node expansions
  from n = 2000 to n = 20 000 before, and 235 → 1036 (the ceiling) after.

  The ceiling covers the predicate-aware part of a query and not the whole of it.
  Unfiltered search has the same missing stop condition and still has it, and a
  filtered query's layer descent navigates unfiltered — so a table with *every*
  row tombstoned, rather than merely every matching one, still costs work
  proportional to the index (2353 node expansions at n = 20 000 against a 1024
  allowance). Closing that is a decision about plain HNSW recall rather than
  about filtering.
- **Fixed: an index scan could return fewer rows than a sequential scan of the
  same query.** A row whose indexed vector is `NULL` cannot go in the graph —
  there is nothing to place or to rank — and was skipped outright. But the index
  is not registered as partial, so PostgreSQL believes it covers every row in the
  table, and a plan that used it silently came back short: on 400 rows plus three
  with a `NULL` embedding, `SELECT count(*) ... WHERE bucket = 7` returned 43 by
  sequential scan and 40 through the index. That is a wrong answer rather than a
  recall trade.

  Those rows are now stored beside the graph with their attribute values, and a
  scan that returns every matching row consults them, applying the predicate
  exactly as it does to a row in the graph. `INSERT`, `VACUUM` and rollback treat
  them like any other indexed row.

  **A ranked scan still omits them, by design.** A `NULL` vector has no distance,
  so it has no place in an `ORDER BY embedding <-> $1` result; an ordered scan is
  already bounded by `brindle.ef_search` and documented as returning at most that
  many rows. Leaving unrankable rows out of a ranking is consistent with that.
  The unordered path is different in kind: its whole purpose is to return every
  matching row.

- **A transaction's inserts are written back to the index once, when it ends,
  rather than once per row.** Every write rewrites the whole stored image, so
  doing that per row made a bulk load quadratic in the table; `INSERT ... SELECT`
  of N rows is now linear. Measured on a 20 000-row index, a row inserted as part
  of a 100-row statement went from 25.5 ms to 0.61 ms — amortization rather than
  elimination, so a larger statement is cheaper per row and a batch of one is no
  batch at all.

  **A single-row `INSERT` is no faster.** It is its own transaction, so it has
  nothing to batch with and rewrites the whole image exactly as before. Not
  rewriting the image per row needs the paged storage this format is a
  placeholder for.

  Consequences worth knowing. A transaction's own rows are visible to it before
  it commits, as before, but they reach the index relation no earlier than the
  first of: a query against that index, a parallel plan, a write to a second
  brindle index, or the end of the transaction (see below) — so a crash before
  any of those leaves the index as it was, which is what rolling that
  transaction back means anyway. A transaction
  that ends with `PREPARE TRANSACTION` writes them at the prepare rather than at
  `COMMIT PREPARED`; a `ROLLBACK PREPARED` after that does **not** take them back
  out, though heap visibility keeps them from being returned. Savepoints do not
  force a write-back, and `ROLLBACK TO` undoes the rows *still staged* when it
  runs — but **anything already written back cannot be taken out again**, and a
  write-back can happen inside a savepoint: a query against that index, a
  statement that plans a parallel scan, or a write to a second brindle index all
  force one. Rows rolled back after that stay in the index as entries pointing at
  dead heap tuples. They return no wrong answers — heap visibility drops them and
  the next `VACUUM` tombstones them — but until then they are bloat, and a
  `plpgsql` loop with an `EXCEPTION` handler keeps its batching only if it does
  not read the index it is writing. **A table with two brindle indexes gets no
  batching at all** — only one index's rows are staged at a time, so writes to a
  second flush the first.

  A `TRUNCATE` or `REINDEX` in the same transaction sets aside whatever that
  transaction had staged for the index rather than writing it over the rebuild.
  Set aside, not discarded: a rebuild inside a subtransaction that later aborts
  is undone, relfilenode and all, and the staged rows belong to the state that
  comes back. They are handed back if that happens and dropped once the rebuild
  is known to stand.

  One consequence of writing at the end rather than per row: a conflict between
  two transactions is reported by the one that commits second, at its `COMMIT`,
  rather than by the statement that caused it. Two sessions inserting different
  vector dimensions into the same empty index is the reachable case.

  **Querying a brindle index inside a transaction that has written to it forces
  the write-back early**, and the batching restarts from there. Staged rows are
  backend-local, so rather than lending them to a scan they are written first —
  and because a parallel worker is a separate process that could not see them
  either way, **any statement that runs a parallel plan also forces the
  write-back**, whether or not it touches a brindle index. A transaction that
  alternates `INSERT` and `SELECT` on the same index therefore gets no batching —
  it pays what it paid before — while one that writes and then reads pays one
  extra write-back. Bulk loads, which do not query what they are filling and plan
  no parallel statements, are unaffected.

  This installs an `ExecutorStart_hook`. It chains to any hook already present,
  so other extensions are unaffected, and it does nothing unless the statement
  needs parallel mode and this transaction has rows staged. `EXPLAIN` without
  `ANALYZE` is excluded, so planning a query stays free of side effects. Note
  that a session running with `debug_parallel_query = on` — which some test
  suites set globally — makes *every* parallel-safe statement force the
  write-back, and so gets no batching at all.

  While a transaction is staging rows it holds a decoded copy of the index, and
  that copy is **not bounded by `brindle.cache_max_mb`** — it exists even when
  that is zero. A write-back that has to replay onto another backend's newer
  image holds two decoded copies plus the encoded blob at its peak. A `TRUNCATE`
  or `REINDEX` also sets the staged graph aside until the rebuild's fate is
  known, so a transaction that rebuilds under several open savepoints can hold
  one decoded graph per savepoint depth — bounded by that depth, not unbounded,
  and released when the transaction ends.
  A large bulk load into a wide-vector index can therefore hold a
  substantial amount of memory for the length of the transaction. Splitting such
  a load into several transactions bounds it, at the cost of one write-back each.
- A backend now keeps one decoded copy of an index in memory and reuses it
  across scans, instead of reading and decoding the whole index for every query.
  On a 100k × 128 index that takes a query from ~58 ms to ~0.3 ms. The first
  scan in a backend still pays the full cost, as does the first after any write
  invalidates the copy, so a connection that issues one query and disconnects
  sees no benefit.
- **The on-disk page layout is now version 3, and an index written by an earlier
  build must be rebuilt with `REINDEX`.** (Distinct from the graph codec version
  named below — they are separate numbers in separate headers, which is worth
  knowing when reading an error message.) The metapage carries a generation
  counter, which is how a backend tells whether the copy it holds is still the
  index — including when another connection wrote to it, which Postgres does not
  otherwise announce; and the image carries the rows that have no vector, for the
  reason in the completeness fix above. Reading an older index reports the format
  it was written in and names `REINDEX`.
- The stored graph codec now carries each row's filterable attribute values
  (codec version 2). Codec version 1 payloads are rejected rather than read as
  attribute-free, because that would let a filtered scan silently return no rows
  instead of failing; an index built by an earlier development build must be
  rebuilt with `REINDEX`. No released version wrote the old format.

### Added

- `brindle.cache_max_mb` (default 256) bounds the decoded index copies a backend
  keeps. The copy is **per backend**, not shared between connections, so the
  real cost is this ceiling times the number of connections that touch an index
  — measured at about 887 bytes per node at 128 dimensions, so a 100k index is
  roughly 89 MB of graph and rather more resident. Zero disables the cache, and
  an index that does not fit is decoded per scan as before.

### Known limitations

- A scan returns at most `brindle.ef_search` rows, so a larger `LIMIT` — or an
  `ORDER BY` with none — comes back short. Raise the setting to see further. The
  ceiling is what makes the ordering guarantee hold.
- The decoded-index cache is per backend and unshared, so a server with many
  connections against a large index holds many copies of it. Lowering
  `brindle.cache_max_mb` bounds each backend; nothing bounds the total. Paged
  storage would put one copy in the shared buffer cache for the whole server.
- Every `INSERT` or `UPDATE` of an indexed row rewrites the whole stored graph,
  so write cost grows with index size. Writes are correct and immediately
  searchable, but a bulk load is still far faster as `COPY` followed by
  `CREATE INDEX` than as inserts into an existing index.
- Filter-aware search exists in the index core but has no SQL surface yet: a
  `WHERE` clause alongside `ORDER BY embedding <-> $1` is applied by the
  executor after the index has already spent its candidate budget.
