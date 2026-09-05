# Changelog

All notable, user-visible changes to Brindle are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/) (pre-1.0, so minor
versions may break).

## [Unreleased]

### Added

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

### Changed

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
  brindle index, or the end of the transaction (see below) — so a crash before any of those leaves the index as it
  was, which is what rolling that transaction back means anyway. A transaction
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
  image holds two decoded copies plus the encoded blob at its peak.
  A large bulk load into a wide-vector index can therefore hold a
  substantial amount of memory for the length of the transaction. Splitting such
  a load into several transactions bounds it, at the cost of one write-back each.
- A backend now keeps one decoded copy of an index in memory and reuses it
  across scans, instead of reading and decoding the whole index for every query.
  On a 100k × 128 index that takes a query from ~58 ms to ~0.3 ms. The first
  scan in a backend still pays the full cost, as does the first after any write
  invalidates the copy, so a connection that issues one query and disconnects
  sees no benefit.
- **The on-disk page layout is now version 2, and an index written by an earlier
  build must be rebuilt with `REINDEX`.** (Distinct from the graph codec version
  named below — they are separate numbers in separate headers, which is worth
  knowing when reading an error message.) The metapage carries a generation
  counter, which is how a backend tells whether the copy it holds is still the
  index — including when another connection wrote to it, which Postgres does not
  otherwise announce. Reading such an index reports the format it was written in
  and names `REINDEX`.
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
