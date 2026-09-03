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
  of N rows is now linear. Measured on a 20 000-row index, a row inserted as
  part of a larger statement went from 26.5 ms to 0.36 ms, and no longer grows
  with the index.

  A single-row `INSERT` is its own transaction, so it has nothing to batch: its
  cost improved (29.8 ms to 3.5 ms on that index, mostly from reusing the cached
  copy) but still grows with the index. Not rewriting the whole image per row
  needs the paged storage this format is a placeholder for.

  Two consequences worth knowing. A transaction's own rows are visible to it
  before it commits, as before, but they reach the index relation at commit — so
  a crash mid-transaction leaves the index as it was, which is what rolling that
  transaction back means anyway. And a `plpgsql` block with an `EXCEPTION`
  handler opens a subtransaction per iteration, which forces a write-back each
  time; such a loop gets no batching, though it is no slower than before.
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
