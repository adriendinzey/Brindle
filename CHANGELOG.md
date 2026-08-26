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

- The stored graph format now carries each row's filterable attribute values
  (format version 2). Version 1 payloads are rejected rather than read as
  attribute-free, because that would let a filtered scan silently return no rows
  instead of failing; an index built by an earlier development build must be
  rebuilt with `REINDEX`. No released version wrote the old format.

### Known limitations

- A scan returns at most `brindle.ef_search` rows, so a larger `LIMIT` — or an
  `ORDER BY` with none — comes back short. Raise the setting to see further. The
  ceiling is what makes the ordering guarantee hold.
- Every `INSERT` or `UPDATE` of an indexed row rewrites the whole stored graph,
  so write cost grows with index size. Writes are correct and immediately
  searchable, but a bulk load is still far faster as `COPY` followed by
  `CREATE INDEX` than as inserts into an existing index.
- Filter-aware search exists in the index core but has no SQL surface yet: a
  `WHERE` clause alongside `ORDER BY embedding <-> $1` is applied by the
  executor after the index has already spent its candidate budget.
