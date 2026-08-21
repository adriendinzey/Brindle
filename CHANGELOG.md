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
  A scan asked for more rows than one search budget holds keeps widening until
  it has returned every matching row, so an unfiltered `ORDER BY` is complete
  rather than truncated.
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

### Known limitations

- A table is effectively read-only once a brindle index exists on it: `INSERT`
  and `UPDATE` raise an error until incremental insert lands. Load the data
  first, then create the index, and `REINDEX` after further loads.
