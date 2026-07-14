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
  reference search, and soft-delete with compaction (not yet wired to a
  Postgres index access method).
- Criterion micro-benchmarks for the distance kernels.
- CI: rustfmt, clippy, and pgrx integration tests on PostgreSQL 16 and 17.
- Repository hygiene: PR/issue templates, Dependabot (cargo + GitHub Actions),
  contributor guide.
