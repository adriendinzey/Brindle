# Brindle — Roadmap

Each phase is sized to be **independently demoable and resume-worthy**. The goal
is a polished P0–P4, not a half-built P0–P5. Ship depth, not breadth.

Legend: ✅ done · 🚧 in progress · ⬜ planned

---

## Phase 0 — Scaffold + exact distance ✅
*"A working PostgreSQL extension in Rust."*

- ✅ pgrx project scaffold, builds & loads as `CREATE EXTENSION brindle`
- ✅ Pure distance kernels (L2², cosine, inner product) with unit tests
- ✅ `#[pg_extern]` distance functions over `real[]`
- ✅ `criterion` micro-benchmark for distance kernels
- ✅ CI (GitHub Actions: fmt, clippy `-D warnings`, `cargo pgrx test` on pg16/pg17)

## Phase 1 — HNSW index access method ✅
*"Implemented a graph-based ANN index integrated with the planner."*

- ✅ In-memory HNSW: insert, layer assignment, greedy search, candidate heap
- ✅ `IndexAmRoutine`: `ambuild`, `aminsert`, `ambeginscan`, `amgettuple`, `amrescan`, `amendscan`
- ✅ `CREATE INDEX ... USING brindle (embedding) WITH (m, ef_construction, gamma)`
- ✅ GUC `brindle.ef_search`
- ✅ `brindle_vector` type + one operator class per metric (operators `<->`, `<=>`, `<#>`)
- ✅ Recall sanity vs brute force, calibrated against deliberately degraded builds
- ✅ Soft delete + `VACUUM` integration (`ambulkdelete`, `amvacuumcleanup`)
- ✅ Measured baseline against pgvector — [BENCHMARKS.md](BENCHMARKS.md)

## Phase 2 — Filter-aware search (the differentiator) 🚧
*"Predicate-aware ANN: high recall under selective filters."* — see [FILTERING.md](FILTERING.md)

- ✅ γ-dense edge construction (`gamma` build param)
- ✅ Inline filterable attributes as **key columns after the vector**
  (equality + numeric range on `bool`/`int2`/`int4`/`int8`/`float4`/`float8`,
  combined with `AND`). *Not `INCLUDE (...)`, which an earlier draft of this
  roadmap specified: a qual on an `INCLUDE` column never reaches the access
  method, so it would be post-filtering — see [FILTERING.md](FILTERING.md) § 3.*
- ✅ Predicate-aware expansion with ACORN-style bridging
- ✅ Reaching matching regions the query is not in — a filter correlated with
  vector position used to return nothing at all
- ⬜ Selectivity sweep proving recall vs pgvector post-filter / iterative scan

## Phase 3 — Durable storage 🚧
*"Crash-safe index in Postgres buffer pages + WAL."*

- ✅ `ambulkdelete` / `amvacuumcleanup` (handle deletes/updates)
- ✅ Index writes are WAL-logged, so the index survives a crash and reaches
  replicas — as whole-fork page images, which is why WAL volume is O(index)
  per write-back
- ⬜ Page layout for graph nodes/edges in the buffer manager — designed in
  [STORAGE.md](STORAGE.md), not built. This is what makes a write touch only
  the pages it changes, and what removes the per-backend in-memory copy a scan
  needs today.
- ⬜ Fine-grained WAL records for inserts

## Phase 4 — Hybrid search 🚧
*"Unified lexical + semantic ranking with RRF."*

- ✅ Reciprocal Rank Fusion in the pure core, with tests
- ⬜ `brindle_hybrid(query_text, query_vec, k, rrf_k)` fusing vector rank +
  Postgres `tsvector` rank via RRF — **no SQL surface yet**
- ⬜ Worked RAG example in `examples/`
- ⬜ (stretch) better lexical scoring than `ts_rank`

## Phase 5 — Quantization + benchmarks ⬜
*"Memory-efficient and measured."*

- ⬜ **RaBitQ** (1-bit) with exact re-rank, then extended B-bit RaBitQ
- ⬜ `ann-benchmarks`-style harness: recall@k vs QPS on SIFT/GIST + a filtered set
- ⬜ Results charts in README (vs pgvector); reproducible scripts
- ⬜ SIMD distance kernels + runtime dispatch

---

## Getting set up

The toolchain install, the WSL2 native-filesystem build loop, editing via VS Code
Remote-WSL, and parallel development with git worktrees are documented in one
place: **[DEVELOPMENT.md](DEVELOPMENT.md)**.

> **TL;DR:** on Windows, build inside WSL2 on the Linux-native filesystem
> (`~/code/brindle`), not under `/mnt/*` — `cargo` over the 9P mount is 5–10×
> slower. `cargo-pgrx` needs the Linux toolchain, so there is no native-Windows
> build path anyway.
