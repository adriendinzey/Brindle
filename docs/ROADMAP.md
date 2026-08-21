# Brindle — Roadmap

Each phase is sized to be **independently demoable and resume-worthy**. The goal
is a polished P0–P4, not a half-built P0–P5. Ship depth, not breadth.

Legend: ✅ done · 🚧 in progress · ⬜ planned

---

## Phase 0 — Scaffold + exact distance 🚧
*"A working PostgreSQL extension in Rust."*

- ✅ pgrx project scaffold, builds & loads as `CREATE EXTENSION brindle`
- ✅ Pure distance kernels (L2², cosine, inner product) with unit tests
- ✅ `#[pg_extern]` distance functions over `real[]`
- ⬜ `criterion` micro-benchmark for distance kernels
- ⬜ CI (GitHub Actions: `cargo test` + `cargo pgrx test`)

## Phase 1 — HNSW index access method ⬜
*"Implemented a graph-based ANN index integrated with the planner."*

- ⬜ In-memory HNSW: insert, layer assignment, greedy search, candidate heap
- ⬜ `IndexAmRoutine`: `ambuild`, `aminsert`, `ambeginscan`, `amgettuple`, `amrescan`, `amendscan`
- ⬜ `CREATE INDEX ... USING brindle (embedding brindle_vector_cosine_ops) WITH (m, ef_construction)`
- ⬜ GUC `brindle.ef_search`
- ⬜ `brindle_vector` type + one operator class per metric (operators `<->`, `<=>`, `<#>`)
- ⬜ Recall sanity vs brute force on a small dataset

## Phase 2 — Filter-aware search (the differentiator) ⬜
*"Predicate-aware ANN: high recall under selective filters."* — see [FILTERING.md](FILTERING.md)

- ⬜ γ-dense edge construction (`gamma` build param)
- ⬜ Inline filterable attributes via `INCLUDE (...)` (label + numeric-range)
- ⬜ Predicate-aware expansion with ACORN-style bridging
- ⬜ Selectivity sweep proving recall vs pgvector post-filter / iterative scan

## Phase 3 — Durable storage ⬜
*"Crash-safe index in Postgres buffer pages + WAL."*

- ⬜ Page layout for graph nodes/edges in the buffer manager
- ⬜ WAL logging of inserts; recovery
- ⬜ `ambulkdelete` / `amvacuumcleanup` (handle deletes/updates)

## Phase 4 — Hybrid search ⬜
*"Unified lexical + semantic ranking with RRF."*

- ⬜ `brindle_hybrid(query_text, query_vec, k, rrf_k)` fusing vector rank +
  Postgres `tsvector` rank via Reciprocal Rank Fusion
- ⬜ Worked RAG example in `examples/`
- ⬜ (stretch) better lexical scoring than `ts_rank`

## Phase 5 — Quantization + benchmarks ⬜
*"Memory-efficient and measured."*

- ⬜ Scalar quantization (f32 → int8), then binary quantization
- ⬜ `ann-benchmarks`-style harness: recall@k vs QPS on SIFT/GIST + a filtered set
- ⬜ Results charts in README (vs pgvector); reproducible scripts

---

## Getting set up

The toolchain install, the WSL2 native-filesystem build loop, editing via VS Code
Remote-WSL, and parallel development with git worktrees are documented in one
place: **[DEVELOPMENT.md](DEVELOPMENT.md)**.

> **TL;DR:** on Windows, build inside WSL2 on the Linux-native filesystem
> (`~/code/brindle`), not under `/mnt/*` — `cargo` over the 9P mount is 5–10×
> slower. `cargo-pgrx` needs the Linux toolchain, so there is no native-Windows
> build path anyway.
