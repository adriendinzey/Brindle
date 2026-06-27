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
- ⬜ `CREATE INDEX ... USING brindle (embedding vector_cosine_ops) WITH (m, ef_construction)`
- ⬜ GUC `brindle.ef_search`
- ⬜ pgvector `vector` type interop (operators `<->`, `<=>`, `<#>`)
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

> **TL;DR:** build in WSL2/Linux/macOS, *not* native Windows, and *not* inside the
> OneDrive folder.

### Why not native Windows / OneDrive?

- `pgrx` (and Postgres extension building generally) is well-supported on
  Linux/macOS and rough on native Windows. Use **WSL2**.
- `cargo` produces a large, constantly-changing `target/` directory. If that lives
  in a OneDrive-synced folder, OneDrive will thrash trying to sync it and can
  corrupt builds. Keep the working copy on the **WSL native filesystem**
  (e.g. `~/code/brindle`), not under `/mnt/c` or `/mnt/d`.

### One-time WSL2 setup

```bash
# inside Ubuntu on WSL2
sudo apt-get update
sudo apt-get install -y build-essential libreadline-dev zlib1g-dev flex bison \
  libxml2-dev libxslt-dev libssl-dev libxml2-utils xsltproc ccache pkg-config clang

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# pgrx
cargo install --locked cargo-pgrx
cargo pgrx init        # downloads & compiles supported Postgres versions (slow, one-time)
```

### Daily loop

```bash
git clone <your-repo-url> ~/code/brindle      # native FS, not /mnt/*
cd ~/code/brindle

cargo test                  # fast: pure-Rust core logic (no Postgres)
cargo pgrx test pg17        # integration tests against a managed Postgres
cargo pgrx run  pg17        # build + install + open psql with brindle loaded
```

```sql
CREATE EXTENSION brindle;
SELECT brindle_l2_distance(ARRAY[1,2,3]::real[], ARRAY[4,5,6]::real[]);
```

### Editing from Windows

You can edit files in VS Code on Windows via the **WSL remote** extension (open the
`~/code/brindle` folder *inside* WSL), which avoids the `/mnt` performance penalty
while keeping a familiar editor.
