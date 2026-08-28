# Brindle

**Filter-aware, hybrid vector search for PostgreSQL — written in Rust & fully vibe-coded from scratch.**

Brindle is a PostgreSQL extension for approximate nearest-neighbor (ANN) vector
search whose design goal is the query production RAG and search systems actually
issue:

```sql
-- "Find the 10 most semantically similar active products under $50 for this tenant"
SELECT id, name
FROM products
WHERE tenant_id = 42 AND status = 'active' AND price < 50      -- structured predicate
ORDER BY embedding <=> $1                                       -- vector similarity
LIMIT 10;
```

Plain HNSW indexes degrade badly on queries like this: filtering *after* the
graph search throws away most of the candidates the index worked to find, while
filtering *before* it means the index isn't used at all. Brindle pushes the
predicate **into** the graph traversal so recall stays high even under selective
filters — and adds first-class **hybrid** (vector + lexical) ranking via
Reciprocal Rank Fusion.

> **Status: early development (Phase 0).** This is a learning-grade project built
> in the open, and *not* production-ready. Nothing in it has been optimized yet:
> the first measured baseline is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md), and
> it shows queries about two orders of magnitude slower than pgvector, almost
> entirely because every scan still deserializes the whole index.
> See [docs/ROADMAP.md](docs/ROADMAP.md) for what works today.

> ⚠️ **On Windows, build inside WSL2 on the Linux-native filesystem.** Clone to
> `~/code/brindle` (ext4) and develop there — not under `/mnt/c` or `/mnt/d`.
> `cargo`/`rustc` touch thousands of small files, and Windows drives are reached
> from WSL over the slow 9P protocol, so builds on `/mnt/*` are commonly 5–10×
> slower than on the native filesystem. Full setup:
> [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Why another Postgres vector extension?

The space is mature — [pgvector](https://github.com/pgvector/pgvector),
[pgvectorscale](https://github.com/timescale/pgvectorscale), and
[VectorChord](https://github.com/tensorchord/VectorChord) are all excellent.
Brindle deliberately does **not** compete on raw QPS or quantization. It targets
the one thing they all still handle awkwardly: **arbitrary metadata filtering
combined with vector search**, plus hybrid lexical+semantic ranking in a single
index. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full rationale and
competitive analysis.

## Design pillars

1. **Filter-aware traversal** — an [ACORN](https://arxiv.org/abs/2403.04871)-style
   HNSW that keeps the matching-node subgraph navigable under predicates.
   ([docs/FILTERING.md](docs/FILTERING.md))
2. **Hybrid by default** — vector + PostgreSQL full-text, fused with RRF.
3. **Honest engineering** — `Result`-based error handling, no `unwrap()` in hot
   paths, zero-allocation distance kernels, benchmark-driven claims.
4. **Drop-in friendly** — its own `brindle_vector` type speaks pgvector's text
   format and operators (`<->`, `<=>`, `<#>`), and `real[]` columns index
   directly, so moving over is low-friction and pulls in no other extension.

## Quick start (dev)

Brindle is built with [`pgrx`](https://github.com/pgcentralfoundation/pgrx).
Building is best done on Linux / WSL2 / macOS (not native Windows).

```bash
# one-time toolchain setup
cargo install --locked cargo-pgrx
cargo pgrx init                      # downloads & builds dev Postgres versions

# from the repo root
cargo pgrx run pg17                  # builds + drops you into psql with brindle loaded
```

```sql
CREATE EXTENSION brindle;
SELECT brindle_l2_distance(ARRAY[1,2,3]::real[], ARRAY[4,5,6]::real[]);  -- 5.196...
```

Full setup notes (toolchain, the WSL2 native-filesystem loop, and parallel
development) live in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Development

Setup and build: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Design and
rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/FILTERING.md](docs/FILTERING.md). Code conventions:
[docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md).

## License

PostgreSQL License (matches the pgvector ecosystem). See `LICENSE`.
