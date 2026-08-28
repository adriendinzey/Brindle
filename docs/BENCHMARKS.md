# Brindle — Performance Baseline

The first honest numbers for the index: how long it takes to build, how long a
query takes, and how much of the true answer it finds. They exist so that later
work — durable paged storage, quantization, filtered traversal — has a *before*
to be measured against.

Read the caveats before quoting anything here. Two of the numbers below say more
about the fixture than about Brindle, and one of them looks alarming until you
know why.

## How to reproduce

```bash
scripts/bench_index.sh                     # uniform fixture
SHAPE=clustered scripts/bench_index.sh     # clustered fixture
```

One command, from a clean worktree. It builds the extension in release mode,
starts the worktree's own Postgres, generates the data, and prints every table
below. Knobs (`ROWS`, `DIMS`, `QUERIES`, `K`, `SHAPE`, `PG`) are documented at the
top of the script.

Nothing here runs in CI. Timing on a shared runner is noise, and no number in
this file should ever gate a merge.

## What was measured

| | |
|---|---|
| Commit | `0fe1a16` (plus this task's benchmark harness) |
| Machine | Intel i7-9700K @ 3.6 GHz, 8 cores, 15 GB RAM, WSL2 (kernel 6.18) |
| PostgreSQL | 17.10, pgrx-managed, default `postgresql.conf` |
| Build | `cargo build --release`, no `target-cpu` flag (baseline x86-64) |
| Dataset | 100 000 rows × 128 dimensions |
| Queries | 100 query vectors per `ef_search` point, k = 10 |
| Index params | defaults: `m = 16`, `ef_construction = 64` |

The `ef_search` sweep starts at 16 rather than 10 on purpose: a scan returns at
most `ef_search` rows, so a sweep point below `k` would measure that ceiling
instead of the graph.

## Results

### Clustered fixture — the one worth comparing against later

100 centroids with local noise, which is roughly how real embeddings sit:
locally dense, globally separated.

| `ef_search` | p50 | p95 | recall@10 |
|---|---|---|---|
| 16 | 86.4 ms | 93.1 ms | 0.814 |
| 32 | 86.1 ms | 88.1 ms | 0.932 |
| 64 | 86.6 ms | 89.1 ms | 0.966 |
| 128 | 85.5 ms | 87.6 ms | 0.974 |
| 256 | 85.7 ms | 88.4 ms | 0.974 |

Build: **112.6 s**. Index: **77 MB** (heap: 56 MB).

### Uniform fixture — the worst case, and a lesson

Every component independent and uniform.

| `ef_search` | p50 | p95 | recall@10 |
|---|---|---|---|
| 16 | 85.2 ms | 88.3 ms | 0.146 |
| 32 | 85.2 ms | 87.5 ms | 0.249 |
| 64 | 85.5 ms | 87.5 ms | 0.375 |
| 128 | 85.9 ms | 87.5 ms | 0.507 |
| 256 | 87.5 ms | 95.4 ms | 0.663 |

Build: **244.4 s**. Index: **77 MB**.

Recall of 0.15 looks like a broken index. It isn't. In 128 uniform dimensions
distances concentrate: the benchmark measures it directly, and the 1000th
nearest neighbour sits at **1.117×** the distance of the 1st. A greedy graph walk
has almost no gradient to follow, so no amount of `ef_search` rescues it. The
clustered fixture's ratio is 1.386, and its recall is 0.97.

This is the number to hold onto when reading anyone's ANN benchmark, including
this one: **recall is a property of the dataset at least as much as of the
index.** Realistic recall against public datasets with published ground truth is
T-062's job, not this file's.

## What these numbers actually say

**Latency is not search cost.** It is flat at ~86 ms across a 16× range of
`ef_search`, on both fixtures. Search is invisible next to the per-scan reload of
the whole 77 MB serialized graph — the interim storage format deserializes the
entire index on every scan. This is the concrete case for paged storage
(`docs/STORAGE.md`, M4), and the number that should move when it lands.

**Recall plateaus at 0.974, and more budget does not help.** Between
`ef_search` 128 and 256 recall does not move. That ceiling is build quality, not
search effort: at `ef_construction = 64` the graph simply does not contain the
edges a wider search would need. Raising build parameters is the lever there, and
measuring that trade is not in this baseline's scope.

**Build is slow and single-threaded.** 112 s for 100k × 128 clustered, 244 s
uniform — structure makes neighbour selection converge faster. Brindle's build
also ignores `maintenance_work_mem` entirely, allocating in backend memory
instead; a production index is expected to respect it, and pgvector does.

**Against pgvector, the gap is two orders of magnitude on latency** and it is
almost all storage. See the comparison below before drawing conclusions about
the algorithm.

## Comparison with pgvector

pgvector 0.8.0, built against the same PostgreSQL, indexing **the same rows**,
answering **the same queries**, scored against **the same ground truth**, at
matched `m = 16` and `ef_construction = 64`. Clustered fixture.

| | Brindle | pgvector | ratio |
|---|---|---|---|
| build | 112.6 s | **15.1 s** | 7.5× slower |
| p50 @ `ef = 64` | 86.6 ms | **0.65 ms** | ~133× slower |
| p95 @ `ef = 64` | 89.1 ms | **0.76 ms** | ~117× slower |
| recall@10 @ `ef = 64` | 0.966 | **0.990** | −0.024 |
| recall@10 @ `ef = 256` | 0.974 | **0.995** | −0.021 |
| index size | **77 MB** | 79 MB | ≈ equal |

pgvector's full curve, for reference:

| `ef_search` | p50 | p95 | recall@10 |
|---|---|---|---|
| 16 | 0.42 ms | 0.55 ms | 0.819 |
| 32 | 0.47 ms | 0.57 ms | 0.954 |
| 64 | 0.65 ms | 0.76 ms | 0.990 |
| 128 | 0.83 ms | 1.00 ms | 0.995 |
| 256 | 1.05 ms | 1.17 ms | 0.995 |

M2's exit criterion asked for "the same order of magnitude, not necessarily
faster yet". **On latency Brindle misses that by two orders of magnitude**, and
that is the honest headline of this baseline.

Where the gap comes from, in order of size:

1. **Storage, almost entirely.** Brindle's latency is flat at ~86 ms whatever
   `ef_search` is, because every scan deserializes the entire 77 MB graph before
   it searches. pgvector reads pages through the buffer manager and its latency
   tracks `ef_search` the way search cost should (0.42 ms → 1.05 ms across the
   sweep). Brindle's *search* is somewhere inside that flat 86 ms and this
   baseline cannot separate it. Paged storage (M4) is the fix, and this row is
   the reason it is the next architectural priority.
2. **Graph quality, modestly.** At matched build parameters pgvector reaches
   0.990 where Brindle reaches 0.966, and 0.995 where Brindle plateaus at 0.974.
   Same `m`, same `ef_construction`, so this is neighbour-selection: pgvector
   prunes candidates with the standard heuristic, and Brindle's selection is
   simpler. Two points of recall, worth a task of its own rather than a
   footnote.
3. **Build time**, 7.5×, with the same caveat as the latency: Brindle rewrites a
   whole serialized image where pgvector writes pages.

### Ways this comparison is not apples to apples

Recorded because a comparison whose asymmetries are hidden is worse than none.

- **SIMD.** pgvector's Makefile compiles with `-march=native -ftree-vectorize
  -fassociative-math`; Brindle is a plain `cargo build --release` at baseline
  x86-64 with no `target-cpu`. pgvector's distance kernels get machine-specific
  vectorization that Brindle's do not. This flatters pgvector on the parts of the
  work that are distance computation — which, given point 1, is not where
  Brindle's time is going.
- **`maintenance_work_mem`.** pgvector builds inside it and warned that it had
  spilled at the default, so the build above was run with 2 GB. Brindle ignores
  the setting entirely and allocates in backend memory — which is itself a defect
  rather than an advantage, and one a production index would not have.
- **Both are the same version's defaults**, not tuned. Neither side got a
  favourable knob.

## What this baseline does not show

- Nothing about **real embeddings**. Synthetic vectors on one machine. Public
  datasets with published ground truth are T-062.
- Nothing about **filtered search**, which is Brindle's actual differentiator.
  That is T-034's measurement.
- Nothing about **concurrency**. Single session, single query at a time.
- Nothing about **cold cache**. Everything here is warm; the first query after a
  restart pays more.
- No **tuning**. Defaults throughout, deliberately: a baseline optimized against
  is not a baseline.

## Follow-ups this baseline argues for

- **Paged storage (M4)** is now quantified, not asserted: ~86 ms of every query
  is graph deserialization, and it is the whole latency gap against pgvector.
- **Neighbour selection** costs ~2 points of recall at matched build parameters.
  Worth its own task rather than a line here.
- **`maintenance_work_mem` is ignored by the build.** Not a tuning knob, a
  missing behaviour: the build allocates without bound in backend memory.
- **A cold-cache number** would be more honest than these warm ones for the
  storage comparison, and impossible to fake once storage is paged.
