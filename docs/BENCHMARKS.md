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
SHAPE=clustered scripts/bench_index.sh              # the headline numbers
scripts/bench_index.sh                              # uniform fixture
SHAPE=clustered PGVECTOR=1 scripts/bench_index.sh   # plus the pgvector comparison
```

One command per row of results, from a clean worktree. It builds the extension in
release mode, starts the worktree's own Postgres, generates the data, and prints
every table below. `PGVECTOR=1` additionally runs the comparison, and needs
pgvector installed into the same PostgreSQL (`make && make install` with
`PG_CONFIG` pointing at the pgrx install). Knobs (`ROWS`, `DIMS`, `QUERIES`, `K`, `SHAPE`, `PG`) are documented at the
top of the script.

Nothing here runs in CI. Timing on a shared runner is noise, and no number in
this file should ever gate a merge.

## What was measured

| | |
|---|---|
| Commit | `4676d42` (this task's harness, on top of `0fe1a16`) |
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
| 1 (control) | 85.6 ms | 87.7 ms | — |
| 16 | 84.5 ms | 88.2 ms | 0.814 |
| 32 | 84.3 ms | 86.6 ms | 0.932 |
| 64 | 85.1 ms | 92.7 ms | 0.966 |
| 128 | 84.4 ms | 88.2 ms | 0.974 |
| 256 | 84.6 ms | 87.0 ms | 0.974 |

Build: **110.6 – 113.6 s** across runs. Index: **77 MB** (heap: 56 MB).

`ef_search = 1` is a control, not an operating point: at a budget below `k` the
scan returns fewer than `k` rows, so it reports timing only and no recall. Its
latency is what a scan costs when the search does nothing — see the storage
finding below.

Recall reproduces to three decimals on every run; latency moves a percent or two
between runs, which is why it is quoted to the nearest tenth and not further.

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
index.** Realistic recall against public datasets with published ground truth
belongs to the ann-benchmarks-style harness, not to this file.

## What these numbers actually say

**Latency is not search cost.** It is flat at ~86 ms across a 16× range of
`ef_search`, on both fixtures. Search is invisible next to the per-scan reload of
the whole 77 MB serialized graph — the interim storage format deserializes the
entire index on every scan. This is the concrete case for paged storage
(`docs/STORAGE.md`), and the number that should move when it lands.

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
almost all storage — measured, not inferred. On recall the two are within a
couple of points, and on the uniform fixture they are indistinguishable. See the
comparison below before drawing any conclusion about the algorithm.

## Comparison with pgvector

pgvector 0.8.0, built against the same PostgreSQL, indexing **the same rows**,
answering **the same queries**, scored against **the same ground truth**, at
matched `m = 16` and `ef_construction = 64`.

### The measurement is not single-run, because one side is not deterministic

Brindle's build uses a fixed RNG seed: its recall is 0.966 at `ef = 64` on every
run, to three decimals, across six runs by two people. pgvector's build is
randomised, and its recall moves between builds:

| `ef_search` | Brindle (6 runs) | pgvector (6 builds) |
|---|---|---|
| 16 | 0.814 | 0.807 – 0.833 |
| 32 | 0.932 | 0.934 – 0.954 |
| 64 | **0.966** | **0.969 – 0.990** |
| 128 | 0.974 | 0.977 – 0.995 |
| 256 | 0.974 | 0.977 – 0.995 |

So the recall gap at `ef = 64` is somewhere between **0.003 and 0.024**, and any
single-run figure for pgvector — including a flattering one — is not a result.
An earlier version of this document quoted the top of that range as though it
were the value; it was one run, and it did not reproduce.

### Latency and build

Clustered fixture, `ef = 64`:

| | Brindle | pgvector |
|---|---|---|
| p50 | 86.6 ms | 0.59 – 0.65 ms |
| p95 | 89.1 ms | 0.66 – 0.76 ms |
| build, parallel (as shipped) | — | 15.1 – 16.9 s |
| build, `max_parallel_maintenance_workers = 0` | 112.6 s | 37.8 – 38.3 s |
| index size | 77 MB | 79 MB |

**Latency: Brindle is ~130× slower.** M2's exit criterion asked for "the same
order of magnitude, not necessarily faster yet", and latency misses that by two.

**Build: ~3× slower, not 7.5×.** pgvector builds in parallel by default and
Brindle is single-threaded, so comparing defaults compares a three-backend build
against a one-backend one. At matched single-threadedness it is 112.6 s against
37.8 s.

### Where the latency gap comes from

Almost entirely storage, and the sweep now measures this rather than inferring it
from flatness. At `ef_search = 1` the search does essentially nothing, so what
remains is the fixed per-scan cost:

| | p50 |
|---|---|
| `ef_search = 1` (control) | 85.6 ms |
| `ef_search = 256` | 86.2 ms |

**Search costs under a millisecond; the other ~85 ms is deserializing the 77 MB
graph on every scan.** pgvector's latency tracks `ef_search` the way search cost
should (0.40 ms → 1.05 ms across the same sweep) because it reads pages through
the buffer manager. Paged storage (`docs/STORAGE.md`) is the fix, and this row is
the reason it is the next architectural priority.

The residual — 2 points of recall at most, possibly a third of one — is
neighbour-selection quality, and worth its own investigation rather than a
footnote here.

### The control that makes the fixture point

Both implementations, same uniform fixture, same queries:

| `ef_search` | Brindle | pgvector |
|---|---|---|
| 16 | 0.146 | 0.147 |
| 32 | 0.249 | 0.239 |
| 64 | 0.375 | 0.374 |
| 128 | 0.507 | 0.528 |
| 256 | 0.663 | 0.675 |

A mature implementation scores the same 0.15 as Brindle does on uniform 128-
dimensional data. That is the strongest available evidence that the number
measures the dataset and not the index — and it is why the recall figures in this
file should never be quoted without the fixture they came from.

### Ways this comparison is still not apples to apples

Recorded because a comparison whose asymmetries are hidden is worse than none.

- **SIMD.** pgvector's Makefile compiles with `-march=native -ftree-vectorize
  -fassociative-math`; Brindle is a plain `cargo build --release` at baseline
  x86-64. pgvector's distance kernels get machine-specific vectorization that
  Brindle's do not — which flatters pgvector on distance computation, though
  given the storage finding above, that is not where Brindle's time goes.
- **`maintenance_work_mem`.** pgvector builds inside it and warned it had spilled
  at the default, so its builds here were given 2 GB. That affects graph quality,
  not just build speed, so it moves the recall column too. Brindle ignores the
  setting entirely and allocates in backend memory — a defect on Brindle's side,
  not an advantage.
- **Harness overhead.** The plpgsql timing loop has a floor of about 0.016 ms.
  That is 0.02% of Brindle's 86 ms and roughly 3% of pgvector's sub-millisecond
  numbers, so the ratio above is if anything conservative.
- **Warm cache throughout.** Both sides. A cold-cache comparison would be more
  informative about storage, and becomes meaningful once Brindle's is paged.

## What this baseline does not show

- Nothing about **real embeddings**. Synthetic vectors on one machine; public
  datasets with published ground truth are a separate piece of work.
- Nothing about **filtered search**, which is Brindle's actual differentiator,
  and which has its own selectivity benchmark ahead of it.
- Nothing about **concurrency**. Single session, single query at a time.
- Nothing about **cold cache**. Everything here is warm; the first query after a
  restart pays more.
- No **tuning**. Defaults throughout, deliberately: a baseline optimized against
  is not a baseline.

## Follow-ups this baseline argues for

- **Paged storage (M4)** is now quantified, not asserted: ~86 ms of every query
  is graph deserialization, and it is the whole latency gap against pgvector.
- **Neighbour selection** costs somewhere between 0.003 and 0.024 of recall at
  matched build parameters — the range, because pgvector's randomised build moves
  between runs. Worth its own investigation, sized honestly.
- **`maintenance_work_mem` is ignored by the build.** Not a tuning knob, a
  missing behaviour: the build allocates without bound in backend memory.
- **A cold-cache number** would be more honest than these warm ones for the
  storage comparison, and impossible to fake once storage is paged.
