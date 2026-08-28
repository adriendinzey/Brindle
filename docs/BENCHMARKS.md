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
SHAPE=clustered PGVECTOR=1 scripts/bench_index.sh   # clustered fixture + comparison
PGVECTOR=1 scripts/bench_index.sh                   # uniform fixture + comparison
```

**Those two commands produced every number in this file, and nothing here is
stitched together from any other run.** Each builds the extension in release
mode, starts the worktree's own Postgres, generates the data, and prints the
tables below. Dropping `PGVECTOR=1` runs the same measurement without the
comparison; keeping it needs pgvector installed into the same PostgreSQL
(`make && make install` with `PG_CONFIG` pointing at the pgrx install). Knobs
(`ROWS`, `DIMS`, `QUERIES`, `K`, `SHAPE`, `PG`) are documented at the top of the
script.

That "one run set" rule is not pedantry. An earlier draft of this file quoted
build times and latencies from three different runs under a single commit stamp,
and the figures contradicted each other — the same configuration appeared as
110.6 s, 112.6 s and 113.6 s in three places, and a p50 appeared as both 85.1 ms
and 86.6 ms. Every table below comes from the two runs named above.

Nothing here runs in CI. Timing on a shared runner is noise, and no number in
this file should ever gate a merge.

## What was measured

| | |
|---|---|
| Commit | `ed958c7` |
| Machine | Intel i7-9700K @ 3.6 GHz, 8 cores, 15 GB RAM, WSL2 (kernel 6.18) |
| PostgreSQL | 17.10, pgrx-managed, default `postgresql.conf` |
| Build | `cargo build --release`, no `target-cpu` flag (baseline x86-64) |
| Dataset | 100 000 rows × 128 dimensions |
| Queries | 100 query vectors per `ef_search` point, k = 10 |
| Index params | defaults: `m = 16`, `ef_construction = 64` |

The `ef_search` sweep starts at 16 rather than 10 on purpose: a scan returns at
most `ef_search` rows, so a sweep point below `k` would measure that ceiling
instead of the graph. The one point below `k` — `ef_search = 1` — is a latency
control that deliberately reports no recall, because a scan returning one row
has recall@10 ≤ 0.100 by construction.

## Results

### Clustered fixture — the one worth comparing against later

100 centroids with local noise, which is roughly how real embeddings sit:
locally dense, globally separated.

| `ef_search` | p50 | p95 | recall@10 |
|---|---|---|---|
| 1 (control) | 82.3 ms | 87.6 ms | — |
| 16 | 82.9 ms | 86.8 ms | 0.814 |
| 32 | 83.1 ms | 87.9 ms | 0.932 |
| 64 | 83.0 ms | 87.3 ms | 0.966 |
| 128 | 83.0 ms | 87.0 ms | 0.974 |
| 256 | 83.4 ms | 88.2 ms | 0.974 |

Build: **114.9 s**. Index: **77 MB** (heap: 56 MB).

Recall reproduces to three decimals on every run — the build takes a fixed RNG
seed. Latency moves a percent or two between runs, which is why it is quoted to
the nearest tenth and no further, and why the search-cost table below is
measured as a paired difference rather than read off this one.

### Uniform fixture — the worst case, and a lesson

Every component independent and uniform.

| `ef_search` | p50 | p95 | recall@10 |
|---|---|---|---|
| 1 (control) | 84.5 ms | 88.9 ms | — |
| 16 | 84.7 ms | 88.5 ms | 0.146 |
| 32 | 85.0 ms | 89.2 ms | 0.249 |
| 64 | 85.2 ms | 88.8 ms | 0.375 |
| 128 | 85.6 ms | 89.1 ms | 0.507 |
| 256 | 86.5 ms | 90.0 ms | 0.663 |

Build: **240.9 s**. Index: **77 MB**.

Recall of 0.15 looks like a broken index. It isn't. In 128 uniform dimensions
distances concentrate: the benchmark measures it directly, and the 1000th
nearest neighbour sits at **1.117×** the distance of the 1st (0.961 → 1.332 on
the clustered fixture, a ratio of **1.386**). A greedy graph walk has almost no
gradient to follow, so no amount of `ef_search` rescues it — and the pgvector
control below scores the same on the same data.

This is the number to hold onto when reading anyone's ANN benchmark, including
this one: **recall is a property of the dataset at least as much as of the
index.** Realistic recall against public datasets with published ground truth
belongs to the ann-benchmarks-style harness, not to this file.

## What the search itself costs

Total latency cannot answer this. The whole sweep spans under two milliseconds
while run-to-run drift is a couple of milliseconds, so comparing p50 columns
across `ef_search` points reads noise — in an earlier draft it read *negative*,
reporting the control as slower than the widest search, which is how the mistake
was caught.

The measurement that works pairs each query against **its own** timing at
`ef_search = 1`, so the fixed per-scan cost cancels, and interleaves the sweep
points inside the query loop so no point gets a colder cache than another. A
warm-up of five real queries runs first; the plan guard uses `EXPLAIN` without
`ANALYZE` and therefore executes nothing.

Median extra milliseconds over that same query's `ef_search = 1`:

| `ef_search` | clustered | uniform |
|---|---|---|
| 16 | +0.35 ms | +0.06 ms |
| 32 | +0.66 ms | +0.43 ms |
| 64 | +0.64 ms | +0.75 ms |
| 128 | +0.61 ms | +1.21 ms |
| 256 | +0.91 ms | +1.97 ms |

**At the widest budget measured, search is 1.1% of a query on the clustered
fixture and 2.3% on the uniform one — and less than that at every narrower
budget.** The remaining 82–85 ms is deserializing the whole 77 MB graph, which
the interim storage format does on every single scan. That is the
concrete case for paged storage (`docs/STORAGE.md`), and the number that should
move when it lands.

The two columns differ in a way worth noting: on uniform data the cost doubles
as the budget doubles, because the walk keeps exploring and spends everything it
is given. On clustered data it is nearly flat from 32 to 128 — the walk
converges early and leaves budget unused, which is the same fact the recall
column reports when it plateaus at 0.974.

## Comparison with pgvector

pgvector 0.8.0, built against the same PostgreSQL, indexing **the same rows**,
answering **the same queries**, scored against **the same ground truth**, at
matched `m = 16` and `ef_construction = 64`.

### Clustered fixture

| `ef_search` | Brindle p50 | pgvector p50 | Brindle recall | pgvector recall |
|---|---|---|---|---|
| 16 | 82.9 ms | 0.41 ms | 0.814 | 0.818 |
| 32 | 83.1 ms | 0.47 ms | 0.932 | 0.927 |
| 64 | 83.0 ms | 0.61 ms | 0.966 | 0.969 |
| 128 | 83.0 ms | 0.79 ms | 0.974 | 0.976 |
| 256 | 83.4 ms | 1.04 ms | 0.974 | 0.976 |

| | Brindle | pgvector |
|---|---|---|
| build, parallel (as shipped) | — | 16.7 s |
| build, `max_parallel_maintenance_workers = 0` | 114.9 s | 38.4 s |
| index size | 77 MB | 79 MB |

**Latency: two orders of magnitude, and the ratio is not a constant.** It is
202× at `ef_search = 16` and 80× at 256 — it *narrows* as the budget grows,
because pgvector's cost scales with the search while Brindle's is dominated by a
fixed 82 ms that the budget does not touch. At `ef_search = 64` it is 136×. M2's
exit criterion asked for "the same order of magnitude, not necessarily faster
yet", and this misses it by two.

**Build: ~3× slower, not ~7×.** pgvector builds in parallel by default and
Brindle is single-threaded, so comparing defaults compares a three-backend build
against a one-backend one. At matched single-threadedness it is 114.9 s against
38.4 s. (The serial setting was verified to take effect, not assumed: the build
runs with zero parallel workers where the default run shows two.)

**Recall is within a few thousandths** — 0.966 against 0.969 at `ef_search = 64`
in this run. Read the next section before quoting that gap as a value.

### Why the pgvector recall figures are one sample and Brindle's are not

Brindle's build takes a fixed RNG seed, and its recall is identical to three
decimals across every run. pgvector's build is randomised, and its recall moves
between builds. Across the eight clustered builds run during this work:

| `ef_search` | Brindle (every run) | pgvector (observed across 8 builds) |
|---|---|---|
| 16 | 0.814 | 0.807 – 0.834 |
| 32 | 0.932 | 0.927 – 0.954 |
| 64 | **0.966** | **0.969 – 0.990** |
| 128 | 0.974 | 0.976 – 0.995 |
| 256 | 0.974 | 0.976 – 0.995 |

**Those are observed minima and maxima over a sample, not bounds.** The
distinction is not theoretical: a six-build range published in an earlier draft
was escaped by the seventh build at one point and by the eighth — the run in the
table above — at three of five points. Expect the next build to fall outside
this range too.

So the recall gap at `ef_search = 64` is somewhere around **0.003 to 0.024**, and
any single-run figure for pgvector, including the one in the comparison table, is
one sample rather than the value. An earlier version of this document quoted the
flattering end of that range as though it were the result.

### The control that makes the fixture point

Both implementations, same uniform fixture, same queries, same ground truth:

| `ef_search` | Brindle | pgvector |
|---|---|---|
| 16 | 0.146 | 0.141 |
| 32 | 0.249 | 0.271 |
| 64 | 0.375 | 0.391 |
| 128 | 0.507 | 0.528 |
| 256 | 0.663 | 0.690 |

A mature implementation scores the same 0.15 as Brindle does on uniform 128-
dimensional data. That is the strongest available evidence that the number
measures the dataset and not the index — and it is why the recall figures in this
file should never be quoted without the fixture they came from.

Both sides also build ~2× slower on this fixture than on the clustered one
(Brindle 240.9 s against 114.9 s; pgvector 72.5 s serial against 38.4 s):
structure makes neighbour selection converge faster, for both implementations.

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
  That is 0.02% of Brindle's 83 ms but roughly 4% of pgvector's 0.41 ms, so the
  latency ratios above are if anything conservative. (It cancels entirely from
  the paired search-cost table, which is one more reason to prefer it.)
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

- **Paged storage (M4)** is now quantified rather than asserted: search accounts
  for at most 2.3% of a query and the remaining 82–85 ms is graph
  deserialization, which is the entire latency gap against pgvector.
- **Recall plateaus at 0.974 and more budget does not help.** Between
  `ef_search` 128 and 256 recall does not move and the search barely spends more
  time. That ceiling is build quality, not search effort: at
  `ef_construction = 64` the graph does not contain the edges a wider search
  would need. Raising build parameters is the lever, and measuring that trade is
  not in this baseline's scope.
- **Neighbour selection** costs somewhere between 0.003 and 0.024 of recall at
  matched build parameters — a range, because pgvector's randomised build moves
  between runs. Worth its own investigation, sized honestly.
- **`maintenance_work_mem` is ignored by the build.** Not a tuning knob, a
  missing behaviour: the build allocates without bound in backend memory.
- **A cold-cache number** would be more honest than these warm ones for the
  storage comparison, and impossible to fake once storage is paged.
