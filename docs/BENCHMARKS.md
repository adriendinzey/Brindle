# Brindle — Performance Baseline

The first honest numbers for the index: how long it takes to build, how long a
query takes, and how much of the true answer it finds. They exist so that later
work — durable paged storage, quantization, filtered traversal — has a *before*
to be measured against.

Read the caveats before quoting anything here. Two of the numbers below say
more about the fixture than about Brindle, and one of them looks alarming until
you know why.

## How to reproduce

```bash
SHAPE=clustered PGVECTOR=1 scripts/bench_index.sh   # clustered + comparison
PGVECTOR=1 scripts/bench_index.sh                   # uniform + comparison
```

**Every measurement in this file comes from those two runs.** A few figures are
deliberately drawn from outside them, and each says so where it appears: the
pgvector recall table across repeated builds, which is multi-run by construction
because its subject
is how far a randomised build moves; the run-to-run ranges quoted for build
time and for the paired search-cost medians, which exist precisely to show how
far those move; the plpgsql timing floor of ~0.016 ms; and the check that
`max_parallel_maintenance_workers = 0` really took effect, an out-of-band look
at `pg_stat_activity`. No figure is silently carried in from a run this file
does not name.

Each command builds the extension in release mode, starts the worktree's own
Postgres, generates the data, and prints the tables below. Dropping
`PGVECTOR=1` runs the same measurement without the comparison; keeping it needs
pgvector installed into the same PostgreSQL (`make && make install` with
`PG_CONFIG` pointing at the pgrx install). Knobs (`ROWS`, `DIMS`, `QUERIES`,
`K`, `SHAPE`, `PG`) are documented at the top of the script.

That "one run set" rule is not pedantry. An earlier draft of this file quoted
build times and latencies from three different runs under a single commit
stamp, and the figures contradicted each other — the same configuration
appeared as 110.6 s, 112.6 s and 113.6 s in three places, and a p50 appeared as
both 85.1 ms and 86.6 ms. Every measurement below comes from the two runs named
above, and the handful of figures that do not are labelled where they appear.

Nothing here runs in CI. Timing on a shared runner is noise, and no number in
this file should ever gate a merge.

## What was measured

| | |
|---|---|
| Commit | `65660ed` |
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
| 1 (control) | 62.5 ms | 74.7 ms | — |
| 16 | 62.4 ms | 80.4 ms | 0.814 |
| 32 | 62.8 ms | 73.8 ms | 0.932 |
| 64 | 62.9 ms | 71.3 ms | 0.966 |
| 128 | 62.9 ms | 68.3 ms | 0.974 |
| 256 | 63.0 ms | 75.5 ms | 0.974 |

Build: **113.1 s**, observed 112.5–116.3 s across runs on this machine — read
it as "under two minutes", not to the tenth. Index: **77 MB** (heap: 56 MB).

Recall reproduces to three decimals on every run — the build takes a fixed RNG
seed, and this column has been identical across three separate measurement
runs. Latency moves a percent or two between runs, which is why it is quoted to
the nearest tenth and no further, and why the search-cost table below is
measured as a paired difference rather than read off this one.

### Uniform fixture — the worst case, and a lesson

Every component independent and uniform.

| `ef_search` | p50 | p95 | recall@10 |
|---|---|---|---|
| 1 (control) | 64.2 ms | 71.4 ms | — |
| 16 | 64.5 ms | 69.6 ms | 0.146 |
| 32 | 64.4 ms | 81.4 ms | 0.249 |
| 64 | 65.2 ms | 77.9 ms | 0.375 |
| 128 | 65.6 ms | 73.9 ms | 0.507 |
| 256 | 66.7 ms | 75.0 ms | 0.663 |

Build: **242.2 s**. Index: **77 MB**.

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

## What changed since the first baseline

The first version of this file measured 82–83 ms per clustered query. The graph
was then stored as one `Vec` per node's vector plus one per node per layer of
neighbors — roughly 400 000 allocations rebuilt on every scan, because a scan
decodes the whole graph before it can walk it. Storing it flat (one buffer
strided by `dim`, neighbor lists in fixed-width slots) cut the decode from
56–60 ms to 25.9 ms.

Flat storage alone made the **tail worse**, not better: p95 went to 125–181 ms
against the original 87–91 ms. Two large contiguous arrays per scan, on top of
the 80 MB buffer the pages were still being copied into, is a lot of large
allocation per query, and large allocations are erratic in a way a median hides.
Decoding straight from the pages removes that buffer, and with it the
regression — p95 is now 68–80 ms, better than where it started.

Both changes are visible above and neither moved recall by a digit, which is the
point: the graph is identical, only its representation and its route into memory
changed. **The remaining ~62 ms is still the whole index being read and decoded
per scan**, so the conclusion the first baseline drew is unchanged — it is just
62 ms of evidence for paged storage now instead of 82.

## What the search itself costs

Total latency cannot answer this. The clustered sweep spans well under a
millisecond of p50 (the uniform one, 2.8 ms) against run-to-run drift of a
couple of milliseconds either way, so comparing p50 columns across `ef_search`
points reads noise — in an earlier draft it read *negative*, reporting the
control as slower than the widest search, which is how the mistake was caught.

The measurement that works pairs each query against **its own** timing at
`ef_search = 1`, so the fixed per-scan cost cancels, and interleaves the sweep
points inside the query loop so no point gets a colder cache than another. A
warm-up of five real queries runs first; the plan guard uses `EXPLAIN` without
`ANALYZE` and therefore executes nothing.

Median extra milliseconds over that same query's `ef_search = 1`:

| `ef_search` | clustered | uniform |
|---|---|---|
| 16 | +0.02 ms | +0.32 ms |
| 32 | +0.34 ms | +0.40 ms |
| 64 | +0.48 ms | +0.86 ms |
| 128 | +0.58 ms | +1.28 ms |
| 256 | +0.59 ms | +2.43 ms |

**Read these as a scale, not as calibrated per-point values, and do not read
the per-point shape at all.** Pairing removes the fixed cost, which is what
makes a sub-millisecond signal visible; a few tenths of a millisecond of
run-to-run movement survive it. Four runs of this same clustered measurement
gave +0.91, +0.36, +0.55 and +0.59 ms at `ef_search = 256` — one flat across the
sweep, three rising. Any story told about the clustered curve's shape is a story
about which run got written down.

Two things do reproduce. **On the clustered fixture every median point is under
a millisecond**, across all four runs — the p95 is a different story, below.
**On the uniform fixture the cost grows with the budget** — roughly doubling as
the budget doubles, out to +2.47 ms — because in 128 uniform dimensions the
walk has no gradient to converge on and spends everything it is given, and
still reaches only 0.663 recall.

It would be tidy to say the clustered walk converges early and leaves its
budget unused, and the recall column forbids it: recall climbs 0.814 → 0.966
between `ef_search` 16 and 64, so the search is plainly finding more in that
range, not coasting. Convergence is a fair description only from 128 to 256,
where recall stops moving. Below that, the honest statement is narrower — the
extra work is real but too small to measure reliably against a 62 ms constant.

The conclusion needs none of that detail: at the widest budget measured, search
is **around 1% of a query on the clustered fixture and under 4% on the uniform
one**. Quoting it to two significant figures would repeat the mistake the table
above warns about, since the numerator moves between runs. The remaining
62–64 ms is reading and decoding the whole 77 MB graph, which the interim
storage format does on every single scan. That is the concrete case for paged
storage (`docs/STORAGE.md`), and the number that should move further when it
lands.

One caveat the medians hide: the same table's p95 paired deltas run 3–16 ms,
many times the median. The "1% of a query" figure is a statement about the
median query, and the tail is worse.

## Comparison with pgvector

pgvector 0.8.0, built against the same PostgreSQL, indexing **the same rows**,
answering **the same queries**, scored against **the same ground truth**, at
matched `m = 16` and `ef_construction = 64`.

### Clustered fixture

Both sides warmed and interleaved the same way. That matters: an earlier draft
measured pgvector with the block sweep this file calls invalid on the Brindle
side, which makes a comparison partly a comparison of protocols.

| `ef_search` | Brindle p50 | pgvector p50 | Brindle recall | pgvector recall |
|---|---|---|---|---|
| 16 | 62.4 ms | 0.57 ms | 0.814 | 0.815 |
| 32 | 62.8 ms | 0.62 ms | 0.932 | 0.940 |
| 64 | 62.9 ms | 0.73 ms | 0.966 | 0.977 |
| 128 | 62.9 ms | 1.03 ms | 0.974 | 0.983 |
| 256 | 63.0 ms | 1.33 ms | 0.974 | 0.983 |

| | Brindle | pgvector |
|---|---|---|
| build, parallel (as shipped) | — | 19.8 s |
| build, `max_parallel_maintenance_workers = 0` | 113.1 s | 45.4 s |
| index size | 77 MB | 79 MB |

**Latency: two orders of magnitude, and the ratio is not a constant.** It is
109× at `ef_search = 16` and 47× at 256 — it *narrows* as the budget grows,
because pgvector's cost scales with the search while Brindle's is dominated by
a fixed 62 ms that the budget does not touch. At `ef_search = 64` it is 86×.
M2's exit criterion asked for "the same order of magnitude, not necessarily
faster yet", and this misses it by two.

**Build: ~2.5× slower, not ~6×.** pgvector builds in parallel by default and
Brindle is single-threaded, so comparing defaults compares a three-backend
build against a one-backend one. At matched single-threadedness it is 113.1 s
against 45.4 s. (The serial setting was verified to take effect rather than
assumed — an out-of-band check, not part of either run above: the build shows
zero parallel workers in `pg_stat_activity` where the default run shows two.)

**Recall is within about a point** — 0.966 against 0.976 at `ef_search = 64` in
this run. Read the next section before quoting that gap as a value.

### Why the pgvector recall figures are one sample and Brindle's are not

Brindle's build takes a fixed RNG seed, and its recall is identical to three
decimals across every run of this benchmark. pgvector's build is randomised,
and its recall moves between builds. Across the eleven clustered builds run
during this work:

| `ef_search` | Brindle (every run) | pgvector (observed across 9 builds) |
|---|---|---|
| 16 | 0.814 | 0.797 – 0.834 |
| 32 | 0.932 | 0.922 – 0.954 |
| 64 | **0.966** | **0.969 – 0.990** |
| 128 | 0.974 | 0.976 – 0.995 |
| 256 | 0.974 | 0.976 – 0.995 |

**Those are observed minima and maxima over a sample, not bounds.** The
distinction is not theoretical, and it keeps happening: a six-build range
published in an earlier draft was escaped by the seventh build at one point, by
the eighth at three of five, and by the tenth — an independent reviewer's — at
two of five, which is why the low ends above moved again. Expect the next build
to fall outside this range too.

So the recall gap at `ef_search = 64` is somewhere around **0.003 to 0.024**,
and any single-run figure for pgvector — including the 0.976 in the table above
— is one sample rather than the value. An earlier version of this document
quoted the flattering end of that range as though it were the result.

### The control that makes the fixture point

Both implementations, same uniform fixture, same queries, same ground truth:

| `ef_search` | Brindle | pgvector |
|---|---|---|
| 16 | 0.146 | 0.133 |
| 32 | 0.249 | 0.234 |
| 64 | 0.375 | 0.379 |
| 128 | 0.507 | 0.514 |
| 256 | 0.663 | 0.674 |

A mature implementation scores the same 0.15 as Brindle does on uniform 128-
dimensional data. That is the strongest available evidence that the number
measures the dataset and not the index — and it is why the recall figures in
this file should never be quoted without the fixture they came from.

Both sides also build ~2× slower on this fixture than on the clustered one
(Brindle 242.2 s against 113.1 s; pgvector 81.9 s serial against 45.4 s):
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
- **Harness overhead.** The plpgsql timing loop has a floor of about 0.016 ms,
  measured separately from the two runs above. That is 0.02% of Brindle's 82 ms
  but roughly 3% of pgvector's 0.57 ms, so the latency ratios above are if
  anything conservative. (It cancels entirely from
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

- **Paged storage (M4)** is now quantified rather than asserted: search is a low
  single-digit percentage of a query and the remaining 62–64 ms is reading and
  decoding the graph, which is the entire latency gap against pgvector.
- **Recall plateaus at 0.974 and more budget does not help.** Between
  `ef_search` 128 and 256 recall does not move. That ceiling is build quality,
  not search effort: at `ef_construction = 64` the graph does not contain the
  edges a wider search would need. Raising build parameters is the lever, and
  measuring that trade is not in this baseline's scope.
- **Neighbour selection** costs somewhere between 0.003 and 0.024 of recall at
  matched build parameters — a range, because pgvector's randomised build moves
  between runs. Worth its own investigation, sized honestly.
- **`maintenance_work_mem` is ignored by the build.** Not a tuning knob, a
  missing behaviour: the build allocates without bound in backend memory.
- **A cold-cache number** would be more honest than these warm ones for the
  storage comparison, and impossible to fake once storage is paged.
