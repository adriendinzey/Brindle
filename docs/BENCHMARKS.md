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
| Commit | `ecbc16c` |
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
| 1 (control) | 0.15 ms | 0.25 ms | — |
| 16 | 0.23 ms | 0.30 ms | 0.814 |
| 32 | 0.27 ms | 0.46 ms | 0.932 |
| 64 | 0.34 ms | 0.48 ms | 0.966 |
| 128 | 0.44 ms | 0.57 ms | 0.974 |
| 256 | 0.56 ms | 0.83 ms | 0.974 |

**These are warm figures, and the distinction now matters.** A backend keeps one
decoded copy of the index, so the first scan pays to read and decode it and the
sweep's 600 queries are answered from that copy. A backend that has just
connected, or one whose copy a writer invalidated, pays **57.9 ms** (p95
65.7 ms) — measured in the same run, and unchanged by any of this. Before the
cache, every query paid it.

Build: **106.0 s**, observed 112.5–116.3 s across runs on this machine — read
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
| 1 (control) | 0.17 ms | 0.25 ms | — |
| 16 | 0.35 ms | 0.48 ms | 0.146 |
| 32 | 0.46 ms | 0.61 ms | 0.249 |
| 64 | 0.75 ms | 0.91 ms | 0.375 |
| 128 | 1.27 ms | 1.56 ms | 0.507 |
| 256 | 2.22 ms | 2.66 ms | 0.663 |

Warm, as above; cold is **59.0 ms** (p95 69.4 ms).

Build: **224.9 s**. Index: **77 MB**.

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
strided by `dim`, neighbor lists in fixed-width slots) is where the latency win
comes from: decoding a 100k × 128 graph goes from 43.9 ms to 28.3 ms, measured
through the same bench on both shapes. A second change stopped assembling the
whole 80 MB blob before decoding, reading the pages in place instead — which
also removed a second copy of every page on the way out.

**Recall did not move by a digit through either change**, which is the point: the
graph is identical, only its representation and its route into memory changed.

An earlier draft of this section claimed the flat layout *alone* made the tail
nearly twice as bad and that the page-streaming change repaired it. **That does
not reproduce** — repeated runs of the intermediate commit put its p95 in line
with the finished branch, and the single measurement behind the claim was an
outlier this file had no business publishing under its own rule about single
runs. What the streaming change is worth is not latency: it removes an 80 MB
allocation from every scan, which is peak backend memory rather than wall clock,
and it is the page-at-a-time shape paged storage needs.

Absolute latency on this machine also moves several percent between sessions —
the same `main` commit measured 82–83 ms here and 92–93 ms in an independent
reviewer's run — so compare figures within a run set, never across them.

**A backend now keeps one decoded copy of the index**, so that ~60 ms is paid
once per backend rather than once per query, and a warm query costs 0.34 ms.
Read the rest of this file with that split in mind: the tables are warm, and the
cold figure beside them is what the first scan in a backend still costs.

The conclusion the first baseline drew is unchanged, only relocated. The whole
index is still read and decoded to answer a query that has no copy to hand — the
cache moves when that happens, not whether. It is also **per backend**, bounded
by `brindle.cache_max_mb` and invisible to every other connection, where paged
storage would put the same working set in the buffer manager once for the whole
server. Trading memory nobody can share for latency is a good trade at one
connection and a worse one at a hundred. How that splits between reading
10 000 pages through the buffer manager and this file's own decode has not been
measured, and should be before anyone sizes the work to remove it.

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
| 16 | +0.08 ms | +0.17 ms |
| 32 | +0.12 ms | +0.30 ms |
| 64 | +0.19 ms | +0.57 ms |
| 128 | +0.28 ms | +1.09 ms |
| 256 | +0.41 ms | +2.04 ms |

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
extra work is real but small.

The conclusion needs none of that detail: at the widest budget measured, search
is now **most of a warm query** — 0.41 of 0.56 ms clustered, 2.04 of 2.22 ms
uniform. That is the shape a query should have, and it is what was hidden while
every scan first rebuilt the index.

What has not gone anywhere is the ~58 ms a scan costs with no copy to hand. It
is off the warm path, not out of the system: it is paid by every newly connected
backend and again whenever a writer invalidates a copy. That remains the case
for paged storage (`docs/STORAGE.md`), which removes it for everyone rather than
for whoever has queried recently.

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
| 16 | 0.23 ms | 0.50 ms | 0.814 | 0.825 |
| 32 | 0.27 ms | 0.49 ms | 0.932 | 0.940 |
| 64 | 0.34 ms | 0.60 ms | 0.966 | 0.978 |
| 128 | 0.44 ms | 0.79 ms | 0.974 | 0.985 |
| 256 | 0.56 ms | 1.07 ms | 0.974 | 0.985 |

| | Brindle | pgvector |
|---|---|---|
| build, parallel (as shipped) | — | 17.0 s |
| build, `max_parallel_maintenance_workers = 0` | 106.0 s | 39.2 s |
| index size | 77 MB | 79 MB |

**Latency: two orders of magnitude, and the ratio is not a constant.** It is
on the warm path Brindle is now **faster**: 0.34 ms against 0.60 ms at
`ef_search = 64`, and ahead at every sweep point. Both sides are answering from
memory, so this is a fair comparison of like with like — with one asymmetry that
belongs in the headline rather than a footnote. **pgvector's working set is in
the shared buffer cache, sized once for the server; Brindle's is a private copy
per backend**, about 89 MB for this index. Brindle is faster here partly by
spending memory pgvector does not.

Cold, Brindle is 57.9 ms against the same 0.60 ms — **96× slower**. Which figure
is the real one depends entirely on whether your connections are long-lived.
The bar this project set itself for a first queryable index was "the same order
of magnitude, not necessarily faster yet", and this misses it by two.

**Build: ~2.75× slower, not ~6.5×.** pgvector builds in parallel by default and
Brindle is single-threaded, so comparing defaults compares a three-backend
build against a one-backend one. At matched single-threadedness it is 114.0 s
against 41.4 s. (The serial setting was verified to take effect rather than
assumed — an out-of-band check, not part of either run above: the build shows
zero parallel workers in `pg_stat_activity` where the default run shows two.)

**Recall is within about a point** — 0.966 against 0.976 at `ef_search = 64` in
this run. Read the next section before quoting that gap as a value.

### Why the pgvector recall figures are one sample and Brindle's are not

Brindle's build takes a fixed RNG seed, and its recall is identical to three
decimals across every run of this benchmark. pgvector's build is randomised,
and its recall moves between builds. Across the twelve clustered builds run
during this work:

| `ef_search` | Brindle (every run) | pgvector (observed across 12 builds) |
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
| 16 | 0.146 | 0.151 |
| 32 | 0.249 | 0.235 |
| 64 | 0.375 | 0.352 |
| 128 | 0.507 | 0.503 |
| 256 | 0.663 | 0.675 |

A mature implementation scores the same 0.15 as Brindle does on uniform 128-
dimensional data. That is the strongest available evidence that the number
measures the dataset and not the index — and it is why the recall figures in
this file should never be quoted without the fixture they came from.

Both sides also build ~2× slower on this fixture than on the clustered one
(Brindle 236.7 s against 114.0 s; pgvector 83.4 s serial against 41.4 s):
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
  measured separately from the two runs above. That is roughly 5% of Brindle's
  0.34 ms and 3% of pgvector's 0.60 ms, so the warm comparison above is if
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

- **Paged storage** is now quantified rather than asserted: search is a low
  the dominant cost of a *cold* scan at ~58 ms, which a per-backend cache moves
  off the warm path without removing — and which paged storage would remove for
  the whole server rather than per connection.
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
