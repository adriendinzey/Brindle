# Brindle — Architecture

This document explains *what* Brindle is, *why* it exists given the crowded
Postgres-vector landscape, and *how* it's structured internally.

---

## 1. Problem statement

Retrieval for AI applications (RAG, semantic search, recommendations) almost
never asks "find the nearest vectors" in isolation. It asks "find the nearest
vectors **that also satisfy these constraints**":

- multi-tenant isolation (`tenant_id = ?`)
- access control (`acl @> ?`)
- recency / range (`created_at > now() - interval '30 days'`, `price < ?`)
- categorical filters (`status = 'active'`, `lang = 'en'`)

It also frequently wants **hybrid** ranking: combine semantic similarity with
exact keyword/lexical matching (BM25-style), because embeddings miss rare tokens,
IDs, and exact phrases.

These two needs — **filtered ANN** and **hybrid ranking** — are exactly where the
current Postgres ecosystem is weakest. Brindle is built around them.

## 2. Competitive landscape (why Brindle is differentiated)

| Project | Lang | Core strength | Filtering story |
|---|---|---|---|
| **pgvector** | C | The default. HNSW + IVFFlat, native WAL-safe storage | Post-filter; v0.8 *iterative scan* mitigates but trades recall/latency |
| **pgvectorscale** | Rust/pgrx | StreamingDiskANN, Statistical Binary Quantization, disk-friendly | Label-set filtering only (not arbitrary predicates) |
| **VectorChord** (ex-`pgvecto.rs`) | Rust/pgrx | RaBitQ quantization, hierarchical k-means, very high QPS@recall | Speed/quant focused |
| **ParadeDB `pg_search`** | Rust/pgrx | True BM25 via Tantivy, hybrid via RRF | Lexical-first; vector is secondary |

**Takeaway:** the heavyweights compete on speed, memory, and scale. The
*arbitrary-predicate filtered search* problem and *unified hybrid ranking* remain
genuinely under-served in open source. That gap is Brindle's reason to exist — and
it's a tractable, well-bounded target for a focused project rather than a
race against funded teams on quantization.

### The filtered-search problem, concretely

With HNSW and the default `ef_search = 40`, a filter that matches 10% of rows
leaves ~4 usable results on average — the graph search spends its budget visiting
nodes the predicate then discards. Increasing `ef_search` recovers recall but
costs latency super-linearly. Pre-filtering into a bitmap and brute-forcing is
fine for tiny result sets and catastrophic for large ones. The principled fix is
**predicate-aware traversal**, the subject of [FILTERING.md](FILTERING.md).

## 3. System overview

```
                     SQL surface
   CREATE INDEX ... USING brindle (embedding vector_cosine_ops)
   SELECT ... ORDER BY embedding <=> $1 WHERE meta_filter
   SELECT brindle_hybrid(query_text => ?, query_vec => ?, k => 10)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  pgrx binding layer                                          │
│   • #[pg_extern] functions (distance, hybrid, utilities)    │
│   • Index Access Method (IndexAmRoutine)  ── Phase 1+        │
│       ambuild · aminsert · ambeginscan · amgettuple ·       │
│       amrescan · amendscan · ambulkdelete · amvacuumcleanup │
└─────────────────────────────────────────────────────────────┘
        │                  │                    │
        ▼                  ▼                    ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│ Graph engine │  │  Filter layer    │  │  Storage layer       │
│ HNSW + ACORN │  │ predicate eval   │  │ Phase 1: in-memory   │
│ γ-dense edges│◄─┤ during traversal │  │   (rebuilt on load)  │
│ entry-point  │  │ • label bitmaps  │  │ Phase 3+: Postgres   │
│ greedy search│  │ • numeric ranges │  │   buffer pages + WAL │
└──────┬───────┘  └──────────────────┘  └──────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Core (pure Rust, no Postgres deps — unit-testable)         │
│   • vector type / views over f32 (later f16, int8)          │
│   • distance kernels: L2², cosine, inner product (SIMD)     │
│   • quantization: scalar → binary (Phase 5)                 │
│   • RRF fusion                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layering principle

The **core** crate logic (distance, graph algorithms, fusion) is written as
*pure Rust over plain slices*, with no `pgrx`/Postgres types. Only the **binding
layer** touches Postgres. This is deliberate:

- the hard algorithmic code is unit-testable with `cargo test` (fast, no Postgres),
- the SIMD hot loops can be benchmarked in isolation with `criterion`,
- the Postgres-specific surface stays thin and auditable.

## 4. Key technical decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language / framework | **Rust + pgrx** | Same stack as pgvectorscale & VectorChord; memory safety in unsafe DB territory |
| Target Postgres | **16 / 17** (test matrix 14–17) | Current; pgrx supports all |
| Vector type | **Brindle's own `brindle_vector`**, laid out and spelled like pgvector's `vector`; `real[]` also indexable | Zero external dependencies, with interop kept through a matching text form and casts |
| Distance kernels | **Stable-Rust autovectorized loops** now; explicit SIMD (`std::arch`/`wide`) in Phase 5 | Avoid nightly; correctness first, then optimize with benchmarks |
| Index storage | **In-memory first**, native buffer/WAL later | Don't block the interesting algorithms on the hardest systems work |
| Filtering | **Indexed label + numeric-range predicates** first; arbitrary SQL pushdown is the research frontier | Achievable and honest; still a real improvement over post-filtering |
| Error handling | `Result` everywhere; `ereport`/`error!` at the boundary; **no `unwrap()` in hot paths** | Production-grade habits; project coding standard |

### On the vector type

Brindle defines its own type, `brindle_vector`: one varlena holding a dimension
count and the components as `f32`, so the distance kernels read the stored bytes
as a `&[f32]` with no copy or deserialization per comparison.

The alternative was to reuse pgvector's `vector` and declare pgvector a
prerequisite extension. That buys drop-in interop for the price of a hard
dependency: every install, CI job, and `cargo pgrx test` run would first have to
build pgvector for each Postgres major. Brindle's differentiator is
filter-aware search, not the type, so the type is not worth a dependency that
users cannot remove.

Interop is kept where it is cheap instead of free:

- the value layout matches pgvector's (`int16` dimension, two padding bytes,
  then the components), and the text form is pgvector's `[1,2,3]`, so
  `embedding::text::brindle_vector` moves a pgvector column over;
- the operators are the familiar ones — `<->` L2, `<#>` (negative) inner
  product, `<=>` cosine — so a query keeps its shape; within Brindle's access
  method each metric also gets its own strategy number (1, 2, 3), so one
  operator family can hold all three;
- casts run to and from `real[]`, which is itself indexable through
  `real_array_l2_ops` for data that already lives in arrays.

`brindle_vector` deliberately has no typmod: a column does not declare its
dimensions, and the index rejects a vector that disagrees with the ones already
in it. Adding `brindle_vector(3)` is a compatible change if column-level
enforcement turns out to be wanted.

An operator class carries the rest of the contract. Each one declares the metric
its index ranks by (`brindle_vector_l2_ops`, `_cosine_ops`, `_ip_ops`) and the
type it indexes, both as support functions the access method reads at build
time — a build has no scan key to take a strategy number from, and index
maintenance runs with a restricted `search_path`, so neither fact can be
recovered from the catalogs alone.

### On storage (the honest tradeoff)

pgvector stores its HNSW graph in Postgres buffer pages, so a write touches only
the pages it changes. Brindle's index lives in the same relation's pages and its
writes *are* WAL-logged — it is crash-safe and reaches replicas — but as
whole-fork page images rather than per-page records, so **WAL volume is O(index)
per write-back** and a scan rebuilds the whole graph in memory before it can walk
it. That is the interim shape, kept deliberately: graph algorithms and filtering
are the differentiating work, and buffer-manager integration is largely
orthogonal systems plumbing. The roadmap calls it out so the tradeoff is never
hidden.

The page layout that replaces the interim blob — metapage, element tuples,
neighbor chunks, and the locking and WAL plan around them — is specified in
[STORAGE.md](STORAGE.md). What the interim blob costs is measured in
[BENCHMARKS.md](BENCHMARKS.md). It used to be 59–61 ms of *every* query; a
per-backend cache of the decoded graph moved it off the warm path, so a scan now
costs 0.34 ms once its backend has a copy and 57.9 ms when it does not. The cost
did not go away — it is paid per backend instead of per query, and bought with a
private ~89 MB copy of the index.

## 5. Module map

As built. The pure core is the top level; `index_am/` is the Postgres boundary.

```
src/
  lib.rs            # pg_module_magic, extension entry, #[pg_extern] surface
  distance.rs       # pure distance kernels + unit tests
  vector.rs         # metric selection / validation over slices
  pg_vector.rs      # the brindle_vector type: layout, I/O, operators
  hnsw.rs           # graph, build, layered search, γ-dense edges,
                    #   predicate-aware traversal, codec
  filter.rs         # predicate model: equality + numeric ranges, AND
  fusion.rs         # RRF over ranked lists
  guc.rs            # session GUCs: brindle.ef_search, brindle.cache_max_mb
  index_am/
    mod.rs          # IndexAmRoutine wiring, ambuild, aminsert, vacuum
    opclass.rs      # operator classes: metric + indexed type
    options.rs      # per-index WITH (m, ef_construction, gamma)
    scan.rs         # ambeginscan/amgettuple/amrescan/amendscan
    attrs.rs        # scan keys → predicate atoms; inline attribute rows
    storage.rs      # the stored index image, paging, and the backend cache
benches/            # criterion micro-benchmarks (distance, graph decode)
```

Not built, and where they will go when they are: `quantize.rs` (RaBitQ, Phase 5)
and the hybrid SQL surface that `fusion.rs` feeds (Phase 4). `hnsw.rs` is one
module rather than the `hnsw/` package this section used to predict — splitting
it is worth doing when it stops being navigable, not before.

## 6. Public SQL surface

```sql
-- Works today: distance functions over real[]
brindle_l2_distance(a real[], b real[])      -> real
brindle_cosine_distance(a real[], b real[])  -> real
brindle_inner_product(a real[], b real[])    -> real

-- Works today: the vector type, its operators, and an index whose
-- operator class picks the metric. Operator spelling is pgvector's:
-- `<->` L2, `<#>` (negative) inner product, `<=>` cosine.
CREATE TABLE docs (id int, embedding brindle_vector);
INSERT INTO docs VALUES (1, '[0.1,0.2,0.3]');   -- or ARRAY[...]::real[]
CREATE INDEX ON docs USING brindle (embedding brindle_vector_cosine_ops);
SELECT id FROM docs ORDER BY embedding <=> $1 LIMIT 10;

-- Works today: build/query knobs
CREATE INDEX ON docs USING brindle (embedding brindle_vector_cosine_ops)
  WITH (m = 16, ef_construction = 64, gamma = 1.0);
SET brindle.ef_search = 64;

-- Works today: filter-aware, with the predicate pushed into the traversal.
-- The filterable columns are key columns *after* the vector -- not INCLUDE, see
-- FILTERING.md § 3 -- and carry a numeric or boolean type. A text label needs
-- the dictionary encoding that is not built yet, and is refused at CREATE INDEX
-- rather than silently ignored.
--
-- Note the operator class travels with the vector column and fixes the metric
-- for the whole index: this one is cosine, so the ORDER BY must use `<=>`.
CREATE TABLE listings (
    id int, tenant_id int, price float8, embedding brindle_vector
);
CREATE INDEX ON listings
    USING brindle (embedding brindle_vector_cosine_ops, tenant_id, price);
SELECT id FROM listings
WHERE tenant_id = 42 AND price < 50
ORDER BY embedding <=> $1 LIMIT 10;

-- Not built: hybrid. `fusion.rs` implements the RRF half; this surface does not
-- exist yet.
SELECT * FROM brindle_hybrid(
  query_text => 'wireless headphones',
  query_vec  => $1,
  k          => 10,
  rrf_k      => 60
);
```

## 7. Non-goals (scope discipline)

- **Not** trying to beat VectorChord/pgvectorscale on raw QPS or memory.
- **Not** building a from-scratch BM25 engine (use Postgres native FTS for the
  lexical side; BM25 quality is a possible later enhancement).
- **Not** a distributed system. Single-node Postgres extension.

Keeping these explicit prevents scope creep — the project's value is depth on
filtering + hybrid, not breadth.

## 8. Testing & validation strategy

- **Unit tests** (`cargo test`) on pure core logic: distance correctness vs a
  naive reference, graph invariants, RRF math.
- **`pg_test`** integration tests (pgrx) for the SQL surface and the index AM.
- **Recall harness** (Phase 5): standard datasets (SIFT, GIST, a filtered
  benchmark), reporting recall@k vs QPS — *no performance claim ships without a
  reproducible number*.

See [ROADMAP.md](ROADMAP.md) for sequencing.
