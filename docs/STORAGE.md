# Brindle — Durable Storage Layout

How a Brindle index lives in PostgreSQL's own storage: buffer pages, read through
the buffer manager, updated in place, and write-ahead logged.

This document is the **contract** for the storage implementation. Everything the
implementation needs to decide about *bytes on pages* is decided here; what is
left open is listed in [§ 12](#12-left-to-the-implementation).

Status: **specification** (format version 2). The shipping code still uses the
interim blob format described in [§ 2](#2-what-this-replaces).

---

## 1. Goals and non-goals

**Goals**

- The graph lives in the index relation's pages and is read through the buffer
  manager — no whole-index deserialization per scan.
- An insert touches only the pages it changes, so its cost (and its WAL volume)
  is proportional to the new node's edges, not to the index.
- Every page mutation can be wrapped in a WAL record, so the index is crash-safe
  and physical replicas see the same bytes.
- Readers never block, and never see a torn structure.
- The layout mirrors proven practice (pgvector's HNSW) wherever there is a
  choice. Brindle's novelty budget belongs to filter-aware search, not to page
  formats.

**Non-goals**

- Beating pgvector on raw build throughput or index size.
- Concurrent writers. Writers are serialized against each other from the start;
  making them concurrent is a later, isolated change ([§ 8.4](#84-why-writers-are-serialized)).
- Storing quantized codes. The element tuple has room reserved for a later
  quantization payload, but nothing is specified here.

---

## 2. What this replaces

The interim blob format (`src/index_am/storage.rs`) serializes the entire
in-memory graph — every vector, every adjacency list — into one opaque blob,
splits it across pages, and re-reads it whole whenever the copy a backend holds
is no longer current:

```
block 0   metapage: magic, version, blob length, generation
block 1   ┐
block 2   │  one byte stream, chunked at the page boundary:
  ...     │  [graph codec][node-id → heap-TID table]
block N   ┘
```

Consequences, all of which this design exists to remove:

| | Interim (blob) | Paged |
|---|---|---|
| Scan startup | read the metapage; deserialize the whole index if this backend has no current copy | read the metapage |
| Scan working set | decoded copies per backend, bounded per backend by `brindle.cache_max_mb` — nothing bounds the total across connections | shared buffer cache, one copy for the server |
| Insert cost | O(index) CPU per *transaction*: load, mutate, write back | O(m · log n) page reads, O(m) page writes |
| Insert WAL | O(index) per transaction — a 4 MB index logs ~4 MB, and a single-row `INSERT` is a transaction | O(m) page images |
| Vacuum tombstone | rewrite the whole image | flip one byte on one page |
| Crash mid-write | image and metapage disagree → index unreadable until `REINDEX` | each record is atomic; worst case is a lost back-edge |
| Cross-backend writes | every reader re-checks the metapage generation, which every writer advances | invalidation is the buffer manager's |
| Reader/writer | one heavyweight lock over the whole image, held across the write-back rather than the transaction | readers take no heavyweight lock |
| Filter attributes | not persisted at all | inline in the element tuple |
| Indexable dimensions | any (up to the type's 16000) | ≤ 2000 ([§ 5](#5-sizing-and-limits)) |
| Max layer-0 degree | 2048 | 1024 ([§ 5](#5-sizing-and-limits)) |
| Inspectability | opaque bytes | ordinary line-pointer items |

The last two rows are the price. Both are consequences of "a tuple fits on a
page", they are the same trade pgvector makes, and both bounds sit well past the
useful range of the knobs (see [§ 5](#5-sizing-and-limits)).

---

## 3. Physical layout

Everything lives in the **main fork**. Unlogged indexes additionally get a
metapage-only **init fork** ([§ 9.4](#94-unlogged-indexes)).

Every page is a standard Postgres page: page header, line-pointer array, items
growing from the end, and 8 bytes of special space. Element and neighbor items
are added with `PageAddItem`, so an item's address is an ordinary
`(BlockNumber, OffsetNumber)` and external tooling (`pageinspect`) sees a
well-formed page. The metapage is the exception — it holds no items, keeping its
struct in the page's content area for the reason
[§ 3.2](#32-metapage-block-0-page-contents) gives.

```
                         ┌─────────────────────────────────────┐
  block 0                │ META      metapage (page contents)  │
                         ├─────────────────────────────────────┤
  block 1                │ ELEMENT   element tuples            │
  block 2                │ ELEMENT   (header, heap TID,        │
    ...                  │            attrs, vector)           │
                         ├─────────────────────────────────────┤
  block k                │ NEIGHBOR  neighbor chunks           │
    ...                  │           (one per node per layer)  │
                         └─────────────────────────────────────┘
```

Element tuples and neighbor chunks live on **separate pages**, tracked by two
insert pointers in the metapage. The separation costs at most one extra page on
a tiny index and buys three things: a sequential sweep of vectors (used by
vacuum, by build, and by the statistics a cleanup pass reports) reads no
adjacency bytes; an edge update never dirties a vector page; and vector pages
stay dense, which is what the distance loop scans.

### 3.1 Page special space

Every page, including the metapage, ends with:

```
offset size field
  0     4   next_free   BlockNumber — free-list link, InvalidBlockNumber if not free
  4     2   kind        1 = meta, 2 = element, 3 = neighbor (never 0)
  6     2   page_id     0x4252 ('BR') — identifies the page as Brindle's
```

`page_id` is the same trick pgvector uses: it makes a page recognizable to
`pageinspect` and turns "we read a page from the wrong relation or fork" into a
loud error instead of a misparse. Every read of an element or neighbor page
validates `page_id` and `kind` before trusting the contents.

Two exceptions, both load-bearing:

- **Block 0 is validated in a fixed order** — magic, then version, then special
  space — never the other way round. A version-1 index has no special space at
  all ([§ 11](#11-migration-from-the-interim-format)), so a special-space check
  that ran first would read past the page end instead of producing the clean
  "rebuild this index" error.
- **An uninitialized page is not an error, and `PageIsNew` is tested first.**
  Relation extension is not transactional: `smgr` extends the file before any
  WAL record is written, so a crash *or an ordinary `ROLLBACK`* between the
  extension and the record leaves a zeroed block in the fork permanently. Every
  reader treats `PageIsNew` as an empty page and moves on; the allocator
  ([§ 7.4](#74-allocation-and-free-space)) treats it as a free page to
  initialize and reuse. A sweep that errored on one would turn a routine
  rollback into an unqueryable index. No page kind is numbered 0 for the same
  reason — a zeroed page must not be able to decode as a valid kind even if
  something checks the kind before the `PageIsNew` test.

### 3.2 Metapage (block 0, page contents)

The metapage is written into the page's content area — directly after the
standard header, at `PageGetContents`, with `pd_lower` marking its end — and
**not** as a line-pointer item. That is where the blob format puts its own metapage
and where pgvector puts its HNSW metapage, and it is what makes the version
check in [§ 11](#11-migration-from-the-interim-format) work: `magic` and
`version` land at the same page offsets (24 and 28) in both formats, so
version-2 code reading a version-1 index finds the field it needs at the
offset it expects, rather than dereferencing a line pointer that format
never wrote.

Fixed offsets, little-endian, exactly as the blob metapage and the graph codec
already store their scalars.

**The paged metapage is version 3, not 2.** The blob format took 2 when it grew
a generation counter, so a paged reader that claimed 2 would find a 24-byte blob
metapage announcing the version it was looking for and parse 80 bytes out of it —
which is precisely the misparse the version field exists to prevent. The number
has to move whenever either format changes shape, and only one of them can hold
a given value.

```
offset size field                notes
  0     4   magic               0x4252_4E44 "BRND"   ─┐ same offsets as the blob
  4     4   version             3                    ─┘ format, so the check works
  8     1   metric              Metric::code()
  9     1   flags               reserved, zero
 10     2   reserved
 12     4   dim                 fixed by the first indexed vector, 0 while empty
 16     4   m                   build parameter
 20     4   ef_construction     build parameter
 24     4   gamma               f32 bits
 28     4   entry_block         entry point, InvalidBlockNumber when empty
 32     2   entry_offset
 34     2   entry_level         the entry point's layer = the graph's top layer
 36     4   element_insert_page block to try first for a new element tuple
 40     4   neighbor_insert_page
 44     4   free_head           head of the free-page chain, or InvalidBlockNumber
 48     8   node_count          hint, see below
 56     8   deleted_count       hint, see below
 64     8   seed                PRNG seed the index was built with
 72     8   rng_state           live PRNG state, advanced by each insert
                                (80 bytes)
```

There is deliberately no separate `max_layer`. The entry point is by
construction a node at the top layer — the core raises both together, in one
place — so a second field could only ever disagree with `entry_level`. Descent
starts at `entry_level`.

`m`, `ef_construction`, `gamma`, `metric`, `dim` and `seed` are written once at
build and never change — an `ALTER INDEX ... SET (...)` still takes effect only
at the next rebuild, exactly as today. The derived degree caps (`m_cap`,
`m0_cap`) and the level normalizer `ml` are **not** stored: they are recomputed
from `m` and `gamma` by the same core function that derives them at build time,
so a stored index and a fresh one can never disagree about them.

`rng_state` is stored because level assignment must survive a restart. The
version-1 codec serializes the graph's PRNG state for the same reason; the
metapage is where it lives now, and an insert advances it under the writer lock.

`node_count` and `deleted_count` are **hints**: they feed planner statistics and
the compaction trigger, and a crash may leave them one behind. They may
*schedule* work — when compaction is worth running, what to report to `ANALYZE` —
but they never decide a query's result. Nothing a correct answer depends on is
read from them; a count that has to be exact comes from a sweep of the element
pages ([§ 6.3](#63-sequential-sweep)), which is a vacuum and statistics path, not
a query one ([§ 6.2](#62-index-scan)).

### 3.3 Element tuple

One per indexed row. Holds everything traversal needs about a node except its
adjacency: the heap pointer it will return, the attributes a predicate is
evaluated against, and the vector a distance is computed from.

```
offset size field
  0     1   kind            1 = element
  1     1   flags           bit 0: deleted (tombstone)
  2     2   level           this node's top layer (0-based)
  4     2   dim
  6     2   attr_count
  8     4   heap_block      ┐ the heap TID this node answers with
 12     2   heap_offset     ┘
 14     2   reserved
 16    6·(level+1)  neighbor_ptr[]   (block, offset) of this node's neighbor
                                     chunk for each layer, layer 0 first
  ·     ·   padding to an 8-byte boundary
  ·    16·attr_count  attrs[]        tag u8, 7 bytes padding, 8-byte payload
                                     (Int i64 / Float f64 / Null)
  ·     4·dim         vector         f32 components, 8-byte aligned
```

Three deliberate choices:

- **The heap TID is inline.** Version 1 keeps a separate node-id → TID table,
  which forces every write path to keep two structures in step and makes
  compaction a renumbering exercise. Storing the TID in the tuple that owns it
  deletes that class of bug outright.
- **Attributes precede the vector.** Predicate-aware traversal rejects a
  non-matching node without reading its vector; putting the attribute row in the
  tuple's first cache lines keeps that rejection cheap. (Version 1's codec does
  not persist attributes at all — it carries a debug assertion that the graph
  reaching it has none. This format closes that gap, which is what lets
  filter-aware search survive a restart.)

  The 16-byte slot is twice what the value needs — three variants and an 8-byte
  payload — and buys a fixed stride with a naturally aligned payload, which is
  what keeps predicate evaluation a subscript rather than a parse. Attribute
  rows are a handful of values per node; the packing is not where this format's
  space goes.

  **Nothing stores an attribute *schema*.** Positions in the row are the index's
  own attribute order — the included columns as the catalog lists them — so the
  mapping lives in the index's tuple descriptor, which every path already has
  open. Storing a copy would create a second source of truth that `ALTER TABLE`
  could desynchronize.
- **The vector is 8-byte aligned** and stored as plain `f32`, so the distance
  kernels read page bytes directly as a `&[f32]` — no copy, no decode, matching
  how `brindle_vector` is already laid out in the heap.

A tombstoned element keeps its slot, its line pointer, and its edges forever;
see invariant I3.

### 3.4 Neighbor chunk

One per node **per layer**, allocated when the node is inserted. A node's chunk
for layer *L* is addressed directly by `neighbor_ptr[L]` in its element tuple,
so a hop costs one page read and no chain walk.

```
offset size field
  0     1   kind          2 = neighbors
  1     1   flags         reserved, zero
  2     2   layer
  4     2   capacity      slots allocated: m0_cap at layer 0, m_cap above
  6     2   count         slots in use, count ≤ capacity
  8    6·capacity  entries[]   (block u32, offset u16) per neighbor —
                               the same 6-byte shape as ItemPointerData
```

The chunk is allocated at full `capacity` and never grows, so adding a back-edge
is a `count += 1` plus a 6-byte write inside one page, and pruning a full list
rewrites entries in place. That is what makes an edge update a single-page,
single-record change.

The 6-byte stride means every entry after the first is 4-byte misaligned — the
same reason `ItemPointerData` is spelled as three `uint16`s rather than a
`uint32` and a `uint16`. Decode entries bytewise; an aligned 32-bit load over
this array is undefined behavior, not an optimization.

Levels are geometrically distributed — a node reaches layer 1 with probability
`1/m`, so at `m = 16` about 15 nodes in 16 exist only at layer 0. The common node
therefore costs exactly one element tuple and one neighbor chunk.

---

## 4. Addressing

**A node's identity is the address of its element tuple: `(BlockNumber,
OffsetNumber)`.** There is no dense node-id space on disk, no id → address
directory, and no id → heap-TID table.

- Packed as a `u64` (`block << 16 | offset`) wherever a node must be hashed,
  compared, or put in a visited set. `InvalidBlockNumber` marks "no node".
- Stable for the life of the index: an element's line pointer is never reused
  while any neighbor list could still name it (invariant I3), so a stored
  neighbor pointer always resolves to a live element tuple. This is what allows
  readers to traverse without any lock beyond the page they are reading.
- Only a full compaction rewrites addresses, and it does so under an exclusive
  lock with nothing else in flight.

The pure core keeps its dense `usize` ids for the in-memory graph — that is an
in-memory detail and stays one. At the boundary between them the core sees an
opaque `NodeRef = u64` ([§ 6.1](#61-the-core-side-accessor)); the in-memory
implementation passes its own ids through it, and the paged implementation packs
addresses into it. Neither knows which the other uses.

---

## 5. Sizing and limits

On the default 8 kB build (`BLCKSZ = 8192`, `MAXALIGN = 8`):

```
page                       8192
− page header                24
− special space               8
− one line pointer            4
= largest single item      8156   (rounded down to MAXALIGN: 8152)
```

**Element tuple** = `16 + 6·(level+1)` → padded to 8 → `+ 16·attr_count + 4·dim`.

At level 0 with no attributes that leaves `(8152 − 24) / 4 = 2032` components,
so:

- **Maximum indexable dimensions: 2000.** Same limit pgvector's HNSW carries,
  and for the same reason. The `brindle_vector` type still accepts up to 16000;
  the cap applies to *indexing*.
- The check happens on the first vector indexed, not at `CREATE INDEX`: a column
  does not declare its width (the type has no typmod, and `real[]` carries none
  either), and `dim` is fixed by the first vector the build sees. So build and
  insert compute the exact tuple size — attributes and levels eat the same
  budget, 8 attributes ≈ 128 bytes ≈ 32 dimensions — and raise an error naming
  the dimension count and the limit if it does not fit. The headline number is
  the guidance; the size check is the rule.

**Neighbor chunk** = `8 + 6·capacity`, so a chunk fits while capacity ≤ 1357.

- **Maximum layer-0 degree: 1024** (down from 2048). The reloption validator's
  ceiling on `2·m·γ` moves to 1024, which still admits `m = 128, γ = 4` and
  `m = 64, γ = 8` — far denser than any useful setting, and 6152 bytes of
  neighbor chunk per node at that. A build that would exceed it is rejected at
  `CREATE INDEX` with the same message shape used today.

Refusing to let a neighbor list span pages is what keeps an edge update atomic
under one buffer lock and inside one WAL record. Chaining chunks would buy
degrees nobody wants and cost that guarantee.

---

## 6. Read paths

### 6.1 The core-side accessor

The traversal algorithm must not be written twice — once over `Vec`s and once
over pages — and it must not import `pgrx`. So the core gains a small
read-access trait, and search becomes generic over it:

```rust
// pure core, no pgrx
pub type NodeRef = u64;

pub trait GraphStore {
    type Error;

    fn params(&self) -> GraphParams;                 // metric, m, caps, dim, ...
    fn entry(&self) -> Option<(NodeRef, usize)>;     // entry point + its layer
    fn is_deleted(&self, node: NodeRef) -> Result<bool, Self::Error>;

    /// Replace `out` with `node`'s neighbors at `layer`.
    fn neighbors(&self, node: NodeRef, layer: usize, out: &mut Vec<NodeRef>)
        -> Result<(), Self::Error>;

    /// Distance from `query` to `node`'s vector.
    fn distance(&self, query: &[f32], node: NodeRef) -> Result<f32, Self::Error>;

    /// Copy `node`'s vector into `out`, for when it must outlive the page pin.
    fn load_vector(&self, node: NodeRef, out: &mut Vec<f32>) -> Result<(), Self::Error>;

    /// Replace `out` with `node`'s attribute row (empty if it has none).
    fn attrs(&self, node: NodeRef, out: &mut Vec<AttrValue>) -> Result<(), Self::Error>;
}
```

The graph search takes `&impl GraphStore` instead of `&self`. The existing
in-memory `Hnsw` implements the trait over its own arrays, so its unit tests,
recall tests and benchmarks keep running unchanged with no database in sight;
the boundary implements it over buffer pages.

`distance` returns a number rather than lending out a `&[f32]` on purpose: the
paged implementation pins a buffer, computes the distance against the page bytes
in place, and unpins before returning. Lending the slice would push buffer-pin
lifetimes into generic core code, and copying the vector out would allocate per
comparison — the one thing the distance path must never do. Callers pass scratch
buffers (`out`) in for the same reason.

`load_vector` is the deliberate escape hatch, and the neighbor-selection
heuristic is why it exists: that heuristic compares candidates *to each other*,
not just to the query, so it needs one node's vector to outlive the page pin on
another's. The pattern is copy a vector into a scratch buffer, then
`distance(&scratch, other)` against the page bytes of the other.

Be honest about what that costs on paged storage. Recomputing a full neighbor
list after a prune copies its base once and compares down the list — one copy
per pass. The selection heuristic does not: its base changes with every
candidate it examines, so it is one `load_vector` per candidate, bounded by
`ef_construction`, at every layer of every insert. That is the most expensive
thing in this design that the interim format got for free, and it is the first
thing to measure once inserts run on pages.

### 6.1.1 The write side

The read trait is not enough. Insert has to run the same descent, layer search
and neighbor-selection heuristic the in-memory build runs — those are private
helpers on the graph today — and then write edges back. Without a mutable
counterpart the boundary would have to reimplement all three, which is exactly
the duplication this section exists to prevent, and the surest way to break
invariant I10.

```rust
pub trait GraphStoreMut: GraphStore {
    /// What the boundary needs to store beside a node and the core never reads
    /// — the heap TID for the paged store, `()` for the in-memory one.
    type Payload;

    /// Draw the next node's level from the index's persistent PRNG, advancing it.
    fn next_level(&mut self) -> Result<usize, Self::Error>;

    /// Reserve a node at `level` with its vector, attribute row and payload,
    /// together with an empty neighbor list per layer. The store decides the
    /// physical write order (see § 7.2 step 3); on return the node exists and
    /// nothing references it yet.
    fn add_node(&mut self, level: usize, vector: &[f32], attrs: &[AttrValue],
                payload: Self::Payload) -> Result<NodeRef, Self::Error>;

    /// Replace `node`'s neighbor list at `layer`. The only edge-write primitive:
    /// adding a back-edge and pruning a full list are both this call.
    fn set_neighbors(&mut self, node: NodeRef, layer: usize, neighbors: &[NodeRef])
        -> Result<(), Self::Error>;

    /// Make `node` the entry point at `level`. Only ever raises the top layer.
    fn set_entry(&mut self, node: NodeRef, level: usize) -> Result<(), Self::Error>;

    /// Set or clear a node's tombstone.
    fn set_deleted(&mut self, node: NodeRef, deleted: bool) -> Result<(), Self::Error>;
}
```

Insert becomes a core free function over the pair:

```rust
pub fn insert_into<S: GraphStoreMut>(store: &mut S, vector: &[f32],
                                     attrs: &[AttrValue], payload: S::Payload)
    -> Result<NodeRef, S::Error>
where
    S::Error: From<HnswError>;
```

The bound is not incidental: the algorithm rejects an empty vector or a
dimension mismatch on its own account, before it ever calls the store, so the
error type has to carry `HnswError` as well as the store's. The generic search
needs the same bound. It carries level assignment, descent, layer search,
neighbor selection and back-edge pruning. `Hnsw::insert` becomes a thin wrapper over it, so the core's
own tests, recall checks and benchmarks keep running against the in-memory
implementation unchanged.

Note what this means for sequencing: **this is a change to the pure core, not
only to the boundary.** The algorithm is untouched — what changes is that it
reads and writes through a trait instead of through its own fields — but the
edit lands in `src/hnsw.rs`, and the storage implementation has to budget for
it.

The core stays `pgrx`-free throughout: `NodeRef` is an opaque `u64`, `Payload`
is an associated type the core never inspects, and nothing in either trait
mentions a buffer, a page or a relation.

### 6.2 Index scan

1. `ambeginscan` reads the metapage once: parameters, entry point and its
   `entry_level`, `node_count`, `deleted_count`. Nothing else is read, and
   nothing is cached across scans.
2. `amrescan` runs the layered search against the paged `GraphStore`. Each hop
   is: read the neighbor chunk (one page), then for each neighbor read its
   element tuple (one page) to compute a distance — and, once filtering is
   wired, to evaluate the predicate before the distance.
3. `amgettuple` returns the heap TID stored inline in the element tuple the
   search already visited.

A scan takes a snapshot of the entry point at its start and does not re-read it.
A concurrent insert that moves the entry point is invisible to an in-flight
scan; the result is an approximate search that started one node off, which is
within what an approximate index promises.

An earlier version of this section specified a scan that widened its budget and
re-searched until the executor stopped asking, backed by an exhaustive sweep so
that an unfiltered `ORDER BY` returned *every* row. **That behavior is gone**,
and the reasoning behind removing it belongs here, because this document is the
spec a storage implementation will follow.

Widening broke ordering. A wider graph search is not a superset of a narrower
one, so re-searching could turn up a row nearer than one already handed to the
executor — and because `amcanorderbyop` makes the planner delete its own sort,
nothing downstream repairs that. A scan now runs **one** search at
`brindle.ef_search`, returns those rows in distance order, and ends, yielding at
most `ef_search` rows.

What that changes for storage:

- **No exhaustive fallback ends a scan.** The sequential sweep
  ([§ 6.3](#63-sequential-sweep)) is still needed for vacuum and for the
  statistics a cleanup pass reports, but it is no longer a query path.
- **The live-count hint is no longer load-bearing for correctness.** It mattered
  because an understated count silently dropped rows from a scan that had
  promised completeness. With no such promise, `deleted_count` and `node_count`
  ([§ 3.2](#32-metapage-block-0-page-contents)) are hints for planning and
  vacuum decisions, where staleness costs accuracy rather than rows.
- **A stale counter can no longer turn a `LIMIT 10` into a full index scan.**
  The O(index) query path this section used to justify is gone.

Reintroducing a widening loop would reintroduce the ordering bug. If completeness
is wanted back, it belongs behind an opt-in mode that documents its ordering as
relaxed — not in the default scan path.

### 6.3 Sequential sweep

Two paths need every element and no adjacency: vacuum, and the statistics a
cleanup pass reports. (A third, the exhaustive fallback that used to end a
widening scan, is gone — see [§ 6.2](#62-index-scan).) Both
sweep element pages in block order, and per page walk line pointers
1..`PageGetMaxOffsetNumber`, skipping items whose `kind` is not `ELEMENT` and
skipping `PageIsNew` pages entirely ([§ 3.1](#31-page-special-space)). Sweeping
is sequential I/O over exactly the vector bytes, which is why elements and
neighbor chunks are on separate pages.

A sweep is the authoritative source for counts and for the live set; the metapage
counters are hints. Since [§ 6.2](#62-index-scan) no longer has a query depending
on either, the distinction now matters for vacuum and statistics rather than for
whether an answer is right.

---

## 7. Write paths

### 7.1 Build

Build stays a two-phase operation, because a bulk build in memory produces a
better graph far faster than incremental page updates would, and because it is
what the code does today:

1. Scan the heap and build the graph in memory exactly as now.
2. **Layout pass:** walk nodes in id order, assigning each an element-tuple
   address by filling element pages sequentially, and each of its layers a
   neighbor-chunk address by filling neighbor pages sequentially. Record the
   id → address map in a `Vec<u64>` indexed by id. This pass computes addresses
   only; no page is written.
3. **Write pass:** emit element tuples and neighbor chunks, translating ids to
   addresses through the map. Pages are filled and released in order.
4. Write the metapage (parameters, entry point, counts, insert pointers, PRNG
   state) last.
5. WAL-log the fork with `log_newpage_range`, as the build path already does.

The build still holds the whole graph in memory, unchanged from today. Making
build spill to disk under `maintenance_work_mem` is a separate, later concern.

`ambuildempty` writes a metapage-only image (no elements, entry point invalid)
to the init fork.

### 7.2 Insert

Under the writer lock ([§ 8](#8-concurrency)), and reading the graph through the
paged `GraphStore`:

1. Read the metapage: parameters, entry point and its `entry_level`, insert
   pointers, PRNG state. Assign the new node's level from the PRNG; write the
   advanced state back with the metapage update in step 6.
2. Run the normal HNSW descent and layer searches to select the new node's
   neighbors at every layer ≤ its level. This is page reads only.
3. Allocate and write the node (`add_node`): one **empty** neighbor chunk per
   layer, then the element tuple naming them (heap TID, attributes, vector,
   `neighbor_ptr[]`). Chunks before the element that points at them, so a write
   split across several records never leaves an element naming a chunk that
   does not exist. Then fill the node's own outgoing edges with `set_neighbors`
   per layer. **Nothing points at the new node yet**, so every prefix of this
   sequence is unreachable garbage rather than a broken graph.
4. For each selected neighbor, add the back-edge: read its chunk under a share
   lock, and if `count < capacity` write the entry under an exclusive lock. If
   the chunk is full, run the same pruning heuristic the in-memory build uses —
   which needs distances, so the candidate vectors are read *before* the
   exclusive lock is taken — then rewrite the list. One page, one record, per
   back-edge.
5. If the new node's level exceeds the current `entry_level`, update the
   metapage's entry point and `entry_level` together — they are one fact.
6. Update the metapage: `node_count`, insert pointers, PRNG state (folded into
   step 5's write when both happen).

Ordering matters and is an invariant (I5): the element and its outgoing edges
are durable before any back-edge names it, and the entry point moves last. A
crash between steps therefore leaves either a node nothing points to — dead
weight until vacuum, no wrong answers — or a node with fewer in-edges than
intended, which costs a little recall and nothing else.

An insert whose transaction later aborts leaves its element in the index. That
is ordinary Postgres index behavior: the scan returns the TID, the executor
finds the heap tuple invisible and discards it, and vacuum tombstones the node
in due course.

### 7.3 Delete and vacuum

Tombstoning is now page-local: find the element (vacuum is already sweeping
them), set `flags` bit 0, log the page. One byte, one page, one record, versus a
whole-image rewrite today. The guarantee is unchanged — a tombstoned node still
routes traffic and is never returned — and so is the reason it exists: a node
must stop naming a heap slot before vacuum may recycle it.

Space reclamation (compacting away tombstoned elements, unlinking their chunks,
threading emptied pages onto `free_head` and reporting them to the free space
map) is a separate task and is not specified here. The layout supports it: page
kinds and the free-list link are in the special space, and `free_head` is in the
metapage.

### 7.4 Allocation and free space

A new item goes on the insert page for its kind (`element_insert_page` /
`neighbor_insert_page`) if it fits. Compare against `MAXALIGN(item_size)`, not
the raw size: `PageAddItem` aligns every item, and `PageGetFreeSpace` has
already subtracted the line pointer the item will need, so the aligned size is
the honest comparison. Otherwise: reuse a page from `free_head` if the chain is
non-empty, else extend the relation; initialize it with the right `kind`; make
it the new insert pointer.

An uninitialized page needs no search. Extension appends, so the only blocks
that can be `PageIsNew` are at the tail of the fork, left by an extension whose
record never landed. The allocator finds one by looking at the last block, not
by scanning; anything it misses is picked up by the next extension, which is
free to initialize and use a zeroed block it finds there.

**Extension does not take the extension lock for you.** The `P_NEW` path of
`ReadBufferExtended` passes `EB_SKIP_EXTENSION_LOCK` — it is documented in
`bufmgr.c` as a backwards-compatibility path — so what actually keeps two
backends from extending at once here is brindle's own writer serialization
([§ 8.3](#83-writers)), nothing in the buffer manager. Prefer
`ExtendBufferedRel`, which takes the lock and scales better, and treat this as
a hard prerequisite of ever relaxing [§ 8.4](#84-why-writers-are-serialized):
concurrent writers without a real extension interlock corrupt the fork.

`free_head` has no producer until compaction lands ([§ 7.3](#73-delete-and-vacuum))
— today the chain is always empty and the branch is dead. It is specified now so
that reclamation does not have to change the metapage format later.

Insert pointers are hints like the counters — a stale one costs a wasted page,
never correctness.

---

## 8. Concurrency

### 8.1 The rule

> **Every path holds at most one buffer content lock at a time.**

The single exception is a WAL record covering several buffers, which locks them
in **ascending block-number order** ([§ 9.2](#92-what-each-record-covers)).

That rule is the whole deadlock argument: a reader never waits for a lock while
holding one, so no cycle can form. It is why step 4 of the insert path reads a
neighbor's candidates under a share lock, releases, computes, and only then
takes the exclusive lock — rather than reading vectors while holding the chunk
locked.

### 8.2 Readers

Scans take **no** heavyweight lock. They take a share content lock on one page,
read what they need (a neighbor list, or a distance and a TID), and release it.
Every pointer they follow resolves (I3), and every list they read is internally
consistent because it is written under an exclusive lock on the same page. What
a reader may see is a *stale* graph — a node whose back-edges are still being
added, an entry point that has since moved — which an approximate index is
entitled to.

### 8.3 Writers

Inserts and vacuum take an `ExclusiveLock` on a designated page-lock block
(block 0 — the same heavyweight lock the interim format uses to arbitrate the
whole image, kept only as a writer mutex). It is held for the duration of one
insert or one vacuum pass.

Note the asymmetry: an insert holds it for a handful of page reads and writes,
but a vacuum pass holds it across a full sequential sweep of every element page,
blocking every `INSERT` on the table for that long. That is no worse than today,
where vacuum rewrites the entire image under the same lock, and it is a bounded
sequential scan rather than a rewrite — but it is a real cost, and it is the
second reason (after insert concurrency) to revisit [§ 8.4](#84-why-writers-are-serialized).

### 8.4 Why writers are serialized

Concurrent inserts into an HNSW graph need per-element locking, a link/unlink
protocol, and an entry-point interlock — a body of work with its own failure
modes, and one that would land in the same change as the page format itself.
Serializing writers keeps this change auditable and makes both the deadlock
argument and the crash argument one paragraph each.

What it costs: concurrent `INSERT`s into an indexed table queue behind each
other. What it does not cost: reads, which were previously blocked by the image
lock and now are not, and insert *latency*, which drops from O(index) to
O(m · log n) page reads.

Relaxing this later is a self-contained change — it touches the locking rules,
not the byte layout — and is the natural next storage task after WAL.

---

## 9. WAL and recovery

Generic WAL (`GenericXLogStart` / `GenericXLogRegisterBuffer` /
`GenericXLogFinish`) is the mechanism. It logs page deltas for arbitrary page
changes with no custom resource manager, and it is the supported route for an
extension.

### 9.1 The constraint that shapes everything

**A generic WAL record covers at most 4 buffers** (`MAX_GENERIC_XLOG_PAGES`).
An insert can touch far more than 4 pages, so an insert is not one atomic
record. It is a *sequence* of records, each atomic on its own, ordered so that
every prefix of the sequence leaves a valid index. That is exactly what the
ordering invariant (I5) buys.

### 9.2 What each record covers

| Record | Buffers | Atomic because |
|---|---|---|
| New node: empty chunks, then the element naming them, then its own edges | ≤ 4 (split by page when more) | nothing references them yet, so a partial write is unreachable garbage |
| One back-edge added or a list pruned | 1 | a neighbor list only ever changes within its own page |
| Entry-point / metapage update | 1 | last, and only after the element it names is durable |
| Tombstone flip | 1 | one byte in one element tuple |
| Build | whole fork via `log_newpage_range` | the relfilenode is not yet visible |

Buffers registered in one record are locked in ascending block order, per
[§ 8.1](#81-the-rule).

One mechanical rule belongs here rather than with the call sequences in
[§ 12](#12-left-to-the-implementation), because getting it wrong is a
correctness bug and not a style one: a **newly initialized page must be
registered with `GENERIC_XLOG_FULL_IMAGE`.** Generic WAL logs the delta between
the page as it was at registration and as it is at finish, so a page that was
`PageInit`ed *before* being registered replays onto a zeroed block during
recovery and loses its header. Every allocation path in
[§ 7.4](#74-allocation-and-free-space) hits this.

### 9.3 Recovery

- **Nothing is cached across transactions**, so recovery rebuilds nothing.
  Parameters, entry point and counters are re-read from the metapage on the next
  scan; the only per-scan state is the entry-point snapshot taken at
  `ambeginscan`.
- After a crash mid-insert, the possible states are: no element (record 1 never
  landed); an element nothing points to (records 2+ never landed); an element
  with some back-edges; or a complete insert whose metapage counter is one
  behind. Every one of them is a valid, queryable index — which is why the
  counters are specified as hints and the sweeps as authoritative.
- The failure mode the interim format has here — a torn whole-image rewrite that
  leaves the metapage's declared length disagreeing with the pages, making the
  index unreadable until `REINDEX` — has no analogue in this design. There is no
  global length to disagree with.

### 9.4 Unlogged indexes

No special-casing is needed on the insert path: `GenericXLogStart` records
`RelationNeedsWAL(relation)` and `GenericXLogFinish` skips the WAL record (while
still dirtying the buffers) when it is false, so one code path serves both
persistences.

The init fork is the exception that must be logged whatever the relation's
persistence, because it is what seeds a valid, empty index after a crash resets
the main fork. `ambuildempty` writes the metapage-only image there and logs it —
the distinction the build path already draws today.

---

## 10. Invariants

Testable statements the implementation must uphold. These are the assertions the
storage tests should be written against.

- **I1** Block 0 is the metapage, has `kind = META`, and carries magic `BRND`
  and the format version. Every read validates both before trusting a byte.
- **I2** Every *initialized* page carries `page_id = 0x4252` and a `kind`; every
  item carries a `kind` byte matching its page. A mismatch is an error, never a
  reinterpretation. A `PageIsNew` block is not a mismatch — extension is not
  transactional, so a zeroed block is a legal state that readers skip and the
  allocator reuses.
- **I2b** Block 0 is checked magic-then-version-then-special-space, so a
  version-1 index produces a clean error instead of an out-of-page read.
- **I3** An element's `(block, offset)` is stable for the life of the index. A
  line pointer is never reused, and a tombstoned element is never removed, while
  any neighbor list could still name it. Only compaction under an exclusive lock
  may break this, and it rewrites every referrer in the same operation.
- **I4** Every pointer in a neighbor chunk resolves to an element tuple with
  `kind = ELEMENT` on a page with `kind = ELEMENT`.
- **I5** An element tuple and its outgoing chunks are durable before any
  back-edge names it; the metapage entry point is updated last and never names
  an element that is not yet durable.
- **I6** `count ≤ capacity` in every chunk, and `capacity` equals the layer's
  derived cap (`m0_cap` at layer 0, `m_cap` above) for the index's `m` and `γ`.
- **I7** Every item fits its page: no element tuple or neighbor chunk spans
  pages, and the metapage struct fits the content area. Build and insert verify
  the size and error before writing.
- **I8** `node_count` and `deleted_count` are hints: they may schedule work —
  when to compact, what to report to `ANALYZE` — but never decide a query's
  result. A stale counter costs accuracy in a plan or an extra pass in vacuum,
  never a wrong answer. (This invariant used to require a scan to sweep the
  element pages before declaring itself drained, back when an unfiltered
  `ORDER BY` promised every row. A scan no longer promises that, and must not
  sweep: see [§ 6.2](#62-index-scan).)
- **I8b** `entry_level` is the level of the node `entry_block`/`entry_offset`
  names, and every element's `dim` equals the metapage's. Both are redundant by
  construction and are checked on read, not trusted.
- **I9** No path holds two buffer content locks except within a single WAL
  record, where buffers are locked in ascending block order.
- **I10** A scan's results and recall are unchanged from the in-memory
  implementation for the same graph — the layout changes where bytes live, never
  which node is nearest.

---

## 11. Migration from the interim format

**There is no in-place upgrade. An index built by the interim format must be
rebuilt with `REINDEX`.**

- The metapage's `version` field is what detects it: an index in an older format
  fails with an error naming the format it was written in and telling the user to
  `REINDEX INDEX <name>` (or `REINDEX TABLE`), rather than misparsing a blob as a
  metapage. The blob format is at 2 and this design takes 3 — every format that
  has ever been written needs its own number, or the check silently passes on the
  wrong layout.
- That check works only because the two formats agree on where to look, which is
  why [§ 3.2](#32-metapage-block-0-page-contents) keeps the metapage in the page
  content area rather than making it a line-pointer item: `magic` and `version`
  sit at page offsets 24 and 28 in both. A version-1 page has no line pointers
  and no special space, so a version-2 reader that consulted either one first
  would read past the end of the page instead of reporting the version — which
  is why the read order in [§ 3.1](#31-page-special-space) is an invariant (I2b)
  and not a suggestion.
- Writing a converter was considered and rejected: it would have to reconstruct
  a graph from a format that is about to be deleted, for an extension that has
  not shipped a release. `REINDEX` produces a better graph anyway.
- The changelog entry for the release carrying this format must say so plainly,
  since it is a user-visible operational step.

What disappears with the blob: the whole-index encode/decode framing, the
node-id → heap-TID table (heap TIDs are inline now), the whole-image rewrite on
insert, and the read lock scans took over the image. The graph's own
`to_bytes`/`from_bytes` codec is not part of this — it is useful for tests and
debugging, and keeping or dropping it is the implementation's call.

---

## 12. Left to the implementation

Deliberately not decided here, because they are code-shaped, not format-shaped:

- The exact `pg_sys` call sequences for buffer read/lock/mark-dirty and
  `PageAddItem`, and where a pgrx-side RAII wrapper is worth having to guarantee
  pins and locks are released on the error paths.
- Whether the paged `GraphStore` keeps a one-entry page cache (the last element
  page read) — a natural win, since neighbors written together tend to share a
  page, and a pure optimization with no format implications.
- Whether the layout pass ([§ 7.1](#71-build)) packs elements by insertion order
  or by layer. Insertion order is the obvious starting point; anything better is
  a measurable question, not a design one.
- The compaction trigger and the free-page reclamation policy (a later task
  owns them; the layout supports both).

---

## References

- pgvector's HNSW storage — metapage, element tuples, separate neighbor tuples,
  page identification: <https://github.com/pgvector/pgvector>
- PostgreSQL: [Database Page Layout](https://www.postgresql.org/docs/current/storage-page-layout.html),
  [Generic WAL Records](https://www.postgresql.org/docs/current/generic-wal.html),
  [Index Access Method Interface](https://www.postgresql.org/docs/current/indexam.html)
- [ARCHITECTURE.md](ARCHITECTURE.md) § "On storage (the honest tradeoff)" — why
  the interim format existed at all
- [FILTERING.md](FILTERING.md) § "Tier 1 — indexed attributes" — what the
  attribute row in an element tuple is for
