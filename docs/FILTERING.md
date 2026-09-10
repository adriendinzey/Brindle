# Brindle — Filter-Aware Search (the differentiator)

This is the heart of the project: making vector search stay accurate when a SQL
predicate also has to hold. It's also the best thing to be able to explain in an
interview, so this doc keeps the *intuition* front and center.

---

## 1. Why naive filtering fails

Given `ORDER BY embedding <=> q WHERE p(row)` you have three naive options:

1. **Post-filter.** Run normal ANN for the top-`ef_search`, then drop rows where
   `p` is false. If `p` is selective (say 5% of rows match), almost everything the
   graph found gets discarded — the index spent its whole budget visiting nodes
   that don't qualify. Recall collapses.
2. **Pre-filter + brute force.** Compute the bitmap of rows where `p` holds, then
   scan them exactly. Great when the bitmap is small; O(n·d) and unusable when the
   predicate matches millions of rows.
3. **Pre-filter + per-predicate index.** Build a separate ANN index per filter
   value. Only works for a tiny, known set of predicates; combinatorial otherwise.

pgvector's **iterative scan** (v0.8) is a smarter version of (1): it keeps pulling
more candidates from the graph until enough pass the filter. It helps, but recall
and latency still degrade as selectivity rises because traversal is **blind to the
predicate** — it can't preferentially walk toward matching regions.

## 2. The idea: predicate-aware traversal (ACORN-style)

Brindle's approach follows [ACORN (Patel et al., 2024)](https://arxiv.org/abs/2403.04871):
keep the graph navigable *even after non-matching nodes are removed*, so the
search can hop across filtered-out nodes instead of dead-ending on them.

Two ingredients:

### (a) γ-dense edges at build time

A normal HNSW node keeps `M` neighbors. If a predicate filters out, say, 90% of
nodes, a node may have **zero** surviving neighbors → the matching subgraph
fragments → traversal gets stuck. ACORN's fix: build with **`M · γ`** candidate
neighbors per node (a denser graph), so that for predicates down to selectivity
~`1/γ`, each node still has enough *matching* neighbors to stay connected.

`γ` (gamma) is the predicate-robustness knob: higher `γ` → more resilient to
selective filters → more memory and build time. Brindle exposes it as a build
option and GUC.

### (b) predicate-filtered expansion at search time

During greedy search, when expanding a node's neighbor list, Brindle evaluates the
predicate on each neighbor and **only admits matching neighbors** to the result
set — but it still *uses* non-matching neighbors as stepping stones (ACORN's
"predicate subgraph traversal") to preserve reachability. Net effect: the
`ef_search` budget is spent entirely on nodes that can actually be answers.

```
for each candidate c popped from the frontier:
    matching = 0
    for n in neighbors(c):                   # neighbors() may be γ-dense
        if predicate.matches(n):             # cheap check against stored attrs
            consider(n); matching += 1       # frontier, and results if live

    if matching < m:                         # too thin to stay navigable
        for n in the non-matching neighbors: # at most m of them
            for nn in neighbors(n):          # ACORN: hop over n, never past nn
                if predicate.matches(nn):
                    consider(nn); matching += 1
                if matching >= m: break

    if matching == 0 and detours_left:       # no match within two hops at all
        for n in neighbors(c):
            consider(n)                      # routes only; never returned
```

**Why it terminates and stays cheap.** One expansion reaches at most two hops — a
node arrived at across a bridge is not itself bridged over *within that
expansion*, though it expands normally once it is popped — so each expansion is
bounded work. What terminates the search is the visited set: a node joins the
frontier only on first sight, so there are at most *n* expansions. The two-hop
scan also stops as soon as it has produced `m` matching neighbors, so an
unselective predicate pays essentially nothing. What bridging mostly spends is
*predicate evaluations* rather than distance computations: a two-hop node is
scored only if it matches, and Tier-1 attributes are tested inline without
touching a vector.

The last clause is the escape hatch for the case γ was supposed to prevent: under
a very selective predicate a node can have no match anywhere in its two-hop
neighborhood. Walking on through non-matching nodes (which still can never be
returned) beats returning nothing. Unlike the two-hop scan, this branch *does*
score non-matching nodes — that is how it walks through them — so the allowance
is charged per node **enqueued**, not per stranded node. Charging per stranded
node would let one unit of allowance queue an entire neighbor list, each member of
which then pays a two-hop scan of its own when popped: work quadratic in the
degree, and so worst on exactly the γ-dense graphs the feature is built around.
With the per-node charge, an unsatisfiable predicate over a 20 000-node graph
costs ~530 expansions and ~610 vector distances per query, and that bill is flat
in the size of the graph.

### (c) a predicate-aware descent, so the search starts somewhere useful

(a) and (b) both work *locally*: they keep the matching subgraph connected around
wherever the search already is. Neither answers the prior question of **where to
start**, and that is a separate failure with a much sharper edge.

The layer descent picks an entry point near the *query*. When the predicate
correlates with position — a tenant whose documents cluster, a price band, a
date range — the matching rows are not near the query at all, and layer 0 cannot
get to them: one hop there covers one node's spacing, so crossing a region of
*n* non-matching nodes costs an allowance proportional to *n*. Measured on a
10 000-node grid whose filter selects columns 46 apart, this returned **zero
rows** — not degraded recall, none — at every `ef_search` below 5000, while an
uncorrelated filter of the same 5% selectivity was answered essentially
perfectly. Raising γ does not help: γ = 1, 4 and 16 all returned nothing.
Densifying edges reconnects a *thinned* neighborhood; it does not move the
search somewhere else.

The upper layers are the answer, because they are built for exactly this — each
one is sparser, so a hop there covers far more ground. So the descent does two
things per layer instead of one:

```
nav  = search_layer(query, [nav], ef=1, layer=lc)              # unfiltered, as before
seed = search_layer(query, [nav, seed], ef=1, layer=lc, pred)  # nearest match on this layer
```

Navigation is untouched: the nearest node to the query does not depend on the
filter, and filtering that walk would move where an unfiltered search lands. The
second call is the new one — a predicate-aware probe, seeded from where
navigation just arrived *and* from what the layer above found, which hands one
foothold in the matching set down to the next layer. Layer 0 is then entered
from both: the usual entry point, and a node that actually matches. It starts
inside the matching region instead of tens of hops from it.

One foothold per layer is enough; carrying 2, 4 or 8 down measured no better,
because what the probe has to get right is *which region*, and the layer below
re-probes from wherever it lands.

### What bounds all this

A filtered search may walk through nodes it can never return, so the result heap
cannot be what stops it — under a selective predicate that heap is exactly what
stays empty. Two allowances do, both totals for the whole search (a per-layer
allowance would multiply by a layer count that grows with the graph) and both
sized from `ef_search`, but as separate multiples of it: the width of the result
beam and the distance a search must cover to *find* results answer different
questions, and tying them together is what made a matching region a few dozen
hops away unreachable at any sane `ef_search`.

| Allowance | Default | Scope | What it bounds |
|---|---|---|---|
| detours | `4 × ef_search` non-matching nodes enqueued | one for the descent, one for layer 0 | the walk through a region with no matches within two hops |
| expansions | `16 × ef_search` nodes popped and expanded | the whole search | the search as a whole, whatever the predicate does |

The detour allowance is *per phase* rather than per search, because the descent
probe and the layer-0 walk spend it on different jobs — finding a region that
matches, then searching inside it — and the probe's job is the one with no
natural limit, since a probe that finds nothing keeps looking. Sharing one pot
lets the probe arrive at layer 0 with nothing left, which is exactly the query
that most needs a fallback there: measured on a 100 000-node graph with 18
matching rows, one pot returned no rows at all where two return some. It is the
split that matters, not the total; both halves stay fixed multiples of
`ef_search`, so the bill is still flat in the size of the graph.

The second exists because the first cannot bound everything. A node that
*matches* but is tombstoned never triggers a detour — it satisfies the predicate
— yet it can never fill the result heap either, and a filtered search
deliberately keeps going while that heap is under-filled. With every match
deleted, nothing about the predicate ends the walk: measured, the search expanded
206 nodes at n = 2000 and 2007 at n = 20 000, i.e. Θ(*n*). The expansion
allowance holds it to 235 and 1036 — the second being the ceiling itself.

The *unfiltered* path has the same shape of hole and still has it: with every row
tombstoned it expands 2143 nodes at n = 2000 and 21 329 at n = 20 000. Bounding
that means changing where an unfiltered search stops, which is a decision about
plain HNSW recall rather than about filtering, so it is left alone here — the
allowances above are drawn on by the filtered path only, and an unfiltered search
is unchanged in results and in cost.

The detour default is 4× rather than 1× because it is measurably free where it
is not needed. On the correlated fixture it lifts recall@10 from 0.93 to 1.00 at
5% selectivity and from 0.67 to 0.90 at 1%; on an *uncorrelated* filter recall
and cost are identical at 1× and 4×, to the distance, because a detour is only
ever charged where two hops turn up no match at all, which there is almost
nowhere. What it does cost is ~3× on a predicate nothing satisfies — 144
expansions per query at 1×, 528 at 4× — a bill that stays flat in the size of
the graph.

## 3. How predicates reach the index

The honest hard part in Postgres is *getting the predicate to the traversal cheaply*.
Brindle ships this in increasing order of difficulty:

### Tier 1 — indexed attributes (first target)

At `CREATE INDEX` time, the user declares which columns participate in filtering
as **key columns after the vector**; Brindle stores those attribute values
**inside the index** next to each vector:

```sql
CREATE INDEX ON docs USING brindle (embedding, tenant_id, status, price);
```

**Not `INCLUDE (...)`, and the distinction is the whole mechanism.** Postgres
matches a `WHERE` clause to an index column only if that column is part of the
*search key*. An `INCLUDE` column is payload: it can satisfy an index-only scan,
but a qual on it never reaches the access method — the planner leaves it as an
executor `Filter`, so the scan returns its `ef_search` candidates and the filter
is applied afterwards. That is post-filtering, which is the thing this design
exists to avoid, and it under-fills `LIMIT k` exactly when the predicate is
selective enough to matter.

An earlier draft of this document specified `INCLUDE`. It was wrong: measured,
the access method received `nkeys = 0` and the plan read `Filter:` rather than
`Index Cond:`.

Filterable columns must have a brindle operator class, which ships for `bool`,
`int2`, `int4`, `int8`, `float4` and `float8`. Anything else is refused at
`CREATE INDEX` with Postgres's own "no default operator class" error rather than
being silently unfilterable.

Supported predicate shapes in Tier 1, as shipped:
- **equality**: `tenant_id = 42` — on any of the numeric types above.
- **range**: `price < 50`, `score BETWEEN 1 AND 9` — `<`, `<=`, `>=`, `>`.
- **conjunctions** of the above (`AND`).

Comparisons work across widths within a family, so `bigint_col = 42` pushes
without the literal needing a cast; integers and floats do not mix, because the
stored value and the bound have to compare as one type.

These are evaluated with zero heap access during traversal — the whole point.

Not yet shipped, and refused at `CREATE INDEX` rather than silently ignored:
**string labels** (`status = 'active'`, which wants the dictionary encoding this
document describes for `AttrValue::Int`) and **dates and timestamps** (which
would map onto the integer path). `OR` and `NOT` are Tier 1 gaps too: the
predicate model has an `And` conjunction only, and anything else stays with the
executor.

Anything the index cannot express is left to the executor rather than dropped,
and the scan reports a recheck for it — so a refused qual costs recall, never
correctness.

### Tier 2 — bitmap handoff

For predicates Brindle doesn't store inline, accept a precomputed **`roaring`-style
bitmap** of qualifying row TIDs (produced by other Postgres indexes via a bitmap
scan) and intersect during traversal. Bridges arbitrary predicates at the cost of
building the bitmap first.

### Tier 3 — arbitrary expression pushdown (research frontier)

Calling back into the executor to evaluate an arbitrary `WHERE` expression per
visited node is the fully general version. It's expensive and fiddly (expression
context, memory contexts, visibility) — documented as a frontier, not promised.
Being explicit about this boundary is part of the project's credibility.

## 4. Parameters

| Knob | Where | Meaning | Trade-off |
|---|---|---|---|
| `m` | build | base neighbors/node | recall vs size |
| `ef_construction` | build | candidate pool at build | build quality vs time |
| `gamma` (γ) | build | edge density multiplier for filter-robustness | filter recall vs memory/build |
| `brindle.ef_search` | query GUC | candidate pool at search | recall vs latency |

`ef_search` is the only query-time knob: the filtered path's two allowances are
fixed multiples of it (§2, "What bounds all this") rather than settings of their
own, so raising it widens the beam and the reach together.

## 5. How we'll prove it works

The claim "Brindle keeps recall under filters" is meaningless without numbers, so
Phase 5 includes a harness that sweeps **predicate selectivity** (100% → 0.1%) and
plots **recall@10** and **QPS** for:

- Brindle (γ-dense + predicate-aware),
- pgvector post-filter,
- pgvector iterative scan,
- brute-force exact (the recall ceiling).

The deliverable is a chart in the README showing where predicate-aware traversal
wins. If it *doesn't* win in some regime, we say so — that honesty is worth more
than a marketing number.

## References

- Patel et al., *ACORN: Performant and Predicate-Agnostic Search Over Vector
  Embeddings and Structured Data* (2024) — https://arxiv.org/abs/2403.04871
- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search
  using HNSW graphs* (2016) — https://arxiv.org/abs/1603.09320
- pgvector iterative scan — https://github.com/pgvector/pgvector
