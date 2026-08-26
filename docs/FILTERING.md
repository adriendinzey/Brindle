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
costs ~0.9 ms and ~36 k predicate evaluations per query, and that bill is flat in
the size of the graph.

## 3. How predicates reach the index

The honest hard part in Postgres is *getting the predicate to the traversal cheaply*.
Brindle ships this in increasing order of difficulty:

### Tier 1 — indexed attributes (first target)

At `CREATE INDEX` time, the user declares which columns participate in filtering;
Brindle stores those attribute values **inside the index** next to each vector:

```sql
CREATE INDEX ON docs USING brindle (embedding vector_cosine_ops)
  INCLUDE (tenant_id, status, price);     -- filterable attrs co-located
```

Supported predicate shapes in Tier 1:
- **equality / label**: `tenant_id = 42`, `status = 'active'` → compact label
  dictionary + per-node label bits, matched with a bitwise test.
- **numeric range**: `price < 50`, `created_at BETWEEN ...` → store the scalar,
  compare during traversal.
- **conjunctions** of the above (`AND`).

These are evaluated with zero heap access during traversal — the whole point.

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
