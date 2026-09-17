# Filtered + hybrid retrieval, in plain SQL

A five-minute, self-contained walkthrough of what makes Brindle different:
**semantic search that respects a `WHERE` clause, and hybrid ranking that fuses
vector similarity with full-text relevance** — the two queries a real
retrieval / RAG pipeline actually issues, both answered by one index and plain
SQL.

The corpus is a tiny product catalog (16 rows). No model download, no API key,
no network: the embeddings are checked in, so you can run every query below and
get exactly the output shown.

## Run it

From the repository root, open a Postgres shell with Brindle loaded:

```bash
cargo pgrx run pg17
```

Then, in the `psql` session it opens (its working directory is the repo root):

```sql
\i examples/rag/setup.sql     -- create the table, load 16 rows, build the indexes
\i examples/rag/queries.sql   -- run the three queries below
```

Already have Brindle installed in your own database? Point `psql` at it and run
the same two files.

## The data

Each product has a text `description` (the lexical signal), an `embedding` (the
semantic signal), and structured columns to filter on:

| column | type | role |
|---|---|---|
| `category_id` | `int` | filter — `1` Audio · `2` Outdoor · `3` Kitchen · `4` Office |
| `price` | `float8` | filter (range) |
| `rating`, `in_stock` | `real`, `bool` | more filter columns to play with |
| `description` | `text` | full-text side of the hybrid search |
| `embedding` | `brindle_vector` | vector side |

The Brindle index puts the vector first and the filter columns next to it — as
**key** columns, which is what lets a `WHERE` reach the graph traversal:

```sql
CREATE INDEX products_embedding_idx
    ON products USING brindle (embedding brindle_vector_cosine_ops, category_id, price);
```

### About the embeddings

They are 8-dimensional **hand-authored topic vectors**, not the output of a
neural model. Each dimension is a named concept and a product's vector is how
strongly it loads on each:

```
[ audio, wearable, outdoor, water, kitchen, office, fitness, wireless ]
```

So `Sport Wireless Earbuds` is `[0.7, 0.8, 0.1, 0.5, 0, 0, 0.8, 0.7]` — strong on
audio / wearable / fitness / wireless. Cosine distance over this space behaves
like semantic similarity while staying small and reproducible. [`embed.py`](embed.py)
is the source of truth for the vectors and prints the `INSERT` block in
`setup.sql`. To use a real embedding model instead, swap the vectors for its
output and widen the column to its dimensionality — the queries are unchanged.
That step is optional; the checked-in vectors keep this example dependency-free.

---

## 1. Plain semantic search

Find products *like* "compact wireless earbuds for workouts", ranked by the
vector alone — the baseline the next query narrows.

```sql
SELECT id, name, category_id, price,
       round((embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]')::numeric, 4) AS distance
FROM products
ORDER BY embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]'
LIMIT 5;
```

```
 id |            name            | category_id | price | distance
----+----------------------------+-------------+-------+----------
  3 | Sport Wireless Earbuds     |           1 | 59.99 |   0.0530
  4 | Noise-Cancelling Earbuds   |           1 |   129 |   0.1343
  1 | Studio Wireless Headphones |           1 | 89.99 |   0.3161
  8 | Rugged Wireless Headphones |           2 |    45 |   0.3729
  2 | Reference Studio Monitors  |           1 |   149 |   0.4097
```

Two of these top matches — the **$129** Noise-Cancelling Earbuds and the
**Outdoor**-category Rugged Wireless Headphones — are about to disappear.

## 2. Filtered semantic search — the differentiator

Same intent, but restricted to Audio products under \$100:

```sql
SELECT id, name, category_id, price,
       round((embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]')::numeric, 4) AS distance
FROM products
WHERE category_id = 1 AND price < 100
ORDER BY embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]'
LIMIT 5;
```

```
 id |            name            | category_id | price | distance
----+----------------------------+-------------+-------+----------
  3 | Sport Wireless Earbuds     |           1 | 59.99 |   0.0530
  1 | Studio Wireless Headphones |           1 | 89.99 |   0.3161
  5 | Waterproof Shower Speaker  |           1 | 24.99 |   0.5424
```

`Noise-Cancelling Earbuds` (price) and `Rugged Wireless Headphones` (category)
are gone, even though they were the 2nd- and 4th-nearest vectors overall. The
predicate is not applied *after* the vector search — it is pushed **into** the
graph traversal, so the search budget is spent only on rows that can be answers.
That is what keeps recall high when the filter is selective, which plain HNSW
indexes struggle with. The plan proves the qual reaches the index:

```sql
EXPLAIN (COSTS OFF)
SELECT id, name FROM products
WHERE category_id = 1 AND price < 100
ORDER BY embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]'
LIMIT 5;
```

```
 Limit
   ->  Index Scan using products_embedding_idx on products
         Index Cond: ((category_id = 1) AND (price < '100'::double precision))
         Order By: (embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]'::brindle_vector)
```

An `Index Cond` (not a `Filter`) is the tell: the predicate is evaluated during
the traversal. (`queries.sql` sets `enable_seqscan = off` because on 16 rows a
sequential scan is genuinely cheaper; on a production-sized table the planner
chooses the index on its own.)

## 3. Hybrid search — vector + full-text, fused with RRF

Vector search finds paraphrases the words miss; full-text search finds exact
terms the vector blurs. `brindle_hybrid` runs both and combines their *rankings*
with Reciprocal Rank Fusion, so a row **both** signals like beats a row only one
of them likes.

Here the lexical query is the phrase `wireless headphones`, and the vector
describes rugged outdoor gear:

```sql
SELECT h.rank, p.id, p.name,
       round(h.score::numeric, 5) AS score, h.vector_rank, h.text_rank
FROM brindle_hybrid(
         'products', 'id', 'embedding', 'tsv',
         'wireless headphones',
         '[0,0.7,0.9,0.6,0,0,0.8,0]'::brindle_vector,
         k => 5, config => 'english'
     ) WITH ORDINALITY AS h(id, score, vector_rank, text_rank, rank)
JOIN products p ON p.id = h.id
ORDER BY h.rank;
```

```
 rank | id |            name            |  score  | vector_rank | text_rank
------+----+----------------------------+---------+-------------+-----------
    1 |  8 | Rugged Wireless Headphones | 0.03202 |           4 |         1
    2 |  1 | Studio Wireless Headphones | 0.03002 |          12 |         2
    3 |  6 | Trail Running Vest         | 0.01639 |           1 |
    4 |  9 | Insulated Water Bottle     | 0.01613 |           2 |
    5 |  7 | Waterproof Hiking Jacket   | 0.01587 |           3 |
```

Read the last two columns — this is the whole point of fusing:

- **Trail Running Vest** is the single *nearest vector* (`vector_rank = 1`) but
  matches none of the words, so it lands only 3rd.
- **Studio Wireless Headphones** matches the words (`text_rank = 2`) but is
  semantically far from "rugged outdoor" (`vector_rank = 12`), so it lands 2nd.
- **Rugged Wireless Headphones** tops the *lexical* ranking (`text_rank = 1`) and
  is a solid vector match (`vector_rank = 4`). It leads neither list on the
  vector side, but being strong on *both* is exactly what RRF rewards — so it
  **wins**.

`vector_rank` / `text_rank` are `NULL` when a row surfaced from only one side —
free explainability for why a result is there. Drop the `query_text`, or pass a
phrase that matches nothing, and the fusion degrades cleanly to vector-only
ranking.

---

## What this demonstrates

- **Filter-aware ANN** — a `WHERE` on indexed columns pushed into the traversal
  (`Index Cond`), not post-filtered, so selective filters keep their recall.
- **Hybrid ranking** — `brindle_hybrid` fusing semantic and lexical rankings with
  RRF, with per-signal ranks for explainability.
- **Plain SQL, one index, no extra services** — the point of doing this inside
  Postgres.

More on the mechanics: [`docs/FILTERING.md`](../../docs/FILTERING.md) for the
filtered traversal, and the `brindle_hybrid` doc comment in
[`src/hybrid.rs`](../../src/hybrid.rs) for the fusion surface.
