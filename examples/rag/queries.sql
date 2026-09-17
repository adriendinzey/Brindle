-- Brindle RAG example — the three queries the walkthrough explains.
-- Load the corpus first (\i setup.sql), then run this file (\i queries.sql) or
-- copy the queries one at a time. Expected output is in README.md.

-- Small tables are faster to scan sequentially, so the planner would skip the
-- index here. Turn seqscan off so these queries exercise the Brindle index (on a
-- production-sized table it would choose the index on its own).
SET enable_seqscan = off;


-- 1) Plain semantic search -----------------------------------------------------
-- "Something like compact wireless earbuds for workouts", ranked by the vector
-- alone. No filter yet — this is the baseline the next query narrows.
SELECT id, name, category_id, price,
       round((embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]')::numeric, 4) AS distance
FROM products
ORDER BY embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]'
LIMIT 5;


-- 2) Filtered semantic search --------------------------------------------------
-- The same query intent, but only Audio (category_id = 1) products under $100.
-- The predicate is pushed INTO the graph traversal, so the search spends its
-- budget on rows that can actually be answers. Note what drops out versus query
-- 1: the $129 Noise-Cancelling Earbuds (price) and the Rugged Wireless
-- Headphones from the Outdoor category (category) are gone, even though they are
-- strong vector matches.
SELECT id, name, category_id, price,
       round((embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]')::numeric, 4) AS distance
FROM products
WHERE category_id = 1 AND price < 100
ORDER BY embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]'
LIMIT 5;

-- Proof the predicate reaches the index rather than being applied afterwards:
-- the plan is an Index Scan on products_embedding_idx with an Index Cond, not a
-- Filter.
EXPLAIN (COSTS OFF)
SELECT id, name
FROM products
WHERE category_id = 1 AND price < 100
ORDER BY embedding <=> '[0.7,0.8,0,0,0,0,0.8,0.7]'
LIMIT 5;


-- 3) Hybrid search (vector + full-text, fused with RRF) ------------------------
-- The lexical query is the phrase "wireless headphones"; the vector describes
-- rugged outdoor gear ([outdoor/water/fitness]). Reciprocal Rank Fusion combines
-- the two rankings, so a row both signals like beats a row only one likes:
--   * Trail Running Vest is the single NEAREST vector, but matches no words.
--   * Studio Wireless Headphones matches the words, but is semantically far.
--   * Rugged Wireless Headphones is strong on BOTH (top word match + a solid
--     vector) — and wins, though it leads neither list on the vector side.
-- vector_rank / text_rank show which signal(s) surfaced each row (NULL = missed).
SELECT h.rank,
       p.id,
       p.name,
       round(h.score::numeric, 5) AS score,
       h.vector_rank,
       h.text_rank
FROM brindle_hybrid(
         'products', 'id', 'embedding', 'tsv',
         'wireless headphones',
         '[0,0.7,0.9,0.6,0,0,0.8,0]'::brindle_vector,
         k => 5,
         config => 'english'
     ) WITH ORDINALITY AS h(id, score, vector_rank, text_rank, rank)
JOIN products p ON p.id = h.id
ORDER BY h.rank;


-- Leave the session as we found it (the seqscan override was only for the demo).
RESET enable_seqscan;
