-- Brindle index baseline: build time, query latency, recall vs ef_search.
--
-- Driven by scripts/bench_index.sh, which sets :rows, :dims, :queries and :k.
-- Everything is timed server-side with clock_timestamp() rather than from the
-- client, so the numbers exclude psql round-trip and connection cost.

\set ON_ERROR_STOP on
\timing off

CREATE EXTENSION IF NOT EXISTS brindle;

DROP TABLE IF EXISTS bench_vectors, bench_queries, bench_timings, bench_recall,
                     bench_truth, bench_centroids;

-- Deterministic data. setseed makes a rerun on this machine reproduce exactly;
-- it does not make two machines agree, which is why the write-up records both.
SELECT setseed(0.42);

-- One vector per row, built by aggregating a row per dimension. The obvious
-- spelling — a scalar subquery over generate_series(1, dims) — does NOT work:
-- it does not reference the outer row, so the planner hoists it to an InitPlan,
-- evaluates it once, and every row gets the *same* vector. That produces a
-- 100k-row table with one distinct value, on which every search is trivially
-- perfect and every number here is meaningless. The check below exists because
-- this benchmark shipped that bug once.
CREATE TABLE bench_vectors (id int primary key, embedding brindle_vector);
CREATE TABLE bench_queries (id int primary key, embedding brindle_vector);

-- Two fixture shapes, because they answer different questions.
--
--   uniform   — every component independent and uniform. In 128 dimensions this
--               is the worst case a graph index can face: distances concentrate
--               so hard that the 1000th neighbour sits ~1.1x the distance of the
--               1st, leaving greedy search no gradient to follow. Recall here
--               measures the fixture, not the implementation.
--   clustered — 100 centroids with local noise, which is roughly how real
--               embeddings sit: locally dense, globally separated. This is the
--               shape whose recall curve is worth comparing against later.
\if :{?clustered}
CREATE TABLE bench_centroids (id int primary key, v real[]);
INSERT INTO bench_centroids
SELECT c, array_agg((random() - 0.5)::real)
FROM generate_series(1, 100) c, generate_series(1, :dims) d
GROUP BY c;

INSERT INTO bench_vectors
SELECT r.i,
       (SELECT array_agg((c.v[ord] + (random() - 0.5) * 0.25)::real ORDER BY ord)
        FROM generate_series(1, :dims) ord)::real[]::brindle_vector
FROM (SELECT i, 1 + (i % 100) AS cid FROM generate_series(1, :rows) i) r
JOIN bench_centroids c ON c.id = r.cid;

INSERT INTO bench_queries
SELECT r.i,
       (SELECT array_agg((c.v[ord] + (random() - 0.5) * 0.25)::real ORDER BY ord)
        FROM generate_series(1, :dims) ord)::real[]::brindle_vector
FROM (SELECT i, 1 + (i % 100) AS cid FROM generate_series(1, :queries) i) r
JOIN bench_centroids c ON c.id = r.cid;
\else
INSERT INTO bench_vectors
SELECT i, array_agg((random() - 0.5)::real)::real[]::brindle_vector
FROM generate_series(1, :rows) i, generate_series(1, :dims) d
GROUP BY i;

INSERT INTO bench_queries
SELECT i, array_agg((random() - 0.5)::real)::real[]::brindle_vector
FROM generate_series(1, :queries) i, generate_series(1, :dims) d
GROUP BY i;
\endif

-- Refuse to measure a degenerate fixture.
DO $$
DECLARE rows_n bigint; distinct_n bigint;
BEGIN
    SELECT count(*), count(DISTINCT embedding::text) INTO rows_n, distinct_n
    FROM bench_vectors;
    IF distinct_n < rows_n THEN
        RAISE EXCEPTION 'fixture is degenerate: % rows but only % distinct vectors',
                        rows_n, distinct_n;
    END IF;
END $$;

VACUUM ANALYZE bench_vectors;

-- ---------------------------------------------------------------- build time
CREATE TABLE bench_timings (phase text, ef int, elapsed_ms double precision);

DO $$
DECLARE started timestamptz;
BEGIN
    started := clock_timestamp();
    CREATE INDEX bench_idx ON bench_vectors USING brindle (embedding);
    INSERT INTO bench_timings
    VALUES ('build', NULL,
            extract(epoch FROM clock_timestamp() - started) * 1000);
END $$;

-- ------------------------------------------------------- latency and recall
--
-- One row per (ef_search, query): how long the index scan took, and how much
-- of the exact top-k it found. The exact side orders by the distance
-- *function*, which the planner cannot answer from the index — the same trick
-- the recall test uses, for the same reason.
CREATE TABLE bench_recall (ef int, query_id int, hits int);

-- The exact top-k does not depend on ef_search, so it is computed once per
-- query rather than once per (ef, query) — five brute-force passes over the
-- whole table per query would otherwise dwarf the thing being measured.
CREATE TABLE bench_truth (query_id int primary key, ids int[]);

-- psql does not substitute :variables inside a dollar-quoted body, so the one
-- parameter the loops need is handed over as a setting instead.
SELECT set_config('bench.k', :'k', false);

DO $$
DECLARE
    k int := current_setting('bench.k')::int;
    q record;
BEGIN
    FOR q IN SELECT id, embedding FROM bench_queries ORDER BY id LOOP
        INSERT INTO bench_truth
        SELECT q.id, array_agg(id)
        FROM (SELECT id FROM bench_vectors
              ORDER BY brindle_vector_l2_distance(embedding, q.embedding)
              LIMIT k) t;
    END LOOP;
END $$;

-- Refuse to report latency for a plan that never touched the index. The
-- benchmark's whole claim is "this is what a brindle index scan costs"; a
-- planner that quietly preferred a sequential scan would make every number
-- below a measurement of something else.
DO $$
DECLARE q brindle_vector; line text; uses_index bool := false;
BEGIN
    SELECT embedding INTO q FROM bench_queries ORDER BY id LIMIT 1;
    PERFORM set_config('brindle.ef_search', '64', false);
    FOR line IN EXPLAIN (COSTS OFF)
        SELECT id FROM bench_vectors ORDER BY embedding <-> q
        LIMIT current_setting('bench.k')::int
    LOOP
        IF line LIKE '%Index Scan using bench_idx%' THEN
            uses_index := true;
        END IF;
    END LOOP;
    IF NOT uses_index THEN
        RAISE EXCEPTION 'the measured query does not use bench_idx; these numbers would not be index numbers';
    END IF;
END $$;

DO $$
DECLARE
    k         int := current_setting('bench.k')::int;
    ef        int;
    q         record;
    started   timestamptz;
    found     int[];
BEGIN
    FOREACH ef IN ARRAY ARRAY[16, 32, 64, 128, 256] LOOP
        PERFORM set_config('brindle.ef_search', ef::text, false);

        FOR q IN SELECT id, embedding FROM bench_queries ORDER BY id LOOP
            -- Timed region is the index scan alone: the exact side is already
            -- computed, and nothing else runs between the two clock reads.
            started := clock_timestamp();
            SELECT array_agg(id) INTO found
            FROM (SELECT id FROM bench_vectors
                  ORDER BY embedding <-> q.embedding
                  LIMIT k) s;
            INSERT INTO bench_timings
            VALUES ('query', ef,
                    extract(epoch FROM clock_timestamp() - started) * 1000);

            INSERT INTO bench_recall
            SELECT ef, q.id,
                   (SELECT count(*) FROM unnest(found) f WHERE f = ANY (t.ids))
            FROM bench_truth t WHERE t.query_id = q.id;
        END LOOP;
    END LOOP;
END $$;

-- ------------------------------------------------------------------ results
\echo ''
\echo '=== build ==='
SELECT round(elapsed_ms)::text || ' ms' AS build_time,
       :rows AS rows,
       :dims AS dims
FROM bench_timings WHERE phase = 'build';

\echo ''
\echo '=== query latency and recall by ef_search ==='
-- Aggregated separately and then joined: a join on ef alone would pair every
-- timing with every recall row, and a percentile over a cartesian product is a
-- number nobody can explain later.
WITH latency AS (
    SELECT ef,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY elapsed_ms) AS p50,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms) AS p95
    FROM bench_timings WHERE phase = 'query' GROUP BY ef
), quality AS (
    SELECT ef, avg(hits)::numeric / :k AS recall FROM bench_recall GROUP BY ef
)
SELECT l.ef AS ef_search,
       round(l.p50::numeric, 2) AS p50_ms,
       round(l.p95::numeric, 2) AS p95_ms,
       round(q.recall, 3) AS recall_at_k
FROM latency l JOIN quality q USING (ef)
ORDER BY l.ef;

\echo ''
\echo '=== distance concentration (how hard this fixture is) ==='
WITH q AS (SELECT embedding FROM bench_queries ORDER BY id LIMIT 1),
     d AS (SELECT brindle_vector_l2_distance(v.embedding, q.embedding) AS dist
           FROM bench_vectors v, q ORDER BY dist LIMIT 1000)
SELECT round(min(dist)::numeric, 3) AS nearest,
       round((array_agg(dist))[1000]::numeric, 3) AS thousandth,
       round(((array_agg(dist))[1000] / min(dist))::numeric, 3) AS ratio
FROM d;

\echo ''
\echo '=== index size ==='
SELECT pg_size_pretty(pg_relation_size('bench_idx')) AS index_size,
       pg_size_pretty(pg_relation_size('bench_vectors')) AS heap_size;
