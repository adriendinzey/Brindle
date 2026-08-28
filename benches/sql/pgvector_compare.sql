-- pgvector side-by-side, run against the fixture index_baseline.sql just built.
--
-- Same rows, same query vectors, same ground truth, same k, same ef sweep, and
-- matched build parameters (m, ef_construction). Run this immediately after
-- index_baseline.sql in the same database, or the tables it reads will not
-- exist.
--
-- One asymmetry that cannot be matched and must not be buried: pgvector's
-- Makefile compiles with -march=native -ftree-vectorize -fassociative-math, so
-- its distance kernels get machine-specific SIMD. Brindle is built by
-- `cargo build --release` with no target-cpu flag, i.e. baseline x86-64. Any
-- latency gap includes that, and it is not a gap in the algorithm.

\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;

-- This file runs in its own psql session, so the setting index_baseline.sql
-- left behind is gone; k comes back in the same way.
SELECT set_config('bench.k', :'k', false);

-- pgvector builds its graph inside maintenance_work_mem and warns loudly when
-- it spills ("no longer fits ... building will take significantly more time").
-- Brindle's build ignores the setting altogether and allocates in backend
-- memory, so leaving the default here would time pgvector against a limit its
-- competitor is not subject to. Raised for the build, and recorded in the
-- write-up as one of the ways this comparison is not apples to apples.
SET maintenance_work_mem = '2GB';

DROP TABLE IF EXISTS pgv_vectors, pgv_timings, pgv_recall;

-- brindle_vector's text form is pgvector's input form, which is what makes this
-- a copy rather than a conversion.
CREATE TABLE pgv_vectors AS
SELECT id, embedding::text::vector AS embedding FROM bench_vectors;
ALTER TABLE pgv_vectors ADD PRIMARY KEY (id);

DO $$
DECLARE dims int;
BEGIN
    SELECT vector_dims(embedding) INTO dims FROM pgv_vectors LIMIT 1;
    EXECUTE format('ALTER TABLE pgv_vectors ALTER COLUMN embedding TYPE vector(%s)', dims);
END $$;

VACUUM ANALYZE pgv_vectors;

CREATE TABLE pgv_timings (phase text, ef int, elapsed_ms double precision);
CREATE TABLE pgv_recall (ef int, query_id int, hits int);

-- Matched to brindle's defaults: m = 16, ef_construction = 64.
DO $$
DECLARE started timestamptz;
BEGIN
    started := clock_timestamp();
    CREATE INDEX pgv_idx ON pgv_vectors USING hnsw (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 64);
    INSERT INTO pgv_timings
    VALUES ('build', NULL, extract(epoch FROM clock_timestamp() - started) * 1000);
END $$;

DO $$
DECLARE
    k       int := current_setting('bench.k')::int;
    ef      int;
    q       record;
    started timestamptz;
    found   int[];
BEGIN
    FOREACH ef IN ARRAY ARRAY[16, 32, 64, 128, 256] LOOP
        PERFORM set_config('hnsw.ef_search', ef::text, false);

        FOR q IN SELECT id, embedding::text::vector AS embedding
                 FROM bench_queries ORDER BY id LOOP
            started := clock_timestamp();
            SELECT array_agg(id) INTO found
            FROM (SELECT id FROM pgv_vectors
                  ORDER BY embedding <-> q.embedding
                  LIMIT k) s;
            INSERT INTO pgv_timings
            VALUES ('query', ef, extract(epoch FROM clock_timestamp() - started) * 1000);

            -- Against the same ground truth brindle was scored on.
            INSERT INTO pgv_recall
            SELECT ef, q.id,
                   (SELECT count(*) FROM unnest(found) f WHERE f = ANY (t.ids))
            FROM bench_truth t WHERE t.query_id = q.id;
        END LOOP;
    END LOOP;
END $$;

\echo ''
\echo '=== pgvector: build ==='
SELECT round((elapsed_ms / 1000.0)::numeric, 1) AS build_seconds
FROM pgv_timings WHERE phase = 'build';

\echo ''
\echo '=== pgvector: query latency and recall by ef_search ==='
WITH latency AS (
    SELECT ef,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY elapsed_ms) AS p50,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms) AS p95
    FROM pgv_timings WHERE phase = 'query' GROUP BY ef
), quality AS (
    SELECT ef, avg(hits)::numeric / current_setting('bench.k')::int AS recall
    FROM pgv_recall GROUP BY ef
)
SELECT l.ef AS ef_search,
       round(l.p50::numeric, 2) AS p50_ms,
       round(l.p95::numeric, 2) AS p95_ms,
       round(q.recall, 3) AS recall_at_k
FROM latency l JOIN quality q USING (ef)
ORDER BY l.ef;

\echo ''
\echo '=== pgvector: index size ==='
SELECT pg_size_pretty(pg_relation_size('pgv_idx')) AS index_size;
