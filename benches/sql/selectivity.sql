-- Filtered search across predicate selectivity: Brindle vs pgvector.
--
-- Driven by scripts/bench_index.sh with SELECTIVITY=1, after index_baseline.sql
-- has built `bench_vectors` and `bench_queries` and pgvector_compare.sql has
-- copied them into `pgv_vectors`. Same rows, same query vectors, same k.
--
-- This is the benchmark the project exists to produce. Three ways to answer
-- "the k nearest rows that also satisfy a predicate":
--
--   brindle    the predicate is pushed into the graph traversal, so the search
--              budget is spent only on rows that can be answers
--   pgv_post   pgvector returns its ef_search nearest rows and the executor
--              filters afterwards -- the naive approach, and the one that
--              collapses as the filter tightens
--   pgv_iter   pgvector 0.8's iterative scan: keep pulling more candidates from
--              the graph until enough pass the filter. Smarter post-filtering,
--              but traversal is still blind to the predicate
--
-- with an exact sequential scan as the recall ceiling.
--
-- TWO LABEL SHAPES, AND THAT IS THE POINT. Both are `ntile(100)`, so both have
-- exactly the same marginal distribution and `bucket <= t` selects exactly t% of
-- rows in each. They differ in one thing only:
--
--   lbl_spread  ntile over a hash of the id -- independent of where the vector
--               sits. This is what every earlier recall test in this repo used.
--   lbl_local   ntile over distance from a fixed reference point -- so matching
--               rows form a *region* of the vector space.
--
-- Real predicates are usually the second kind: category correlates with content,
-- price with product type, tenant with topic. Measuring only the first is how
-- this project shipped a search that returned *zero* rows on a correlated filter
-- for four tasks running -- see docs/FILTERING.md § 2(c).

\set ON_ERROR_STOP on
\timing off

DROP TABLE IF EXISTS sel_truth, sel_timings, sel_recall, sel_points;

SELECT set_config('bench.k', :'k', false);

-- ------------------------------------------------------------------ labels

ALTER TABLE bench_vectors ADD COLUMN IF NOT EXISTS lbl_spread int;
ALTER TABLE bench_vectors ADD COLUMN IF NOT EXISTS lbl_local int;

-- A fixed reference point for the correlated label: the first query vector, so
-- the matching regions sit somewhere the queries actually are.
CREATE TEMP TABLE sel_ref AS SELECT embedding FROM bench_queries ORDER BY id LIMIT 1;

UPDATE bench_vectors v
SET lbl_spread = s.b_spread, lbl_local = s.b_local
FROM (
    SELECT id,
           ntile(100) OVER (ORDER BY hashint4(id))              AS b_spread,
           ntile(100) OVER (ORDER BY embedding <-> (SELECT embedding FROM sel_ref))
                                                                AS b_local
    FROM bench_vectors
) s
WHERE v.id = s.id;

-- The fixture is only a fixture if the two labels really do differ in
-- correlation. Measured as the spread of matching rows' distance-from-reference:
-- a correlated label concentrates them, an uncorrelated one does not.
DO $$
DECLARE spread_hi double precision; local_hi double precision;
BEGIN
    SELECT max(d) INTO local_hi FROM (
        SELECT embedding <-> (SELECT embedding FROM sel_ref) AS d
        FROM bench_vectors WHERE lbl_local <= 5) s;
    SELECT max(d) INTO spread_hi FROM (
        SELECT embedding <-> (SELECT embedding FROM sel_ref) AS d
        FROM bench_vectors WHERE lbl_spread <= 5) s;
    IF local_hi >= spread_hi THEN
        RAISE EXCEPTION
            'the two labels are not differently correlated: at 5%% selectivity '
            'the local label reaches % from the reference and the spread label %',
            local_hi, spread_hi;
    END IF;
END $$;

ALTER TABLE pgv_vectors ADD COLUMN IF NOT EXISTS lbl_spread int;
ALTER TABLE pgv_vectors ADD COLUMN IF NOT EXISTS lbl_local int;
UPDATE pgv_vectors p
SET lbl_spread = v.lbl_spread, lbl_local = v.lbl_local
FROM bench_vectors v WHERE v.id = p.id;

VACUUM ANALYZE bench_vectors;
VACUUM ANALYZE pgv_vectors;

-- ------------------------------------------------------------------ indexes

-- Brindle: the attribute columns are KEY columns after the vector, which is what
-- makes a qual on them reach the traversal at all (docs/FILTERING.md § 3).
DROP INDEX IF EXISTS sel_brindle_idx;
CREATE INDEX sel_brindle_idx ON bench_vectors
    USING brindle (embedding, lbl_spread, lbl_local) WITH (m = 16, ef_construction = 64);

-- pgvector: one index on the vector alone, which is all pgvector can do. The
-- filter is the executor's problem either way -- that is the comparison.
--
-- maintenance_work_mem is raised for the same reason pgvector_compare.sql
-- raises it: pgvector builds its graph inside that limit and warns loudly when
-- it spills ("no longer fits ... building will take significantly more time"),
-- while brindle ignores the setting and allocates in backend memory. Leaving
-- the default would hand pgvector a *degraded graph* and then report its recall
-- as if the algorithm produced it. The first run of this file did exactly that
-- and the numbers were quietly unfair.
DROP INDEX IF EXISTS sel_pgv_idx;
DO $$
DECLARE built_bytes bigint; limit_bytes bigint;
BEGIN
    SET LOCAL maintenance_work_mem = '2GB';
    CREATE INDEX sel_pgv_idx ON pgv_vectors
        USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);

    -- ...and it must not have spilled anyway. The build only WARNS when it does,
    -- which is easy to lose in a long run, so check it here where the LOCAL
    -- setting is still in force -- outside this block it has already reverted.
    built_bytes := pg_relation_size('sel_pgv_idx');
    limit_bytes := pg_size_bytes(current_setting('maintenance_work_mem'));
    IF built_bytes > limit_bytes THEN
        RAISE EXCEPTION
            'the pgvector index is % against a maintenance_work_mem of %, so its '
            'build spilled: its recall below would measure that limit rather than '
            'pgvector. Raise the setting or lower the row count.',
            pg_size_pretty(built_bytes), pg_size_pretty(limit_bytes);
    END IF;
END $$;

-- The baseline's own index would also match `ORDER BY embedding <-> q`, and the
-- planner may prefer it; drop it so the measured plan is unambiguous.
DROP INDEX IF EXISTS bench_idx;

-- ------------------------------------------------- ground truth (exact, once)

CREATE TABLE sel_points (shape text, sel int);
INSERT INTO sel_points
SELECT shape, sel FROM unnest(ARRAY['spread', 'local']) shape,
                       unnest(ARRAY[50, 10, 5, 1]) sel;

CREATE TABLE sel_truth (shape text, sel int, query_id int, ids int[],
                        PRIMARY KEY (shape, sel, query_id));

-- Exact, and it must stay exact: if an index ever served this the whole
-- benchmark would be comparing the index against itself.
DO $$
DECLARE line text; uses_index bool := false; q brindle_vector;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_bitmapscan = off;
    SELECT embedding INTO q FROM bench_queries ORDER BY id LIMIT 1;
    FOR line IN EXPLAIN (COSTS OFF)
        SELECT id FROM bench_vectors WHERE lbl_spread <= 5
        ORDER BY brindle_vector_l2_distance(embedding, q) LIMIT 10
    LOOP
        IF line LIKE '%Index Scan%' THEN uses_index := true; END IF;
    END LOOP;
    IF uses_index THEN
        RAISE EXCEPTION 'the ground-truth query is using an index; it is not an exact baseline';
    END IF;
END $$;

DO $$
DECLARE
    k int := current_setting('bench.k')::int;
    p record; q record;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_bitmapscan = off;
    FOR p IN SELECT shape, sel FROM sel_points LOOP
        FOR q IN SELECT id, embedding FROM bench_queries ORDER BY id LOOP
            EXECUTE format(
                'INSERT INTO sel_truth SELECT %L, %s, %s, array_agg(id) FROM ('
                '  SELECT id FROM bench_vectors WHERE %I <= %s'
                '  ORDER BY brindle_vector_l2_distance(embedding, $1) LIMIT %s) t',
                p.shape, p.sel, q.id, 'lbl_' || p.shape, p.sel, k)
            USING q.embedding;
        END LOOP;
    END LOOP;
END $$;

-- A selectivity point is only interesting if the exact answer has k rows to
-- find. Below that, every engine is measured against a short ceiling.
DO $$
DECLARE thin int;
BEGIN
    SELECT count(*) INTO thin FROM sel_truth
    WHERE coalesce(array_length(ids, 1), 0) < current_setting('bench.k')::int;
    IF thin > 0 THEN
        RAISE EXCEPTION
            '% ground-truth answers hold fewer than k rows; lower the selectivity '
            'points or raise the row count', thin;
    END IF;
END $$;

-- ------------------------------------------------------------------ measure

CREATE TABLE sel_timings (engine text, shape text, sel int, ef int,
                          elapsed_ms double precision, query_id int);
CREATE TABLE sel_recall (engine text, shape text, sel int, ef int,
                         query_id int, hits int);

-- Every measured plan must actually use the index it claims to. A planner that
-- fell back to a sequential scan would turn this into a comparison of seq scans.
DO $$
DECLARE line text; ok bool; q brindle_vector; qv text;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT embedding INTO q FROM bench_queries ORDER BY id LIMIT 1;
    qv := q::text;

    ok := false;
    FOR line IN EXECUTE format(
        'EXPLAIN (COSTS OFF) SELECT id FROM bench_vectors WHERE lbl_spread <= 5 '
        'ORDER BY embedding <-> %L::brindle_vector LIMIT 10', qv)
    LOOP
        IF line LIKE '%sel_brindle_idx%' THEN ok := true; END IF;
        IF line LIKE '%Filter: (lbl_spread%' THEN
            RAISE EXCEPTION 'brindle is post-filtering: the qual did not reach the index:%',
                E'\n' || line;
        END IF;
    END LOOP;
    IF NOT ok THEN
        RAISE EXCEPTION 'the brindle arm does not use sel_brindle_idx';
    END IF;

    ok := false;
    FOR line IN EXECUTE format(
        'EXPLAIN (COSTS OFF) SELECT id FROM pgv_vectors WHERE lbl_spread <= 5 '
        'ORDER BY embedding <-> %L::vector LIMIT 10', qv)
    LOOP
        IF line LIKE '%sel_pgv_idx%' THEN ok := true; END IF;
    END LOOP;
    IF NOT ok THEN
        RAISE EXCEPTION 'the pgvector arm does not use sel_pgv_idx';
    END IF;
END $$;

DO $$
DECLARE
    k       int := current_setting('bench.k')::int;
    efs     int[] := ARRAY[64, 256];
    ef      int;
    p       record;
    q       record;
    started timestamptz;
    found   int[];
    warm    int;
    col     text;
BEGIN
    SET LOCAL enable_seqscan = off;

    -- Warm every engine before anything is timed, so no arm pays a cold cache
    -- that the others do not. Brindle in particular decodes the whole index on
    -- a backend's first scan (docs/BENCHMARKS.md), which would otherwise land
    -- entirely on whichever point is measured first.
    PERFORM set_config('brindle.ef_search', '64', false);
    PERFORM set_config('hnsw.ef_search', '64', false);
    FOR q IN SELECT id, embedding FROM bench_queries ORDER BY id LIMIT 3 LOOP
        SELECT count(*) INTO warm FROM (
            SELECT id FROM bench_vectors WHERE lbl_spread <= 5
            ORDER BY embedding <-> q.embedding LIMIT k) s;
        SELECT count(*) INTO warm FROM (
            SELECT id FROM pgv_vectors WHERE lbl_spread <= 5
            ORDER BY embedding <-> q.embedding::text::vector LIMIT k) s;
    END LOOP;

    -- Query outermost so every engine and every point shares cache state; a
    -- block sweep would hand the first block a colder cache than the last.
    FOR q IN SELECT id, embedding FROM bench_queries ORDER BY id LOOP
        FOR p IN SELECT shape, sel FROM sel_points LOOP
            col := 'lbl_' || p.shape;
            FOREACH ef IN ARRAY efs LOOP

                -- brindle: predicate pushed into the traversal
                PERFORM set_config('brindle.ef_search', ef::text, false);
                started := clock_timestamp();
                EXECUTE format(
                    'SELECT array_agg(id) FROM (SELECT id FROM bench_vectors '
                    'WHERE %I <= %s ORDER BY embedding <-> $1 LIMIT %s) s',
                    col, p.sel, k) INTO found USING q.embedding;
                INSERT INTO sel_timings VALUES ('brindle', p.shape, p.sel, ef,
                    extract(epoch FROM clock_timestamp() - started) * 1000, q.id);
                INSERT INTO sel_recall
                SELECT 'brindle', p.shape, p.sel, ef, q.id,
                       (SELECT count(*) FROM unnest(coalesce(found, '{}')) f WHERE f = ANY (t.ids))
                FROM sel_truth t
                WHERE t.shape = p.shape AND t.sel = p.sel AND t.query_id = q.id;

                -- pgvector post-filter: ef_search candidates, then the executor
                PERFORM set_config('hnsw.ef_search', ef::text, false);
                PERFORM set_config('hnsw.iterative_scan', 'off', false);
                started := clock_timestamp();
                EXECUTE format(
                    'SELECT array_agg(id) FROM (SELECT id FROM pgv_vectors '
                    'WHERE %I <= %s ORDER BY embedding <-> $1 LIMIT %s) s',
                    col, p.sel, k) INTO found USING q.embedding::text::vector;
                INSERT INTO sel_timings VALUES ('pgv_post', p.shape, p.sel, ef,
                    extract(epoch FROM clock_timestamp() - started) * 1000, q.id);
                INSERT INTO sel_recall
                SELECT 'pgv_post', p.shape, p.sel, ef, q.id,
                       (SELECT count(*) FROM unnest(coalesce(found, '{}')) f WHERE f = ANY (t.ids))
                FROM sel_truth t
                WHERE t.shape = p.shape AND t.sel = p.sel AND t.query_id = q.id;

                -- pgvector iterative scan: keep pulling until k pass the filter
                PERFORM set_config('hnsw.iterative_scan', 'relaxed_order', false);
                started := clock_timestamp();
                EXECUTE format(
                    'SELECT array_agg(id) FROM (SELECT id FROM pgv_vectors '
                    'WHERE %I <= %s ORDER BY embedding <-> $1 LIMIT %s) s',
                    col, p.sel, k) INTO found USING q.embedding::text::vector;
                INSERT INTO sel_timings VALUES ('pgv_iter', p.shape, p.sel, ef,
                    extract(epoch FROM clock_timestamp() - started) * 1000, q.id);
                INSERT INTO sel_recall
                SELECT 'pgv_iter', p.shape, p.sel, ef, q.id,
                       (SELECT count(*) FROM unnest(coalesce(found, '{}')) f WHERE f = ANY (t.ids))
                FROM sel_truth t
                WHERE t.shape = p.shape AND t.sel = p.sel AND t.query_id = q.id;

                PERFORM set_config('hnsw.iterative_scan', 'off', false);
            END LOOP;
        END LOOP;
    END LOOP;
END $$;

-- ------------------------------------------------------------------ results

\echo
\echo '=== recall@k by selectivity (uncorrelated label: matches spread everywhere) ==='
SELECT r.sel AS "sel %", r.ef AS ef,
       round(avg(CASE WHEN r.engine = 'brindle'  THEN r.hits END)::numeric / :k, 3) AS brindle,
       round(avg(CASE WHEN r.engine = 'pgv_post' THEN r.hits END)::numeric / :k, 3) AS pgv_post,
       round(avg(CASE WHEN r.engine = 'pgv_iter' THEN r.hits END)::numeric / :k, 3) AS pgv_iter
FROM sel_recall r WHERE r.shape = 'spread'
GROUP BY r.sel, r.ef ORDER BY r.sel DESC, r.ef;

\echo
\echo '=== recall@k by selectivity (CORRELATED label: matches form a region) ==='
SELECT r.sel AS "sel %", r.ef AS ef,
       round(avg(CASE WHEN r.engine = 'brindle'  THEN r.hits END)::numeric / :k, 3) AS brindle,
       round(avg(CASE WHEN r.engine = 'pgv_post' THEN r.hits END)::numeric / :k, 3) AS pgv_post,
       round(avg(CASE WHEN r.engine = 'pgv_iter' THEN r.hits END)::numeric / :k, 3) AS pgv_iter
FROM sel_recall r WHERE r.shape = 'local'
GROUP BY r.sel, r.ef ORDER BY r.sel DESC, r.ef;

-- Machine-readable, for the chart generator. Written only when the driver asks
-- for it, so running this file by hand does not litter the working tree.
\if :{?chart_csv}
COPY (SELECT r.shape, r.sel, r.ef, r.engine, round(avg(r.hits)::numeric / (SELECT current_setting('bench.k')::int), 4) AS recall, round((SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY t.elapsed_ms) FROM sel_timings t WHERE t.shape = r.shape AND t.sel = r.sel AND t.ef = r.ef AND t.engine = r.engine)::numeric, 4) AS p50_ms FROM sel_recall r GROUP BY r.shape, r.sel, r.ef, r.engine ORDER BY r.shape, r.sel DESC, r.ef, r.engine) TO :'chart_csv' WITH (FORMAT csv, HEADER);
\endif

\echo
\echo '=== median latency, ms (and QPS at one connection) ==='
SELECT t.shape, t.sel AS "sel %", t.ef AS ef, t.engine,
       round(percentile_cont(0.5) WITHIN GROUP (ORDER BY t.elapsed_ms)::numeric, 3) AS p50_ms,
       round((1000.0 / nullif(percentile_cont(0.5) WITHIN GROUP (ORDER BY t.elapsed_ms), 0))::numeric, 0) AS qps
FROM sel_timings t
GROUP BY t.shape, t.sel, t.ef, t.engine
ORDER BY t.shape, t.sel DESC, t.ef, t.engine;
