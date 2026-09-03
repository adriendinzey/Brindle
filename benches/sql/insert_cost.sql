-- What an insert into a brindle index costs, against the size of that index.
--
-- Two shapes, because they degrade differently and only one of them is fixable
-- inside the interim format:
--
--   single  — one row, its own transaction. Each pays to load, mutate and write
--             back the whole stored image, so cost grows with the index.
--   bulk    — many rows in one statement. Today that is the single-row cost N
--             times over, which makes a bulk load quadratic in the table.
--
-- Not a CI gate. Timing on a shared runner is noise.

\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS brindle;

-- psql does not substitute :vars inside a dollar-quoted body, so the dimension
-- comes through a setting the block can read instead.
SELECT set_config('bench.dims', :'dims', false);

DROP TABLE IF EXISTS ins_bench;
CREATE TABLE ins_bench (id int, embedding real[]);
-- Autovacuum must not rewrite the image underneath the measurement.
ALTER TABLE ins_bench SET (autovacuum_enabled = off);

CREATE TABLE ins_results (
    rows_in_index int,
    shape         text,
    ms_per_row    double precision
);

DO $$
DECLARE
    -- Small on purpose. Each row rewrites the whole stored image, so the
    -- measurement is quadratic in the fixture and a realistic size would take
    -- longer to measure than anyone will wait — which is the finding, not an
    -- inconvenience. The shape is visible well before it becomes unbearable.
    sizes   int[] := ARRAY[2000, 8000, 20000];
    n       int;
    dims    int := current_setting('bench.dims')::int;
    started timestamptz;
    j       int;
    single  int := 3;
    batch   int := 30;
BEGIN
    FOREACH n IN ARRAY sizes LOOP
        TRUNCATE ins_bench;
        INSERT INTO ins_bench
        SELECT g, (SELECT array_agg((random() - 0.5)::real) FROM generate_series(1, dims))
        FROM generate_series(1, n) AS g;

        DROP INDEX IF EXISTS ins_bench_idx;
        CREATE INDEX ins_bench_idx ON ins_bench USING brindle (embedding);

        -- Single rows, each its own statement.
        started := clock_timestamp();
        FOR j IN 1..single LOOP
            INSERT INTO ins_bench
            SELECT n + j, (SELECT array_agg((random() - 0.5)::real)
                           FROM generate_series(1, dims));
        END LOOP;
        INSERT INTO ins_results
        VALUES (n, 'single',
                extract(epoch FROM clock_timestamp() - started) * 1000 / single);

        -- The same number of rows again, in one statement.
        started := clock_timestamp();
        INSERT INTO ins_bench
        SELECT n + single + g, (SELECT array_agg((random() - 0.5)::real)
                                FROM generate_series(1, dims))
        FROM generate_series(1, batch) AS g;
        INSERT INTO ins_results
        VALUES (n, 'bulk',
                extract(epoch FROM clock_timestamp() - started) * 1000 / batch);
    END LOOP;
END $$;

\echo ''
\echo '=== insert cost per row, by index size ==='
SELECT rows_in_index,
       round(max(ms_per_row) FILTER (WHERE shape = 'single')::numeric, 2) AS single_ms,
       round(max(ms_per_row) FILTER (WHERE shape = 'bulk')::numeric, 2)   AS bulk_ms
FROM ins_results
GROUP BY rows_in_index
ORDER BY rows_in_index;
