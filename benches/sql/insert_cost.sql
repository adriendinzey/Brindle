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

DROP TABLE IF EXISTS ins_results;
CREATE TABLE ins_results (
    rows_in_index int,
    shape         text,
    ms_per_row    double precision
);

-- A procedure rather than a DO block, because a procedure can COMMIT. Rows are
-- written back when a transaction ends, so timing them inside one transaction
-- measures everything except the write-back — which is the only cost this task
-- changed. An earlier version of this file did exactly that, and published
-- single-row numbers about a third of the truth.
--
-- Small fixtures on purpose: the pre-change behaviour is quadratic in the
-- fixture, so a realistic size takes longer to measure than anyone will wait.
CREATE OR REPLACE PROCEDURE measure_inserts()
LANGUAGE plpgsql AS $proc$
DECLARE
    sizes   int[] := ARRAY[2000, 8000, 20000];
    n       int;
    dims    int := current_setting('bench.dims')::int;
    started timestamptz;
    j       int;
    single  int := 3;
    batch   int := 100;
BEGIN
    FOREACH n IN ARRAY sizes LOOP
        TRUNCATE ins_bench;
        INSERT INTO ins_bench
        SELECT g, (SELECT array_agg((random() - 0.5)::real) FROM generate_series(1, dims))
        FROM generate_series(1, n) AS g;
        COMMIT;

        DROP INDEX IF EXISTS ins_bench_idx;
        CREATE INDEX ins_bench_idx ON ins_bench USING brindle (embedding);
        COMMIT;

        -- One row per transaction, which is what an autocommitted INSERT is.
        -- The COMMIT inside the loop is the whole point: it forces the
        -- write-back that makes this the real cost.
        started := clock_timestamp();
        FOR j IN 1..single LOOP
            INSERT INTO ins_bench
            SELECT n + j, (SELECT array_agg((random() - 0.5)::real)
                           FROM generate_series(1, dims));
            COMMIT;
        END LOOP;
        INSERT INTO ins_results
        VALUES (n, 'single',
                extract(epoch FROM clock_timestamp() - started) * 1000 / single);
        COMMIT;

        -- The same rows again in one statement, committed once, so the
        -- write-back is inside the timed region here too and amortized.
        started := clock_timestamp();
        INSERT INTO ins_bench
        SELECT n + single + g, (SELECT array_agg((random() - 0.5)::real)
                                FROM generate_series(1, dims))
        FROM generate_series(1, batch) AS g;
        COMMIT;
        INSERT INTO ins_results
        VALUES (n, 'bulk',
                extract(epoch FROM clock_timestamp() - started) * 1000 / batch);
        COMMIT;
    END LOOP;
END $proc$;

CALL measure_inserts();

\echo ''
\echo '=== insert cost per row, by index size ==='
SELECT rows_in_index,
       round(max(ms_per_row) FILTER (WHERE shape = 'single')::numeric, 2) AS single_ms,
       round(max(ms_per_row) FILTER (WHERE shape = 'bulk')::numeric, 2)   AS bulk_ms
FROM ins_results
GROUP BY rows_in_index
ORDER BY rows_in_index;
