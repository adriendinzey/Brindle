-- A real VACUUM against a brindle-indexed table.
--
-- This case cannot be written as a `#[pg_test]`: that harness wraps each test in
-- a transaction, and Postgres refuses to run VACUUM inside a transaction block.
-- The existing Rust test drives `index_bulk_delete` directly, which covers the
-- callback but never reaches amvacuumcleanup — VACUUM is what calls that.
--
-- What it asserts is observable rather than log-scraped: Postgres refreshes an
-- index's pg_class.relpages/reltuples from whatever amvacuumcleanup returns, and
-- skips the refresh entirely if it returns NULL. So correct, current stats after
-- VACUUM are proof the callback ran and reported. Make it return NULL and this
-- case fails.

CREATE TABLE vac_stats (id int, embedding real[]);
-- ANALYZE refreshes these same stats on its own, so an autoanalyze landing
-- mid-case would either trip the precondition below or satisfy the assertion
-- that follows it without VACUUM having done anything.
ALTER TABLE vac_stats SET (autovacuum_enabled = off);
INSERT INTO vac_stats
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 500) i;

CREATE INDEX vac_stats_idx ON vac_stats USING brindle (embedding);

-- Grow the index past its build size. Each insert rewrites the stored image, so
-- the relation gets larger while pg_class keeps saying what it was at build.
INSERT INTO vac_stats
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(501, 900) i;

DO $$
DECLARE stated bigint; actual bigint;
BEGIN
    SELECT relpages INTO stated FROM pg_class WHERE relname = 'vac_stats_idx';
    SELECT pg_relation_size('vac_stats_idx') / current_setting('block_size')::int
      INTO actual;
    IF stated = actual THEN
        RAISE EXCEPTION
            'precondition failed: stats are already current (% pages), so the '
            'assertion after VACUUM would pass whether or not it refreshed them',
            stated;
    END IF;
END $$;

VACUUM vac_stats;

DO $$
DECLARE stated bigint; actual bigint; tuples real; live bigint;
BEGIN
    SELECT relpages, reltuples INTO stated, tuples
      FROM pg_class WHERE relname = 'vac_stats_idx';
    SELECT pg_relation_size('vac_stats_idx') / current_setting('block_size')::int
      INTO actual;
    SELECT count(*) INTO live FROM vac_stats;

    IF stated <> actual THEN
        RAISE EXCEPTION
            'VACUUM did not refresh relpages: pg_class says %, relation is % pages. '
            'amvacuumcleanup either was not reached or returned NULL',
            stated, actual;
    END IF;
    IF tuples <> live THEN
        RAISE EXCEPTION
            'VACUUM did not refresh reltuples: pg_class says %, table holds % rows',
            tuples, live;
    END IF;
END $$;

-- The index must still answer correctly afterwards.
DO $$
DECLARE found int[]; nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(id ORDER BY id) INTO found
    FROM (SELECT id FROM vac_stats
          ORDER BY embedding <-> ARRAY[10.0, 11.0]::real[] LIMIT 5) s;
    -- Separately: an aggregate carrying its own ORDER BY does not preserve the
    -- subquery's, so the nearest row has to be asked for on its own.
    SELECT id INTO nearest FROM vac_stats
    ORDER BY embedding <-> ARRAY[10.0, 11.0]::real[] LIMIT 1;
    -- Sorted, because ids 9/11 and 8/12 sit at identical distances from the
    -- query and their relative order is not something the index promises.
    IF found IS DISTINCT FROM ARRAY[8, 9, 10, 11, 12] THEN
        RAISE EXCEPTION 'index answered % after VACUUM, expected the five nearest ids', found;
    END IF;
    IF nearest <> 10 THEN
        RAISE EXCEPTION 'nearest row after VACUUM was %, expected the exact match 10', nearest;
    END IF;
END $$;
