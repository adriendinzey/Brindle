-- A transaction must see the rows it inserted, including when the scan that
-- looks for them runs inside a parallel worker.
--
-- Staged rows live in backend-local memory. A parallel worker is a separate
-- process, so it cannot see them however carefully they are handed around
-- inside this one: it opens the index, reads the image on disk, and finds a
-- graph that predates the insert. The answer comes back short with no error.
--
-- `amcanparallel = false` does not prevent this. It stops one index scan being
-- split across workers; it does not stop the whole scan running inside a single
-- worker under a Gather, which is what a parallel plan over this shape does.
--
-- The fix is not to make the staged graph visible across processes but to stop
-- lending it out at all: a scan writes back first, so what every reader sees --
-- this backend, a worker, or another session after commit -- is the image.

CREATE TABLE pw (id int, embedding real[]);
ALTER TABLE pw SET (autovacuum_enabled = off);
INSERT INTO pw SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 300) i;
CREATE INDEX pw_idx ON pw USING brindle (embedding);

-- Renamed in Postgres 16; brindle supports 14 and up, so ask the server which
-- one it has rather than hardcoding either name.
SELECT name AS force_parallel_guc
FROM pg_settings
WHERE name IN ('debug_parallel_query', 'force_parallel_mode') \gset

-- One transaction throughout: the point is what an uncommitted insert is
-- visible to, so a commit anywhere before the assertion would make this pass
-- whether or not the mechanism works. The queries are run at psql level rather
-- than inside a DO block because `:variables` are not substituted into a
-- dollar-quoted body.
BEGIN;
SET LOCAL enable_seqscan = off;
SELECT set_config(:'force_parallel_guc', 'on', true) \gset force_

-- The plan guard runs BEFORE the insert, so that nothing at all sits between
-- staging the row and the query that asserts on it.
--
-- Two separate traps meet here. The query has to be shaped so the scan really
-- reaches a worker -- wrapping it in a scalar subquery makes it an InitPlan,
-- which the *leader* evaluates before dispatching, so no worker ever opens the
-- index. And the guard itself runs EXPLAIN, which goes through the same
-- ExecutorStart hook as the real query; while EXPLAIN without ANALYZE is
-- excluded from the flush, running it after the insert would still leave the
-- assertion resting on that exclusion holding rather than on the mechanism
-- under test. Ordering it first makes both moot.
DO $$
DECLARE line text; plan text := '';
BEGIN
    FOR line IN EXECUTE
        'EXPLAIN SELECT id FROM pw '
        'ORDER BY embedding <-> ARRAY[1201.0, 1202.0]::real[] LIMIT 1'
    LOOP
        plan := plan || line || E'\n';
    END LOOP;
    IF plan NOT LIKE '%Gather%' THEN
        RAISE EXCEPTION 'plan is not parallel, so this proves nothing:%', E'\n' || plan;
    END IF;
    IF plan NOT LIKE '%Index Scan%' THEN
        RAISE EXCEPTION 'plan does not use the index, so this proves nothing:%', E'\n' || plan;
    END IF;
    IF plan LIKE '%InitPlan%' THEN
        RAISE EXCEPTION
            'the index scan is an InitPlan, which the leader runs before '
            'dispatching -- no worker opens the index:%', E'\n' || plan;
    END IF;
END $$;

INSERT INTO pw VALUES (1201, ARRAY[1201.0::real, 1202.0::real]);

-- Nothing between the insert above and this query. A serial scan, or any other
-- statement that triggers the write-back, would satisfy this as a side effect
-- and the case could never fail on its own claim.
SELECT id AS parallel_nearest FROM pw
ORDER BY embedding <-> ARRAY[1201.0, 1202.0]::real[] LIMIT 1 \gset
SELECT set_config(:'force_parallel_guc', 'off', true) \gset force_

-- Only now the serial one, which must agree.
SELECT id AS serial_nearest FROM pw
ORDER BY embedding <-> ARRAY[1201.0, 1202.0]::real[] LIMIT 1 \gset

SELECT set_config('brindle_test.serial_nearest', :'serial_nearest', true) \gset carry_
SELECT set_config('brindle_test.parallel_nearest', :'parallel_nearest', true) \gset carry_

DO $$
DECLARE
    serial_nearest   int := current_setting('brindle_test.serial_nearest')::int;
    parallel_nearest int := current_setting('brindle_test.parallel_nearest')::int;
BEGIN
    IF parallel_nearest IS DISTINCT FROM 1201 THEN
        RAISE EXCEPTION
            'a parallel worker did not see a row this transaction staged: '
            'nearest to [1201,1202] is % in a worker', parallel_nearest;
    END IF;
    IF serial_nearest IS DISTINCT FROM 1201 THEN
        RAISE EXCEPTION
            'a row inserted in this transaction is not findable in it at all: '
            'nearest to [1201,1202] is %', serial_nearest;
    END IF;
END $$;

COMMIT;

-- And it is really in the index afterwards, rather than having been visible
-- only because something read the heap.
DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM pw
    ORDER BY embedding <-> ARRAY[1201.0, 1202.0]::real[] LIMIT 1;
    IF nearest IS DISTINCT FROM 1201 THEN
        RAISE EXCEPTION 'row 1201 is not in the committed index: nearest is %', nearest;
    END IF;
END $$;
