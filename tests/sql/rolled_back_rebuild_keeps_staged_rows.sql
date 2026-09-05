-- A rebuild that is itself rolled back must not take the transaction's staged
-- rows with it.
--
-- `ambuild` discards whatever the transaction had staged for the index, on the
-- grounds that a TRUNCATE emptied the table and a REINDEX indexed those rows
-- itself. That holds only while the rebuild stands. TRUNCATE and REINDEX inside
-- a subtransaction that aborts are undone -- Postgres puts the old relfilenode
-- back, for the heap and the index both -- and the staged rows belong to the
-- state that comes back. Dropped at `ambuild`, they are gone for good: the rows
-- stay live in the heap with nothing pointing at them, and no vacuum repairs it,
-- because ambulkdelete only removes entries for tuples that are *dead*.
--
-- Which is also why this cannot be tested the usual way. The rows are live, so a
-- sequential scan finds them whatever the index holds, and the heap-visibility
-- recheck that hides rolled-back rows elsewhere does not apply. The signal that
-- discriminates is how many rows an index scan can reach against how many the
-- heap holds, with seqscan off.

CREATE TABLE rbk (id int, embedding real[]);
ALTER TABLE rbk SET (autovacuum_enabled = off);
INSERT INTO rbk SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 200) i;
CREATE INDEX rbk_idx ON rbk USING brindle (embedding);

-- 1. TRUNCATE inside an explicit savepoint that is rolled back.
BEGIN;
INSERT INTO rbk VALUES (9999, ARRAY[9999.0::real, 10000.0::real]);
SAVEPOINT s;
TRUNCATE rbk;
ROLLBACK TO s;
COMMIT;

-- 2. REINDEX, same shape.
BEGIN;
INSERT INTO rbk VALUES (8888, ARRAY[8888.0::real, 8889.0::real]);
SAVEPOINT r;
REINDEX INDEX rbk_idx;
ROLLBACK TO r;
COMMIT;

-- 3. No explicit savepoint at all: a plpgsql block whose EXCEPTION handler
--    swallows an error raised after the truncate. This is the shape that turns
--    up without anyone reaching for a savepoint.
BEGIN;
INSERT INTO rbk VALUES (7777, ARRAY[7777.0::real, 7778.0::real]);
DO $$ BEGIN TRUNCATE rbk; RAISE EXCEPTION 'undo'; EXCEPTION WHEN OTHERS THEN NULL; END $$;
COMMIT;

-- 4. A rebuild in an inner savepoint that is RELEASEd, inside an outer one that
--    is then rolled back. The rebuild does not survive, so neither may the
--    discard: setting the staging aside has to follow the rebuild up the
--    subtransaction stack as it is released, or it is orphaned at a depth
--    nothing will ever abort and the rows are lost exactly as in (1).
BEGIN;
INSERT INTO rbk VALUES (6060, ARRAY[6060.0::real, 6061.0::real]);
SAVEPOINT k1;
SAVEPOINT k2;
TRUNCATE rbk;
RELEASE SAVEPOINT k2;
ROLLBACK TO k1;
COMMIT;

DO $$
DECLARE missing int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 1000;
    SELECT array_agg(want) INTO missing FROM unnest(ARRAY[9999, 8888, 7777, 6060]) want
    WHERE want IS DISTINCT FROM (
        SELECT id FROM rbk
        ORDER BY embedding <-> ARRAY[want::real, (want + 1)::real] LIMIT 1);
    IF missing IS NOT NULL THEN
        RAISE EXCEPTION
            'rows staged before a rolled-back rebuild are not in the index: % -- '
            'they are live in the heap with nothing pointing at them, and no '
            'vacuum will repair that', missing;
    END IF;
END $$;

-- Every live row must be reachable through the index, not just the three probes:
-- the loss is the whole staged batch, not one row.
DO $$
DECLARE heap_rows bigint; reachable bigint;
BEGIN
    -- Counted before seqscan is disabled: a keyless count would otherwise be
    -- planned as a brindle index scan, which has no ORDER BY to work from.
    SELECT count(*) INTO heap_rows FROM rbk;
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 5000;
    SELECT count(*) INTO reachable FROM (
        SELECT id FROM rbk ORDER BY embedding <-> ARRAY[1.0, 2.0]::real[] LIMIT 5000) s;
    IF reachable <> heap_rows THEN
        RAISE EXCEPTION
            'index reaches % of % live rows after rolled-back rebuilds',
            reachable, heap_rows;
    END IF;
END $$;

-- And a rebuild that *stands* must still discard, which is the behaviour the
-- stash is not allowed to break. Two shapes, because they reach different arms:
-- a truncate at top level never involves a subtransaction at all, while one
-- inside a RELEASEd savepoint goes through the subtransaction-commit arm, where
-- setting the staging aside has to become the parent's business rather than
-- being handed back.
BEGIN;
INSERT INTO rbk VALUES (5555, ARRAY[5555.0::real, 5556.0::real]);
SAVEPOINT k;
TRUNCATE rbk;
RELEASE SAVEPOINT k;
COMMIT;

DO $$
DECLARE found bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT count(*) INTO found
    FROM (SELECT id FROM rbk
          ORDER BY embedding <-> ARRAY[5555.0, 5556.0]::real[] LIMIT 5) s;
    IF found <> 0 THEN
        RAISE EXCEPTION
            'index returned % rows from a table truncated inside a released '
            'savepoint: staging set aside by the rebuild was handed back even '
            'though the rebuild stood', found;
    END IF;
END $$;

INSERT INTO rbk SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 50) i;

-- Note on what is *not* tested here: staging set aside by a rebuild that
-- commits is cleared at transaction end, and that clearing is about memory
-- rather than correctness, so no case asserts on it. At COMMIT Postgres commits
-- every open subtransaction in turn, which re-parents the entry up the chain to
-- the top-level subtransaction id -- and no SUBXACT_EVENT_ABORT_SUB ever fires
-- with that id, because a top-level abort is XACT_EVENT_ABORT instead. So an
-- entry left behind could never be handed to a later transaction; it would only
-- hold a decoded graph for the life of the backend. An earlier version of this
-- file asserted otherwise and passed with the clearing removed, which is the
-- same defect this suite keeps producing: a block that cannot fail.

BEGIN;
INSERT INTO rbk VALUES (6666, ARRAY[6666.0::real, 6667.0::real]);
TRUNCATE rbk;
COMMIT;

DO $$
DECLARE found bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT count(*) INTO found
    FROM (SELECT id FROM rbk
          ORDER BY embedding <-> ARRAY[6666.0, 6667.0]::real[] LIMIT 5) s;
    IF found <> 0 THEN
        RAISE EXCEPTION
            'index returned % rows from a truncated table: staging set aside by '
            'the rebuild was restored even though the rebuild committed', found;
    END IF;
END $$;
