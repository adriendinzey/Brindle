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

-- 5. More than one rebuild mapping to the same savepoint. Only the *first* set
--    aside can hold rows staged before that savepoint began: staging is a single
--    slot, so a second rebuild at the same depth is necessarily setting aside
--    rows staged after the first, inside the subtransaction, which roll back
--    with it. Restoring the later one hands back rows that must go and strands
--    the ones that must stay -- the same silent loss as (1), reached three ways.
--
--    5a needs nothing unusual at all: two ordinary tables, truncated once each,
--    inside one savepoint that rolls back.
CREATE TABLE rbk2 (id int, embedding real[]);
ALTER TABLE rbk2 SET (autovacuum_enabled = off);
INSERT INTO rbk2 SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 150) i;
CREATE INDEX rbk2_idx ON rbk2 USING brindle (embedding);

BEGIN;
INSERT INTO rbk VALUES (5010, ARRAY[5010.0::real, 5011.0::real]);
SAVEPOINT m1;
TRUNCATE rbk;
INSERT INTO rbk2 VALUES (5011, ARRAY[5011.0::real, 5012.0::real]);
TRUNCATE rbk2;
ROLLBACK TO m1;
COMMIT;

-- 5b: two rebuilds of the same index under one savepoint.
BEGIN;
INSERT INTO rbk VALUES (5020, ARRAY[5020.0::real, 5021.0::real]);
SAVEPOINT m2;
TRUNCATE rbk;
INSERT INTO rbk VALUES (5021, ARRAY[5021.0::real, 5022.0::real]);
TRUNCATE rbk;
ROLLBACK TO m2;
COMMIT;

-- 5c: a rebuild in a RELEASEd inner savepoint, re-parented onto an outer one
--     that already holds an entry of its own.
BEGIN;
INSERT INTO rbk VALUES (5030, ARRAY[5030.0::real, 5031.0::real]);
SAVEPOINT m3;
SAVEPOINT m4;
TRUNCATE rbk;
RELEASE SAVEPOINT m4;
INSERT INTO rbk VALUES (5031, ARRAY[5031.0::real, 5032.0::real]);
TRUNCATE rbk;
ROLLBACK TO m3;
COMMIT;

DO $$
DECLARE missing int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 2000;
    SELECT array_agg(want) INTO missing FROM unnest(ARRAY[5010, 5020, 5030]) want
    WHERE want IS DISTINCT FROM (
        SELECT id FROM rbk
        ORDER BY embedding <-> ARRAY[want::real, (want + 1)::real] LIMIT 1);
    IF missing IS NOT NULL THEN
        RAISE EXCEPTION
            'rows staged before a savepoint holding more than one rebuild are '
            'not in the index: % -- a later rebuild''s staging was handed back '
            'in place of the one that predates the savepoint', missing;
    END IF;
END $$;

-- The second table's own row was staged *after* the savepoint, so the rollback
-- takes it out of the heap too and its absence proves nothing. What must hold is
-- that dropping its set-aside staging left the rest of that index intact.
DO $$
DECLARE heap_rows bigint; reachable bigint;
BEGIN
    SELECT count(*) INTO heap_rows FROM rbk2;   -- before seqscan is disabled
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 2000;
    SELECT count(*) INTO reachable FROM (
        SELECT id FROM rbk2 ORDER BY embedding <-> ARRAY[1.0, 2.0]::real[] LIMIT 2000) s;
    IF reachable <> heap_rows THEN
        RAISE EXCEPTION
            'index on the second table reaches % of % live rows', reachable, heap_rows;
    END IF;
END $$;

-- 6. A rebuild in a savepoint that is RELEASEd onto a subtransaction which
--    already holds set-aside staging. The release has to drop the inner entry
--    rather than move it, or the two collide and the stash holds two writes for
--    one subtransaction -- the state that makes the restore ambiguous and that
--    (5) shows losing rows. Nothing here rolls back, so the assertions below are
--    about the end state; the invariant itself is what this shape pins.
CREATE TABLE rbk3 (id int, embedding real[]);
ALTER TABLE rbk3 SET (autovacuum_enabled = off);
INSERT INTO rbk3 SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 120) i;
CREATE INDEX rbk3_idx ON rbk3 USING brindle (embedding);

BEGIN;
INSERT INTO rbk3 VALUES (6100, ARRAY[6100.0::real, 6101.0::real]);
TRUNCATE rbk3;                      -- sets staging aside at the top level
INSERT INTO rbk3 VALUES (6101, ARRAY[6101.0::real, 6102.0::real]);
SAVEPOINT p;
TRUNCATE rbk3;                      -- sets aside again, one level down
RELEASE SAVEPOINT p;                -- re-parents onto an entry that already exists
INSERT INTO rbk3 VALUES (6102, ARRAY[6102.0::real, 6103.0::real]);
COMMIT;

DO $$
DECLARE heap_rows bigint; reachable bigint; nearest int;
BEGIN
    SELECT count(*) INTO heap_rows FROM rbk3;      -- before seqscan is disabled
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 2000;
    SELECT count(*) INTO reachable FROM (
        SELECT id FROM rbk3 ORDER BY embedding <-> ARRAY[6102.0, 6103.0]::real[] LIMIT 2000) s;
    IF reachable <> heap_rows THEN
        RAISE EXCEPTION
            'index reaches % of % live rows after a rebuild released onto one '
            'that had already set staging aside', reachable, heap_rows;
    END IF;
    SELECT id INTO nearest FROM rbk3
    ORDER BY embedding <-> ARRAY[6102.0, 6103.0]::real[] LIMIT 1;
    IF nearest IS DISTINCT FROM 6102 THEN
        RAISE EXCEPTION 'the row inserted after the released rebuild is missing: %', nearest;
    END IF;
END $$;

-- Note on what is *not* tested here: staging set aside by a rebuild that stands
-- is cleared at transaction end, and no case asserts on that clearing. It does
-- matter -- subtransaction ids restart with every transaction, so an entry left
-- behind at a live subid could be handed to a later transaction, which would
-- both clobber that transaction's own staging and rewrite a different index from
-- a graph staged against a relation that no longer exists. What makes it
-- unreachable is the rule above: an abort drains *every* entry for its subid and
-- a release re-parents or drops them, so nothing is ever left at a subid below
-- the top level, and no SUBXACT_EVENT_ABORT_SUB fires with the top-level id --
-- a top-level abort is XACT_EVENT_ABORT instead.
--
-- An earlier version of this file asserted the cross-transaction hazard directly
-- and passed with the clearing removed, which was taken as evidence the hazard
-- did not exist. It was not: the shape that strands an entry needs two rebuilds
-- at one savepoint, which (5) above did not then cover.

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
