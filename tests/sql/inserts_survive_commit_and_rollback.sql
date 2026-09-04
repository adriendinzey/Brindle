-- Deferred inserts must land on commit and vanish on rollback.
--
-- Rows are applied to a graph held for the transaction and written back when it
-- ends, so nothing here can be checked inside a `#[pg_test]`: that harness rolls
-- its transaction back, which is the case this file is largely about.

CREATE TABLE defer_t (id int, embedding real[]);
ALTER TABLE defer_t SET (autovacuum_enabled = off);
INSERT INTO defer_t
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 300) i;
CREATE INDEX defer_idx ON defer_t USING brindle (embedding);

-- Committed rows are findable, and are on disk rather than in a backend's head:
-- the check runs after the transaction that wrote them has ended.
BEGIN;
INSERT INTO defer_t
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(5000, 5099) i;

-- Still inside the transaction: it must see its own rows.
DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM defer_t
    ORDER BY embedding <-> ARRAY[5050.0, 5051.0]::real[] LIMIT 1;
    IF nearest <> 5050 THEN
        RAISE EXCEPTION
            'a transaction cannot see its own pending insert: nearest to '
            '[5050,5051] was %, expected 5050', nearest;
    END IF;
END $$;
COMMIT;

DO $$
DECLARE nearest int; total bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM defer_t
    ORDER BY embedding <-> ARRAY[5050.0, 5051.0]::real[] LIMIT 1;
    IF nearest <> 5050 THEN
        RAISE EXCEPTION 'committed rows are not in the index: got %', nearest;
    END IF;
    SELECT count(*) INTO total
    FROM (SELECT id FROM defer_t
          ORDER BY embedding <-> ARRAY[5050.0, 5051.0]::real[] LIMIT 40) s;
    IF total <> 40 THEN
        RAISE EXCEPTION 'only % of 40 committed rows came back', total;
    END IF;
END $$;

-- Rolled back rows must not be returned. This one does not discriminate on its
-- own: MVCC hides index entries for aborted heap tuples, so it passes whether or
-- not the staged rows were discarded. It is here because the behaviour is worth
-- stating, not because it would catch a regression — the in-transaction
-- visibility check above and the post-commit check are what do that.
BEGIN;
INSERT INTO defer_t
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(9000, 9099) i;
ROLLBACK;

DO $$
DECLARE leaked int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(id) INTO leaked
    FROM (SELECT id FROM defer_t
          ORDER BY embedding <-> ARRAY[9050.0, 9051.0]::real[] LIMIT 20) s
    WHERE id >= 9000;
    IF leaked IS NOT NULL THEN
        RAISE EXCEPTION 'rolled-back rows are in the index: %', leaked;
    END IF;
END $$;

-- Rolling back to a savepoint must undo only what came after it. The rows before
-- the savepoint are flushed when the subtransaction opens, which is what makes
-- discarding the rest a correct undo rather than a partial one.
BEGIN;
INSERT INTO defer_t SELECT 6001, ARRAY[6001.0::real, 6002.0::real];
SAVEPOINT sp;
INSERT INTO defer_t SELECT 6002, ARRAY[6003.0::real, 6004.0::real];
ROLLBACK TO SAVEPOINT sp;
COMMIT;

DO $$
DECLARE kept int; dropped int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO kept FROM defer_t
    ORDER BY embedding <-> ARRAY[6001.0, 6002.0]::real[] LIMIT 1;
    IF kept <> 6001 THEN
        RAISE EXCEPTION
            'the row inserted before the savepoint was lost: nearest to '
            '[6001,6002] was %', kept;
    END IF;
    SELECT id INTO dropped FROM defer_t
    ORDER BY embedding <-> ARRAY[6003.0, 6004.0]::real[] LIMIT 1;
    -- As with the rollback above, MVCC would hide this row anyway; the
    -- assertion that carries weight is the one before it, that the row from
    -- *before* the savepoint survived. A stale flush would lose that one.
    IF dropped = 6002 THEN
        RAISE EXCEPTION 'the row after the savepoint survived its rollback';
    END IF;
END $$;
