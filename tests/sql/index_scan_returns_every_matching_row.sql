-- An index scan must return the same rows a sequential scan does.
--
-- A row whose indexed vector is NULL cannot go in the graph -- there is nothing
-- to place or to rank. That part is right. The bug is claiming completeness
-- anyway: the row is skipped at build and insert time, no index predicate is
-- registered, so PostgreSQL believes the index covers every row in the table and
-- a plan that uses it silently returns fewer.
--
-- That is a wrong answer rather than a recall trade, and it outranks every
-- recall number in this project.
--
-- It became reachable when the attribute columns became search keys: before
-- that, `amrescan` errored on any scan without an ORDER BY, so no plan could ask
-- this index for "every row matching a qual". Now one can.
--
-- The ordered path is deliberately not asserted here. A NULL vector has no
-- distance, and an ordered scan is already documented as returning at most
-- `brindle.ef_search` rows -- omitting unrankable rows from a ranking is
-- consistent with that. The keyless path is different in kind: its whole purpose
-- is to return every matching row.

CREATE TABLE nv (id int, bucket int, embedding real[]);
ALTER TABLE nv SET (autovacuum_enabled = off);
INSERT INTO nv SELECT i, i % 10, ARRAY[i::real, (i + 1)::real]
FROM generate_series(1, 400) i;

-- Rows that satisfy the predicate and have no vector at all.
INSERT INTO nv VALUES (901, 7, NULL), (902, 7, NULL), (903, 7, NULL);

CREATE INDEX nv_idx ON nv USING brindle (embedding, bucket);

-- The premise: the index is not registered as partial, so PostgreSQL believes it
-- covers the whole table. If that ever changes, this case should say so rather
-- than quietly testing nothing.
DO $$
BEGIN
    IF (SELECT indpred IS NOT NULL FROM pg_index
         WHERE indexrelid = 'nv_idx'::regclass) THEN
        RAISE EXCEPTION
            'nv_idx is a partial index; this case assumes it claims to cover '
            'every row, which is what makes an omission a wrong answer';
    END IF;
END $$;

DO $$
DECLARE seq_rows bigint; idx_rows bigint; line text; plan text := '';
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_bitmapscan = off;
    SET LOCAL enable_seqscan = on;
    SELECT count(*) INTO seq_rows FROM nv WHERE bucket = 7;

    SET LOCAL enable_seqscan = off;
    SET LOCAL enable_indexscan = on;
    SET LOCAL enable_bitmapscan = on;
    FOR line IN EXECUTE 'EXPLAIN SELECT id FROM nv WHERE bucket = 7' LOOP
        plan := plan || line || E'\n';
    END LOOP;
    IF plan NOT LIKE '%nv_idx%' THEN
        RAISE EXCEPTION
            'the qual did not reach the index, so this proves nothing:%',
            E'\n' || plan;
    END IF;
    SELECT count(*) INTO idx_rows FROM nv WHERE bucket = 7;

    IF idx_rows <> seq_rows THEN
        RAISE EXCEPTION
            'index scan returned % rows where a sequential scan returns % -- '
            'the index omits rows it claims to cover', idx_rows, seq_rows;
    END IF;
END $$;

-- The omitted rows must be the NULL-vector ones specifically, and they must come
-- back with their attributes intact rather than as bare tids.
DO $$
DECLARE missing bigint; wrong bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    CREATE TEMP TABLE got AS SELECT id FROM nv WHERE bucket = 7;

    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;
    SELECT count(*) INTO missing
    FROM nv WHERE embedding IS NULL AND bucket = 7
      AND id NOT IN (SELECT id FROM got);
    IF missing <> 0 THEN
        RAISE EXCEPTION '% NULL-vector rows are missing from the index scan', missing;
    END IF;

    SELECT count(*) INTO wrong
    FROM got JOIN nv USING (id) WHERE nv.bucket IS DISTINCT FROM 7;
    IF wrong <> 0 THEN
        RAISE EXCEPTION
            'the index returned % rows that fail the predicate -- a NULL-vector '
            'row must still be filtered on its attributes', wrong;
    END IF;
END $$;

-- A NULL-vector row that no longer satisfies the predicate must not come back.
DO $$
DECLARE leaked bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT count(*) INTO leaked FROM nv WHERE bucket = 3 AND embedding IS NULL;
    IF leaked <> 0 THEN
        RAISE EXCEPTION
            '% NULL-vector rows came back for a bucket none of them are in', leaked;
    END IF;
END $$;

-- They must survive a round trip through the stored index, not just live in the
-- in-memory copy the building session happens to hold.
DO $$
DECLARE idx_rows bigint; seq_rows bigint;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;
    SELECT count(*) INTO seq_rows FROM nv WHERE bucket = 7;
    SET LOCAL enable_seqscan = off;
    SET LOCAL enable_indexscan = on;
    SELECT count(*) INTO idx_rows FROM nv WHERE bucket = 7;
    IF idx_rows <> seq_rows THEN
        RAISE EXCEPTION
            'after reload the index returns % rows against %', idx_rows, seq_rows;
    END IF;
END $$;

-- And a NULL-vector row inserted *after* the build must be found too: the
-- insert path skips these rows for the same reason the build does.
INSERT INTO nv VALUES (904, 7, NULL);

DO $$
DECLARE idx_rows bigint; seq_rows bigint;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;
    SELECT count(*) INTO seq_rows FROM nv WHERE bucket = 7;
    SET LOCAL enable_seqscan = off;
    SET LOCAL enable_indexscan = on;
    SELECT count(*) INTO idx_rows FROM nv WHERE bucket = 7;
    IF idx_rows <> seq_rows THEN
        RAISE EXCEPTION
            'a NULL-vector row inserted after the build is missing: index % '
            'against sequential %', idx_rows, seq_rows;
    END IF;
END $$;

-- ...and must disappear when the row does.
DELETE FROM nv WHERE id = 904;
VACUUM nv;

DO $$
DECLARE idx_rows bigint; seq_rows bigint;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;
    SELECT count(*) INTO seq_rows FROM nv WHERE bucket = 7;
    SET LOCAL enable_seqscan = off;
    SET LOCAL enable_indexscan = on;
    SELECT count(*) INTO idx_rows FROM nv WHERE bucket = 7;
    IF idx_rows <> seq_rows THEN
        RAISE EXCEPTION
            'after deleting a NULL-vector row and vacuuming, the index returns % '
            'against sequential %', idx_rows, seq_rows;
    END IF;
END $$;
