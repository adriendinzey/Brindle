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

-- The index reports how many entries it holds, and CREATE INDEX writes that to
-- pg_class.reltuples. A row with no vector is still an entry, so a count of
-- graph nodes alone tells the planner the index is smaller than it is -- and
-- disagrees with what amvacuumcleanup reports for this same non-partial index on
-- the first VACUUM, making the number flip depending on which ran last.
DO $$
DECLARE stated real; live bigint;
BEGIN
    SELECT reltuples INTO stated FROM pg_class WHERE relname = 'nv_idx';
    SELECT count(*) INTO live FROM nv;
    IF stated <> live THEN
        RAISE EXCEPTION
            'the index reported % entries for a table of % rows -- the rows with '
            'no vector are in the index but not in its count', stated, live;
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

-- The ordered path deliberately does the opposite, and that decision is pinned
-- here rather than left to a comment. A NULL vector has no distance, so a row
-- carrying one has no position in an `ORDER BY embedding <-> $1` result; letting
-- one into a ranked stream would be an ordering violation, and `amgettuple` sets
-- `xs_recheckorderby = false`, so nothing above the AM would repair it.
--
-- The LIMIT must exceed the number of rows that *can* be ranked (40 in bucket
-- 7), or the assertion is vacuous: unrankable rows would be appended after the
-- ranked ones, and a LIMIT that fills from the ranked ones alone never reaches
-- them. Confirmed by mutation -- at LIMIT 20 this passes with the ordered path
-- deliberately broken.
DO $$
DECLARE ranked int[]; unrankable int[]; line text; plan text := '';
BEGIN
    SET LOCAL enable_seqscan = off;
    FOR line IN EXECUTE
        'EXPLAIN SELECT id FROM nv WHERE bucket = 7
          ORDER BY embedding <-> ARRAY[350.0, 351.0]::real[] LIMIT 100' LOOP
        plan := plan || line || E'\n';
    END LOOP;
    IF plan NOT LIKE '%nv_idx%' OR plan NOT LIKE '%Order By%' THEN
        RAISE EXCEPTION
            'this is not an ordered index scan, so it proves nothing:%',
            E'\n' || plan;
    END IF;

    SELECT array_agg(id) INTO ranked FROM (
        SELECT id FROM nv WHERE bucket = 7
         ORDER BY embedding <-> ARRAY[350.0, 351.0]::real[] LIMIT 100
    ) q;
    IF ranked IS NULL THEN
        RAISE EXCEPTION
            'the ordered scan returned nothing, so the assertion below would '
            'pass whatever the index did with the NULL-vector rows';
    END IF;

    SELECT array_agg(id) INTO unrankable
    FROM unnest(ranked) AS id WHERE id >= 900;
    IF unrankable IS NOT NULL THEN
        RAISE EXCEPTION
            'a ranked scan returned %, which have no vector and therefore no '
            'distance to be ordered by', unrankable;
    END IF;
END $$;

-- A second backend must see them too.
--
-- Re-running the count in *this* session would prove nothing: the first block
-- above already decoded the stored image, nothing has written since, so the
-- cached copy answers and no page is touched. An earlier version of this case
-- did exactly that and asserted nothing.
--
-- This reads through a connection with no cached copy and no staging buffer of
-- its own, which is what an ordinary reader is. It shares the decode path with
-- the block above rather than reaching past it, so it is breadth, not a second
-- independent signal; the staged-and-replayed paths are covered by
-- `staged_null_vector_rows_follow_the_transaction`.
CREATE EXTENSION IF NOT EXISTS dblink;

DO $$
DECLARE conn text; idx_rows bigint; seq_rows bigint;
BEGIN
    conn := 'dbname=' || current_database() ||
            ' port=' || current_setting('port') ||
            ' host=' || (string_to_array(current_setting('unix_socket_directories'), ','))[1];

    SELECT count(*) INTO seq_rows FROM nv WHERE bucket = 7;

    SELECT n INTO idx_rows FROM dblink(conn,
        $inner$SET enable_seqscan = off;
               SELECT count(*) FROM nv WHERE bucket = 7$inner$) AS t(n bigint);

    IF idx_rows <> seq_rows THEN
        RAISE EXCEPTION
            'a second backend reading the stored image sees % rows against %',
            idx_rows, seq_rows;
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
--
-- Counting after the DELETE would pass whether or not VACUUM dropped the entry:
-- Postgres rechecks heap visibility for every TID an index scan returns, so a
-- stale one is silently discarded and the count is identical either way. The
-- observable failure is the one `vacuum_frees_line_pointers_safely` is built
-- around: VACUUM recycles the line pointer, a new row lands in the freed slot,
-- and a stale entry then resolves to that live row -- a wrong answer the
-- visibility recheck cannot catch. So: delete, vacuum, reclaim the slot with a
-- row in a *different* bucket, and look for it coming back for bucket 7.
DELETE FROM nv WHERE id = 904;
VACUUM nv;

-- Enough rows to claim the freed slot, none of them in bucket 7.
INSERT INTO nv SELECT i, 3, ARRAY[i::real, (i + 1)::real]
FROM generate_series(9000, 9019) i;

DO $$
DECLARE bad int[]; idx_rows bigint; seq_rows bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(id) INTO bad FROM (
        SELECT id FROM nv WHERE bucket = 7
    ) q WHERE id >= 9000;
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION
            'the index returned % for bucket 7 -- a stale entry for the deleted '
            'NULL-vector row resolved to whatever reclaimed its line pointer',
            bad;
    END IF;

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
