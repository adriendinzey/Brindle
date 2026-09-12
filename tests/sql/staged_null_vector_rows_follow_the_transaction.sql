-- A row with no vector is staged, rolled back and replayed like any other row.
--
-- Inserts are held in a per-transaction buffer and written back once when the
-- transaction ends, so a row with no vector travels a second path beside the
-- graph's: its own staged sequence, its own savepoint bookkeeping, and its own
-- replay when another backend wrote while this transaction was open. Every SQL
-- case that exercises staging inserts vectors only, so none of that was covered.
--
-- Both halves below are built so the two sequences have *different* lengths at
-- the savepoint. That is the point: one count cannot say where the rollback line
-- falls in two sequences, and a fixture where the counts happen to agree passes
-- on code that uses the graph's number for both.

CREATE EXTENSION IF NOT EXISTS dblink;

CREATE TABLE sv (id int, bucket int, embedding real[]);
-- Autovacuum must not run this test's vacuum for it, before the assertions.
ALTER TABLE sv SET (autovacuum_enabled = off);
INSERT INTO sv SELECT i, i % 10, ARRAY[i::real, (i + 1)::real]
FROM generate_series(1, 300) i;
INSERT INTO sv VALUES (901, 7, NULL), (902, 7, NULL);

CREATE INDEX sv_idx ON sv USING brindle (embedding, bucket);

-- ---------------------------------------------------------------------------
-- A savepoint rolls back the vectorless rows staged after it, and only those.
-- ---------------------------------------------------------------------------

BEGIN;
INSERT INTO sv VALUES (903, 7, NULL);                       -- vectorless 1, graph 0
INSERT INTO sv VALUES (907, 7, NULL);                       -- vectorless 2, graph 0
INSERT INTO sv VALUES (905, 7, ARRAY[5, 6]::real[]);        -- vectorless 2, graph 1
SAVEPOINT s;
INSERT INTO sv VALUES (904, 7, NULL);                       -- vectorless 3, graph 1
INSERT INTO sv VALUES (906, 7, ARRAY[7, 8]::real[]);        -- vectorless 3, graph 2
ROLLBACK TO s;
COMMIT;

-- The mark is (1 graph row, 2 vectorless rows). Carrying the graph's number
-- across to the other sequence keeps one of 903/907 and drops the other.
DO $$
DECLARE got int[]; idx_rows bigint; seq_rows bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(id ORDER BY id) INTO got
    FROM sv WHERE bucket = 7 AND id BETWEEN 900 AND 999;
    IF got IS DISTINCT FROM ARRAY[901, 902, 903, 905, 907] THEN
        RAISE EXCEPTION
            'after ROLLBACK TO the index holds %, expected {901,902,903,905,907}',
            got;
    END IF;

    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;
    SELECT count(*) INTO seq_rows FROM sv WHERE bucket = 7;
    SET LOCAL enable_seqscan = off;
    SET LOCAL enable_indexscan = on;
    SELECT count(*) INTO idx_rows FROM sv WHERE bucket = 7;
    IF idx_rows <> seq_rows THEN
        RAISE EXCEPTION 'index returns % rows against sequential %',
            idx_rows, seq_rows;
    END IF;
END $$;

-- The rolled-back rows must be gone from the index, not merely hidden by the
-- heap visibility recheck -- which discards a stale TID silently, so the counts
-- above agree whether or not the rewind happened. Reclaim the freed slots and
-- look for a stale entry resolving to whatever landed in them.
VACUUM sv;
INSERT INTO sv SELECT i, 3, ARRAY[i::real, (i + 1)::real]
FROM generate_series(9000, 9039) i;

DO $$
DECLARE bad int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(id) INTO bad FROM sv WHERE bucket = 7 AND id >= 9000;
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION
            'bucket 7 returned % -- an entry for a rolled-back row survived and '
            'now resolves to whatever reclaimed its line pointer', bad;
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- Another backend writing mid-transaction: the staged rows replay onto its
-- image, and the ones this transaction merely loaded are not replayed again.
-- ---------------------------------------------------------------------------

BEGIN;
INSERT INTO sv VALUES (908, 7, NULL);

-- A genuinely separate session commits, moving the generation this transaction
-- loaded at. Deliberately not a query against sv_idx: any scan of this index
-- would flush the staged row early and the replay path would never run.
SELECT dblink_exec(
    'dbname=' || current_database() ||
    ' port=' || current_setting('port') ||
    ' host=' || (string_to_array(current_setting('unix_socket_directories'), ','))[1],
    $inner$INSERT INTO sv VALUES (909, 4, ARRAY[9, 10]::real[])$inner$);

COMMIT;

-- Replaying the whole staged buffer rather than just this transaction's own
-- additions appends 901/902/903/907 a second time, because the image reloaded
-- for the replay already holds them. Every one of them then comes back twice.
DO $$
DECLARE idx_rows bigint; seq_rows bigint; dupes int[];
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;
    SELECT count(*) INTO seq_rows FROM sv WHERE bucket = 7;

    SET LOCAL enable_seqscan = off;
    SET LOCAL enable_indexscan = on;
    SELECT count(*) INTO idx_rows FROM sv WHERE bucket = 7;
    IF idx_rows <> seq_rows THEN
        RAISE EXCEPTION
            'after a concurrent write the index returns % rows against '
            'sequential % -- rows the image already held were replayed onto it '
            'again', idx_rows, seq_rows;
    END IF;

    SELECT array_agg(id) INTO dupes FROM (
        SELECT id FROM sv WHERE bucket = 7 GROUP BY id HAVING count(*) > 1
    ) q;
    IF dupes IS NOT NULL THEN
        RAISE EXCEPTION 'the index returned % more than once', dupes;
    END IF;

    -- And the row this transaction actually staged is there.
    PERFORM 1 FROM sv WHERE bucket = 7 AND id = 908;
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'the vectorless row staged before the concurrent write is missing';
    END IF;
END $$;
