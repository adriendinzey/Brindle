-- ROLLBACK TO a savepoint must undo the rows staged after it and keep the rows
-- staged before it.
--
-- Two ways to test this that do not work, both tried:
--
--   * "the rolled-back row is not returned" holds whatever the index contains.
--     The row is dead in the heap and Postgres rechecks heap visibility for
--     every TID an index scan returns.
--   * the line-pointer-recycling trick used by
--     vacuum_frees_line_pointers_safely does not reach this either. Freeing the
--     slots means running VACUUM, and VACUUM calls ambulkdelete, which
--     tombstones exactly those entries on the way past -- so the stale nodes are
--     cleaned up before they can produce a wrong answer, and the case passes
--     with the rollback disabled.
--
-- What does discriminate is the size of the stored graph. The metapage records
-- the blob length, so an index that kept the rolled-back rows is measurably
-- larger than one built from the same surviving rows without them. Compare
-- against that control rather than against a hardcoded number, which would only
-- be asserting the current encoding.

CREATE EXTENSION IF NOT EXISTS pageinspect;

-- The blob length the metapage records, in bytes. Little-endian u64 at offset 32
-- of page 0: 24 bytes of page header, then magic (4) and version (4).
CREATE FUNCTION blob_len(idx text) RETURNS bigint LANGUAGE sql AS $$
    SELECT ('x' || encode(
        substring(
            (SELECT string_agg(b, '' ORDER BY i DESC)
             FROM (SELECT i, substring(get_raw_page(idx, 0) from i for 1) AS b
                   FROM generate_series(33, 40) i) s),
            1, 8), 'hex'))::bit(64)::bigint
$$;

-- The index under test: 400 rows, then a transaction that keeps three rows and
-- rolls back fifty, with a nested savepoint, a released savepoint, and a reused
-- savepoint name.
CREATE TABLE sp (id int, embedding real[]);
ALTER TABLE sp SET (autovacuum_enabled = off);
INSERT INTO sp SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 400) i;
CREATE INDEX sp_idx ON sp USING brindle (embedding);

BEGIN;
INSERT INTO sp VALUES (500, ARRAY[500.0::real, 501.0::real]);    -- before s1: kept
SAVEPOINT s1;
INSERT INTO sp SELECT i, ARRAY[i::real, (i + 1)::real]
FROM generate_series(8000, 8024) i;                              -- rolled back
SAVEPOINT s2;
INSERT INTO sp SELECT i, ARRAY[i::real, (i + 1)::real]
FROM generate_series(8025, 8049) i;                              -- rolled back
ROLLBACK TO s1;
INSERT INTO sp VALUES (501, ARRAY[501.0::real, 502.0::real]);    -- after the rollback: kept
SAVEPOINT s3;
INSERT INTO sp VALUES (502, ARRAY[502.0::real, 503.0::real]);    -- released: kept
RELEASE SAVEPOINT s3;
COMMIT;

-- The control: the same 403 surviving rows, no savepoints anywhere.
CREATE TABLE sp_ctl (id int, embedding real[]);
ALTER TABLE sp_ctl SET (autovacuum_enabled = off);
INSERT INTO sp_ctl SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 400) i;
CREATE INDEX sp_ctl_idx ON sp_ctl USING brindle (embedding);
BEGIN;
INSERT INTO sp_ctl VALUES (500, ARRAY[500.0::real, 501.0::real]);
INSERT INTO sp_ctl VALUES (501, ARRAY[501.0::real, 502.0::real]);
INSERT INTO sp_ctl VALUES (502, ARRAY[502.0::real, 503.0::real]);
COMMIT;

DO $$
DECLARE got bigint; want bigint;
BEGIN
    got  := blob_len('sp_idx');
    want := blob_len('sp_ctl_idx');
    IF got <> want THEN
        RAISE EXCEPTION
            'stored graph is % bytes against a control of % for the same '
            'surviving rows: rows staged inside a rolled-back savepoint reached '
            'the index', got, want;
    END IF;
END $$;

-- And the rows that were *not* rolled back must really be there, so the check
-- above is not satisfied by having dropped everything.
DO $$
DECLARE missing int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(want) INTO missing FROM unnest(ARRAY[500, 501, 502]) want
    WHERE want IS DISTINCT FROM (
        SELECT id FROM sp
        ORDER BY embedding <-> ARRAY[want::real, (want + 1)::real] LIMIT 1);
    IF missing IS NOT NULL THEN
        RAISE EXCEPTION
            'rows staged outside the rolled-back savepoint are missing from the '
            'index: %', missing;
    END IF;
END $$;

-- Staging begun *inside* the rolled-back subtransaction has no mark to fall back
-- on, and must discard all of it rather than keeping it.
CREATE TABLE sp2 (id int, embedding real[]);
ALTER TABLE sp2 SET (autovacuum_enabled = off);
INSERT INTO sp2 SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 400) i;
CREATE INDEX sp2_idx ON sp2 USING brindle (embedding);

SELECT blob_len('sp2_idx') AS sp2_before \gset

BEGIN;
-- The savepoint opens before anything is staged, so there is no mark to fall
-- back on and the rollback has to discard all of it.
SAVEPOINT outer_sp;
INSERT INTO sp2 SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(8100, 8149) i;
ROLLBACK TO outer_sp;
COMMIT;

-- Read *after* the commit. Staged rows are not written until the transaction
-- ends, so a reading taken inside it cannot observe what this is asserting --
-- which is how an earlier version of this block passed against code that let
-- the rolled-back rows through.
SELECT blob_len('sp2_idx') AS sp2_after \gset
SELECT set_config('brindle_test.sp2_before', :'sp2_before', false) \gset carry_
SELECT set_config('brindle_test.sp2_after', :'sp2_after', false) \gset carry_

DO $$
DECLARE
    before_len bigint := current_setting('brindle_test.sp2_before')::bigint;
    after_len  bigint := current_setting('brindle_test.sp2_after')::bigint;
BEGIN
    IF after_len <> before_len THEN
        RAISE EXCEPTION
            'stored graph grew from % to % bytes across a subtransaction that '
            'staged rows and rolled back: they survived their own rollback',
            before_len, after_len;
    END IF;
END $$;

-- A concurrent commit between staging and the rollback must not cost the
-- transaction its own earlier rows.
--
-- The marks that say where each savepoint began are counts of staged rows, not
-- positions in the stored graph, precisely because of this: rolling back
-- reloads the image, another backend may have grown it in the meantime, and a
-- position recorded against the old image means something else against the new
-- one. Recorded as positions, every mark lands below the reloaded base and the
-- next ROLLBACK TO discards rows that were committed -- live in the heap, absent
-- from the index, with nothing to repair them.
CREATE EXTENSION IF NOT EXISTS dblink;
CREATE TABLE cw (id int, embedding real[]);
ALTER TABLE cw SET (autovacuum_enabled = off);
INSERT INTO cw SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 400) i;
CREATE INDEX cw_idx ON cw USING brindle (embedding);

SELECT set_config('brindle_test.conn',
                  'host=' || current_setting('unix_socket_directories')
                  || ' port=' || current_setting('port')
                  || ' dbname=' || current_database(), false) \gset carry_

BEGIN;
INSERT INTO cw VALUES (1001, ARRAY[1001.0::real, 1002.0::real]);  -- before a: must survive
SAVEPOINT a;
INSERT INTO cw VALUES (1002, ARRAY[1002.0::real, 1003.0::real]);  -- before b: must survive
SAVEPOINT b;
INSERT INTO cw VALUES (1003, ARRAY[1003.0::real, 1004.0::real]);  -- after b: rolled back
ROLLBACK TO b;
-- Another backend grows the stored image while this transaction holds staged
-- rows and an outstanding rewind.
SELECT dblink_exec(current_setting('brindle_test.conn'),
                   'INSERT INTO cw SELECT i, ARRAY[i::real, (i + 1)::real]
                    FROM generate_series(2001, 2100) i');
INSERT INTO cw VALUES (1004, ARRAY[1004.0::real, 1005.0::real]);  -- forces the rewind to be carried out
ROLLBACK TO b;
COMMIT;

DO $$
DECLARE missing int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(want) INTO missing FROM unnest(ARRAY[1001, 1002]) want
    WHERE want IS DISTINCT FROM (
        SELECT id FROM cw
        ORDER BY embedding <-> ARRAY[want::real, (want + 1)::real] LIMIT 1);
    IF missing IS NOT NULL THEN
        RAISE EXCEPTION
            'rows committed before the rolled-back savepoint are missing from '
            'the index: % -- they are live in the heap and nothing will repair '
            'them', missing;
    END IF;
END $$;

-- And the other backend's rows, and the rolled-back ones, are as they should be.
DO $$
DECLARE nearest int; leaked int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM cw
    ORDER BY embedding <-> ARRAY[2050.0, 2051.0]::real[] LIMIT 1;
    IF nearest IS DISTINCT FROM 2050 THEN
        RAISE EXCEPTION 'the concurrent backend''s rows are not in the index: got %', nearest;
    END IF;
    SELECT count(*) INTO leaked FROM cw WHERE id IN (1003, 1004);
    IF leaked <> 0 THEN
        RAISE EXCEPTION 'rolled-back rows % are live in the heap', leaked;
    END IF;
END $$;
