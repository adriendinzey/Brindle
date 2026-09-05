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

DO $$
DECLARE before_len bigint; after_len bigint;
BEGIN
    before_len := blob_len('sp2_idx');
    BEGIN
        -- The savepoint opens before anything is staged, so there is no mark.
        INSERT INTO sp2 SELECT i, ARRAY[i::real, (i + 1)::real]
        FROM generate_series(8100, 8149) i;
        RAISE EXCEPTION 'roll back';
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;
    after_len := blob_len('sp2_idx');
    IF after_len <> before_len THEN
        RAISE EXCEPTION
            'stored graph grew from % to % bytes across a subtransaction that '
            'staged rows and rolled back: they survived their own rollback',
            before_len, after_len;
    END IF;
END $$;
