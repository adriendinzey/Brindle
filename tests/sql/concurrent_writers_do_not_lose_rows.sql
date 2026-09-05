-- Two transactions writing the same index must both end up in it.
--
-- Pending rows are held in memory and written back at commit, and the exclusive
-- lock is deliberately not held across the transaction — holding it would make
-- one writer block every other for its whole duration. The cost of that choice
-- is that another backend can write in between, so the flush compares the
-- metapage generation against the one it started from and, when they differ,
-- replays its rows onto the image that is actually there rather than writing
-- over it.
--
-- Without that, the later commit silently drops the earlier one's rows: a lost
-- update, and the reason this case exists.

CREATE EXTENSION IF NOT EXISTS dblink;

CREATE TABLE conc_t (id int, embedding real[]);
ALTER TABLE conc_t SET (autovacuum_enabled = off);
INSERT INTO conc_t
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 300) i;
CREATE INDEX conc_idx ON conc_t USING brindle (embedding);

BEGIN;
-- This transaction's row, held pending.
INSERT INTO conc_t SELECT 4001, ARRAY[4001.0::real, 4002.0::real];

-- Another backend commits its own row while ours is still open and unwritten.
SELECT dblink_exec(
    'dbname=' || current_database() ||
    ' port=' || current_setting('port') ||
    ' host=' || (string_to_array(current_setting('unix_socket_directories'), ',')) [1],
    $inner$INSERT INTO conc_t SELECT 4002, ARRAY[4003.0::real, 4004.0::real]$inner$);
COMMIT;

-- Both rows must be findable. Ours went in last, so it is the one that would
-- have overwritten theirs.
DO $$
DECLARE ours int; theirs int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO ours FROM conc_t
    ORDER BY embedding <-> ARRAY[4001.0, 4002.0]::real[] LIMIT 1;
    IF ours <> 4001 THEN
        RAISE EXCEPTION 'this transaction''s own row is missing: got %', ours;
    END IF;

    SELECT id INTO theirs FROM conc_t
    ORDER BY embedding <-> ARRAY[4003.0, 4004.0]::real[] LIMIT 1;
    IF theirs <> 4002 THEN
        RAISE EXCEPTION
            'the other session''s row was lost: nearest to [4003,4004] is %, '
            'so this commit wrote over an image it had not read', theirs;
    END IF;
END $$;

-- The same race, but with the index moved between tablespaces before the commit.
-- A move changes the relfilenode without rebuilding anything, so a flush that
-- treated any relfilenode change as a rebuild would discard our staged row here
-- while the other session's was already on disk — losing ours silently.
DROP TABLESPACE IF EXISTS brindle_conc_ts;
CREATE TABLESPACE brindle_conc_ts LOCATION :'tablespace_dir';

BEGIN;
INSERT INTO conc_t SELECT 4101, ARRAY[4101.0::real, 4102.0::real];
SELECT dblink_exec(
    'dbname=' || current_database() ||
    ' port=' || current_setting('port') ||
    ' host=' || (string_to_array(current_setting('unix_socket_directories'), ',')) [1],
    $inner$INSERT INTO conc_t SELECT 4102, ARRAY[4103.0::real, 4104.0::real]$inner$);
ALTER INDEX conc_idx SET TABLESPACE brindle_conc_ts;
COMMIT;

DO $$
DECLARE ours int; theirs int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO ours FROM conc_t
    ORDER BY embedding <-> ARRAY[4101.0, 4102.0]::real[] LIMIT 1;
    SELECT id INTO theirs FROM conc_t
    ORDER BY embedding <-> ARRAY[4103.0, 4104.0]::real[] LIMIT 1;
    IF ours <> 4101 THEN
        RAISE EXCEPTION
            'our staged row was dropped when the index moved tablespace: '
            'nearest to [4101,4102] is %', ours;
    END IF;
    IF theirs <> 4102 THEN
        RAISE EXCEPTION 'the other session''s row was lost across the move: %', theirs;
    END IF;
END $$;

ALTER INDEX conc_idx SET TABLESPACE pg_default;
DROP TABLESPACE IF EXISTS brindle_conc_ts;
