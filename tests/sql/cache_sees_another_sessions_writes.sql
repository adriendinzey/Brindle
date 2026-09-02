-- A cached index must not answer from a copy another session has invalidated.
--
-- A scan keeps the decoded graph in this backend so later queries skip the
-- decode. Nothing tells it when a *different* backend writes: Postgres sends no
-- relcache invalidation for ordinary DML, so a cache keyed on the relation alone
-- would keep serving the index as it was. That is not a stale statistic, it is a
-- wrong answer — a row that exists and cannot be found.
--
-- The guard is a generation counter in the metapage, bumped by every writer and
-- checked by every reader. This is the test that it works, and it needs two real
-- backends: a `#[pg_test]` rolls its transaction back, so nothing it does is
-- visible to anyone else, and it cannot observe anyone either.

CREATE EXTENSION IF NOT EXISTS dblink;

CREATE TABLE cache_x (id int, embedding real[]);
INSERT INTO cache_x
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 300) i;
CREATE INDEX cache_x_idx ON cache_x USING brindle (embedding);

-- Populate this backend's cache, and confirm the row we are about to add is
-- genuinely absent first — otherwise the assertion at the end proves nothing.
DO $$
DECLARE found int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO found FROM cache_x
    ORDER BY embedding <-> ARRAY[9999.0, 10000.0]::real[] LIMIT 1;
    IF found = 9999 THEN
        RAISE EXCEPTION 'precondition failed: row 9999 already present';
    END IF;
END $$;

-- A second backend inserts. dblink runs it on its own connection, so this is a
-- genuinely separate session committing behind our back.
SELECT dblink_exec(
    'dbname=' || current_database() ||
    ' port=' || current_setting('port') ||
    ' host=' || (string_to_array(current_setting('unix_socket_directories'), ',')) [1],
    $inner$INSERT INTO cache_x
           SELECT 9999, ARRAY[9999.0::real, 10000.0::real]$inner$);

-- This backend still holds the graph it decoded before that insert. It must
-- notice and re-read rather than answer from it.
DO $$
DECLARE found int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO found FROM cache_x
    ORDER BY embedding <-> ARRAY[9999.0, 10000.0]::real[] LIMIT 1;
    IF found IS DISTINCT FROM 9999 THEN
        RAISE EXCEPTION
            'cached index missed a row another session committed: nearest to '
            '[9999,10000] came back as %, expected 9999', found;
    END IF;
END $$;

-- And a delete from the other session must stop being answered too.
SELECT dblink_exec(
    'dbname=' || current_database() ||
    ' port=' || current_setting('port') ||
    ' host=' || (string_to_array(current_setting('unix_socket_directories'), ',')) [1],
    $inner$DELETE FROM cache_x WHERE id = 9999$inner$);

DO $$
DECLARE found int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO found FROM cache_x
    ORDER BY embedding <-> ARRAY[9999.0, 10000.0]::real[] LIMIT 1;
    IF found = 9999 THEN
        RAISE EXCEPTION 'cached index still returns a row another session deleted';
    END IF;
END $$;

-- REINDEX writes a new relfilenode, so a cached copy keyed on the old one can
-- never be mistaken for the new index. Asserting that deleted rows stop being
-- returned would prove nothing — Postgres rechecks heap visibility for every TID
-- an index scan hands back, so they disappear whether or not the index knows.
-- An earlier version of this block did exactly that and passed against a
-- knowingly stale cache with no REINDEX at all.
--
-- What a stale copy actually does is return *too few* rows: it still holds the
-- deleted nodes, the recheck drops them, and the scan comes up short. So count.
DELETE FROM cache_x WHERE id <= 150;
REINDEX INDEX cache_x_idx;

DO $$
DECLARE got int[]; want int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(id ORDER BY id) INTO got
    FROM (SELECT id FROM cache_x
          ORDER BY embedding <-> ARRAY[151.0, 152.0]::real[] LIMIT 20) s;
    SELECT array_agg(i ORDER BY i) INTO want FROM generate_series(151, 170) i;
    IF got IS DISTINCT FROM want THEN
        RAISE EXCEPTION
            'after REINDEX the index returned % for the 20 nearest to [151,152], '
            'expected %; a copy cached before the rebuild would come up short',
            got, want;
    END IF;
END $$;

-- TRUNCATE also writes a new relfilenode, and empties the table. A stale copy
-- would still hold every node; the recheck hides them, so again count rather
-- than look for leaks — then refill and require the new rows to be findable,
-- which a copy from before the truncate cannot do.
TRUNCATE cache_x;
INSERT INTO cache_x
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(7000, 7099) i;

DO $$
DECLARE nearest int; total bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT count(*) INTO total
    FROM (SELECT id FROM cache_x
          ORDER BY embedding <-> ARRAY[7050.0, 7051.0]::real[] LIMIT 10) s;
    IF total <> 10 THEN
        RAISE EXCEPTION
            'after TRUNCATE and refill the index returned % of 10 rows; a copy '
            'cached before the truncate does not contain them', total;
    END IF;
    SELECT id INTO nearest FROM cache_x
    ORDER BY embedding <-> ARRAY[7050.0, 7051.0]::real[] LIMIT 1;
    IF nearest <> 7050 THEN
        RAISE EXCEPTION 'nearest after TRUNCATE and refill was %, expected 7050', nearest;
    END IF;
END $$;

-- Dropping and recreating the index is the third way its identity changes, and
-- the acceptance criteria name it separately.
DROP INDEX cache_x_idx;
CREATE INDEX cache_x_idx ON cache_x USING brindle (embedding);
INSERT INTO cache_x SELECT 7500, ARRAY[7500.0::real, 7501.0::real];

DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM cache_x
    ORDER BY embedding <-> ARRAY[7500.0, 7501.0]::real[] LIMIT 1;
    IF nearest <> 7500 THEN
        RAISE EXCEPTION
            'after DROP and CREATE the index answered %, expected 7500', nearest;
    END IF;
END $$;
