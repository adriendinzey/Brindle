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

-- REINDEX replaces the image wholesale, and the replacement starts its
-- generation over — so a copy cached at the same number would match by
-- coincidence rather than by being current. Postgres invalidates the relcache
-- entry, which drops the cache with it; this checks that it really does, since
-- the generation alone would not catch it.
DELETE FROM cache_x WHERE id <= 150;
REINDEX INDEX cache_x_idx;

DO $$
DECLARE leaked int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(id) INTO leaked
    FROM (SELECT id FROM cache_x
          ORDER BY embedding <-> ARRAY[1.0, 2.0]::real[] LIMIT 40) s
    WHERE id <= 150;
    IF leaked IS NOT NULL THEN
        RAISE EXCEPTION
            'index returned rows % after REINDEX; a cached copy survived the '
            'rebuild that removed them', leaked;
    END IF;
END $$;

-- TRUNCATE empties the table and rebuilds the index; nothing may survive it.
TRUNCATE cache_x;

DO $$
DECLARE remaining bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT count(*) INTO remaining
    FROM (SELECT id FROM cache_x
          ORDER BY embedding <-> ARRAY[1.0, 2.0]::real[] LIMIT 10) s;
    IF remaining <> 0 THEN
        RAISE EXCEPTION 'index returned % rows after TRUNCATE', remaining;
    END IF;
END $$;
