-- A rebuilt index must not be mistaken for the one cached before it.
--
-- The generation counter catches a *write*, but REINDEX does not write to the
-- existing image — it builds a new relfilenode whose generation starts over at
-- 1. So a copy cached at generation 1, of an index nobody has written to since
-- it was built, matches the rebuilt one by coincidence. The cache is therefore
-- keyed on the physical relation as well, and this is the case that shows the
-- key is load-bearing rather than belt-and-braces.
--
-- Getting both sides to generation 1 is the whole setup: DELETE marks heap
-- tuples dead without touching the index (only VACUUM does that), so the
-- generation stays where CREATE INDEX left it while the rows underneath change.
--
-- A surviving stale copy shows up as *too few* rows, not wrong ones: it still
-- holds the deleted nodes, and Postgres rechecks heap visibility for every TID
-- an index scan returns, so they are dropped on the way out and the scan comes
-- up short. Count, do not look for leaks.

CREATE TABLE rebuilt (id int, embedding real[]);
ALTER TABLE rebuilt SET (autovacuum_enabled = off);
INSERT INTO rebuilt
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 400) i;
CREATE INDEX rebuilt_idx ON rebuilt USING brindle (embedding);

-- Cache it, at the generation CREATE INDEX wrote.
DO $$
DECLARE sink int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO sink FROM rebuilt
    ORDER BY embedding <-> ARRAY[200.0, 201.0]::real[] LIMIT 1;
END $$;

-- Rows go away; the index is not told, so its generation does not move.
DELETE FROM rebuilt WHERE id <= 200;
REINDEX INDEX rebuilt_idx;

DO $$
DECLARE got bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    -- Query where the deleted rows *were*. A stale copy walks straight to them,
    -- the recheck drops every one, and the scan comes up short. Asking near the
    -- surviving rows would not discriminate: they are in both copies.
    SELECT count(*) INTO got
    FROM (SELECT id FROM rebuilt
          ORDER BY embedding <-> ARRAY[1.0, 2.0]::real[] LIMIT 30) s;
    IF got <> 30 THEN
        RAISE EXCEPTION
            'asked for the 30 nearest live rows after REINDEX and got %; a copy '
            'cached before the rebuild still holds the deleted nodes, and the '
            'heap recheck drops them on the way out', got;
    END IF;
END $$;
