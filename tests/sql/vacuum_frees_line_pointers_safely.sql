-- A real VACUUM, then re-use of the slots it freed.
--
-- Deleting rows is not enough to test anything through SQL: Postgres rechecks
-- heap visibility for every TID an index scan returns, so a deleted row cannot
-- surface whether or not the index dropped its reference. An earlier version of
-- this case asserted exactly that and passed with no VACUUM at all.
--
-- The failure that *is* observable is the one ambulkdelete exists to prevent.
-- VACUUM recycles a dead tuple's line pointer once every index has confirmed it
-- dropped its references; an index node left pointing at a recycled slot then
-- resolves to whatever new row lands there. That is a wrong answer, not a miss,
-- and the visibility recheck cannot catch it because the row it finds is live.
--
-- So: delete, vacuum, then insert new rows that reclaim the freed slots, and
-- check the index never returns one of them for a query it does not match.

CREATE TABLE vac_recycle (id int, embedding real[]);
-- Autovacuum must not run this test's vacuum for it, at a moment of its
-- choosing, before the assertions below.
ALTER TABLE vac_recycle SET (autovacuum_enabled = off);

INSERT INTO vac_recycle
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 400) i;
CREATE INDEX vac_recycle_idx ON vac_recycle USING brindle (embedding);

-- Free a contiguous run of slots near the front of the heap.
DELETE FROM vac_recycle WHERE id <= 100;
VACUUM vac_recycle;

-- Reclaim them with rows that live far away in vector space. If any index node
-- still points at a recycled line pointer, a query near the *old* occupants now
-- resolves to one of these.
INSERT INTO vac_recycle
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(9000, 9099) i;

DO $$
DECLARE bad int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    -- Ask near where the deleted rows used to sit. Every answer must be a row
    -- that genuinely lies near [1, 2] — never one of the far-away newcomers.
    SELECT array_agg(id) INTO bad
    FROM (SELECT id FROM vac_recycle
          ORDER BY embedding <-> ARRAY[1.0, 2.0]::real[] LIMIT 20) s
    WHERE id >= 9000;
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION
            'index returned recycled rows % for a query near [1,2]: a node still '
            'points at a line pointer VACUUM freed', bad;
    END IF;
END $$;

-- And the newcomers must be findable on their own terms, so the reclaim did not
-- simply leave them out of the index.
DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM vac_recycle
    ORDER BY embedding <-> ARRAY[9050.0, 9051.0]::real[] LIMIT 1;
    IF nearest <> 9050 THEN
        RAISE EXCEPTION 'nearest to the reclaimed region was %, expected 9050', nearest;
    END IF;
END $$;
