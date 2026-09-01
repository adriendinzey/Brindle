-- A partial index must not have the heap's tuple count reported as its own.
--
-- The cleanup callback reports what a vacuum already counted over the heap
-- rather than reading the index to ask it. For a full index that is the same
-- number. For a partial one it is not: `indpred` means most of those heap tuples
-- have no entry here, so reporting the heap's total overstates the index by
-- however selective the predicate is — and an overstated index is one the
-- planner avoids, which is the opposite of why it was made partial.
--
-- Needs a committed database twice over: VACUUM cannot run inside a transaction
-- block, and DISABLE_PAGE_SKIPPING is what makes the refresh path run at all.

CREATE TABLE vac_partial (id int, embedding real[]);
ALTER TABLE vac_partial SET (autovacuum_enabled = off);
INSERT INTO vac_partial
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 2000) i;

-- One tenth of the table.
CREATE INDEX vac_partial_idx ON vac_partial USING brindle (embedding)
WHERE id <= 200;

VACUUM (DISABLE_PAGE_SKIPPING) vac_partial;

DO $$
DECLARE tuples real; indexed bigint; heap bigint;
BEGIN
    SELECT reltuples INTO tuples FROM pg_class WHERE relname = 'vac_partial_idx';
    SELECT count(*) INTO indexed FROM vac_partial WHERE id <= 200;
    SELECT count(*) INTO heap FROM vac_partial;

    IF tuples = heap THEN
        RAISE EXCEPTION
            'partial index reports % tuples, the whole heap, when only % rows '
            'match its predicate', tuples, indexed;
    END IF;
    -- Anything between "untouched" and the true count is defensible; claiming
    -- the heap's total is not.
    IF tuples > indexed AND tuples <> -1 THEN
        RAISE EXCEPTION
            'partial index reports % tuples, more than the % matching its predicate',
            tuples, indexed;
    END IF;
END $$;

-- And it must still answer correctly.
DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM vac_partial WHERE id <= 200
    ORDER BY embedding <-> ARRAY[50.0, 51.0]::real[] LIMIT 1;
    IF nearest <> 50 THEN
        RAISE EXCEPTION 'partial index answered % after VACUUM, expected 50', nearest;
    END IF;
END $$;
