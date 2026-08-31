-- A real VACUUM reclaiming deleted rows, end to end.
--
-- The Rust suite covers this by calling `index_bulk_delete` directly, because a
-- `#[pg_test]` cannot run VACUUM. That is a fair unit test of the callback and
-- proves nothing about the path VACUUM actually takes to reach it: which rows
-- Postgres decides are dead, when it decides to call the AM at all, and what it
-- does with the result. This exercises that path.

CREATE TABLE vac_dead (id int, embedding real[]);
INSERT INTO vac_dead
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 400) i;
CREATE INDEX vac_dead_idx ON vac_dead USING brindle (embedding);

DELETE FROM vac_dead WHERE id <= 100;
VACUUM vac_dead;

-- A scan must not return the deleted rows. Forcing the index scan matters:
-- against a sequential scan this would pass no matter what the index holds.
DO $$
DECLARE leaked int[];
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT array_agg(id) INTO leaked
    FROM (SELECT id FROM vac_dead
          ORDER BY embedding <-> ARRAY[1.0, 2.0]::real[] LIMIT 50) s
    WHERE id <= 100;
    IF leaked IS NOT NULL THEN
        RAISE EXCEPTION 'deleted rows still reachable through the index: %', leaked;
    END IF;
END $$;

-- Rows that survived must still be findable, so the vacuum did not take live
-- neighbours down with the dead ones.
DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM vac_dead
    ORDER BY embedding <-> ARRAY[101.0, 102.0]::real[] LIMIT 1;
    IF nearest <> 101 THEN
        RAISE EXCEPTION 'nearest live row was %, expected the exact match 101', nearest;
    END IF;
END $$;

-- And the index's reported tuple count must reflect the removal, which is the
-- AM's own accounting rather than the heap's.
DO $$
DECLARE tuples real; live bigint;
BEGIN
    SELECT reltuples INTO tuples FROM pg_class WHERE relname = 'vac_dead_idx';
    SELECT count(*) INTO live FROM vac_dead;
    IF tuples <> live THEN
        RAISE EXCEPTION 'index reports % tuples after VACUUM, table holds %', tuples, live;
    END IF;
END $$;
