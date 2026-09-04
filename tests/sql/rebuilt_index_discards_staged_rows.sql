-- Rows staged for an index that is rebuilt before the transaction ends must be
-- dropped, not written over the rebuild.
--
-- The oid survives a rebuild, so opening it at flush time says nothing about
-- whether this is still the same index. TRUNCATE and REINDEX both give the
-- relation a new relfilenode while keeping the oid, and the generation cannot
-- distinguish them either: a rebuild starts counting again, so a graph staged at
-- generation 1 compares equal to the fresh one.
--
-- Writing it anyway leaves the index pointing at heap rows that no longer exist,
-- which is not a wrong answer but a hard error — every later scan reads past the
-- end of the file and the table is unusable until someone reindexes it.

CREATE TABLE rb (id int, embedding real[]);
ALTER TABLE rb SET (autovacuum_enabled = off);
INSERT INTO rb SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 300) i;
CREATE INDEX rb_idx ON rb USING brindle (embedding);

-- Insert, then truncate the table out from under the staged row.
BEGIN;
INSERT INTO rb SELECT 9001, ARRAY[9001.0::real, 9002.0::real];
TRUNCATE rb;
COMMIT;

DO $$
DECLARE found bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    -- The scan must simply find nothing. If the staged graph was written over
    -- the truncated index, this raises instead: the entries point into a heap
    -- with no blocks left.
    SELECT count(*) INTO found
    FROM (SELECT id FROM rb
          ORDER BY embedding <-> ARRAY[9001.0, 9002.0]::real[] LIMIT 5) s;
    IF found <> 0 THEN
        RAISE EXCEPTION
            'index returned % rows from a truncated table; a graph staged before '
            'the truncate was written over the rebuilt index', found;
    END IF;
END $$;

-- REINDEX in the same transaction is the milder version of the same mistake.
-- The heap is untouched by a rebuild, so staged rows written over it still point
-- at real tuples and no scan breaks; what is lost is the rebuild itself, quietly
-- undone by a graph staged before it.
--
-- There is deliberately no assertion on the scan here. A non-concurrent rebuild
-- scans with SnapshotAny and so indexes rows that are deleted but not yet
-- vacuumed, which both outcomes contain — and those entries consume the search
-- budget on the way out, so a row count near the deleted region says more about
-- `ef_search` than about which graph won. Asserting on it would be a test that
-- fails for the wrong reason, which is worse than not testing it. The size check
-- below is the honest, if weaker, signal.
INSERT INTO rb SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 300) i;
REINDEX INDEX rb_idx;

-- The row inserted alongside the rebuild must be findable either way, which is
-- the part that would be a correctness bug rather than wasted work.
BEGIN;
INSERT INTO rb SELECT 9002, ARRAY[9003.0::real, 9004.0::real];
REINDEX INDEX rb_idx;
COMMIT;

DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM rb
    ORDER BY embedding <-> ARRAY[9003.0, 9004.0]::real[] LIMIT 1;
    IF nearest IS DISTINCT FROM 9002 THEN
        RAISE EXCEPTION
            'a row inserted in the same transaction as a REINDEX is not in the '
            'index: nearest to [9003,9004] is %', nearest;
    END IF;
END $$;
