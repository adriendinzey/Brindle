-- A WHERE clause on an attribute column must reach the graph traversal, not be
-- applied to its results afterwards.
--
-- Correctness alone cannot tell those apart: post-filtering returns correct rows
-- too, just fewer of them. So this checks the two things that differ.
--
-- 1. The plan says `Index Cond`, not `Filter`. Postgres only matches a qual to
--    an index column if that column is part of the *search key* -- an INCLUDE
--    column is payload, and a qual on one never reaches the access method. The
--    attributes are therefore key columns after the vector.
-- 2. At low selectivity a bounded search that filters afterwards cannot fill k.
--    With ef_search = 200 over 20 000 rows at 1% selectivity, the 200 nearest
--    rows overall contain about two matches, so a post-filtered LIMIT 10 returns
--    about two rows. Filtering *during* traversal returns ten.
--
-- Both matter. The first alone would pass if the predicate reached the access
-- method and were then ignored; the second alone would pass if the executor
-- happened to be given enough candidates.

CREATE TABLE sel (id int, bucket int, embedding real[]);
ALTER TABLE sel SET (autovacuum_enabled = off);
INSERT INTO sel
SELECT i, i % 100, ARRAY[(i % 1000)::real, (i / 1000)::real, (i % 37)::real]
FROM generate_series(1, 20000) i;
CREATE INDEX sel_idx ON sel USING brindle (embedding, bucket);

-- The exact answer, from a scan that cannot use the index.
SET enable_indexscan = off;
SET enable_seqscan = on;
CREATE TABLE truth AS
SELECT id FROM sel WHERE bucket = 7
ORDER BY embedding <-> ARRAY[500.0, 10.0, 20.0]::real[] LIMIT 10;
RESET enable_indexscan;
RESET enable_seqscan;

DO $$
DECLARE line text; plan text := '';
BEGIN
    SET LOCAL enable_seqscan = off;
    FOR line IN EXECUTE
        'EXPLAIN SELECT id FROM sel WHERE bucket = 7 '
        'ORDER BY embedding <-> ARRAY[500.0, 10.0, 20.0]::real[] LIMIT 10'
    LOOP
        plan := plan || line || E'\n';
    END LOOP;
    IF plan NOT LIKE '%Index Scan%' THEN
        RAISE EXCEPTION 'plan does not use the index, so this proves nothing:%', E'\n' || plan;
    END IF;
    IF plan NOT LIKE '%Index Cond%' THEN
        RAISE EXCEPTION
            'the predicate did not reach the index -- it is an executor filter, '
            'which is the post-filtering this index exists to avoid:%', E'\n' || plan;
    END IF;
END $$;

DO $$
DECLARE returned bigint; violations bigint; hits bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 200;

    CREATE TEMP TABLE got AS
    SELECT id FROM sel WHERE bucket = 7
    ORDER BY embedding <-> ARRAY[500.0, 10.0, 20.0]::real[] LIMIT 10;

    -- Filled: a post-filtered scan comes up short here, by construction.
    SELECT count(*) INTO returned FROM got;
    IF returned <> 10 THEN
        RAISE EXCEPTION
            'asked for 10 matching rows at 1%% selectivity and got % -- the '
            'search is not filtering during traversal', returned;
    END IF;

    -- Correct: the hard constraint. A row that fails the filter is never
    -- acceptable, whatever the recall.
    SELECT count(*) INTO violations
    FROM got JOIN sel USING (id) WHERE sel.bucket IS DISTINCT FROM 7;
    IF violations <> 0 THEN
        RAISE EXCEPTION 'index returned % rows that fail the pushed predicate', violations;
    END IF;

    -- And good: the card's bar is 0.85 at 10% selectivity; this is 1%.
    SELECT count(*) INTO hits FROM got WHERE id IN (SELECT id FROM truth);
    IF hits < 9 THEN
        RAISE EXCEPTION
            'recall@10 among matching rows is %, below the 0.85 the card asks for',
            hits / 10.0;
    END IF;
END $$;

-- A NULL attribute satisfies nothing, matching SQL's treatment of comparisons
-- against NULL -- and it must not be dropped from the index either, or the rows
-- around it would lose their neighbours.
CREATE TABLE nulls (id int, bucket int, embedding real[]);
ALTER TABLE nulls SET (autovacuum_enabled = off);
INSERT INTO nulls
SELECT i, CASE WHEN i % 3 = 0 THEN NULL ELSE i % 10 END, ARRAY[i::real, (i + 1)::real]
FROM generate_series(1, 600) i;
CREATE INDEX nulls_idx ON nulls USING brindle (embedding, bucket);

DO $$
DECLARE violations bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 200;
    SELECT count(*) INTO violations FROM (
        SELECT id FROM nulls WHERE bucket = 4
        ORDER BY embedding <-> ARRAY[300.0, 301.0]::real[] LIMIT 20) s
    JOIN nulls USING (id) WHERE nulls.bucket IS DISTINCT FROM 4;
    IF violations <> 0 THEN
        RAISE EXCEPTION 'a NULL attribute satisfied a filter (% rows)', violations;
    END IF;
END $$;

-- A qual this access method does not claim stays the executor's job. It must
-- still be applied -- never silently dropped.
DO $$
DECLARE violations bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 200;
    SELECT count(*) INTO violations FROM (
        SELECT id FROM sel WHERE bucket <> 7
        ORDER BY embedding <-> ARRAY[500.0, 10.0, 20.0]::real[] LIMIT 20) s
    JOIN sel USING (id) WHERE sel.bucket = 7;
    IF violations <> 0 THEN
        RAISE EXCEPTION 'an unsupported qual was dropped: % rows fail it', violations;
    END IF;
END $$;

-- An index with no attribute columns at all still works, unfiltered.
CREATE TABLE plain (id int, embedding real[]);
ALTER TABLE plain SET (autovacuum_enabled = off);
INSERT INTO plain SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 400) i;
CREATE INDEX plain_idx ON plain USING brindle (embedding);

DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM plain
    ORDER BY embedding <-> ARRAY[200.0, 201.0]::real[] LIMIT 1;
    IF nearest IS DISTINCT FROM 200 THEN
        RAISE EXCEPTION 'a vector-only index stopped working: nearest is %', nearest;
    END IF;
END $$;
