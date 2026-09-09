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

-- A qual that *reaches* this access method and is refused by it must still be
-- applied, by the executor, on the rows the scan hands back.
--
-- `bucket <> 7` will not do for this, though it looks like it should: `<>` is in
-- no brindle operator family, so the planner never makes it a scan key and the
-- access method never sees it. An earlier version of this block used exactly
-- that and therefore tested Postgres's own executor filter -- it passed with the
-- recheck plumbing removed entirely, which is how a real defect shipped.
--
-- What does reach the access method and get refused is a scan key whose value
-- turns out to be NULL at run time. `ExecIndexEvalRuntimeKeys` marks it
-- SK_ISNULL, `atom_from_key` declines it (a comparison against NULL is never
-- true, so it cannot be expressed as an atom), and the scan must then set the
-- recheck flag -- or every row it returns is one the query excluded.
DO $$
DECLARE returned bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 200;
    SELECT count(*) INTO returned FROM (
        SELECT id FROM sel WHERE bucket = (SELECT NULL::int)
        ORDER BY embedding <-> ARRAY[500.0, 10.0, 20.0]::real[] LIMIT 5) s;
    IF returned <> 0 THEN
        RAISE EXCEPTION
            'a scan key the index refused was dropped instead of rechecked: '
            '% rows came back for `bucket = NULL`, which is never true', returned;
    END IF;
END $$;

-- And the same shape as a nest-loop parameter, which is how it turns up in real
-- queries: an optional filter whose driving value is NULL for some outer rows.
DO $$
DECLARE leaked bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 200;
    CREATE TEMP TABLE probe(v int);
    INSERT INTO probe VALUES (7), (NULL);
    SELECT count(*) INTO leaked FROM probe p
    LEFT JOIN LATERAL (
        SELECT id FROM sel WHERE bucket = p.v
        ORDER BY embedding <-> ARRAY[500.0, 10.0, 20.0]::real[] LIMIT 3) s ON true
    WHERE p.v IS NULL AND s.id IS NOT NULL;
    IF leaked <> 0 THEN
        RAISE EXCEPTION
            'a NULL nest-loop parameter returned % rows; the refused scan key '
            'was not rechecked', leaked;
    END IF;
END $$;

-- A qual the planner never offers the index at all stays the executor's job.
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

-- Comparisons across widths within a family must push *and* be exact.
--
-- The integer half of this survives almost any mistake by accident: integer
-- datums sign-extend, so reading a narrow one as a wider type round-trips for
-- small values. The float half does not. A `float8` column compared against a
-- `float4` literal is the one pairing where reading the argument as the
-- *column's* type instead of its own reinterprets the bit pattern, and the
-- result is a returned row that fails the predicate — which is the one thing
-- this index may never do.
CREATE TABLE widths (
    id int, i2 int2, i8 int8, f4 float4, f8 float8, embedding real[]
);
ALTER TABLE widths SET (autovacuum_enabled = off);
INSERT INTO widths
SELECT i, (i % 300)::int2, (i % 500)::int8, (i % 100)::float4, (i % 100)::float8,
       ARRAY[(i % 200)::real, (i / 200)::real]
FROM generate_series(1, 4000) i;
CREATE INDEX widths_idx ON widths USING brindle (embedding, i2, i8, f4, f8);

DO $$
DECLARE
    shapes text[] := ARRAY[
        'i2 = 42::int8', 'i2 < 42::int4', 'i8 = 42::int4', 'i8 > 400::int2',
        'f4 = 1.5::float8', 'f4 > 50::float8', 'f8 = 1.5::float4', 'f8 > 0.1::float4'
    ];
    shape text; plan text; line text; via_index bigint; via_heap bigint;
BEGIN
    SET LOCAL brindle.ef_search = 6000;
    FOREACH shape IN ARRAY shapes LOOP
        -- It has to reach the access method, or the rest measures the executor.
        plan := '';
        SET LOCAL enable_seqscan = off;
        FOR line IN EXECUTE
            'EXPLAIN SELECT count(*) FROM widths WHERE ' || shape
        LOOP
            plan := plan || line || E'\n';
        END LOOP;
        IF plan NOT LIKE '%Index Cond%' THEN
            RAISE EXCEPTION
                'the qual `%` did not reach the index -- a cross-type comparison '
                'silently degraded to post-filtering:%', shape, E'\n' || plan;
        END IF;
        EXECUTE 'SELECT count(*) FROM widths WHERE ' || shape INTO via_index;

        SET LOCAL enable_indexscan = off;
        SET LOCAL enable_seqscan = on;
        EXECUTE 'SELECT count(*) FROM widths WHERE ' || shape INTO via_heap;
        RESET enable_indexscan;

        IF via_index <> via_heap THEN
            RAISE EXCEPTION
                'the qual `%` returns % rows through the index against % from a '
                'heap scan -- the argument is being read as the wrong type',
                shape, via_index, via_heap;
        END IF;
    END LOOP;
END $$;

-- A NaN bound must not be pushed. The core orders floats by IEEE 754, where NaN
-- compares equal to nothing; PostgreSQL gives floats a total order in which
-- `'NaN' = 'NaN'` is true and NaN sorts above everything. Pushing such a bound
-- would answer a different question than the query asked, so the scan refuses it
-- and the executor -- which has the right semantics -- decides.
--
-- The rows a *stored* NaN would add are a separate, filed gap; this asserts only
-- the query-side half, by comparing against a heap scan rather than a constant.
CREATE TABLE nan_t (id int, f8 float8, embedding real[]);
ALTER TABLE nan_t SET (autovacuum_enabled = off);
INSERT INTO nan_t
SELECT i, CASE WHEN i % 200 = 0 THEN 'NaN'::float8 ELSE (i % 100)::float8 END,
       ARRAY[(i % 200)::real, (i / 200)::real]
FROM generate_series(1, 404) i;
CREATE INDEX nan_idx ON nan_t USING brindle (embedding, f8);

DO $$
DECLARE
    shapes text[] := ARRAY[
        'f8 = ''NaN''::float8', 'f8 < ''NaN''::float8', 'f8 > ''NaN''::float8'
    ];
    shape text; via_index bigint; via_heap bigint;
BEGIN
    SET LOCAL brindle.ef_search = 2000;
    FOREACH shape IN ARRAY shapes LOOP
        SET LOCAL enable_seqscan = off;
        SET LOCAL enable_indexscan = on;
        EXECUTE 'SELECT count(*) FROM nan_t WHERE ' || shape INTO via_index;
        SET LOCAL enable_indexscan = off;
        SET LOCAL enable_seqscan = on;
        EXECUTE 'SELECT count(*) FROM nan_t WHERE ' || shape INTO via_heap;
        IF via_index <> via_heap THEN
            RAISE EXCEPTION
                'the qual `%` returns % rows through the index against % from a '
                'heap scan -- a NaN bound was pushed with IEEE semantics instead '
                'of being refused', shape, via_index, via_heap;
        END IF;
    END LOOP;
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
