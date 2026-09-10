-- A comparison on a float column must return the same rows through this index
-- as through a sequential scan, including when a row stores NaN.
--
-- PostgreSQL gives floats a total order so that btree works: `'NaN' = 'NaN'` is
-- true and NaN sorts above every other value, so `'NaN' >= 1` is true. The core
-- used to order floats by IEEE 754 instead, where NaN compares with nothing, so
-- a stored NaN failed a comparison SQL would satisfy.
--
-- That was documented as costing rows and never wrong ones. It does not survive
-- negation, and the anti-join below is the case that shows why: rows the index
-- drops from the inner side come back as *extra* rows in the result. A count
-- that is too high cannot be explained away as a recall trade.
--
-- Both directions are asserted, and both against a sequential scan of the same
-- query rather than against a constant -- the index must agree with PostgreSQL,
-- not with a number someone typed here.

CREATE TABLE fz (id int, score float8, embedding real[]);
ALTER TABLE fz SET (autovacuum_enabled = off);
INSERT INTO fz
SELECT i, (i % 100)::float8, ARRAY[i::real, (i + 1)::real]
FROM generate_series(1, 400) i;
INSERT INTO fz VALUES
    (901, 'NaN'::float8, ARRAY[10.0, 11.0]::real[]),
    (902, 'NaN'::float8, ARRAY[20.0, 21.0]::real[]);
CREATE INDEX fz_idx ON fz USING brindle (embedding, score);

-- The premise: PostgreSQL really does hold NaN >= 1. If a future PostgreSQL
-- changed that, this case should say so rather than quietly test nothing.
DO $$
DECLARE pg_says boolean;
BEGIN
    SELECT 'NaN'::float8 >= 1 INTO pg_says;
    IF NOT pg_says THEN
        RAISE EXCEPTION 'this PostgreSQL does not sort NaN above 1; the rule under test has changed';
    END IF;
END $$;

-- The qual must actually reach the access method, or this tests the executor.
DO $$
DECLARE line text; plan text := '';
BEGIN
    SET LOCAL enable_seqscan = off;
    FOR line IN EXECUTE
        'EXPLAIN SELECT id FROM fz WHERE score >= 1 '
        'ORDER BY embedding <-> ARRAY[10.0, 11.0]::real[] LIMIT 500'
    LOOP
        plan := plan || line || E'\n';
    END LOOP;
    IF plan NOT LIKE '%Index Scan%' OR plan NOT LIKE '%Index Cond%' THEN
        RAISE EXCEPTION 'the float bound did not reach the index:%', E'\n' || plan;
    END IF;
END $$;

-- A stored NaN must not be dropped from an ordinary comparison.
--
-- `ef_search` is raised above the row count deliberately: an ordered brindle
-- scan returns at most its budget, so at the default 64 this query would come up
-- short for a reason that has nothing to do with NaN, and the case would pass or
-- fail on the wrong mechanism.
DO $$
BEGIN
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.ef_search = 2000;
    CREATE TEMP TABLE got_idx AS
    SELECT id FROM fz WHERE score >= 1
    ORDER BY embedding <-> ARRAY[10.0, 11.0]::real[] LIMIT 500;
END $$;

DO $$
DECLARE seq_rows bigint; idx_rows bigint; missing_nan bigint;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;

    SELECT count(*) INTO seq_rows FROM fz WHERE score >= 1;
    SELECT count(*) INTO idx_rows FROM got_idx;
    IF idx_rows <> seq_rows THEN
        RAISE EXCEPTION
            'index returned % rows for `score >= 1`, sequential scan returned % '
            '-- the index disagrees with PostgreSQL', idx_rows, seq_rows;
    END IF;

    SELECT count(*) INTO missing_nan
    FROM fz WHERE score = 'NaN'::float8 AND id NOT IN (SELECT id FROM got_idx);
    IF missing_nan <> 0 THEN
        RAISE EXCEPTION '% row(s) storing NaN were dropped by the index', missing_nan;
    END IF;
END $$;

-- ...and the negation, where dropped rows become invented ones. Before the fix
-- the sequential scan answered 4 and the index answered 6.
DO $$
DECLARE anti_seq bigint; anti_idx bigint;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;
    SELECT count(*) INTO anti_seq FROM fz a
     WHERE NOT EXISTS (SELECT 1 FROM fz b WHERE b.id = a.id AND b.score >= 1);

    SET LOCAL enable_seqscan = off;
    SET LOCAL enable_indexscan = on;
    -- No ef_search here on purpose: an ORDER BY-less scan takes the keyless
    -- path, which enumerates every matching node and has no budget to run out
    -- of. Setting one would imply a truncation risk this path does not have.
    SELECT count(*) INTO anti_idx FROM fz a
     WHERE NOT EXISTS (SELECT 1 FROM fz b WHERE b.id = a.id AND b.score >= 1);

    IF anti_idx <> anti_seq THEN
        RAISE EXCEPTION
            'anti-join returned % rows through the index and % by sequential '
            'scan -- rows the index dropped came back as extra rows',
            anti_idx, anti_seq;
    END IF;
END $$;

-- A NaN *bound* is answerable now too, and used to be refused at the boundary.
-- `score < 'NaN'` selects every non-NaN row; `score >= 'NaN'` selects the NaN
-- rows alone.
--
-- These, and everything below, go through the **keyless** path -- a count with no
-- ORDER BY, which enumerates every matching node instead of walking the graph.
-- That is deliberate. `score >= 'NaN'` matches 2 rows of 402, and an ordered ANN
-- scan asked for 2 needles among 402 nodes is measuring *recall*, not the
-- comparison: whether it finds them depends on where they sit relative to the
-- query, so the case would pass or fail on the fixture's geometry. An earlier
-- draft of this file did exactly that and passed only because both NaN rows
-- happened to be the two nearest the query point. The keyless path has no such
-- variable, which is what makes it the honest place to pin semantics.
DO $$
DECLARE
    shape text; shapes text[] := ARRAY[
        'score < ''NaN''::float8', 'score >= ''NaN''::float8', 'score = ''NaN''::float8'
    ];
    line text; plan text; via_index bigint; via_heap bigint;
BEGIN
    FOREACH shape IN ARRAY shapes LOOP
        SET LOCAL enable_seqscan = off;
        SET LOCAL enable_indexscan = on;

        plan := '';
        FOR line IN EXECUTE 'EXPLAIN SELECT count(*) FROM fz WHERE ' || shape LOOP
            plan := plan || line || E'\n';
        END LOOP;
        IF plan NOT LIKE '%Index Cond%' THEN
            RAISE EXCEPTION
                'the qual `%` did not reach the index, so comparing it against a '
                'heap scan proves nothing:%', shape, E'\n' || plan;
        END IF;
        EXECUTE 'SELECT count(*) FROM fz WHERE ' || shape INTO via_index;

        SET LOCAL enable_indexscan = off;
        SET LOCAL enable_seqscan = on;
        EXECUTE 'SELECT count(*) FROM fz WHERE ' || shape INTO via_heap;

        IF via_index <> via_heap THEN
            RAISE EXCEPTION
                'the qual `%` returns % rows through the index against % from a '
                'heap scan', shape, via_index, via_heap;
        END IF;
    END LOOP;
END $$;

-- -0.0 = 0.0 in SQL. `f64::total_cmp` is the obvious way to write "NaN sorts
-- above everything" and gets this wrong, so the case is here to keep anyone
-- (including a future simplification) from reaching for it.
--
-- float4 as well as float8: PostgreSQL compares them with a different function
-- (`float4_cmp`), the datum is widened before it reaches the core, and the
-- operator family declares cross-type members -- so `float4_col = 0.0`, where
-- the literal resolves to float8, exercises a path float8 alone does not.
CREATE TABLE zed (id int, s8 float8, s4 float4, embedding real[]);
ALTER TABLE zed SET (autovacuum_enabled = off);
INSERT INTO zed VALUES
    (1, -0.0::float8, -0.0::float4, ARRAY[1.0, 1.0]::real[]),
    (2,  0.0::float8,  0.0::float4, ARRAY[2.0, 2.0]::real[]),
    (3,  1.0::float8,  1.0::float4, ARRAY[3.0, 3.0]::real[]);
INSERT INTO zed
SELECT i, (i % 7)::float8, (i % 7)::float4, ARRAY[i::real, i::real]
FROM generate_series(10, 300) i;
CREATE INDEX zed_idx ON zed USING brindle (embedding, s8, s4);

DO $$
DECLARE
    shape text; shapes text[] := ARRAY[
        's8 = 0.0', 's8 >= 0.0', 's4 = 0.0', 's4 >= 0.0',
        's4 = 0.0::float4', 's4 >= 1::float8'
    ];
    line text; plan text; via_index bigint; via_heap bigint;
BEGIN
    FOREACH shape IN ARRAY shapes LOOP
        SET LOCAL enable_seqscan = off;
        SET LOCAL enable_indexscan = on;

        plan := '';
        FOR line IN EXECUTE 'EXPLAIN SELECT count(*) FROM zed WHERE ' || shape LOOP
            plan := plan || line || E'\n';
        END LOOP;
        IF plan NOT LIKE '%Index Cond%' THEN
            RAISE EXCEPTION
                'the qual `%` did not reach the index:%', shape, E'\n' || plan;
        END IF;
        EXECUTE 'SELECT count(*) FROM zed WHERE ' || shape INTO via_index;

        SET LOCAL enable_indexscan = off;
        SET LOCAL enable_seqscan = on;
        EXECUTE 'SELECT count(*) FROM zed WHERE ' || shape INTO via_heap;

        IF via_index <> via_heap THEN
            RAISE EXCEPTION
                '`%` returns % rows through the index against % from a heap scan',
                shape, via_index, via_heap;
        END IF;
    END LOOP;
END $$;
