-- A filter correlated with position must still be answered.
--
-- Every recall case that came before this one labels rows independently of
-- where their vector sits, so matching rows are sprinkled through every
-- neighbourhood and a search that can only look near the query still finds
-- ten of them. That is the easy half of filtered search, and it hid the hard
-- half completely: when the predicate selects a *region* the query is not in,
-- traversal has to cross the non-matching ground in between.
--
-- Here `embedding` is a 500 x 20 grid and `price = i % 100`, so `price < 5`
-- keeps 25 of the 500 x-positions. The query sits at x = 250, whose nearest
-- matching column is 46 away with roughly two thousand non-matching rows
-- nearer than it. This returned *zero* rows -- not a short answer, an empty
-- one -- at every ef_search below 5000, while an uncorrelated filter of the
-- same selectivity returned ten.
--
-- Three things are asserted, because the first two can each pass for the wrong
-- reason: that ten rows come back (a reachability failure returns none), that
-- none of them fails the predicate (the constraint that outranks recall), and
-- that they are the *right* ten (reaching some far matching column is not the
-- same as reaching the nearest one -- an earlier version of the traversal
-- returned ten qualifying rows from a column 150 away and scored 0 recall).
--
-- Deliberately at the default ef_search: needing 5000 is the bug.

CREATE TABLE reach (id int, price int, embedding real[]);
ALTER TABLE reach SET (autovacuum_enabled = off);
INSERT INTO reach
SELECT i, i % 100, ARRAY[(i % 500)::real, (i / 500)::real]
FROM generate_series(1, 10000) i;
CREATE INDEX reach_idx ON reach USING brindle (embedding, price);

-- The exact answer, from a scan that cannot use the index.
SET enable_indexscan = off;
SET enable_seqscan = on;
CREATE TABLE reach_truth AS
SELECT id FROM reach WHERE price < 5
ORDER BY embedding <-> ARRAY[250.0, 10.0]::real[] LIMIT 10;
RESET enable_indexscan;
RESET enable_seqscan;

-- The premise: the matching rows really are far away. If a fixture change ever
-- moved them next to the query this case would pass without testing anything,
-- which is the failure mode the harness header warns about.
--
-- Off the index, deliberately. Postgres rewrites `min(expr)` into
-- `ORDER BY expr LIMIT 1`, which this index can serve -- so the first draft of
-- this block measured the traversal it was supposed to be independent of, and
-- reported "the nearest match is 0 rows away" when the traversal returned
-- nothing at all.
DO $$
DECLARE nearest double precision; nearer_non_matching bigint;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;

    SELECT min(embedding <-> ARRAY[250.0, 10.0]::real[]) INTO nearest
    FROM reach WHERE price < 5;
    IF nearest < 40 THEN
        RAISE EXCEPTION
            'fixture is not testing reach: the nearest matching row is % away', nearest;
    END IF;
    SELECT count(*) INTO nearer_non_matching FROM reach
    WHERE price >= 5 AND embedding <-> ARRAY[250.0, 10.0]::real[] < nearest;
    IF nearer_non_matching < 1000 THEN
        RAISE EXCEPTION
            'fixture is not testing reach: only % non-matching rows lie between '
            'the query and the nearest match', nearer_non_matching;
    END IF;
END $$;

DO $$
DECLARE line text; plan text := '';
BEGIN
    SET LOCAL enable_seqscan = off;
    FOR line IN EXECUTE
        'EXPLAIN SELECT id FROM reach WHERE price < 5 '
        'ORDER BY embedding <-> ARRAY[250.0, 10.0]::real[] LIMIT 10'
    LOOP
        plan := plan || line || E'\n';
    END LOOP;
    IF plan NOT LIKE '%Index Scan%' OR plan NOT LIKE '%Index Cond%' THEN
        RAISE EXCEPTION
            'the predicate did not reach the index, so this proves nothing:%',
            E'\n' || plan;
    END IF;
END $$;

-- The query under test, and only that, under the index.
DO $$
BEGIN
    SET LOCAL enable_seqscan = off;

    CREATE TEMP TABLE reach_got AS
    SELECT id FROM reach WHERE price < 5
    ORDER BY embedding <-> ARRAY[250.0, 10.0]::real[] LIMIT 10;
END $$;

-- Everything that judges the answer runs off the index, so no assertion here
-- can be satisfied by the thing it is judging.
DO $$
DECLARE returned bigint; violations bigint; hits bigint;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;

    SELECT count(*) INTO returned FROM reach_got;
    IF returned <> 10 THEN
        RAISE EXCEPTION
            'asked for 10 matching rows and got % -- the search could not reach '
            'the matching region at the default ef_search', returned;
    END IF;

    SELECT count(*) INTO violations
    FROM reach_got JOIN reach USING (id) WHERE reach.price >= 5;
    IF violations <> 0 THEN
        RAISE EXCEPTION 'index returned % rows that fail the pushed predicate', violations;
    END IF;

    SELECT count(*) INTO hits FROM reach_got WHERE id IN (SELECT id FROM reach_truth);
    IF hits < 9 THEN
        RAISE EXCEPTION
            'recall@10 over a correlated filter is %, below the 0.85 the design '
            'promises -- the search reached *a* matching region, not the nearest',
            hits / 10.0;
    END IF;
END $$;

-- The control: the same graph, the same query, a filter that is *not* correlated
-- with position. This is the case every earlier recall test measures, it was
-- already good, and the reach work must leave it that way -- the allowances are
-- only ever drawn on where two hops turn up no match at all, which here is
-- almost nowhere.
--
-- Two things about this fixture are deliberate.
--
-- The label is a *hash* of the id, not an arithmetic function of it. The x
-- coordinate is `i % 500` and 100 divides 500, so any `(a*i + b) % 100` is a
-- function of `i % 500` and therefore constant down each column -- the same
-- correlated shape as `reach` wearing a disguise. The first draft used
-- `(i * 7919) % 100` and selected 25 whole columns, just with the nearest one 8
-- away instead of 46; a real regression in uncorrelated filtering would have
-- passed it.
--
-- The filter keeps the same 5% as the correlated half, so the two differ in
-- exactly one thing: whether the label correlates with position.
--
-- It was pinned at 10% for a while, because at 5% this lattice ran into a
-- *different* limit -- the matching subgraph fragments into components of a few
-- nodes, and a traversal that only bridges out of a node with no matching
-- neighbour explores one component and stops, returning 4 rows of 10 whatever
-- ef_search said. That is fixed now, and this control moves back to 5% because
-- a control at a gentler selectivity than the case it controls for is only
-- half a control.
CREATE TABLE spread (id int, price int, embedding real[]);
ALTER TABLE spread SET (autovacuum_enabled = off);
INSERT INTO spread
SELECT i, (hashint4(i) % 100 + 100) % 100, ARRAY[(i % 500)::real, (i / 500)::real]
FROM generate_series(1, 10000) i;
CREATE INDEX spread_idx ON spread USING brindle (embedding, price);

SET enable_indexscan = off;
SET enable_seqscan = on;
CREATE TABLE spread_truth AS
SELECT id FROM spread WHERE price < 5
ORDER BY embedding <-> ARRAY[250.0, 10.0]::real[] LIMIT 10;
RESET enable_indexscan;
RESET enable_seqscan;

-- ...and the control is only a control if the label really did spread. A
-- columnar one puts every match in a handful of x positions; this must reach
-- most of the 500.
DO $$
DECLARE columns_hit bigint; matches bigint;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;

    SELECT count(DISTINCT embedding[1]), count(*) INTO columns_hit, matches
    FROM spread WHERE price < 5;
    IF columns_hit < 300 THEN
        RAISE EXCEPTION
            'the control is not uncorrelated: % matching rows fall in only % of '
            'the 500 x positions', matches, columns_hit;
    END IF;
END $$;

DO $$
BEGIN
    SET LOCAL enable_seqscan = off;

    CREATE TEMP TABLE spread_got AS
    SELECT id FROM spread WHERE price < 5
    ORDER BY embedding <-> ARRAY[250.0, 10.0]::real[] LIMIT 10;
END $$;

DO $$
DECLARE returned bigint; hits bigint;
BEGIN
    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;

    SELECT count(*) INTO returned FROM spread_got;
    IF returned <> 10 THEN
        RAISE EXCEPTION 'uncorrelated filter regressed: % of 10 rows', returned;
    END IF;

    SELECT count(*) INTO hits FROM spread_got WHERE id IN (SELECT id FROM spread_truth);
    IF hits < 10 THEN
        RAISE EXCEPTION 'uncorrelated recall@10 regressed to %', hits / 10.0;
    END IF;
END $$;
