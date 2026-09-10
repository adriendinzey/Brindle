-- The README's quickstart, verbatim, plus the claim its prose makes about the
-- plan.
--
-- This exists because the README had drifted badly by the time anyone checked:
-- it described the project as "early development (Phase 0)" with four phases
-- shipped, and quoted a performance figure two orders of magnitude out of date.
-- A status line cannot be tested, but the snippet a reader will actually paste
-- can be, and so can the plan the surrounding prose promises.
--
-- **If you change the quickstart in README.md, change it here too, and the
-- reverse.** The two are a copy of each other on purpose: `psql` cannot read
-- Markdown, so the alternative is no check at all.
--
-- `CREATE EXTENSION brindle` is omitted only because the harness has already run
-- it; everything else is character-for-character what the README prints.

CREATE TABLE products (
    id        bigserial PRIMARY KEY,
    tenant_id int,
    price     float8,
    embedding real[]
);

INSERT INTO products (tenant_id, price, embedding)
SELECT i % 10, (i % 100)::float8, ARRAY[(i % 500)::real, (i / 500)::real]
FROM generate_series(1, 5000) i;

CREATE INDEX products_embedding_idx
    ON products USING brindle (embedding, tenant_id, price);

SELECT id, price
FROM products
WHERE tenant_id = 7 AND price < 50
ORDER BY embedding <-> ARRAY[250, 10]::real[]
LIMIT 10;

-- "EXPLAIN on that query should show `Index Scan using products_embedding_idx`
--  with an `Index Cond`" -- the README says so, so it has to be true. The
--  seqscan disable mirrors the parenthetical the README gives for small tables.
DO $$
DECLARE line text; plan text := '';
BEGIN
    SET LOCAL enable_seqscan = off;
    FOR line IN EXECUTE
        'EXPLAIN SELECT id, price FROM products '
        'WHERE tenant_id = 7 AND price < 50 '
        'ORDER BY embedding <-> ARRAY[250, 10]::real[] LIMIT 10'
    LOOP
        plan := plan || line || E'\n';
    END LOOP;
    IF plan NOT LIKE '%Index Scan using products_embedding_idx%' THEN
        RAISE EXCEPTION
            'the README quickstart does not produce the Index Scan it claims:%',
            E'\n' || plan;
    END IF;
    IF plan NOT LIKE '%Index Cond%' THEN
        RAISE EXCEPTION
            'the README claims the predicate reaches the index, but the plan '
            'applies it afterwards -- which is the post-filtering the README '
            'says this design avoids:%', E'\n' || plan;
    END IF;
END $$;

-- And the query actually answers: ten rows, none failing the predicate. A plan
-- check alone would pass on an index that reached the traversal and then
-- returned nothing useful.
DO $$
DECLARE returned bigint; violations bigint;
BEGIN
    SET LOCAL enable_seqscan = off;
    CREATE TEMP TABLE got AS
    SELECT id FROM products WHERE tenant_id = 7 AND price < 50
    ORDER BY embedding <-> ARRAY[250, 10]::real[] LIMIT 10;

    SET LOCAL enable_indexscan = off;
    SET LOCAL enable_seqscan = on;
    SELECT count(*) INTO returned FROM got;
    IF returned <> 10 THEN
        RAISE EXCEPTION 'the README quickstart returns % rows, not 10', returned;
    END IF;

    SELECT count(*) INTO violations
    FROM got JOIN products USING (id)
    WHERE tenant_id <> 7 OR price >= 50;
    IF violations <> 0 THEN
        RAISE EXCEPTION '% returned rows fail the quickstart predicate', violations;
    END IF;
END $$;
