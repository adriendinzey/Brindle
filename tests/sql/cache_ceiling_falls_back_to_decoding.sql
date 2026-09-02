-- An index over the ceiling must still work, by decoding per scan.
--
-- The cache is per backend and shared with nobody, so its cost is the ceiling
-- times the connections that touch the index. Exceeding it has to degrade to the
-- behaviour from before there was a cache — decode every scan — rather than to
-- an unbounded cache, which is how a cache becomes an outage.

CREATE TABLE ceil_t (id int, embedding real[]);
INSERT INTO ceil_t
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 500) i;
CREATE INDEX ceil_idx ON ceil_t USING brindle (embedding);

-- Zero disables caching outright: the documented way to turn it off.
SET brindle.cache_max_mb = 0;

DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM ceil_t
    ORDER BY embedding <-> ARRAY[42.0, 43.0]::real[] LIMIT 1;
    IF nearest <> 42 THEN
        RAISE EXCEPTION 'uncached scan answered %, expected 42', nearest;
    END IF;
END $$;

-- Writes still land, and are still seen, with no cache in play.
INSERT INTO ceil_t SELECT 9001, ARRAY[9001.0::real, 9002.0::real];

DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM ceil_t
    ORDER BY embedding <-> ARRAY[9001.0, 9002.0]::real[] LIMIT 1;
    IF nearest <> 9001 THEN
        RAISE EXCEPTION 'uncached scan missed a row just inserted, got %', nearest;
    END IF;
END $$;

-- Zero must stop this backend *using* a copy it already holds, not merely stop
-- it keeping new ones. An earlier version only checked that answers stayed
-- correct, which they did — from the very cache the setting was supposed to have
-- given up. Timing is what distinguishes the two, so the fixture is large enough
-- for a decode to dwarf a search by a wide margin.
CREATE TABLE ceil_big (id int, embedding real[]);
INSERT INTO ceil_big
SELECT i, ARRAY[(i % 977)::real, (i % 331)::real, i::real]
FROM generate_series(1, 20000) i;
CREATE INDEX ceil_big_idx ON ceil_big USING brindle (embedding);

DO $$
DECLARE
    started timestamptz; warm_ms float8; cold_ms float8; sink int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SET LOCAL brindle.cache_max_mb = 256;
    -- Prime the cache, then time a scan served from it.
    SELECT id INTO sink FROM ceil_big ORDER BY embedding <-> ARRAY[5.0,5.0,5.0]::real[] LIMIT 1;
    started := clock_timestamp();
    SELECT id INTO sink FROM ceil_big ORDER BY embedding <-> ARRAY[7.0,7.0,7.0]::real[] LIMIT 1;
    warm_ms := extract(epoch FROM clock_timestamp() - started) * 1000;

    SET LOCAL brindle.cache_max_mb = 0;
    started := clock_timestamp();
    SELECT id INTO sink FROM ceil_big ORDER BY embedding <-> ARRAY[9.0,9.0,9.0]::real[] LIMIT 1;
    cold_ms := extract(epoch FROM clock_timestamp() - started) * 1000;

    IF cold_ms < warm_ms * 5 THEN
        RAISE EXCEPTION
            'setting the ceiling to zero did not stop the cache being used: '
            'a scan took %ms with it disabled against %ms with it enabled, and '
            'decoding this index should cost far more than searching it',
            round(cold_ms::numeric, 2), round(warm_ms::numeric, 2);
    END IF;
END $$;

-- Turning it back on mid-session must not serve anything from before.
SET brindle.cache_max_mb = 256;

DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM ceil_t
    ORDER BY embedding <-> ARRAY[9001.0, 9002.0]::real[] LIMIT 1;
    IF nearest <> 9001 THEN
        RAISE EXCEPTION 'scan after re-enabling the cache answered %, expected 9001', nearest;
    END IF;
END $$;
