-- A vector too wide for the index must be refused when it goes in.
--
-- The decoder caps the dimension it will accept, because it sizes a buffer from
-- that field before anything relates it to how many bytes the payload holds.
-- If the build accepted a wider vector, CREATE INDEX would succeed and every
-- later read, insert and vacuum would fail on the way back — an index that
-- exists and cannot be used. The refusal has to happen at the boundary going in.
--
-- The real[] opclass is the path that mattered: brindle_vector validates its own
-- input, real[] did not.

CREATE TABLE wide (id int, embedding real[]);

DO $$
DECLARE ok bool := false;
BEGIN
    INSERT INTO wide
    SELECT 1, (SELECT array_agg(1.0::real) FROM generate_series(1, 16001));
    BEGIN
        CREATE INDEX wide_idx ON wide USING brindle (embedding);
    EXCEPTION WHEN OTHERS THEN
        ok := true;
        IF SQLERRM NOT LIKE '%16001%' OR SQLERRM NOT LIKE '%16000%' THEN
            RAISE EXCEPTION
                'refused, but the message names neither the width nor the limit: %',
                SQLERRM;
        END IF;
    END;
    IF NOT ok THEN
        RAISE EXCEPTION
            'CREATE INDEX accepted a 16001-dimension vector; the index it built '
            'could not have been read back';
    END IF;
END $$;

-- The limit itself must still work.
DELETE FROM wide;
INSERT INTO wide
SELECT 1, (SELECT array_agg(1.0::real) FROM generate_series(1, 16000));
CREATE INDEX wide_idx ON wide USING brindle (embedding);

DO $$
DECLARE found int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO found FROM wide
    ORDER BY embedding <-> (SELECT array_agg(1.0::real)::real[] FROM generate_series(1, 16000))
    LIMIT 1;
    IF found <> 1 THEN
        RAISE EXCEPTION 'an index at exactly the limit did not answer, got %', found;
    END IF;
END $$;
