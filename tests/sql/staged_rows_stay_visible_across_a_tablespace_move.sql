-- A transaction that has staged rows and then moves the index must still see
-- them.
--
-- `ALTER INDEX ... SET TABLESPACE` gives the relation a new relfilenumber while
-- keeping its oid. Anything that identifies the staged rows by relfilenode
-- therefore stops matching halfway through the transaction, and the rows go
-- invisible to their own writer -- while the write-back, which matches on the
-- oid, still lands them at commit. The result is a transaction that cannot read
-- what it just wrote and no error to say so.

CREATE TABLE mv (id int, embedding real[]);
ALTER TABLE mv SET (autovacuum_enabled = off);
INSERT INTO mv SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 300) i;
CREATE INDEX mv_idx ON mv USING brindle (embedding);

-- The harness makes an empty directory per case; the drop first because a
-- tablespace is cluster-wide and a case that failed before its own cleanup
-- leaves the catalog entry for the next run.
DROP TABLESPACE IF EXISTS brindle_visible_ts;
CREATE TABLESPACE brindle_visible_ts LOCATION :'tablespace_dir';

-- No scan between the insert and the move. That ordering is the whole test: a
-- scan writes staged rows back, so one placed before the move would leave
-- nothing staged to lose and the case would pass without exercising anything.
BEGIN;
SET LOCAL enable_seqscan = off;
INSERT INTO mv VALUES (9101, ARRAY[9101.0::real, 9102.0::real]);
ALTER INDEX mv_idx SET TABLESPACE brindle_visible_ts;

-- Stage a second row after the move as well: the first exercises finding an
-- already-staged graph under a changed relfilenode, the second exercises
-- staging into it.
INSERT INTO mv VALUES (9103, ARRAY[9103.0::real, 9104.0::real]);

SELECT id AS after_move FROM mv
ORDER BY embedding <-> ARRAY[9101.0, 9102.0]::real[] LIMIT 1 \gset
SELECT id AS second_staged FROM mv
ORDER BY embedding <-> ARRAY[9103.0, 9104.0]::real[] LIMIT 1 \gset
SELECT set_config('brindle_test.after_move', :'after_move', true) \gset carry_
SELECT set_config('brindle_test.second_staged', :'second_staged', true) \gset carry_

DO $$
DECLARE
    after_move    int := current_setting('brindle_test.after_move')::int;
    second_staged int := current_setting('brindle_test.second_staged')::int;
BEGIN
    IF after_move IS DISTINCT FROM 9101 THEN
        RAISE EXCEPTION
            'a row staged before a tablespace move went invisible to its own '
            'transaction after it: nearest to [9101,9102] is %', after_move;
    END IF;
    IF second_staged IS DISTINCT FROM 9103 THEN
        RAISE EXCEPTION
            'a row staged after a tablespace move is not visible to its own '
            'transaction: nearest to [9103,9104] is %', second_staged;
    END IF;
END $$;

COMMIT;

-- Both rows must be in the committed index, on the new tablespace.
DO $$
DECLARE first_id int; second_id int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO first_id FROM mv
    ORDER BY embedding <-> ARRAY[9101.0, 9102.0]::real[] LIMIT 1;
    SELECT id INTO second_id FROM mv
    ORDER BY embedding <-> ARRAY[9103.0, 9104.0]::real[] LIMIT 1;
    IF first_id IS DISTINCT FROM 9101 OR second_id IS DISTINCT FROM 9103 THEN
        RAISE EXCEPTION
            'rows staged around a tablespace move are missing from the committed '
            'index: got % and %, expected 9101 and 9103', first_id, second_id;
    END IF;
END $$;

ALTER INDEX mv_idx SET TABLESPACE pg_default;
DROP TABLE mv;
DROP TABLESPACE IF EXISTS brindle_visible_ts;
