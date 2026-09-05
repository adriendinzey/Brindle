-- A two-phase committed insert must reach the index.
--
-- Rows are staged in the backend that inserts them and written back when the
-- transaction ends. `PREPARE TRANSACTION` is an ending the deferral has to
-- notice: it fires its own event, not the ordinary pre-commit one, and the
-- session that prepared is normally gone before anyone runs COMMIT PREPARED. A
-- backend that dropped its staged rows at prepare would therefore lose them for
-- good — the heap keeps the row, the index never hears about it, and no error is
-- reported to anybody.
--
-- Needs three sessions and a real commit, so it cannot be a `#[pg_test]`. The
-- prepared transaction is deliberately committed from a *different* connection,
-- because committing from the same one hides the bug: the staged rows are still
-- sitting in that backend and get written by the next transaction's commit.

CREATE EXTENSION IF NOT EXISTS dblink;

DO $$
BEGIN
    IF current_setting('max_prepared_transactions')::int = 0 THEN
        RAISE EXCEPTION
            'this case needs max_prepared_transactions > 0; the harness sets it '
            'when it creates the cluster';
    END IF;
END $$;

CREATE TABLE tpc (id int, embedding real[]);
ALTER TABLE tpc SET (autovacuum_enabled = off);
INSERT INTO tpc
SELECT i, ARRAY[i::real, (i + 1)::real] FROM generate_series(1, 200) i;
CREATE INDEX tpc_idx ON tpc USING brindle (embedding);

-- Prepare on one connection and leave it. dblink closes the connection when the
-- call returns, which is the point: the staged rows have to survive the backend
-- that made them.
SELECT dblink_exec(
    'dbname=' || current_database() ||
    ' port=' || current_setting('port') ||
    ' host=' || (string_to_array(current_setting('unix_socket_directories'), ',')) [1],
    $inner$BEGIN;
           INSERT INTO tpc SELECT 7777, ARRAY[7777.0::real, 7778.0::real];
           PREPARE TRANSACTION 'brindle_tpc'$inner$);

-- Commit it from somewhere else entirely.
SELECT dblink_exec(
    'dbname=' || current_database() ||
    ' port=' || current_setting('port') ||
    ' host=' || (string_to_array(current_setting('unix_socket_directories'), ',')) [1],
    $inner$COMMIT PREPARED 'brindle_tpc'$inner$);

DO $$
DECLARE nearest int;
BEGIN
    SET LOCAL enable_seqscan = off;
    SELECT id INTO nearest FROM tpc
    ORDER BY embedding <-> ARRAY[7777.0, 7778.0]::real[] LIMIT 1;
    IF nearest IS DISTINCT FROM 7777 THEN
        RAISE EXCEPTION
            'a two-phase committed row never reached the index: nearest to '
            '[7777,7778] is %, and the row is in the heap with nothing pointing '
            'at it', nearest;
    END IF;
END $$;
