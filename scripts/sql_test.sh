#!/usr/bin/env bash
#
# SQL tests against a real, committed database.
#
# `cargo pgrx test` wraps every case in a transaction it rolls back, which makes
# a class of behavior structurally untestable: VACUUM (Postgres refuses to run it
# inside a transaction block), anything needing a real commit, anything needing a
# second session to observe the first, and anything involving a restart. This
# runs cases outside any such transaction so those things can be tested at all.
#
# It does not replace `cargo pgrx test`. Everything that fits inside a rolled-back
# transaction belongs there, where assertions are in Rust and setup is cheaper.
#
# Usage:
#   scripts/sql_test.sh                 # every case
#   scripts/sql_test.sh vacuum          # cases whose filename contains "vacuum"
#
# Knobs (environment):
#   PG    Postgres major to run against  (default 17)
#   KEEP  set to 1 to leave each case's database behind for inspection
#
# Writing a case: drop a .sql file in tests/sql/. It runs against its own fresh
# database with the extension already created, under ON_ERROR_STOP, so any error
# fails the case. Assert with plpgsql and RAISE EXCEPTION — the message is what
# the failure report shows, so make it say what was expected and what was found.
# Expected-output files are deliberately not used: they fail on incidental
# formatting and say "files differ" when they do.

set -euo pipefail

PG="${PG:-17}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# shellcheck disable=SC1091
. "$repo_root/scripts/lib/pgenv.sh"
pgenv_load "$repo_root" "$PG"

filter="${1:-}"
mapfile -t cases < <(find tests/sql -name '*.sql' | sort | { [[ -n "$filter" ]] && grep -- "$filter" || cat; })
if [[ ${#cases[@]} -eq 0 ]]; then
  echo "error: no cases matched${filter:+ filter '$filter'}" >&2
  exit 1
fi

echo "==> installing the extension into pg$PG"
cargo pgrx install --release --pg-config "$pg_config" >/dev/null

echo "==> starting pg$PG (port $port)"
cargo pgrx start "pg$PG" >/dev/null

echo "==> $(basename "$pg_config" >/dev/null; "$bindir/postgres" --version | awk '{print $3}'), ${#cases[@]} case(s)"
echo

failed=0
for case_file in "${cases[@]}"; do
  name="$(basename "$case_file" .sql)"
  # A database per case, so cases cannot see each other's committed state and
  # the suite passes in any order and when re-run. Nothing here rolls back.
  db="brindle_t_${name}"
  "$bindir/dropdb" --host "$host" --port "$port" --if-exists "$db" >/dev/null 2>&1 || true
  "$bindir/createdb" --host "$host" --port "$port" "$db"
  "$bindir/psql" --host "$host" --port "$port" --dbname "$db" --quiet --no-psqlrc \
    --set=ON_ERROR_STOP=1 -c 'CREATE EXTENSION brindle' >/dev/null

  output_file="$(mktemp)"
  if "$bindir/psql" --host "$host" --port "$port" --dbname "$db" --quiet --no-psqlrc \
       --set=ON_ERROR_STOP=1 --file "$case_file" >"$output_file" 2>&1; then
    printf 'ok    %s\n' "$name"
  else
    printf 'FAIL  %s\n' "$name"
    # The whole output, indented: a harness that prints "it failed" and nothing
    # else costs more time than it saves.
    sed 's/^/        /' "$output_file"
    failed=$((failed + 1))
  fi
  rm -f "$output_file"
  if [[ "${KEEP:-0}" != "1" ]]; then
    "$bindir/dropdb" --host "$host" --port "$port" --if-exists "$db" >/dev/null 2>&1 || true
  else
    echo "        (kept database $db)"
  fi
done

echo
if [[ $failed -gt 0 ]]; then
  echo "==> $failed of ${#cases[@]} case(s) failed"
  exit 1
fi
echo "==> all ${#cases[@]} case(s) passed"
