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
# `while read` rather than mapfile: stock macOS ships bash 3.2, which has neither
# mapfile nor readarray, and ONBOARDING lists macOS as supported.
cases=()
while IFS= read -r case_file; do
  # Match the case name, not the directory path — otherwise a filter like "sql"
  # matches every file by way of "tests/sql/".
  name="$(basename "$case_file" .sql)"
  if [[ -z "$filter" || "$name" == *"$filter"* ]]; then
    cases+=("$case_file")
  fi
done < <(find tests/sql -name '*.sql' | sort)
if [[ ${#cases[@]} -eq 0 ]]; then
  echo "error: no cases matched${filter:+ filter '$filter'}" >&2
  exit 1
fi

# The pgrx *install prefix* is shared across worktrees — every worktree's
# Postgres loads brindle.so from the same ~/.pgrx/<ver>/pgrx-install — so this
# clobbers a sibling task's build while it runs. Same caveat as the benchmark
# driver; do not run them concurrently.
#
# Not --release: these cases assert behaviour, not timing, and an optimized build
# here means CI pays for a second full compile after the debug one that
# `cargo pgrx test` already did.
echo "==> installing the extension into pg$PG"
cargo pgrx install --pg-config "$pg_config" >/dev/null

# Two-phase commit is off by default, and one case needs it. Setting it costs
# nothing when unused and cannot be changed without a restart, so it goes in
# before the cluster starts rather than being skipped around.
# `cargo pgrx start` creates the data directory on first use, so the setting has
# to go in after that and before the case that needs it — writing it only when
# the file already exists leaves a fresh machine failing on its own assertion.
cargo pgrx start "pg$PG" >/dev/null
datadir="$PGRX_HOME/data-$PG"
if ! grep -q '^max_prepared_transactions' "$datadir/postgresql.conf"; then
  echo "max_prepared_transactions = 10" >>"$datadir/postgresql.conf"
  cargo pgrx stop "pg$PG" >/dev/null 2>&1 || true
fi

echo "==> starting pg$PG (port $port)"
cargo pgrx start "pg$PG" >/dev/null

# Some cases move an index between tablespaces, and a tablespace needs an empty
# directory Postgres owns — which SQL cannot create for itself, and which may not
# live inside the data directory. Cases read the path from `:tablespace_dir`.
#
# One directory per case, not one for the suite: two tablespaces cannot share a
# location, so a shared directory works only until a second case wants one and
# then fails on whichever runs later.
tablespace_root="$PGRX_HOME/sql-test-tablespace-$PG"
rm -rf "$tablespace_root"
mkdir -p "$tablespace_root"

echo "==> $("$bindir/postgres" --version | awk '{print $3}'), ${#cases[@]} case(s)"
echo

failed=0
for case_file in "${cases[@]}"; do
  name="$(basename "$case_file" .sql)"
  # A database per case, so cases cannot see each other's committed state and
  # the suite passes in any order and when re-run. Nothing here rolls back.
  # Postgres truncates identifiers at NAMEDATALEN (63) without complaint, which
  # would silently run two long case names against one database.
  db="brindle_t_${name}"
  if [[ ${#db} -gt 63 ]]; then
    echo "error: case name '$name' makes a database name over 63 characters" >&2
    exit 1
  fi
  "$bindir/dropdb" --host "$host" --port "$port" --if-exists "$db" >/dev/null 2>&1 || true
  "$bindir/createdb" --host "$host" --port "$port" "$db"
  tablespace_dir="$tablespace_root/$name"
  mkdir -p "$tablespace_dir"
  "$bindir/psql" --host "$host" --port "$port" --dbname "$db" --quiet --no-psqlrc \
    --set=ON_ERROR_STOP=1 -c 'CREATE EXTENSION brindle' >/dev/null

  output_file="$(mktemp)"
  if "$bindir/psql" --host "$host" --port "$port" --dbname "$db" --quiet --no-psqlrc \
       --set=ON_ERROR_STOP=1 --set=tablespace_dir="$tablespace_dir" \
       --file "$case_file" >"$output_file" 2>&1; then
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
echo "    Postgres is still running; stop it with: scripts/pgx.sh stop pg$PG"
