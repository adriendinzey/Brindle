#!/usr/bin/env bash
#
# Brindle index performance baseline: build time, query latency, recall.
#
# Runs against the worktree's own Postgres data directory (the one scripts/pgx.sh
# manages). Note the *install prefix* is still shared: `cargo pgrx install` writes
# brindle.so into ~/.pgrx/<ver>/pgrx-install, which every worktree's Postgres
# loads from — so do not run this while another task is installing or testing.
#
# Usage:
#   scripts/bench_index.sh                    # 100k x 128, 100 queries, k = 10
#   ROWS=10000 DIMS=64 scripts/bench_index.sh # a smaller, faster run
#
# Knobs (environment):
#   ROWS     rows in the indexed table          (default 100000)
#   DIMS     vector dimensions                  (default 128)
#   QUERIES  query vectors per ef_search point  (default 100)
#   K        neighbors requested per query      (default 10)
#   PGVECTOR set to 1 to also run the matched pgvector comparison (needs the
#            vector extension installed into the same Postgres)
#   INSERTS  set to 1 to also measure insert cost against index size, after the
#            query sweep — a different question, and a slow one, so it is opt-in
#   SELECTIVITY  set to 1 to also run the filtered-search sweep (brindle vs
#            pgvector post-filter vs pgvector iterative scan) across predicate
#            selectivity, for both an uncorrelated and a correlated label.
#            Requires PGVECTOR=1, since the comparison is the point.
#   SHAPE    set to "clustered" for 100 gaussian-ish clusters; unset means
#            uniform random, which in high dimensions is the worst case a graph
#            index can face and says more about the fixture than the index
#   PG       Postgres major to use              (default 17)
#
# The ef_search sweep lives in benches/sql/index_baseline.sql. Every point that
# reports recall is above K on purpose: a scan returns at most ef_search rows, so
# a sweep point below K would measure that ceiling rather than the graph. The one
# point below K (ef = 1) is a latency control and deliberately reports no recall.
#
# This is a local baseline, not a CI gate. Timing on a shared runner is noise,
# and nothing here should ever gate a merge.

set -euo pipefail

ROWS="${ROWS:-100000}"
DIMS="${DIMS:-128}"
QUERIES="${QUERIES:-100}"
K="${K:-10}"
PG="${PG:-17}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# shellcheck disable=SC1091
. "$repo_root/scripts/lib/pgenv.sh"

case "${SHAPE:-uniform}" in
  clustered) clustered_flag="--set=clustered=1" ;;
  uniform)   clustered_flag="" ;;
  *) echo "error: SHAPE must be 'uniform' or 'clustered', got '$SHAPE'" >&2; exit 1 ;;
esac

pgenv_load "$repo_root" "$PG"

echo "==> installing the extension into pg$PG"
cargo pgrx install --release --pg-config "$pg_config" >/dev/null

echo "==> starting pg$PG (port $port)"
cargo pgrx start "pg$PG" >/dev/null

db="brindle_bench"
"$bindir/dropdb" --host "$host" --port "$port" --if-exists "$db" >/dev/null 2>&1 || true
"$bindir/createdb" --host "$host" --port "$port" "$db"

cat <<EOF

Brindle index baseline
  commit   $(git rev-parse --short HEAD)$([[ -n "$(git status --porcelain)" ]] && echo ' (dirty tree)')
  postgres $("$bindir/postgres" --version | awk '{print $3}')
  rows     $ROWS x $DIMS dims
  queries  $QUERIES per ef_search point, k = $K
EOF

# clustered_flag is deliberately unquoted: it must expand to no argument at all
# when SHAPE is uniform, which "$clustered_flag" would not do.
# shellcheck disable=SC2086
"$bindir/psql" --host "$host" --port "$port" --dbname "$db" --quiet --no-psqlrc \
  --set=ON_ERROR_STOP=1 \
  --set=rows="$ROWS" --set=dims="$DIMS" \
  --set=queries="$QUERIES" --set=k="$K" \
  ${clustered_flag} \
  --file benches/sql/index_baseline.sql

if [[ "${INSERTS:-0}" == "1" ]]; then
  echo
  echo "==> insert cost against index size"
  "$bindir/psql" --host "$host" --port "$port" --dbname "$db" --quiet --no-psqlrc \
    --set=ON_ERROR_STOP=1 --set=dims="$DIMS" \
    --file benches/sql/insert_cost.sql
fi

if [[ "${PGVECTOR:-0}" == "1" ]]; then
  echo
  echo "==> pgvector comparison (same rows, same queries, same ground truth)"
  "$bindir/psql" --host "$host" --port "$port" --dbname "$db" --quiet --no-psqlrc \
    --set=ON_ERROR_STOP=1 --set=k="$K" \
    --file benches/sql/pgvector_compare.sql
fi

if [[ "${SELECTIVITY:-0}" == "1" ]]; then
  if [[ "${PGVECTOR:-0}" != "1" ]]; then
    echo "error: SELECTIVITY=1 needs PGVECTOR=1 — the comparison is the benchmark" >&2
    exit 1
  fi
  echo
  echo "==> filtered search across selectivity (brindle vs pgvector post-filter vs iterative)"
  chart_csv="$(mktemp -t brindle-selectivity-XXXXXX.csv)"
  "$bindir/psql" --host "$host" --port "$port" --dbname "$db" --quiet --no-psqlrc \
    --set=ON_ERROR_STOP=1 --set=k="$K" --set=chart_csv="$chart_csv" \
    --file benches/sql/selectivity.sql

  # The chart is generated from the numbers just measured, with the Python
  # standard library only: this repo has no plotting dependency and adding one
  # for a single figure would make "one command regenerates everything" false.
  if python3 benches/chart.py "$chart_csv" docs/assets/selectivity.svg \
       "$ROWS x $DIMS, ${SHAPE:-uniform}, k=$K"; then
    echo "==> chart written to docs/assets/selectivity.svg"
  fi
  echo "    raw numbers: $chart_csv"

  # The committed write-up can be checked against this run, but only if the run
  # was captured -- this script prints to stdout rather than to a file, so the
  # check belongs to whoever redirected it. Stale figures have survived several
  # review rounds of this benchmark by looking plausible beside the tables they
  # contradict, so it is worth doing.
  echo "    to check the write-up against this run:"
  echo "      <this command> 2>&1 | tee run.log && python3 benches/verify_writeup.py run.log"
fi

echo
echo "==> done. Postgres is still running; stop it with:"
echo "    scripts/pgx.sh stop pg$PG"
