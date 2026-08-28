#!/usr/bin/env bash
#
# Brindle index performance baseline: build time, query latency, recall.
#
# Runs against the worktree's own Postgres (the one scripts/pgx.sh manages), so
# it never touches a shared cluster and can run while other tasks are building.
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
#   SHAPE    set to "clustered" for 100 gaussian-ish clusters; unset means
#            uniform random, which in high dimensions is the worst case a graph
#            index can face and says more about the fixture than the index
#   PG       Postgres major to use              (default 17)
#
# The ef_search sweep lives in benches/sql/index_baseline.sql. Every point in it
# is above K on purpose: a scan returns at most ef_search rows, so a sweep point
# below K would measure that ceiling rather than the graph.
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

# Load the worktree's isolated PGRX_HOME / CARGO_TARGET_DIR, exactly as
# scripts/pgx.sh does, so the bench uses this task's Postgres and build dir.
if [[ -f "$repo_root/.cargo/worktree-env" ]]; then
  # shellcheck disable=SC1091
  . "$repo_root/.cargo/worktree-env"
fi

pg_config="$(cargo pgrx info pg-config "pg$PG" 2>/dev/null || true)"
if [[ -z "$pg_config" ]]; then
  echo "error: pg$PG is not initialized for this PGRX_HOME." >&2
  echo "       run: cargo pgrx init --pg$PG download" >&2
  exit 1
fi
bindir="$("$pg_config" --bindir)"
# pgrx derives the running port from base_port in PGRX_HOME/config.toml, and
# exposes no command to print it — read it the same way pgrx does.
port="$(awk -v major="$PG" '/^base_port/ { print $3 + major }' "$PGRX_HOME/config.toml")"
if [[ -z "$port" ]]; then
  echo "error: could not read base_port from $PGRX_HOME/config.toml" >&2
  exit 1
fi

echo "==> installing the extension into pg$PG"
cargo pgrx install --release --pg-config "$pg_config" >/dev/null

echo "==> starting pg$PG (port $port)"
cargo pgrx start "pg$PG" >/dev/null

# pgrx points the postmaster's unix socket at PGRX_HOME rather than /tmp, so
# every client needs --host; without it psql looks in the default socket dir and
# reports the server as not running.
host="$PGRX_HOME"

db="brindle_bench"
"$bindir/dropdb" --host "$host" --port "$port" --if-exists "$db" >/dev/null 2>&1 || true
"$bindir/createdb" --host "$host" --port "$port" "$db"

cat <<EOF

Brindle index baseline
  commit   $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')
  postgres $("$bindir/postgres" --version | awk '{print $3}')
  rows     $ROWS x $DIMS dims
  queries  $QUERIES per ef_search point, k = $K
EOF

"$bindir/psql" --host "$host" --port "$port" --dbname "$db" --quiet --no-psqlrc \
  --set=ON_ERROR_STOP=1 \
  --set=rows="$ROWS" --set=dims="$DIMS" \
  --set=queries="$QUERIES" --set=k="$K" \
  ${SHAPE:+--set=clustered=1} \
  --file benches/sql/index_baseline.sql

echo
echo "==> done. Postgres is still running; stop it with:"
echo "    scripts/pgx.sh stop pg$PG"
