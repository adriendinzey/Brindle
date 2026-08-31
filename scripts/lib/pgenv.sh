# Shared discovery for the worktree's pgrx-managed Postgres.
#
# Sourced, not executed. Sets: PGRX_HOME, pg_config, bindir, port, host.
# Callers set PG first (the major version) and must have `set -u` tolerated —
# every variable here is assigned before use.
#
# This exists because two drivers need the same three non-obvious facts, and a
# second copy of them drifts: the port is derived from a config key pgrx does
# not expose, a stock config.toml omits that key entirely, and pgrx puts the
# unix socket in PGRX_HOME rather than /tmp.

pgenv_load() {
  local repo_root="$1"
  local major="$2"

  # The worktree's isolated PGRX_HOME / CARGO_TARGET_DIR, exactly as
  # scripts/pgx.sh loads them, so a task uses its own Postgres and build dir.
  if [[ -f "$repo_root/.cargo/worktree-env" ]]; then
    # shellcheck disable=SC1091
    . "$repo_root/.cargo/worktree-env"
  fi
  # Outside a task worktree PGRX_HOME is unset, and `set -u` would kill the
  # caller with "unbound variable" instead of the clearer errors below.
  : "${PGRX_HOME:=$HOME/.pgrx}"
  export PGRX_HOME

  pg_config="$(cargo pgrx info pg-config "pg$major" 2>/dev/null || true)"
  if [[ -z "$pg_config" ]]; then
    echo "error: pg$major is not initialized for this PGRX_HOME ($PGRX_HOME)." >&2
    echo "       run: cargo pgrx init --pg$major download" >&2
    return 1
  fi
  bindir="$("$pg_config" --bindir)"

  # pgrx derives the running port from base_port in PGRX_HOME/config.toml and
  # exposes no command to print it, so read it the way pgrx does. A stock
  # config.toml has no base_port key at all (only [configs]); pgrx falls back to
  # 28800 there, and so must this, or the caller dies outside a task worktree —
  # exactly the case the PGRX_HOME default above exists to support. `|| true`
  # keeps a missing file from tripping `set -e` before the fallback applies.
  local base_port
  base_port="$(awk '/^base_port/ { print $3 }' "$PGRX_HOME/config.toml" 2>/dev/null || true)"
  port=$(( ${base_port:-28800} + major ))

  # pgrx points the postmaster's unix socket at PGRX_HOME rather than /tmp, so
  # every client needs --host; without it psql looks in the default socket dir
  # and reports the server as not running.
  host="$PGRX_HOME"
}
