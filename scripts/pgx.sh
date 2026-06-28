#!/usr/bin/env bash
#
# Run `cargo pgrx` with the current worktree's isolated environment.
#
# Worktrees created by scripts/worktree.sh drop a .cargo/worktree-env file with the
# right CARGO_TARGET_DIR (and PGRX_HOME, for an isolated Postgres). cargo's own
# [env] table does not propagate to the external `cargo pgrx` subcommand, and shell
# exports don't persist across separate commands, so this wrapper re-loads that env
# on every call. Outside a managed worktree it just runs `cargo pgrx` unchanged.
#
# Usage:  scripts/pgx.sh <pgrx-subcommand> [args]     e.g.  scripts/pgx.sh test pg17
set -euo pipefail

dir="$PWD"
while [[ "$dir" != "/" ]]; do
  if [[ -f "$dir/.cargo/worktree-env" ]]; then
    # shellcheck disable=SC1091
    . "$dir/.cargo/worktree-env"
    break
  fi
  dir="$(dirname "$dir")"
done

exec cargo pgrx "$@"
