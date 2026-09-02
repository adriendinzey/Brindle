# Brindle — Development setup

Brindle is a single `pgrx` crate, so **all** compilation, tests, and benchmarks
need the Linux Postgres toolchain (`cargo-pgrx` does not install on native
Windows). On Windows, that means **WSL2**. There is no native-Windows build path
to preserve — only editing and git.

## Where to put the working tree

Clone to the **Linux-native filesystem** (ext4), e.g. `~/code/brindle` — **not**
under `/mnt/c` or `/mnt/d`.

Why it matters: `cargo`/`rustc` touch thousands of small files on every build, and
WSL reaches Windows drives over the 9P protocol. Builds on `/mnt/*` are commonly
**5–10× slower** than on the native filesystem. Keep the large, constantly-churning
`target/` directory off the Windows side entirely. (`target/` is git-ignored and
should never be committed.)

## One-time setup (inside WSL2 / Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y build-essential libreadline-dev zlib1g-dev flex bison \
  libxml2-dev libxslt-dev libssl-dev libxml2-utils xsltproc ccache pkg-config clang

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# pgrx — the version must match the pgrx pin in Cargo.toml
cargo install --locked cargo-pgrx
cargo pgrx init        # downloads & compiles supported Postgres versions (slow, one-time)
```

## Clone and the daily loop

```bash
git clone https://github.com/adriendinzey/Brindle.git ~/code/brindle   # ext4, not /mnt/*
cd ~/code/brindle

cargo test                  # fast: pure-Rust core logic (no Postgres)
cargo pgrx test pg17        # integration tests against a managed Postgres
scripts/sql_test.sh         # SQL tests against a real, committed database
cargo pgrx run  pg17        # build + install + open psql with brindle loaded
```

```sql
CREATE EXTENSION brindle;
SELECT brindle_l2_distance(ARRAY[1,2,3]::real[], ARRAY[4,5,6]::real[]);  -- 5.196...
```

## Two kinds of SQL test

`cargo pgrx test` is the default and should stay that way: assertions are in
Rust, setup is cheap, and each case is wrapped in a transaction that rolls back,
so cases cannot see each other.

That wrapper is also a ceiling. Anything needing a **real commit** cannot be
tested through it — most visibly `VACUUM`, which Postgres refuses to run inside a
transaction block, but equally a second session observing the first, or anything
that has to survive a restart. Reaching for a workaround (calling the callback
`VACUUM` would have called, and checking the rest by hand) tests the callback but
not the path Postgres takes to it.

`scripts/sql_test.sh` exists for exactly those cases and nothing else:

```bash
scripts/sql_test.sh              # every case
scripts/sql_test.sh vacuum       # cases whose filename contains "vacuum"
PG=16 scripts/sql_test.sh        # against pg16
KEEP=1 scripts/sql_test.sh       # leave each case's database behind to inspect
```

It installs the extension, starts the worktree's own Postgres, and runs each
`tests/sql/*.sql` file **in its own fresh database** — so cases cannot see each
other's committed state, the suite passes in any order, and re-running it is
safe. It runs in CI on pg16 and pg17 beside `cargo pgrx test`.

### Writing a case

Drop a `.sql` file in `tests/sql/`. It runs under `ON_ERROR_STOP`, against a
database with the extension already created, so any error fails the case. Assert
in plpgsql and `RAISE EXCEPTION` when something is wrong:

```sql
DO $$
DECLARE stated bigint; actual bigint;
BEGIN
    SELECT relpages INTO stated FROM pg_class WHERE relname = 'my_idx';
    SELECT pg_relation_size('my_idx') / current_setting('block_size')::int INTO actual;
    IF stated <> actual THEN
        RAISE EXCEPTION 'pg_class says % pages, relation is %', stated, actual;
    END IF;
END $$;
```

Three things worth knowing:

- **The exception message is the failure report.** Say what was expected and what
  was found; a case that fails with "assertion failed" wastes the next person's
  afternoon. There are no expected-output files on purpose — they break on
  incidental formatting and tell you only that two files differ.
- **Assert observable behaviour, not log output.** The `VACUUM` case checks that
  `pg_class` stats were refreshed, because a vacuum refreshes them only from what
  `amvacuumcleanup` returns. (`ANALYZE` refreshes them by its own route, which is
  why that case disables autovacuum on its table.)
- **A test that cannot fail is worse than none, and it is easy to write one by
  accident.** The first version of the recycled-line-pointer case asserted that
  deleted rows stop being returned — which passes with no `VACUUM` at all, because
  Postgres rechecks heap visibility for every TID an index scan hands back. It
  looked like a strong test and tested nothing. Before trusting a case, break the
  thing it covers and watch it go red; if it stays green, the assertion is
  measuring something else.

## Editing from Windows

Edit with VS Code's **Remote - WSL** extension: open the `~/code/brindle` folder
*inside* WSL (the UI runs on Windows, the files and toolchain stay on Linux). This
keeps editing fast without the `/mnt` performance penalty. For occasional plain
editing you can also reach the tree at `\\wsl.localhost\<distro>\home\<user>\code\brindle`,
but do your builds inside WSL.

## Parallel development with git worktrees

Several branches can be developed **at the same time** — each as if it were a
separate engineer — without sharing a working directory. Two checkouts in the same
directory would stomp on each other's uncommitted files and can't sit on different
branches at once. The fix is a **git worktree per task**: the same repository and
history, but a separate directory + branch + build sandbox.

One command gives a task a fully isolated sandbox:

```bash
scripts/worktree.sh new <slug>   # branch task/<slug> + worktree + build dir + its own Postgres
scripts/worktree.sh ls           # list active sandboxes and their Postgres ports
scripts/worktree.sh rm  <slug>   # stop pg, remove the worktree, delete the branch (after merge)
```

`new` branches off the latest `origin/main`, creates the worktree on ext4, and sets
up **three** layers of isolation automatically. Then just work in that directory:

```bash
cd ~/code/brindle-wt/<slug>
cargo test                  # pure-core — isolated build dir, parallel-safe
scripts/pgx.sh test pg17    # pgrx/SQL tests — isolated Postgres, parallel-safe
```

Use `scripts/pgx.sh` in place of `cargo pgrx` inside a worktree: it loads that
worktree's isolated environment first (cargo's own `[env]` table does not reach the
`cargo pgrx` subcommand, so the wrapper does it). Plain `cargo` commands need no
wrapper — the generated `.cargo/config.toml` already redirects their build output.

### Isolation, layer by layer

| Resource | Isolation | How |
|---|---|---|
| Files + branch | ✅ automatic | the worktree (`scripts/worktree.sh new`) |
| Build artifacts | ✅ automatic | per-worktree `~/.cargo-target/<slug>` via generated `.cargo/config.toml` (a shared `target/` corrupts under concurrent `cargo`) |
| Postgres instance | ✅ automatic | per-worktree `PGRX_HOME=~/.pgrx-<slug>` on an auto-allocated free port; run pgrx via `scripts/pgx.sh` |

The isolated Postgres is **cheap**: it reuses the already-compiled install (no
recompile) and only creates a ~40 MB data directory per worktree, lazily on first
`pgx.sh` use. So `cargo pgrx test` of the same major can run in many worktrees at
once without the usual fixed-port collision. If you don't need it (pure-core task,
or you'll serialize pgrx tests), pass `--shared-pg` to skip it.

This is self-service: the person (or agent) doing a task runs `new` themselves as
their first step and works in the sandbox it prints — there's no separate setup
hand-off. `rm` tears the whole thing down (stops Postgres, removes worktree + branch).

Any git-ignored local docs in the main clone (e.g. `tasks/`, `design/`, `CLAUDE.md`)
are symlinked into each worktree, so those references resolve by their normal paths
and a shared board (like a task tracker) stays one source of truth across worktrees.

### Branch → PR → merge (conflicts are approved by a human)

Each task is developed on its own branch and merged to `main` via a pull request
that CI and review gate:

1. `scripts/worktree.sh new <slug>` (branches `task/<slug>` off latest `origin/main`).
2. Implement → self-review → commit. **Stage only the files your task touched**
   (`git add <paths>`); never `git add -A`. If you see changes you didn't make,
   leave them alone — they belong to another branch.
3. `git push -u origin task/<slug>` and open a PR. CI must be green.
4. **Merging is a human decision.** A branch never merges itself, and **merge
   conflicts are never auto-resolved** — if a PR conflicts with `main`, the
   maintainer reviews and resolves (or asks for a rebase) before merging.
5. After merge: `scripts/worktree.sh rm <slug>`.

To enforce step 4 on GitHub (require a PR + passing CI + up-to-date branch, and block
force-push/deletion of `main`) — **already applied on this repo**:

```bash
gh api -X PUT repos/adriendinzey/Brindle/branches/main/protection \
  -H "Accept: application/vnd.github+json" --input - <<'JSON'
{
  "required_status_checks": { "strict": true,
    "contexts": ["rustfmt", "clippy + test (pg16)", "clippy + test (pg17)"] },
  "enforce_admins": false,
  "required_pull_request_reviews": { "required_approving_review_count": 0 },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": true
}
JSON
```

> **Solo note:** `required_approving_review_count` is **0**, not 1 — GitHub won't let you
> approve your *own* PR, so requiring 1 makes every PR unmergeable for a single
> maintainer. CI-green + up-to-date is the real gate. `enforce_admins:false` keeps an
> owner escape hatch if a required check name ever drifts; set it to `true` to force
> *everyone* (you included) through the PR path. Required checks must match the CI job
> names exactly: `rustfmt`, `clippy + test (pg16)`, `clippy + test (pg17)`.
