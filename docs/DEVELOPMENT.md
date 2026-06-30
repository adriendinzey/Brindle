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
cargo pgrx run  pg17        # build + install + open psql with brindle loaded
```

```sql
CREATE EXTENSION brindle;
SELECT brindle_l2_distance(ARRAY[1,2,3]::real[], ARRAY[4,5,6]::real[]);  -- 5.196...
```

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
