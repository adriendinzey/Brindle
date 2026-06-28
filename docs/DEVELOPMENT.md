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

A helper script automates the setup:

```bash
scripts/worktree.sh new <slug>   # branch task/<slug> + worktree at ~/code/brindle-wt/<slug>
scripts/worktree.sh ls           # list active task worktrees
scripts/worktree.sh rm  <slug>   # remove the worktree and delete its branch (after merge)
```

`new` branches off the latest `origin/main`, creates the worktree on ext4, and
writes a per-worktree `.cargo/config.toml` so every `cargo` command there builds
into its own target directory (`~/.cargo-target/<slug>`). Open your editor/session
*in that directory* and work only on that branch.

### Isolation, layer by layer

| Resource | Isolation | How |
|---|---|---|
| Files + branch | ✅ automatic | the worktree (`scripts/worktree.sh new`) |
| Build artifacts | ✅ automatic | per-worktree `~/.cargo-target/<slug>` (a shared `target/` corrupts under concurrent `cargo`) |
| Postgres test instance | ⚠️ shared by default | see below |

`cargo pgrx test` boots a managed Postgres from `PGRX_HOME` (`~/.pgrx`) on a fixed
per-version port, so **two concurrent `cargo pgrx test` of the same major collide.**
Options, cheapest first:

- **(a) Serialize pgrx tests** — run only one `cargo pgrx test` at a time. Pure-core
  `cargo test` work parallelizes freely and is most of the early parallel work, so
  this is usually enough.
- **(b) Per-worktree `PGRX_HOME` + a unique port** — for heavy parallelism, pass
  `--pg-port <N>` to `scripts/worktree.sh new`; it wires a separate
  `PGRX_HOME=~/.pgrx-<slug>` into the worktree's cargo env. Initialize it once with
  `PGRX_HOME=~/.pgrx-<slug> cargo pgrx init` (the script prints the exact command).
- **(c) A Docker dev-container per worktree** (its own Postgres) — full isolation,
  heaviest.

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

To enforce step 4 on GitHub (require a PR + passing CI, block direct pushes to
`main`):

```bash
gh api -X PUT repos/adriendinzey/Brindle/branches/main/protection \
  -H "Accept: application/vnd.github+json" \
  -f 'required_status_checks[strict]=true' \
  -F 'enforce_admins=false' \
  -f 'required_pull_request_reviews[required_approving_review_count]=1' \
  -f 'restrictions=null'
```
