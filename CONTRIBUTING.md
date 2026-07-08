# Contributing to Brindle

Thanks for your interest in Brindle! It's an early-stage project built in the
open — issues and pull requests are welcome.

## Building and testing

Brindle is a Rust PostgreSQL extension built with
[pgrx](https://github.com/pgcentralfoundation/pgrx). Build on **Linux, WSL2, or
macOS** — there is no native-Windows build path. On Windows, work inside WSL2 on
the Linux-native filesystem (e.g. `~/code/brindle`, not `/mnt/*`); the full
toolchain setup, the reasons behind it, and the parallel-worktree workflow are
in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

The daily loop:

```bash
cargo test              # pure-Rust core tests (no Postgres needed)
cargo pgrx test pg17    # SQL/integration tests against a managed Postgres
cargo pgrx run  pg17    # build + open psql with the extension loaded
cargo fmt --all         # formatting (CI enforces --check)
cargo clippy --no-default-features --features pg17 -- -D warnings
```

## Project layout

Algorithms (distance kernels, the HNSW graph, rank fusion) live in pure-Rust
modules with no Postgres imports, so they're unit-testable without a database;
the pgrx boundary (`src/lib.rs`) adapts them to SQL. Please keep new algorithmic
logic in the pure layer. Design rationale:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/FILTERING.md](docs/FILTERING.md). Code conventions:
[docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md) — highlights: `Result`-based
error handling (no `unwrap()` on non-test paths), no allocation in the distance
hot loops, comments explain *why*, not *what*.

## Contribution flow

1. **Open (or pick) an issue** describing the bug or feature — the issue
   templates ask for what a reviewer needs.
2. **Branch off `main`** and keep the change small and focused: one logical
   change per PR.
3. **Add tests** for new behavior and run the loop above locally.
4. **Open a pull request** and fill in the template. The *files touched* list
   matters — it's how overlap between in-flight PRs gets spotted early.
5. **CI must be green.** `main` is protected: changes land only via PR; the
   required checks (`rustfmt`, `clippy + test (pg16)`, `clippy + test (pg17)`)
   must pass; the branch must be up to date with `main` (rebase rather than
   merge — linear history is enforced); force-pushes and deletion of `main` are
   blocked. The maintainer does the merging.

## Commit messages

`type(scope): summary`, where `type` is one of
`feat | fix | test | docs | refactor | chore | bench | ci` — for example
`feat(hnsw): in-memory graph construction`. Add a body when the change needs
context.

## Changelog

User-visible changes get a line under **Unreleased** in
[CHANGELOG.md](CHANGELOG.md); release notes are drawn from it.

## License

Brindle is released under the [PostgreSQL License](LICENSE). By contributing,
you agree that your contributions are licensed under it.
