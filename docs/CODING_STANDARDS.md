# Brindle — Coding Standards

Conventions for all code in this repository. They're enforced in review.

## Comments & documentation

- **Write self-documenting code first.** Clear names, small focused functions, and
  obvious control flow beat comments. Reach for a better name or a small refactor
  before reaching for a comment.
- **Comment only when necessary, and explain _why_, not _what_.** Justified
  comments: a non-obvious invariant, a subtle algorithm step, a deliberate
  performance trade-off, a safety justification, or a link to a paper/issue.
  Don't narrate code that already states what it does.
- **Public API gets doc comments (`///`).** Document what a caller needs:
  behavior, errors, units, and edge cases (e.g. "distances are squared for L2").
  This is the documentation users actually read.
- **Every `unsafe` block gets a comment stating the invariant it upholds.**
- Prefer one clear doc comment over scattered inline noise.

## No internal tracking in production code

The repo's planning/orchestration lives outside the shipped code and must never
leak into it:

- **No task IDs in source or comments** (no `T-011`, `T-0xx`, etc.).
- **No references to internal-only files** (the task tracker, AI working
  instructions) anywhere in `src/`, doc comments, or user-facing docs.
- Use plain **`TODO:`** / **`FIXME:`** describing the actual work — not a ticket
  number. Write `// TODO: read ef_search from a GUC instead of this constant`,
  never `// TODO(T-024)`.
- Commit messages describe the change; they don't reference the internal tracker.

## Error handling

- Return `Result` for anything fallible in the core. **No
  `unwrap()`/`expect()`/`panic!`/`unreachable!` on non-test paths.**
- Convert errors to Postgres `ERROR`s only at the pgrx boundary (`error!`), with
  clear messages.
- `#[cfg(test)]` code may use `unwrap`/`expect` freely.

## Performance

- The per-vector distance hot loop must not allocate.
- Measure before optimizing (criterion benches); don't add complexity on a hunch.

## Rust hygiene

- `cargo fmt` clean; `cargo clippy` with no warnings for the crate.
- Keep pure-algorithm modules free of `pgrx`/Postgres imports so they stay
  unit-testable without a database; confine Postgres glue to the boundary layer
  (`lib.rs`, the index AM).

## Testing

- Unit-test pure logic beside the module (`#[cfg(test)]`); SQL/integration tests
  run via `cargo pgrx test`.
- Quality/recall claims must be backed by an assertion or a reproducible number,
  not prose.
