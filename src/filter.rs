//! Tier-1 filter predicates over inline node attributes.
//!
//! Pure Rust with no Postgres dependency: the index stores a small row of
//! [`AttrValue`]s beside each node, and traversal evaluates a [`Predicate`]
//! against that row to decide whether a node can be an answer. The Postgres
//! boundary is responsible for translating `INCLUDE` columns and `WHERE` scan
//! keys into this representation — e.g. dictionary-encoding a text category to
//! an [`AttrValue::Int`] — so the core stays type-agnostic and unit-testable.
//!
//! Scope is ACORN's Tier 1: equality and numeric-range atoms combined with
//! `AND`. `OR`/`NOT`, bitmap handoff, and arbitrary expression pushdown are
//! deliberately out of scope here.

use std::cmp::Ordering;
use std::ops::Bound;

/// A single filterable attribute value stored beside a node.
///
/// Deliberately Postgres-independent and cache-friendly: a tag plus at most 8
/// bytes of payload. String categories are expected to be dictionary-encoded to
/// [`AttrValue::Int`] by the caller, so matching never touches the heap.
///
/// `Float` equality and ordering follow **PostgreSQL's** total order, not IEEE
/// 754: `'NaN' = 'NaN'` is true and NaN sorts above every other value, so
/// `'NaN' > 1e308` is true. `Null` is the only value that orders with nothing,
/// so it satisfies no atom, which keeps [`Predicate::matches`] total.
///
/// An earlier version followed IEEE 754 here, on the reasoning that a NaN row
/// failing an atom "costs rows, never wrong ones". That does not survive
/// negation: measured on 400 rows plus two storing NaN, `score >= 1` returned
/// 398 rows by sequential scan and 396 through the index, and the `NOT EXISTS`
/// form of the same predicate returned 4 and **6** — the dropped rows came back
/// as invented ones. This order decides which rows an index returns, so it has
/// to be the order SQL uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttrValue {
    /// Signed integer: ids, counts, booleans (`0`/`1`), or dictionary-encoded
    /// labels. Supports equality and range.
    Int(i64),
    /// Floating-point scalar: prices, scores, epoch timestamps. Supports
    /// equality and range.
    Float(f64),
    /// SQL `NULL` / absent value. Never satisfies any atom.
    Null,
}

impl AttrValue {
    /// Order two values *within the same numeric type*, the way PostgreSQL's
    /// btree does. Returns `None` only when they aren't order-comparable at all
    /// — different variants, or a `Null` — which range matching treats as "does
    /// not match".
    ///
    /// Floats use [`pg_float_cmp`], which places `NaN` above every other value
    /// and equal to itself. That is PostgreSQL's rule, not IEEE 754's, and it is
    /// deliberate: this order decides which rows an index returns, and an index
    /// whose answer differs from a sequential scan's is wrong however defensible
    /// its arithmetic.
    #[inline]
    fn num_cmp(&self, other: &AttrValue) -> Option<Ordering> {
        match (self, other) {
            (AttrValue::Int(a), AttrValue::Int(b)) => Some(a.cmp(b)),
            (AttrValue::Float(a), AttrValue::Float(b)) => Some(pg_float_cmp(*a, *b)),
            _ => None,
        }
    }

    /// Whether this value can take part in an ordered comparison at all. Only
    /// `Null` (absent) cannot, so it satisfies no range atom — not even one with
    /// both sides unbounded. A `NaN` orders fine under PostgreSQL's rule; see
    /// [`AttrValue::num_cmp`].
    #[inline]
    fn is_orderable(&self) -> bool {
        !matches!(self, AttrValue::Null)
    }
}

/// Order two floats the way PostgreSQL's `float8_cmp` does: ordinary comparison
/// where it is defined, and `NaN` above everything and equal to itself where it
/// is not.
///
/// Not `f64::total_cmp`, which is the obvious-looking shortcut and is wrong
/// here: it ranks `-0.0` below `0.0`, while SQL holds them equal. Comparing
/// normally first gets that right and leaves only the NaN cases, which is
/// exactly the shape of PostgreSQL's own implementation.
#[inline]
fn pg_float_cmp(a: f64, b: f64) -> Ordering {
    match a.partial_cmp(&b) {
        Some(ord) => ord,
        // `partial_cmp` is `None` only when one side is NaN.
        None if a.is_nan() && b.is_nan() => Ordering::Equal,
        None if a.is_nan() => Ordering::Greater,
        None => Ordering::Less,
    }
}

/// Type-strict equality with PostgreSQL's semantics: distinct variants never
/// match, `Null` never matches, and `NaN` equals itself — which `==` on `f64`
/// does not, hence [`pg_float_cmp`] rather than `==`. Not `total_cmp` either;
/// that one's doc says why.
#[inline]
fn eq_matches(value: &AttrValue, target: &AttrValue) -> bool {
    match (value, target) {
        (AttrValue::Int(a), AttrValue::Int(b)) => a == b,
        (AttrValue::Float(a), AttrValue::Float(b)) => pg_float_cmp(*a, *b) == Ordering::Equal,
        _ => false,
    }
}

/// Match one side of a range. The caller guarantees `value.is_orderable()`, so
/// an unbounded side is trivially satisfied; a bounded side fails on a type
/// mismatch (`num_cmp` → `None`).
#[inline]
fn lower_matches(value: &AttrValue, lo: &Bound<AttrValue>) -> bool {
    match lo {
        Bound::Unbounded => true,
        Bound::Included(b) => matches!(value.num_cmp(b), Some(Ordering::Greater | Ordering::Equal)),
        Bound::Excluded(b) => matches!(value.num_cmp(b), Some(Ordering::Greater)),
    }
}

#[inline]
fn upper_matches(value: &AttrValue, hi: &Bound<AttrValue>) -> bool {
    match hi {
        Bound::Unbounded => true,
        Bound::Included(b) => matches!(value.num_cmp(b), Some(Ordering::Less | Ordering::Equal)),
        Bound::Excluded(b) => matches!(value.num_cmp(b), Some(Ordering::Less)),
    }
}

/// One Tier-1 predicate term. Columns are referenced by position in a node's
/// attribute row (the same order the caller stored them in).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Atom {
    /// `row[col] == value`. Type-strict: an `Int` never equals a `Float`, and
    /// `Null` never equals anything.
    Eq { col: usize, value: AttrValue },
    /// `lo ≤/< row[col] </≤ hi`, each side inclusive, exclusive, or unbounded.
    /// `price < 50` is `Range { col, lo: Unbounded, hi: Excluded(Int(50)) }`.
    /// Bounds should share the column's numeric type; a type mismatch (or a
    /// `Null` value) fails the atom.
    Range {
        col: usize,
        lo: Bound<AttrValue>,
        hi: Bound<AttrValue>,
    },
}

impl Atom {
    #[inline]
    fn col(&self) -> usize {
        match self {
            Atom::Eq { col, .. } | Atom::Range { col, .. } => *col,
        }
    }

    /// Evaluate against a node's attribute row. A column that is out of range
    /// (missing attribute) or `Null` fails the atom.
    #[inline]
    fn matches(&self, row: &[AttrValue]) -> bool {
        let value = match row.get(self.col()) {
            Some(v) => v,
            None => return false,
        };
        match self {
            Atom::Eq { value: target, .. } => eq_matches(value, target),
            // A non-orderable value (`Null`) satisfies no range, including one
            // with unbounded sides.
            Atom::Range { lo, hi, .. } => {
                value.is_orderable() && lower_matches(value, lo) && upper_matches(value, hi)
            }
        }
    }
}

/// A Tier-1 predicate: the trivial match-all, or a conjunction of atoms.
///
/// Only `AND` is modelled because that is all Tier 1 supports; the enum is the
/// extension point for later tiers (`OR`/`NOT`) and gives traversal a cheap
/// [`Predicate::All`] fast path.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    /// Matches every node — the no-filter case.
    All,
    /// Matches a node iff *every* atom matches. An empty `And` also matches all,
    /// but prefer [`Predicate::All`] for that.
    And(Vec<Atom>),
}

impl Predicate {
    /// Evaluate against a node's attribute row. Allocation-free: it borrows the
    /// row and the atoms and returns a `bool` without touching the heap.
    #[inline]
    pub fn matches(&self, row: &[AttrValue]) -> bool {
        match self {
            Predicate::All => true,
            Predicate::And(atoms) => atoms.iter().all(|atom| atom.matches(row)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eq(col: usize, value: AttrValue) -> Atom {
        Atom::Eq { col, value }
    }

    fn range(col: usize, lo: Bound<AttrValue>, hi: Bound<AttrValue>) -> Atom {
        Atom::Range { col, lo, hi }
    }

    #[test]
    fn eq_int_match_and_mismatch() {
        let row = [AttrValue::Int(42), AttrValue::Int(7)];
        assert!(eq(0, AttrValue::Int(42)).matches(&row));
        assert!(!eq(0, AttrValue::Int(43)).matches(&row));
        assert!(eq(1, AttrValue::Int(7)).matches(&row));
    }

    #[test]
    fn eq_is_type_strict() {
        let row = [AttrValue::Int(42)];
        // 42 the int is not 42.0 the float.
        assert!(!eq(0, AttrValue::Float(42.0)).matches(&row));
        let row = [AttrValue::Float(1.5)];
        assert!(eq(0, AttrValue::Float(1.5)).matches(&row));
        assert!(!eq(0, AttrValue::Int(1)).matches(&row));
    }

    #[test]
    fn null_never_matches_equality() {
        let row = [AttrValue::Null];
        assert!(!eq(0, AttrValue::Null).matches(&row));
        assert!(!eq(0, AttrValue::Int(0)).matches(&row));
    }

    /// NaN follows PostgreSQL's total order, not IEEE 754's: equal to itself,
    /// and above every other value. Each case below is what `psql` answers for
    /// the same comparison — the point of the rule is that those two agree, so
    /// an index scan and a sequential scan return the same rows.
    #[test]
    fn nan_follows_postgres_ordering() {
        let nan = [AttrValue::Float(f64::NAN)];

        // SELECT 'NaN'::float8 = 'NaN'  → true
        assert!(eq(0, AttrValue::Float(f64::NAN)).matches(&nan));
        // SELECT 'NaN'::float8 = 1      → false
        assert!(!eq(0, AttrValue::Float(1.0)).matches(&nan));

        // SELECT 'NaN'::float8 >= 1     → true  (NaN sorts above everything)
        assert!(range(0, Bound::Included(AttrValue::Float(1.0)), Bound::Unbounded).matches(&nan));
        // SELECT 'NaN'::float8 <= 1e308 → false
        assert!(!range(
            0,
            Bound::Unbounded,
            Bound::Included(AttrValue::Float(1e308))
        )
        .matches(&nan));
        // An unbounded range holds every non-NULL value, NaN included.
        assert!(range(0, Bound::Unbounded, Bound::Unbounded).matches(&nan));

        // ...and a NaN *bound* is answerable too, so the index layer no longer
        // has to refuse one: SELECT 1::float8 < 'NaN' → true.
        let one = [AttrValue::Float(1.0)];
        assert!(range(
            0,
            Bound::Unbounded,
            Bound::Excluded(AttrValue::Float(f64::NAN))
        )
        .matches(&one));
        // SELECT 1::float8 >= 'NaN' → false
        assert!(!range(
            0,
            Bound::Included(AttrValue::Float(f64::NAN)),
            Bound::Unbounded
        )
        .matches(&one));

        // -0.0 and 0.0 are equal in SQL, and `f64::total_cmp` — the obvious way
        // to write this rule — says otherwise. This case is why the comparison
        // is PostgreSQL's shape and not that one-liner.
        let neg_zero = [AttrValue::Float(-0.0)];
        assert!(eq(0, AttrValue::Float(0.0)).matches(&neg_zero));
        assert!(
            range(0, Bound::Included(AttrValue::Float(0.0)), Bound::Unbounded).matches(&neg_zero)
        );
    }

    #[test]
    fn range_inclusive_and_exclusive_bounds() {
        let row = [AttrValue::Int(50)];
        // price < 50  → excludes 50
        assert!(!range(0, Bound::Unbounded, Bound::Excluded(AttrValue::Int(50))).matches(&row));
        // price <= 50 → includes 50
        assert!(range(0, Bound::Unbounded, Bound::Included(AttrValue::Int(50))).matches(&row));
        // 10 <= price < 100
        assert!(range(
            0,
            Bound::Included(AttrValue::Int(10)),
            Bound::Excluded(AttrValue::Int(100))
        )
        .matches(&row));
        // 50 < price  → excludes 50
        assert!(!range(0, Bound::Excluded(AttrValue::Int(50)), Bound::Unbounded).matches(&row));
    }

    #[test]
    fn range_on_floats() {
        let row = [AttrValue::Float(19.99)];
        assert!(range(0, Bound::Unbounded, Bound::Excluded(AttrValue::Float(20.0))).matches(&row));
        assert!(!range(0, Bound::Included(AttrValue::Float(20.0)), Bound::Unbounded).matches(&row));
    }

    #[test]
    fn range_type_mismatch_and_null_fail() {
        let row = [AttrValue::Int(5)];
        // Float bounds against an Int value never match.
        assert!(!range(
            0,
            Bound::Included(AttrValue::Float(0.0)),
            Bound::Included(AttrValue::Float(10.0))
        )
        .matches(&row));
        let row = [AttrValue::Null];
        assert!(!range(0, Bound::Unbounded, Bound::Unbounded).matches(&row));
    }

    #[test]
    fn unbounded_range_is_present_and_numeric() {
        assert!(range(0, Bound::Unbounded, Bound::Unbounded).matches(&[AttrValue::Int(0)]));
        assert!(range(0, Bound::Unbounded, Bound::Unbounded).matches(&[AttrValue::Float(-1.0)]));
        assert!(!range(0, Bound::Unbounded, Bound::Unbounded).matches(&[AttrValue::Null]));
    }

    #[test]
    fn missing_column_fails() {
        let row = [AttrValue::Int(1)];
        assert!(!eq(5, AttrValue::Int(1)).matches(&row));
        assert!(!range(5, Bound::Unbounded, Bound::Unbounded).matches(&row));
        // An empty row misses every column.
        assert!(!eq(0, AttrValue::Int(1)).matches(&[]));
    }

    #[test]
    fn conjunction_requires_all_atoms() {
        // tenant_id = 42 AND price < 50
        let pred = Predicate::And(vec![
            eq(0, AttrValue::Int(42)),
            range(1, Bound::Unbounded, Bound::Excluded(AttrValue::Int(50))),
        ]);
        assert!(pred.matches(&[AttrValue::Int(42), AttrValue::Int(30)]));
        assert!(!pred.matches(&[AttrValue::Int(42), AttrValue::Int(50)])); // price fails
        assert!(!pred.matches(&[AttrValue::Int(1), AttrValue::Int(30)])); // tenant fails
        assert!(!pred.matches(&[AttrValue::Int(1), AttrValue::Int(99)])); // both fail
    }

    #[test]
    fn all_and_empty_and_match_everything() {
        assert!(Predicate::All.matches(&[]));
        assert!(Predicate::All.matches(&[AttrValue::Null, AttrValue::Int(3)]));
        // An empty conjunction is vacuously true, like `All`.
        assert!(Predicate::And(vec![]).matches(&[AttrValue::Null]));
    }
}
