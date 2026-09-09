//! Filterable attribute columns: the bridge between a SQL `WHERE` clause and
//! [`crate::filter::Predicate`].
//!
//! The vector is key column 1; every key column after it is a *filterable
//! attribute*, stored inline beside the vector so traversal can test a predicate
//! without touching the heap ([`docs/FILTERING.md`] Tier 1).
//!
//! They are key columns rather than `INCLUDE` columns for one reason: Postgres
//! matches a `WHERE` clause to an index column only if that column is part of
//! the search key. An `INCLUDE` column is payload — it can serve an index-only
//! scan, but a qual on it never reaches the access method, so the planner leaves
//! it as an executor `Filter` and the scan is post-filtered. That is the
//! behaviour this index exists to improve on, so the attributes have to be keys.

use crate::filter::{Atom, AttrValue, Predicate};
use pgrx::datum::FromDatum;
use pgrx::prelude::*;
use std::ops::Bound;

/// Strategy numbers for attribute operator classes, following the btree
/// convention so the declarations read the way a reader expects.
const STRATEGY_LT: u16 = 1;
const STRATEGY_LE: u16 = 2;
const STRATEGY_EQ: u16 = 3;
const STRATEGY_GE: u16 = 4;
const STRATEGY_GT: u16 = 5;

/// The highest strategy number any brindle operator class uses.
pub const MAX_STRATEGY: u16 = STRATEGY_GT;

/// How many key columns follow the vector.
///
/// # Safety
/// `index` must be an open brindle index relation.
pub unsafe fn count(index: pg_sys::Relation) -> usize {
    ((*(*index).rd_index).indnkeyatts as usize).saturating_sub(1)
}

/// The type an attribute column was declared to hold, by its position among the
/// attribute columns (0-based, so column 0 is index key column 2).
///
/// Read from `rd_opcintype` rather than the tuple descriptor: it is the type the
/// operator class was declared for, which is the same type the scan keys will
/// carry.
///
/// # Safety
/// `index` must be an open brindle index relation and `col` within [`count`].
unsafe fn column_type(index: pg_sys::Relation, col: usize) -> pg_sys::Oid {
    *(*index).rd_opcintype.add(col + 1)
}

/// Convert one attribute datum to the core's value type.
///
/// `None` means the type is not one this index can filter on, which is a build
/// error rather than something to paper over — a predicate that silently matched
/// nothing would be worse than refusing the column.
///
/// # Safety
/// `datum` must be a valid datum of type `typoid`, or `is_null` must be true.
pub unsafe fn value_from_datum(
    typoid: pg_sys::Oid,
    datum: pg_sys::Datum,
    is_null: bool,
) -> Option<AttrValue> {
    if is_null {
        // Stored, not skipped: a row whose attribute is NULL still occupies a
        // node, and `AttrValue::Null` is what makes every atom fail for it —
        // matching SQL, where a comparison against NULL is never true.
        return Some(AttrValue::Null);
    }
    let value = match typoid {
        pg_sys::BOOLOID => AttrValue::Int(bool::from_datum(datum, false)? as i64),
        pg_sys::INT2OID => AttrValue::Int(i16::from_datum(datum, false)? as i64),
        pg_sys::INT4OID => AttrValue::Int(i32::from_datum(datum, false)? as i64),
        pg_sys::INT8OID => AttrValue::Int(i64::from_datum(datum, false)?),
        pg_sys::FLOAT4OID => AttrValue::Float(f32::from_datum(datum, false)? as f64),
        pg_sys::FLOAT8OID => AttrValue::Float(f64::from_datum(datum, false)?),
        _ => return None,
    };
    Some(value)
}

/// Whether two supported types compare within one [`AttrValue`] variant.
///
/// The operator families keep integers with integers and floats with floats, so
/// this only ever refuses a pairing someone added later without teaching
/// [`value_from_datum`] about it.
fn same_family(a: pg_sys::Oid, b: pg_sys::Oid) -> bool {
    let integral = |t: pg_sys::Oid| {
        matches!(
            t,
            pg_sys::BOOLOID | pg_sys::INT2OID | pg_sys::INT4OID | pg_sys::INT8OID
        )
    };
    let floating = |t: pg_sys::Oid| matches!(t, pg_sys::FLOAT4OID | pg_sys::FLOAT8OID);
    (integral(a) && integral(b)) || (floating(a) && floating(b))
}

/// Read the attribute row for one heap tuple, in key-column order.
///
/// `values`/`isnull` are the arrays Postgres passes to a build callback or
/// `aminsert`, which hold one entry per index column with the vector first.
///
/// # Safety
/// `values` and `isnull` must each have at least `count(index) + 1` entries.
pub unsafe fn row_from_datums(
    index: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
) -> Vec<AttrValue> {
    let n = count(index);
    let mut row = Vec::with_capacity(n);
    for col in 0..n {
        let typoid = column_type(index, col);
        // Column 0 of the attribute row is index column 2, hence the offset.
        let datum = *values.add(col + 1);
        let is_null = *isnull.add(col + 1);
        match value_from_datum(typoid, datum, is_null) {
            Some(value) => row.push(value),
            None => error!(
                "brindle: column {} of the index has type {}, which cannot be filtered on",
                col + 2,
                typoid.as_u32()
            ),
        }
    }
    row
}

/// What a scan's keys became, and whether the executor still has to check them.
pub struct PushedPredicate {
    /// The atoms traversal can test for itself.
    pub predicate: Predicate,
    /// True when at least one key could not be represented, so the rows this
    /// scan returns are a *superset* of the ones the query wants.
    ///
    /// Dropping an atom only ever widens a conjunction, so the index still
    /// returns every matching row — never fewer. Postgres is told to recheck,
    /// which is what keeps a qual this AM cannot express from becoming a wrong
    /// answer rather than merely a slower one.
    pub recheck: bool,
}

/// Translate one scan key into an atom over the attribute row.
///
/// `None` means "this AM cannot express it", which the caller turns into a
/// recheck — never into a silently dropped filter.
///
/// # Safety
/// `key` must be an initialized scan key belonging to `index`.
unsafe fn atom_from_key(index: pg_sys::Relation, key: &pg_sys::ScanKeyData) -> Option<Atom> {
    let flags = key.sk_flags as u32;
    // A row-wise comparison, a ScalarArrayOp, or an ordering key is not a simple
    // scalar test; `amsearchnulls = false` should keep NULL keys away, but a
    // NULL argument makes every strict comparison unknown, so it is refused here
    // too rather than being read as a value.
    if flags & (pg_sys::SK_ISNULL | pg_sys::SK_ROW_HEADER | pg_sys::SK_SEARCHARRAY) != 0 {
        return None;
    }
    // Column 1 is the vector; a key on it is an ordering key, handled elsewhere.
    let col = (key.sk_attno as usize).checked_sub(2)?;
    if col >= count(index) {
        return None;
    }
    // Read the argument as *its own* type, not the column's. A cross-type
    // operator — `bigint_col = 7`, where the literal is `int4` — carries the
    // right-hand type in `sk_subtype`, and reading that datum as the column's
    // type would reinterpret four bytes as eight. `value_from_datum` refusing an
    // unknown type is what keeps this safe if the families ever gain a member
    // this code does not know.
    let arg_type = if key.sk_subtype == pg_sys::InvalidOid {
        column_type(index, col)
    } else {
        key.sk_subtype
    };
    let value = value_from_datum(arg_type, key.sk_argument, false)?;
    // Both sides must land in the same `AttrValue` variant, or the comparison is
    // not one the core can make: an `Int` never orders against a `Float`, so
    // such an atom would silently match nothing rather than fail. The operator
    // families pair integers with integers and floats with floats, so this
    // guards against a member added later rather than a case reachable today.
    if !same_family(column_type(index, col), arg_type) {
        return None;
    }

    let atom = match key.sk_strategy {
        STRATEGY_EQ => Atom::Eq { col, value },
        STRATEGY_LT => Atom::Range {
            col,
            lo: Bound::Unbounded,
            hi: Bound::Excluded(value),
        },
        STRATEGY_LE => Atom::Range {
            col,
            lo: Bound::Unbounded,
            hi: Bound::Included(value),
        },
        STRATEGY_GT => Atom::Range {
            col,
            lo: Bound::Excluded(value),
            hi: Bound::Unbounded,
        },
        STRATEGY_GE => Atom::Range {
            col,
            lo: Bound::Included(value),
            hi: Bound::Unbounded,
        },
        _ => return None,
    };
    Some(atom)
}

/// Assemble the predicate a scan's keys describe.
///
/// # Safety
/// `keys` must hold `nkeys` initialized scan keys belonging to `index`.
pub unsafe fn predicate_from_keys(
    index: pg_sys::Relation,
    keys: pg_sys::ScanKey,
    nkeys: usize,
) -> PushedPredicate {
    let mut atoms = Vec::new();
    let mut recheck = false;
    for i in 0..nkeys {
        // SAFETY: the caller guarantees `nkeys` initialized keys at `keys`.
        let key = &*keys.add(i);
        match atom_from_key(index, key) {
            Some(atom) => atoms.push(atom),
            None => recheck = true,
        }
    }
    let predicate = if atoms.is_empty() {
        Predicate::All
    } else {
        Predicate::And(atoms)
    };
    PushedPredicate { predicate, recheck }
}

// One operator class per filterable type. They declare comparisons only: the
// access method never calls a support function on an attribute column, so none
// is required. Each is DEFAULT for its type, so `USING brindle (embedding, col)`
// resolves without the user naming a class.
//
// `float4`/`float8` are here for completeness, but note that NaN satisfies no
// atom — matching SQL, where a comparison involving NaN is not true.
extension_sql!(
    r#"
CREATE OPERATOR FAMILY brindle_integer_ops USING brindle;
CREATE OPERATOR FAMILY brindle_float_ops USING brindle;

CREATE OPERATOR CLASS brindle_int2_ops DEFAULT FOR TYPE int2
    USING brindle FAMILY brindle_integer_ops AS OPERATOR 1 <, OPERATOR 2 <=, OPERATOR 3 =, OPERATOR 4 >=, OPERATOR 5 >;
CREATE OPERATOR CLASS brindle_int4_ops DEFAULT FOR TYPE int4
    USING brindle FAMILY brindle_integer_ops AS OPERATOR 1 <, OPERATOR 2 <=, OPERATOR 3 =, OPERATOR 4 >=, OPERATOR 5 >;
CREATE OPERATOR CLASS brindle_int8_ops DEFAULT FOR TYPE int8
    USING brindle FAMILY brindle_integer_ops AS OPERATOR 1 <, OPERATOR 2 <=, OPERATOR 3 =, OPERATOR 4 >=, OPERATOR 5 >;
CREATE OPERATOR CLASS brindle_float4_ops DEFAULT FOR TYPE float4
    USING brindle FAMILY brindle_float_ops AS OPERATOR 1 <, OPERATOR 2 <=, OPERATOR 3 =, OPERATOR 4 >=, OPERATOR 5 >;
CREATE OPERATOR CLASS brindle_float8_ops DEFAULT FOR TYPE float8
    USING brindle FAMILY brindle_float_ops AS OPERATOR 1 <, OPERATOR 2 <=, OPERATOR 3 =, OPERATOR 4 >=, OPERATOR 5 >;
CREATE OPERATOR CLASS brindle_bool_ops DEFAULT FOR TYPE bool USING brindle AS
    OPERATOR 1 <, OPERATOR 2 <=, OPERATOR 3 =, OPERATOR 4 >=, OPERATOR 5 >;

ALTER OPERATOR FAMILY brindle_integer_ops USING brindle ADD
    OPERATOR 1 < (int2, int4),
    OPERATOR 2 <= (int2, int4),
    OPERATOR 3 = (int2, int4),
    OPERATOR 4 >= (int2, int4),
    OPERATOR 5 > (int2, int4),
    OPERATOR 1 < (int2, int8),
    OPERATOR 2 <= (int2, int8),
    OPERATOR 3 = (int2, int8),
    OPERATOR 4 >= (int2, int8),
    OPERATOR 5 > (int2, int8),
    OPERATOR 1 < (int4, int2),
    OPERATOR 2 <= (int4, int2),
    OPERATOR 3 = (int4, int2),
    OPERATOR 4 >= (int4, int2),
    OPERATOR 5 > (int4, int2),
    OPERATOR 1 < (int4, int8),
    OPERATOR 2 <= (int4, int8),
    OPERATOR 3 = (int4, int8),
    OPERATOR 4 >= (int4, int8),
    OPERATOR 5 > (int4, int8),
    OPERATOR 1 < (int8, int2),
    OPERATOR 2 <= (int8, int2),
    OPERATOR 3 = (int8, int2),
    OPERATOR 4 >= (int8, int2),
    OPERATOR 5 > (int8, int2),
    OPERATOR 1 < (int8, int4),
    OPERATOR 2 <= (int8, int4),
    OPERATOR 3 = (int8, int4),
    OPERATOR 4 >= (int8, int4),
    OPERATOR 5 > (int8, int4);
ALTER OPERATOR FAMILY brindle_float_ops USING brindle ADD
    OPERATOR 1 < (float4, float8),
    OPERATOR 2 <= (float4, float8),
    OPERATOR 3 = (float4, float8),
    OPERATOR 4 >= (float4, float8),
    OPERATOR 5 > (float4, float8),
    OPERATOR 1 < (float8, float4),
    OPERATOR 2 <= (float8, float4),
    OPERATOR 3 = (float8, float4),
    OPERATOR 4 >= (float8, float4),
    OPERATOR 5 > (float8, float4);
"#,
    name = "brindle_attribute_opclasses",
    requires = [brindle_amhandler],
);
