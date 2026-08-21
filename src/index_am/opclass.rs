//! Operator classes: how `CREATE INDEX` tells the access method which distance
//! metric an index ranks by, and which type its column holds.
//!
//! Every brindle operator class declares three support functions:
//!
//! - **1** — the ranking distance function: exactly the value the graph orders
//!   by, so an operator class and the index built from it cannot disagree. For
//!   L2 that is the *squared* distance, which the graph uses because the square
//!   root doesn't change an ordering; the `<->` operator still hands callers a
//!   true Euclidean distance.
//! - **2** — the metric, as [`Metric::code`]. A build reads it instead of
//!   hardcoding a metric.
//! - **3** — the indexed type, as [`VectorKind::code`], which is how the access
//!   method knows to read a datum as an array or as a vector.
//!
//! Both facts come from support functions rather than from the catalogs
//! because an operator class is the only thing that knows them before the first
//! vector is read: a build has no scan key to take a strategy number from, and
//! index maintenance runs with a restricted `search_path`, so resolving the
//! type by name from inside a build finds nothing. It also leaves room for an
//! operator class this crate doesn't ship.
//!
//! Each metric gets its own ordering-operator strategy number — 1 = `<->` (L2),
//! 2 = `<#>` (inner product), 3 = `<=>` (cosine) — so one operator family can
//! hold all three unambiguously. The operator symbols themselves are pgvector's,
//! so a query keeps its shape across the two.

use pgrx::prelude::*;
use pgrx::{pg_sys, FromDatum};

use crate::vector::Metric;

/// Support-function number of the proc reporting an operator class's metric.
pub const METRIC_PROC_NUM: u16 = 2;

/// Support-function number of the proc reporting the type an operator class
/// indexes.
pub const KIND_PROC_NUM: u16 = 3;

/// Support functions a brindle operator class provides, i.e. the access
/// method's `amsupport`.
pub const SUPPORT_PROCS: u16 = KIND_PROC_NUM;

/// The type an indexed column holds, and therefore how its datums are read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VectorKind {
    RealArray,
    Vector,
}

impl VectorKind {
    /// Code an operator class reports through support function
    /// [`KIND_PROC_NUM`]. Existing codes must never be renumbered: an operator
    /// class outside this crate can use them too.
    pub fn code(self) -> i32 {
        match self {
            VectorKind::RealArray => 0,
            VectorKind::Vector => 1,
        }
    }

    /// Inverse of [`VectorKind::code`]; `None` for an unknown code.
    pub fn from_code(code: i32) -> Option<VectorKind> {
        match code {
            0 => Some(VectorKind::RealArray),
            1 => Some(VectorKind::Vector),
            _ => None,
        }
    }
}

#[pg_extern(immutable, parallel_safe)]
fn brindle_l2_metric() -> i32 {
    Metric::L2.code() as i32
}

#[pg_extern(immutable, parallel_safe)]
fn brindle_cosine_metric() -> i32 {
    Metric::Cosine.code() as i32
}

#[pg_extern(immutable, parallel_safe)]
fn brindle_inner_product_metric() -> i32 {
    Metric::InnerProduct.code() as i32
}

#[pg_extern(immutable, parallel_safe)]
fn brindle_real_array_kind() -> i32 {
    VectorKind::RealArray.code()
}

#[pg_extern(immutable, parallel_safe)]
fn brindle_vector_kind() -> i32 {
    VectorKind::Vector.code()
}

/// The metric `index`'s operator class selects.
///
/// # Safety
/// `index` must be an open index relation of the brindle access method.
pub unsafe fn index_metric(index: pg_sys::Relation) -> Metric {
    let code = support_code(index, METRIC_PROC_NUM, "metric");
    u8::try_from(code)
        .ok()
        .and_then(Metric::from_code)
        .unwrap_or_else(|| error!("brindle: operator class reports an unknown metric"))
}

/// The type `index`'s operator class is declared for.
///
/// # Safety
/// `index` must be an open index relation of the brindle access method.
pub unsafe fn index_kind(index: pg_sys::Relation) -> VectorKind {
    let code = support_code(index, KIND_PROC_NUM, "indexed type");
    let kind = VectorKind::from_code(code)
        .unwrap_or_else(|| error!("brindle: operator class reports an unknown indexed type"));

    // The kind decides how a raw datum is reinterpreted, so an operator class
    // that reports the wrong one would have the access method read an array's
    // bytes as a vector header, or the reverse. The type the class was declared
    // for is the cross-check available here.
    let declared_for_array = *(*index).rd_opcintype == pg_sys::FLOAT4ARRAYOID;
    if declared_for_array != (kind == VectorKind::RealArray) {
        error!("brindle: operator class does not match the type of the indexed column");
    }
    kind
}

/// Call one of the operator class's descriptive support functions.
///
/// # Safety
/// `index` must be an open index relation of the brindle access method.
unsafe fn support_code(index: pg_sys::Relation, procnum: u16, describes: &str) -> i32 {
    // Attribute 1 is the only one: the access method is single-column.
    let support_proc = pg_sys::index_getprocid(index, 1, procnum);
    if support_proc == pg_sys::InvalidOid {
        error!("brindle: operator class declares no {describes} (support function {procnum})");
    }

    let code = pg_sys::OidFunctionCall0Coll(support_proc, pg_sys::InvalidOid);
    i32::from_datum(code, false)
        .unwrap_or_else(|| error!("brindle: operator class reports no {describes}"))
}

extension_sql!(
    r#"
CREATE OPERATOR CLASS brindle_vector_l2_ops
    DEFAULT FOR TYPE brindle_vector USING brindle AS
    OPERATOR 1 <-> (brindle_vector, brindle_vector) FOR ORDER BY float_ops,
    FUNCTION 1 brindle_vector_l2_squared_distance(brindle_vector, brindle_vector),
    FUNCTION 2 (brindle_vector, brindle_vector) brindle_l2_metric(),
    FUNCTION 3 (brindle_vector, brindle_vector) brindle_vector_kind();

CREATE OPERATOR CLASS brindle_vector_ip_ops
    FOR TYPE brindle_vector USING brindle AS
    OPERATOR 2 <#> (brindle_vector, brindle_vector) FOR ORDER BY float_ops,
    FUNCTION 1 brindle_vector_negative_inner_product(brindle_vector, brindle_vector),
    FUNCTION 2 (brindle_vector, brindle_vector) brindle_inner_product_metric(),
    FUNCTION 3 (brindle_vector, brindle_vector) brindle_vector_kind();

CREATE OPERATOR CLASS brindle_vector_cosine_ops
    FOR TYPE brindle_vector USING brindle AS
    OPERATOR 3 <=> (brindle_vector, brindle_vector) FOR ORDER BY float_ops,
    FUNCTION 1 brindle_vector_cosine_distance(brindle_vector, brindle_vector),
    FUNCTION 2 (brindle_vector, brindle_vector) brindle_cosine_metric(),
    FUNCTION 3 (brindle_vector, brindle_vector) brindle_vector_kind();
"#,
    name = "brindle_vector_opclasses",
    requires = [
        brindle_amhandler,
        "brindle_vector_operators",
        brindle_vector_l2_squared_distance,
        brindle_vector_cosine_distance,
        brindle_vector_negative_inner_product,
        brindle_l2_metric,
        brindle_cosine_metric,
        brindle_inner_product_metric,
        brindle_vector_kind,
    ],
);
