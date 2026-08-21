//! `brindle_vector` — the extension's own vector type.
//!
//! One varlena per value: the standard 4-byte header, a dimension count, then
//! the components as `f32`. The payload therefore starts at a fixed offset and
//! is read as a plain `&[f32]`, so the distance kernels in [`crate::distance`]
//! see the stored bytes directly — no per-call copy, no deserialization.
//!
//! The layout deliberately matches pgvector's `vector` (`dim` as an `int16`
//! followed by two padding bytes), and the text form is pgvector's `[1,2,3]`,
//! so values move between the two types through `::text` without a converter.
//! Brindle does not *depend* on pgvector, though: see `docs/ARCHITECTURE.md`
//! § "on the vector type".

use core::ffi::CStr;
use core::fmt;
use core::ptr::NonNull;
use core::slice;
use std::ffi::CString;

use pgrx::callconv::{Arg, ArgAbi, BoxRet, FcInfo};
use pgrx::pgrx_sql_entity_graph::metadata::{
    ArgumentError, Returns, ReturnsError, SqlMapping, SqlTranslatable,
};
use pgrx::prelude::*;
use pgrx::{varlena, Array, FromDatum, IntoDatum};

use crate::distance;
use crate::or_error;

/// Name of the type in SQL.
const SQL_NAME: &str = "brindle_vector";

/// Largest vector the type accepts. Matches pgvector's limit: past a few
/// thousand dimensions a graph index is the wrong tool anyway, and the cap
/// keeps the dimension count in the 16 bits the header reserves for it.
pub const MAX_DIM: usize = 16_000;

/// Header of a `brindle_vector` value. The components follow it contiguously.
#[repr(C)]
struct VectorHeader {
    varlena_header: i32,
    dim: u16,
    /// Padding so the components start 8-byte aligned, as in pgvector.
    unused: u16,
}

/// Byte offset of the first component.
const DATA_OFFSET: usize = size_of::<VectorHeader>();

/// Why a value cannot be a `brindle_vector`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorError {
    /// A vector needs at least one component.
    Empty,
    /// More components than [`MAX_DIM`].
    TooManyDimensions(usize),
    /// Component `.0` is `NaN` or infinite; neither ranks.
    NonFinite(usize),
    /// The literal is not `[x, y, ...]`.
    Syntax(&'static str),
    /// Component `.0` is not a number.
    InvalidElement(usize),
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorError::Empty => write!(f, "vector must have at least one dimension"),
            VectorError::TooManyDimensions(dim) => {
                write!(f, "vector has {dim} dimensions, the maximum is {MAX_DIM}")
            }
            VectorError::NonFinite(index) => {
                write!(f, "vector element {index} must be a finite number")
            }
            VectorError::Syntax(expected) => {
                write!(f, "malformed vector literal: {expected}")
            }
            VectorError::InvalidElement(index) => {
                write!(f, "vector element {index} is not a number")
            }
        }
    }
}

impl std::error::Error for VectorError {}

/// Check the invariants every stored vector holds: non-empty, within
/// [`MAX_DIM`], and finite throughout.
pub fn validate(values: &[f32]) -> Result<(), VectorError> {
    if values.is_empty() {
        return Err(VectorError::Empty);
    }
    if values.len() > MAX_DIM {
        return Err(VectorError::TooManyDimensions(values.len()));
    }
    match values.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(VectorError::NonFinite(index)),
        None => Ok(()),
    }
}

/// Parse pgvector's text form, `[1,2,3]`, tolerating surrounding whitespace.
pub fn parse_literal(input: &str) -> Result<Vec<f32>, VectorError> {
    let body = input
        .trim()
        .strip_prefix('[')
        .ok_or(VectorError::Syntax("expected '[' at the start"))?
        .strip_suffix(']')
        .ok_or(VectorError::Syntax("expected ']' at the end"))?
        .trim();
    if body.is_empty() {
        return Err(VectorError::Empty);
    }

    let mut values = Vec::with_capacity(body.split(',').count());
    for (index, field) in body.split(',').enumerate() {
        let value: f32 = field
            .trim()
            .parse()
            .map_err(|_| VectorError::InvalidElement(index))?;
        values.push(value);
    }
    validate(&values)?;
    Ok(values)
}

/// Render the text form [`parse_literal`] accepts.
pub fn format_literal(values: &[f32]) -> String {
    use fmt::Write;

    // Rust's float formatting is shortest-round-trip, like float4out.
    let mut out = String::with_capacity(2 + values.len() * 8);
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let _ = write!(out, "{value}");
    }
    out.push(']');
    out
}

/// A `brindle_vector` datum.
///
/// It borrows the palloc'd, detoasted value rather than copying it, so it is
/// only valid inside the function call that produced it — never store one past
/// that call.
pub struct BrindleVector(NonNull<VectorHeader>);

impl BrindleVector {
    /// Copy `values` into a new palloc'd value in the current memory context.
    pub fn from_slice(values: &[f32]) -> Result<Self, VectorError> {
        validate(values)?;

        let size = DATA_OFFSET + size_of_val(values);
        // SAFETY: palloc0 either returns `size` writable, MAXALIGN'd bytes in
        // the current memory context or raises an ERROR, so the header and the
        // component array below are in bounds and aligned. The components are
        // copied from a slice that cannot overlap a fresh allocation.
        unsafe {
            let header = pg_sys::palloc0(size).cast::<VectorHeader>();
            varlena::set_varsize_4b(header.cast(), size as i32);
            (*header).dim = values.len() as u16;
            core::ptr::copy_nonoverlapping(
                values.as_ptr(),
                header.cast::<u8>().add(DATA_OFFSET).cast::<f32>(),
                values.len(),
            );
            Ok(Self(NonNull::new_unchecked(header)))
        }
    }

    /// Number of components.
    pub fn dim(&self) -> usize {
        // SAFETY: the constructors above only produce a pointer to a header
        // whose length was checked against its dimension count.
        unsafe { (*self.0.as_ptr()).dim as usize }
    }

    /// The components, borrowed from the value itself.
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `dim` components follow the header — checked on the way in by
        // `from_detoasted`, established by construction in `from_slice` — and
        // the allocation outlives this borrow of `self`.
        unsafe {
            slice::from_raw_parts(
                self.0.as_ptr().cast::<u8>().add(DATA_OFFSET).cast::<f32>(),
                self.dim(),
            )
        }
    }

    /// Wrap a fully detoasted value, rejecting one whose length disagrees with
    /// its dimension count before anything reads past the header.
    ///
    /// # Safety
    /// `header` must point at a detoasted (4-byte header, uncompressed)
    /// `brindle_vector` value.
    unsafe fn from_detoasted(header: NonNull<VectorHeader>) -> Self {
        let dim = (*header.as_ptr()).dim as usize;
        let expected = DATA_OFFSET + dim * size_of::<f32>();
        if varlena::varsize(header.as_ptr().cast()) != expected {
            error!("brindle: corrupt brindle_vector value");
        }
        Self(header)
    }
}

/// Copy a stored value's components out of its datum, releasing the copy that
/// detoasting a compressed, out-of-line, or short-header value makes.
///
/// Reading a value through [`FromDatum`] leaves that copy for the surrounding
/// memory context to reclaim, which is right inside an expression — the
/// executor resets that context per row — but wrong for a caller that reads
/// every row of a table under one context, like an index build.
///
/// # Safety
/// `datum` must be a non-null, valid `brindle_vector` value.
pub unsafe fn components_from_datum(datum: pg_sys::Datum) -> Vec<f32> {
    let stored = datum.cast_mut_ptr::<pg_sys::varlena>();
    let detoasted = pg_sys::pg_detoast_datum(stored);
    let vector =
        BrindleVector::from_detoasted(NonNull::new_unchecked(detoasted.cast::<VectorHeader>()));

    let components = vector.as_slice().to_vec();
    if !core::ptr::eq(detoasted, stored) {
        pg_sys::pfree(detoasted.cast());
    }
    components
}

/// OID of the SQL type, resolved through the current `search_path`.
///
/// This is the conversion path for values crossing into SQL from Rust (SPI and
/// friends), which run with the caller's search path. The index access method
/// deliberately does not use it: index maintenance runs with a restricted
/// search path, so it asks the operator class what it indexes instead.
pub fn type_oid() -> pg_sys::Oid {
    const NAME: &CStr = c"brindle_vector";

    // SAFETY: a NUL-terminated name is all the lookup needs; it reports "not
    // found" as InvalidOid rather than raising.
    let oid = unsafe { pg_sys::TypenameGetTypid(NAME.as_ptr()) };
    if oid == pg_sys::InvalidOid {
        error!("brindle: type {SQL_NAME} is not visible in search_path");
    }
    oid
}

impl FromDatum for BrindleVector {
    unsafe fn from_polymorphic_datum(
        datum: pg_sys::Datum,
        is_null: bool,
        _typoid: pg_sys::Oid,
    ) -> Option<Self> {
        if is_null {
            return None;
        }
        let value = NonNull::new(datum.cast_mut_ptr::<pg_sys::varlena>())?;
        // SAFETY: the caller guarantees a brindle_vector datum. Detoasting is
        // what makes the fixed layout hold: a stored value may arrive
        // compressed, out of line, or with a 1-byte header, and only the
        // detoasted form has its components where `as_slice` looks for them.
        let detoasted = pg_sys::pg_detoast_datum(value.as_ptr());
        Some(Self::from_detoasted(NonNull::new_unchecked(
            detoasted.cast::<VectorHeader>(),
        )))
    }
}

impl IntoDatum for BrindleVector {
    /// Hands back the value's own pointer. Safe for a value this crate just
    /// built; a value read from a datum still points at storage the caller
    /// owns, so returning one unchanged would hand out a borrow — copy it with
    /// [`BrindleVector::from_slice`] first.
    fn into_datum(self) -> Option<pg_sys::Datum> {
        Some(pg_sys::Datum::from(self.0.as_ptr()))
    }

    fn type_oid() -> pg_sys::Oid {
        type_oid()
    }
}

unsafe impl<'fcx> ArgAbi<'fcx> for BrindleVector {
    unsafe fn unbox_arg_unchecked(arg: Arg<'_, 'fcx>) -> Self {
        // Every function taking a brindle_vector is STRICT, so Postgres filters
        // NULL arguments out before the call.
        arg.unbox_arg_using_from_datum()
            .unwrap_or_else(|| error!("brindle: unexpected NULL {SQL_NAME} argument"))
    }
}

unsafe impl BoxRet for BrindleVector {
    unsafe fn box_into<'fcx>(self, fcinfo: &mut FcInfo<'fcx>) -> pgrx::datum::Datum<'fcx> {
        match self.into_datum() {
            Some(datum) => fcinfo.return_raw_datum(datum),
            None => fcinfo.return_null(),
        }
    }
}

unsafe impl SqlTranslatable for BrindleVector {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(SQL_NAME.into()))
    }

    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(SQL_NAME.into())))
    }
}

extension_sql!(
    "CREATE TYPE brindle_vector;",
    name = "brindle_vector_shell",
    creates = [Type(BrindleVector)],
);

/// Text input: `[1,2,3]`.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_vector_in(input: &CStr) -> BrindleVector {
    let literal = input
        .to_str()
        .unwrap_or_else(|_| error!("brindle: {SQL_NAME} input must be valid UTF-8"));
    let values = parse_literal(literal).unwrap_or_else(|e| error!("brindle: {e}"));
    BrindleVector::from_slice(&values).unwrap_or_else(|e| error!("brindle: {e}"))
}

/// Text output, in the form [`brindle_vector_in`] reads back.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_vector_out(vector: BrindleVector) -> CString {
    CString::new(format_literal(vector.as_slice()))
        .unwrap_or_else(|_| error!("brindle: {SQL_NAME} output contains an interior NUL"))
}

// STORAGE = external keeps large vectors out of line but never compressed:
// float bit patterns don't compress, so pglz would only burn CPU. This is
// pgvector's choice too.
extension_sql!(
    r#"
CREATE TYPE brindle_vector (
    INTERNALLENGTH = variable,
    INPUT = brindle_vector_in,
    OUTPUT = brindle_vector_out,
    STORAGE = external
);
"#,
    name = "brindle_vector_type",
    requires = [
        "brindle_vector_shell",
        brindle_vector_in,
        brindle_vector_out
    ],
);

/// Euclidean (L2) distance — the value `<->` returns.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_vector_l2_distance(a: BrindleVector, b: BrindleVector) -> f32 {
    or_error(distance::l2_distance(a.as_slice(), b.as_slice()))
}

/// Squared Euclidean distance: same ordering as `<->` without the square root,
/// which is how the index ranks internally.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_vector_l2_squared_distance(a: BrindleVector, b: BrindleVector) -> f32 {
    or_error(distance::l2_squared(a.as_slice(), b.as_slice()))
}

/// Cosine distance (`1 - cosine_similarity`) — the value `<=>` returns.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_vector_cosine_distance(a: BrindleVector, b: BrindleVector) -> f32 {
    or_error(distance::cosine_distance(a.as_slice(), b.as_slice()))
}

/// Negative inner product — the value `<#>` returns, so that smaller is nearer.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_vector_negative_inner_product(a: BrindleVector, b: BrindleVector) -> f32 {
    or_error(distance::negative_inner_product(a.as_slice(), b.as_slice()))
}

/// Convert a `real[]`, so existing array-shaped data can move to the type.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_vector_from_real_array(values: Array<f32>) -> BrindleVector {
    let values = one_dimensional(values);
    let values = match values.as_slice() {
        Ok(values) => values,
        Err(_) => error!("brindle: vector must not contain NULL elements"),
    };
    BrindleVector::from_slice(values).unwrap_or_else(|e| error!("brindle: {e}"))
}

/// Reject the array shapes a vector cannot represent. Postgres stores a
/// multi-dimensional array as one flat run of elements, so `ARRAY[[1,2],[3,4]]`
/// would otherwise become a four-component vector instead of an error. An
/// empty array falls through to the emptiness check, which says more.
fn one_dimensional(values: Array<f32>) -> Array<f32> {
    let array = values.into_array_type();
    // SAFETY: into_array_type returns the detoasted ArrayType this argument was
    // unboxed from, which stays valid for the rest of the call; re-wrapping an
    // already-detoasted value copies nothing.
    unsafe {
        if (*array).ndim > 1 {
            error!("brindle: vector must be a one-dimensional array");
        }
        Array::from_polymorphic_datum(pg_sys::Datum::from(array), false, pg_sys::FLOAT4ARRAYOID)
            .unwrap_or_else(|| error!("brindle: could not read real[] value"))
    }
}

/// Convert back to a `real[]`, for callers that still speak arrays.
#[pg_extern(immutable, strict, parallel_safe)]
fn brindle_vector_to_real_array(vector: BrindleVector) -> Vec<f32> {
    vector.as_slice().to_vec()
}

// Operator symbols and their meanings follow pgvector so queries port over
// unchanged. All three are symmetric, hence each is its own commutator.
extension_sql!(
    r#"
CREATE OPERATOR <-> (
    LEFTARG = brindle_vector, RIGHTARG = brindle_vector,
    FUNCTION = brindle_vector_l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <#> (
    LEFTARG = brindle_vector, RIGHTARG = brindle_vector,
    FUNCTION = brindle_vector_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <=> (
    LEFTARG = brindle_vector, RIGHTARG = brindle_vector,
    FUNCTION = brindle_vector_cosine_distance,
    COMMUTATOR = '<=>'
);

CREATE CAST (real[] AS brindle_vector)
    WITH FUNCTION brindle_vector_from_real_array(real[]) AS ASSIGNMENT;
CREATE CAST (brindle_vector AS real[])
    WITH FUNCTION brindle_vector_to_real_array(brindle_vector);
"#,
    name = "brindle_vector_operators",
    requires = [
        "brindle_vector_type",
        brindle_vector_l2_distance,
        brindle_vector_cosine_distance,
        brindle_vector_negative_inner_product,
        brindle_vector_from_real_array,
        brindle_vector_to_real_array,
    ],
);

#[cfg(test)]
mod literal_tests {
    use super::*;

    #[test]
    fn parses_pgvector_text_form() {
        assert_eq!(parse_literal("[1,2,3]").unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(parse_literal(" [ 1 , -2.5 ] ").unwrap(), vec![1.0, -2.5]);
        assert_eq!(parse_literal("[1e3]").unwrap(), vec![1000.0]);
    }

    #[test]
    fn round_trips_through_text() {
        let values = vec![1.0, -2.5, 0.125, 1e10];
        assert_eq!(parse_literal(&format_literal(&values)).unwrap(), values);
    }

    #[test]
    fn rejects_malformed_literals() {
        assert!(matches!(
            parse_literal("1,2,3"),
            Err(VectorError::Syntax(_))
        ));
        assert!(matches!(
            parse_literal("[1,2,3"),
            Err(VectorError::Syntax(_))
        ));
        assert_eq!(parse_literal("[]"), Err(VectorError::Empty));
        assert_eq!(parse_literal("[1,x]"), Err(VectorError::InvalidElement(1)));
    }

    #[test]
    fn rejects_values_that_cannot_rank() {
        assert_eq!(parse_literal("[1,NaN]"), Err(VectorError::NonFinite(1)));
        assert_eq!(parse_literal("[inf]"), Err(VectorError::NonFinite(0)));
        assert_eq!(validate(&[]), Err(VectorError::Empty));
        assert_eq!(
            validate(&vec![0.0; MAX_DIM + 1]),
            Err(VectorError::TooManyDimensions(MAX_DIM + 1))
        );
    }
}

// The pgrx test harness looks every #[pg_test] up in a schema named `tests`.
#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    fn text_of(query: &str) -> String {
        Spi::get_one::<String>(query)
            .expect("SPI failed")
            .expect("null result")
    }

    fn distance_of(query: &str) -> f32 {
        Spi::get_one::<f32>(query)
            .expect("SPI failed")
            .expect("null result")
    }

    #[pg_test]
    fn text_round_trips() {
        assert_eq!(text_of("SELECT '[1,2,3]'::brindle_vector::text"), "[1,2,3]");
        assert_eq!(
            text_of("SELECT ' [ 1 , -2.5 ] '::brindle_vector::text"),
            "[1,-2.5]"
        );
    }

    #[pg_test]
    fn casts_to_and_from_real_arrays() {
        assert_eq!(
            text_of("SELECT ARRAY[1,2,3]::real[]::brindle_vector::text"),
            "[1,2,3]"
        );
        assert_eq!(
            text_of("SELECT ('[1,2,3]'::brindle_vector::real[])::text"),
            "{1,2,3}"
        );
    }

    #[pg_test]
    fn columns_accept_arrays_without_an_explicit_cast() {
        Spi::run("CREATE TABLE t_cast (embedding brindle_vector)").expect("create");
        Spi::run("INSERT INTO t_cast VALUES (ARRAY[1,2,3]::real[])").expect("insert");
        assert_eq!(text_of("SELECT embedding::text FROM t_cast"), "[1,2,3]");
    }

    #[pg_test]
    fn operators_return_their_metrics() {
        let l2 = distance_of("SELECT '[1,2,3]'::brindle_vector <-> '[4,5,6]'::brindle_vector");
        assert!((l2 - 5.196_152).abs() < 1e-4, "got {l2}");

        let cosine = distance_of("SELECT '[1,0]'::brindle_vector <=> '[0,1]'::brindle_vector");
        assert!((cosine - 1.0).abs() < 1e-4, "got {cosine}");

        let ip = distance_of("SELECT '[1,2,3]'::brindle_vector <#> '[4,5,6]'::brindle_vector");
        assert!((ip + 32.0).abs() < 1e-4, "got {ip}");
    }

    #[pg_test]
    fn squared_distance_matches_the_operator() {
        let squared = distance_of(
            "SELECT brindle_vector_l2_squared_distance('[1,2,3]'::brindle_vector,
                                                       '[4,5,6]'::brindle_vector)",
        );
        assert!((squared - 27.0).abs() < 1e-4, "got {squared}");
    }

    #[pg_test]
    fn survives_the_toast_round_trip() {
        // Wide enough that the value is stored out of line, which is the path
        // detoasting exists for.
        Spi::run("CREATE TABLE t_toast (embedding brindle_vector)").expect("create");
        Spi::run(
            "INSERT INTO t_toast
             SELECT ('[' || string_agg(i::text, ',') || ']')::brindle_vector
             FROM generate_series(1, 4000) i",
        )
        .expect("insert");
        let distance = distance_of("SELECT embedding <-> embedding FROM t_toast");
        assert_eq!(distance, 0.0);
    }

    #[pg_test(error = "brindle: malformed vector literal: expected '[' at the start")]
    fn rejects_a_malformed_literal() {
        Spi::run("SELECT '1,2,3'::brindle_vector").expect("cast");
    }

    #[pg_test(error = "brindle: vector must be a one-dimensional array")]
    fn rejects_a_multi_dimensional_array() {
        Spi::run("SELECT ARRAY[[1,2],[3,4]]::real[]::brindle_vector").expect("cast");
    }

    #[pg_test(error = "brindle: vector element 1 must be a finite number")]
    fn rejects_a_non_finite_element() {
        Spi::run("SELECT '[1,NaN]'::brindle_vector").expect("cast");
    }

    #[pg_test(error = "brindle: vector dimension mismatch: 3 != 2")]
    fn rejects_mismatched_dimensions() {
        Spi::run("SELECT '[1,2,3]'::brindle_vector <-> '[1,2]'::brindle_vector").expect("distance");
    }
}
