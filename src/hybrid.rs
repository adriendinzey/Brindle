//! Hybrid search: fuse vector similarity and full-text relevance with RRF.
//!
//! [`brindle_hybrid`] is the SQL surface over [`crate::fusion`]. It runs two
//! ranked searches over one table — nearest-neighbour by the Brindle index and
//! lexical relevance by Postgres full-text search — and fuses their *ranks*
//! (not their scores) with Reciprocal Rank Fusion, so a row that both signals
//! agree on outranks a row only one of them ranks highly.
//!
//! # Why the plumbing lives here and the math lives in `fusion`
//!
//! The fusion arithmetic is pure and unit-tested in [`crate::fusion`]. This
//! module is the Postgres boundary: it resolves the caller's table and columns,
//! issues two `SPI` queries, maps row identities to the `u64` keys fusion
//! speaks, and turns the fused ranking back into rows. Nothing here re-derives
//! the fusion math.
//!
//! # Safety of the dynamic SQL
//!
//! The two searches run against a caller-named table and columns, so the SQL is
//! built at call time. Every *value* — the query vector, the query text, the
//! text-search config, the per-side depth — is a bound parameter (`$1`..), never
//! interpolated. Every *identifier* — the table, the id/vector/text columns, and
//! the distance operator's schema — is quoted by Postgres itself (`quote_ident`,
//! and the operator is read from the catalog), so no caller string can alter the
//! statement. See [`resolve`].
//!
//! # Choosing the distance operator
//!
//! The vector search must order by the *same* operator the index was built with,
//! or the planner will not use the index. Rather than take a metric parameter
//! that could disagree with the index, [`resolve`] reads the ordering operator
//! straight off the Brindle index's operator class, and requires such an index.

use std::collections::HashMap;

use pgrx::prelude::*;
use pgrx::spi::{SpiClient, SpiError};
use pgrx::{IntoDatum, PgOid};

use crate::fusion;
use crate::pg_vector::{self, BrindleVector, VectorError};

/// Why a hybrid search could not run.
///
/// The variants that name a column or relation carry it, so the boundary can
/// raise a message that points at the caller's own identifier.
#[derive(Debug)]
enum HybridError {
    /// `relation` did not resolve to a table in the current `search_path`.
    RelationNotFound(String),
    /// A named column is not present on the relation.
    ColumnMissing { role: &'static str, column: String },
    /// The id column is not an integer type, so it cannot key fusion losslessly.
    IdNotInteger { column: String, found: String },
    /// The vector column is not `brindle_vector`.
    VectorNotVector { column: String, found: String },
    /// The text column is neither `tsvector` nor a text type.
    TextNotSearchable { column: String, found: String },
    /// No Brindle index covers the vector column, so no metric can be inferred.
    NoBrindleIndex { column: String },
    /// Brindle indexes with *different* metrics cover the vector column, so the
    /// metric cannot be inferred without guessing which one the caller meant.
    AmbiguousMetric { column: String, operators: String },
    /// `k` (rows returned) or `n` (per-side depth) was below 1.
    NonPositive { param: &'static str, value: i32 },
    /// The query vector could not be re-encoded to bind to the search.
    Vector(VectorError),
    /// Fusion rejected its parameters (e.g. a non-finite `rrf_k`).
    Fusion(fusion::FusionError),
    /// An SPI call failed.
    Spi(SpiError),
}

impl std::fmt::Display for HybridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HybridError::RelationNotFound(rel) => write!(f, "relation \"{rel}\" does not exist"),
            HybridError::ColumnMissing { role, column } => {
                write!(
                    f,
                    "{role} column \"{column}\" does not exist on the relation"
                )
            }
            HybridError::IdNotInteger { column, found } => write!(
                f,
                "id column \"{column}\" is {found}, but must be an integer type \
                 (smallint, integer, or bigint)"
            ),
            HybridError::VectorNotVector { column, found } => write!(
                f,
                "vector column \"{column}\" is {found}, but must be brindle_vector"
            ),
            HybridError::TextNotSearchable { column, found } => write!(
                f,
                "text column \"{column}\" is {found}, but must be tsvector or a \
                 text type (text, varchar, char)"
            ),
            HybridError::NoBrindleIndex { column } => write!(
                f,
                "hybrid search needs a brindle index on vector column \"{column}\" \
                 (its operator class selects the distance metric)"
            ),
            HybridError::AmbiguousMetric { column, operators } => write!(
                f,
                "vector column \"{column}\" has brindle indexes with different \
                 metrics ({operators}); drop one so the metric is unambiguous"
            ),
            HybridError::NonPositive { param, value } => {
                write!(f, "{param} must be at least 1, got {value}")
            }
            HybridError::Vector(e) => write!(f, "{e}"),
            HybridError::Fusion(e) => write!(f, "{e}"),
            HybridError::Spi(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for HybridError {}

/// Whether the lexical column is already a `tsvector` or raw text that must be
/// lexed per query.
#[derive(Clone, Copy)]
enum TextKind {
    /// A `tsvector` column: `@@` and `ts_rank` read it directly, and a GIN index
    /// can serve the match.
    Tsvector,
    /// A `text`/`varchar`/`char` column: `to_tsvector` lexes it in the query, so
    /// the same config lexes both sides but no index can help.
    Text,
}

/// The catalog facts a hybrid search needs, resolved in one round trip: the
/// safely-quoted identifiers to build SQL from, and the ordering operator the
/// vector index was built with.
struct Resolved {
    /// `quote_ident(schema).quote_ident(table)` — ready to drop into a `FROM`.
    rel_ident: String,
    /// `quote_ident(id_column)`.
    id_ident: String,
    /// `quote_ident(vector_column)`.
    vec_ident: String,
    /// `quote_ident(text_column)`.
    text_ident: String,
    /// `quote_ident(schema).<op>` for the index's ordering operator, ready to
    /// drop inside `OPERATOR(...)`.
    vec_operator: String,
    text_kind: TextKind,
}

/// The sign bit, flipped when mapping an `i64` id to its `u64` fusion key.
///
/// Two's-complement `i64` XOR this is an *order-preserving* bijection onto
/// `u64`: it maps `i64::MIN..=i64::MAX` onto `0..=u64::MAX` in the same order.
/// So fusion's ascending-`u64` tie-break (for rows with identical fused scores)
/// falls out as ascending id, even when ids are negative — and the round trip
/// back is exact, since XOR is its own inverse.
const ID_SIGN: u64 = 1 << 63;

fn key_of(id: i64) -> u64 {
    (id as u64) ^ ID_SIGN
}

fn id_of(key: u64) -> i64 {
    (key ^ ID_SIGN) as i64
}

/// One fused result row: the caller's id, the fused score, and the row's 1-based
/// rank within each source (absent when the source did not surface it).
type HybridRow = (i64, f64, Option<i32>, Option<i32>);

/// Fuse a Brindle vector search and a Postgres full-text search over `relation`
/// into one ranking with Reciprocal Rank Fusion.
///
/// `id_column` (a *unique* integer key), `vector_column` (a `brindle_vector`
/// with a brindle index), and `text_column` (a `tsvector`, or raw text) name the
/// data. `query_vec` drives the vector side and `query_text` the lexical side;
/// either may find nothing, in which case fusion reduces to the other. Returns
/// the top `k` rows as `(id, score, vector_rank, text_rank)`, best first.
///
/// The id column must be unique: fusion keys on the id, so two rows sharing one
/// would be coalesced into a single result.
///
/// The vector operator is taken from the index's operator class, so the ORDER BY
/// matches the metric the data was indexed under. `n` rows are pulled from each
/// side before fusing (default `max(4·k, 40)`); the vector side is additionally
/// capped by `brindle.ef_search`. `rrf_k` is RRF's damping constant. `config`
/// names the text-search configuration, defaulting to the database's.
#[allow(clippy::too_many_arguments)]
#[pg_extern(stable)]
fn brindle_hybrid(
    relation: &str,
    id_column: &str,
    vector_column: &str,
    text_column: &str,
    query_text: Option<String>,
    query_vec: BrindleVector,
    k: default!(i32, 10),
    n: default!(Option<i32>, "NULL"),
    rrf_k: default!(f64, 60),
    config: default!(Option<String>, "NULL"),
) -> TableIterator<
    'static,
    (
        name!(id, i64),
        name!(score, f64),
        name!(vector_rank, Option<i32>),
        name!(text_rank, Option<i32>),
    ),
> {
    match run(
        relation,
        id_column,
        vector_column,
        text_column,
        query_text.as_deref(),
        &query_vec,
        k,
        n,
        rrf_k,
        config.as_deref(),
    ) {
        Ok(rows) => TableIterator::new(rows),
        Err(e) => error!("brindle: {e}"),
    }
}

/// The whole search as a `Result`, so the `#[pg_extern]` boundary is the only
/// place that raises — keeping `unwrap`/`error!` out of the logic.
#[allow(clippy::too_many_arguments)]
fn run(
    relation: &str,
    id_column: &str,
    vector_column: &str,
    text_column: &str,
    query_text: Option<&str>,
    query_vec: &BrindleVector,
    k: i32,
    n: Option<i32>,
    rrf_k: f64,
    config: Option<&str>,
) -> Result<Vec<HybridRow>, HybridError> {
    if k < 1 {
        return Err(HybridError::NonPositive {
            param: "k",
            value: k,
        });
    }
    // Default the per-side depth to a few multiples of k so RRF has enough
    // overlap to reward consensus, with a floor of 40 so a small k still fuses a
    // useful window. The vector side is separately capped by `brindle.ef_search`
    // (default 64), so realising a depth beyond that needs a larger GUC.
    let depth = n.unwrap_or_else(|| k.saturating_mul(4).max(40));
    if depth < 1 {
        return Err(HybridError::NonPositive {
            param: "n",
            value: depth,
        });
    }

    // Skipping an empty lexical query keeps the SPI round trip off the path and
    // leaves the text list empty, so fusion falls back to the vector ranking.
    let query_text = query_text.filter(|t| !t.trim().is_empty());

    Spi::connect(|client| {
        let resolved = resolve(&client, relation, id_column, vector_column, text_column)?;

        let vector_ids = vector_ids(&client, &resolved, query_vec, depth)?;
        let text_ids = match query_text {
            Some(text) => text_ids(&client, &resolved, text, config, depth)?,
            None => Vec::new(),
        };

        fuse(&vector_ids, &text_ids, rrf_k, k)
    })
}

/// Resolve the relation and columns to quoted identifiers and the index's
/// ordering operator, in one SPI round trip, validating types as it goes.
///
/// Every subquery keys off `to_regclass($1)`, so a relation not visible in the
/// caller's `search_path` resolves to NULL and every other field with it — one
/// clean "does not exist" rather than a cascade.
fn resolve(
    client: &SpiClient<'_>,
    relation: &str,
    id_column: &str,
    vector_column: &str,
    text_column: &str,
) -> Result<Resolved, HybridError> {
    // $1 relation, $2 id column, $3 vector column, $4 text column. quote_ident on
    // the column names is always non-null (it quotes whatever string it is
    // given); the *_type columns are null exactly when the column is absent,
    // which is how existence is checked without a second query. The operator
    // subquery unnests the index's key/opclass vectors with ordinality (cast to
    // real arrays so the 1-based ordinality is unambiguous) to read the ordering
    // operator of the opclass on the vector column, schema-qualified so it
    // resolves under any search_path.
    const SQL: &str = r#"
        WITH rel AS (SELECT to_regclass($1) AS oid)
        SELECT
          (SELECT quote_ident(n.nspname) || '.' || quote_ident(c.relname)
             FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.oid = (SELECT oid FROM rel))                        AS rel_ident,
          quote_ident($2)                                              AS id_ident,
          quote_ident($3)                                              AS vec_ident,
          quote_ident($4)                                              AS text_ident,
          (SELECT t.typname::text FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
            WHERE a.attrelid = (SELECT oid FROM rel) AND a.attname = $2
              AND NOT a.attisdropped)                                  AS id_type,
          (SELECT t.typname::text FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
            WHERE a.attrelid = (SELECT oid FROM rel) AND a.attname = $3
              AND NOT a.attisdropped)                                  AS vec_type,
          (SELECT t.typname::text FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
            WHERE a.attrelid = (SELECT oid FROM rel) AND a.attname = $4
              AND NOT a.attisdropped)                                  AS text_type,
          -- Every *distinct* ordering operator of a brindle index on the
          -- vector column, joined by '|'. One means an unambiguous metric; more
          -- than one means indexes with different metrics cover the column, and
          -- the caller must disambiguate rather than have one silently chosen.
          (SELECT string_agg(DISTINCT quote_ident(non.nspname) || '.' || o.oprname, '|'
                              ORDER BY quote_ident(non.nspname) || '.' || o.oprname)
             FROM pg_index i
             JOIN pg_class ic ON ic.oid = i.indexrelid
             JOIN pg_am am ON am.oid = ic.relam AND am.amname = 'brindle'
             JOIN unnest(i.indkey::int2[])  WITH ORDINALITY AS k(attnum, ord) ON true
             JOIN unnest(i.indclass::oid[]) WITH ORDINALITY AS cl(opc, ord2)  ON cl.ord2 = k.ord
             JOIN pg_attribute att ON att.attrelid = i.indrelid
               AND att.attnum = k.attnum AND att.attname = $3
             JOIN pg_opclass oc ON oc.oid = cl.opc
             JOIN pg_amop ao ON ao.amopfamily = oc.opcfamily
               AND ao.amoppurpose = 'o' AND ao.amoplefttype = oc.opcintype
             JOIN pg_operator o ON o.oid = ao.amopopr
             JOIN pg_namespace non ON non.oid = o.oprnamespace
            WHERE i.indrelid = (SELECT oid FROM rel))                 AS vec_operators
    "#;

    let args = vec![
        text_arg(Some(relation)),
        text_arg(Some(id_column)),
        text_arg(Some(vector_column)),
        text_arg(Some(text_column)),
    ];
    let table = client
        .select(SQL, None, Some(args))
        .map_err(HybridError::Spi)?;
    // The outer SELECT has no FROM, so it yields exactly one row whatever the
    // subqueries find.
    let row = table
        .into_iter()
        .next()
        .ok_or_else(|| HybridError::RelationNotFound(relation.to_string()))?;

    let rel_ident = row
        .get::<String>(1)
        .map_err(HybridError::Spi)?
        .ok_or_else(|| HybridError::RelationNotFound(relation.to_string()))?;
    let id_ident = row
        .get::<String>(2)
        .map_err(HybridError::Spi)?
        .unwrap_or_default();
    let vec_ident = row
        .get::<String>(3)
        .map_err(HybridError::Spi)?
        .unwrap_or_default();
    let text_ident = row
        .get::<String>(4)
        .map_err(HybridError::Spi)?
        .unwrap_or_default();
    let id_type = row.get::<String>(5).map_err(HybridError::Spi)?;
    let vec_type = row.get::<String>(6).map_err(HybridError::Spi)?;
    let text_type = row.get::<String>(7).map_err(HybridError::Spi)?;
    let vec_operators = row.get::<String>(8).map_err(HybridError::Spi)?;

    // id column: present and an integer type.
    match id_type.as_deref() {
        None => {
            return Err(HybridError::ColumnMissing {
                role: "id",
                column: id_column.to_string(),
            })
        }
        Some("int2" | "int4" | "int8") => {}
        Some(found) => {
            return Err(HybridError::IdNotInteger {
                column: id_column.to_string(),
                found: found.to_string(),
            })
        }
    }

    // vector column: present and brindle_vector. Checked before the index, so a
    // real[] column with its own brindle opclass gets a "wrong type" message
    // rather than a query that fails on a mismatched operator later.
    match vec_type.as_deref() {
        None => {
            return Err(HybridError::ColumnMissing {
                role: "vector",
                column: vector_column.to_string(),
            })
        }
        Some("brindle_vector") => {}
        Some(found) => {
            return Err(HybridError::VectorNotVector {
                column: vector_column.to_string(),
                found: found.to_string(),
            })
        }
    }

    // text column: present and a searchable type.
    let text_kind = match text_type.as_deref() {
        None => {
            return Err(HybridError::ColumnMissing {
                role: "text",
                column: text_column.to_string(),
            })
        }
        Some("tsvector") => TextKind::Tsvector,
        Some("text" | "varchar" | "bpchar") => TextKind::Text,
        Some(found) => {
            return Err(HybridError::TextNotSearchable {
                column: text_column.to_string(),
                found: found.to_string(),
            })
        }
    };

    // One distinct ordering operator → the metric is unambiguous. None → no
    // brindle index covers the column. More than one → indexes with different
    // metrics, which must not be resolved by guessing.
    let operators = vec_operators.unwrap_or_default();
    let vec_operator = match operators.split('|').filter(|s| !s.is_empty()).count() {
        0 => {
            return Err(HybridError::NoBrindleIndex {
                column: vector_column.to_string(),
            })
        }
        1 => operators,
        _ => {
            return Err(HybridError::AmbiguousMetric {
                column: vector_column.to_string(),
                operators,
            })
        }
    };

    Ok(Resolved {
        rel_ident,
        id_ident,
        vec_ident,
        text_ident,
        vec_operator,
        text_kind,
    })
}

/// The vector side: the `depth` nearest ids to `query_vec`, nearest first.
///
/// A bare `ORDER BY col <op> $1 LIMIT n` is exactly the shape the Brindle index
/// answers, so the planner drives it with an index ordering scan; the ids come
/// back in distance order and their position is their rank.
///
/// No secondary sort key is added: it would cost the index ordering scan (the
/// plan the whole shape exists for). So among rows at an *identical* distance
/// the order — and thus the vector rank handed to fusion — is the graph's, not
/// a stable id order. In practice exact-distance ties are rare, and fusion's own
/// ascending-id tie-break still makes the final output deterministic whenever
/// two rows end on the same fused score.
fn vector_ids(
    client: &SpiClient<'_>,
    r: &Resolved,
    query_vec: &BrindleVector,
    depth: i32,
) -> Result<Vec<i64>, HybridError> {
    let sql = format!(
        "SELECT ({id})::bigint FROM {rel} WHERE {vec} IS NOT NULL \
         ORDER BY {vec} OPERATOR({op}) $1 LIMIT $2",
        id = r.id_ident,
        rel = r.rel_ident,
        vec = r.vec_ident,
        op = r.vec_operator,
    );
    // Re-encode into a fresh value this call owns, rather than lending out the
    // argument's own storage as a bind datum.
    let query = BrindleVector::from_slice(query_vec.as_slice()).map_err(HybridError::Vector)?;
    let args = vec![
        (PgOid::from(pg_vector::type_oid()), query.into_datum()),
        int8_arg(depth as i64),
    ];
    fetch_ids(client, &sql, args)
}

/// The lexical side: the `depth` best full-text matches for `text`, best first.
///
/// `websearch_to_tsquery` never raises on arbitrary user input (it accepts
/// quotes, `OR`, and `-negation`), which is what a query box needs. The config
/// defaults to the database's when `config` is NULL. The `id` tiebreak makes the
/// order — and so the rank fed to fusion — deterministic when `ts_rank` ties.
fn text_ids(
    client: &SpiClient<'_>,
    r: &Resolved,
    text: &str,
    config: Option<&str>,
    depth: i32,
) -> Result<Vec<i64>, HybridError> {
    // $1 is the query text. When a config is given it binds as $2 and the depth
    // as $3; otherwise the single-argument forms use the database's default
    // config and the depth binds as $2. The tsvector column is read directly;
    // a raw text column is lexed with the *same* config as the query, so both
    // sides of the match agree.
    let (tsquery, lexed, args, limit) = if let Some(cfg) = config {
        let tsquery = "websearch_to_tsquery($2::regconfig, $1)".to_string();
        let lexed = match r.text_kind {
            TextKind::Tsvector => r.text_ident.clone(),
            TextKind::Text => format!("to_tsvector($2::regconfig, {})", r.text_ident),
        };
        let args = vec![
            text_arg(Some(text)),
            text_arg(Some(cfg)),
            int8_arg(depth as i64),
        ];
        (tsquery, lexed, args, "$3")
    } else {
        let tsquery = "websearch_to_tsquery($1)".to_string();
        let lexed = match r.text_kind {
            TextKind::Tsvector => r.text_ident.clone(),
            TextKind::Text => format!("to_tsvector({})", r.text_ident),
        };
        let args = vec![text_arg(Some(text)), int8_arg(depth as i64)];
        (tsquery, lexed, args, "$2")
    };
    let sql = format!(
        "SELECT ({id})::bigint FROM {rel} \
         WHERE {lexed} @@ {tsquery} \
         ORDER BY ts_rank({lexed}, {tsquery}) DESC, ({id})::bigint ASC \
         LIMIT {limit}",
        id = r.id_ident,
        rel = r.rel_ident,
    );
    fetch_ids(client, &sql, args)
}

/// Run a query whose first column is a `bigint` id and collect the ids in row
/// order, dropping any NULL (which a well-formed id column never produces).
fn fetch_ids(
    client: &SpiClient<'_>,
    sql: &str,
    args: Vec<(PgOid, Option<pg_sys::Datum>)>,
) -> Result<Vec<i64>, HybridError> {
    let table = client
        .select(sql, None, Some(args))
        .map_err(HybridError::Spi)?;
    let mut ids = Vec::new();
    for row in table {
        if let Some(id) = row.get::<i64>(1).map_err(HybridError::Spi)? {
            ids.push(id);
        }
    }
    Ok(ids)
}

/// Fuse the two ranked id lists with RRF and take the top `k`, carrying each
/// row's 1-based rank within each source for explainability.
fn fuse(
    vector_ids: &[i64],
    text_ids: &[i64],
    rrf_k: f64,
    k: i32,
) -> Result<Vec<HybridRow>, HybridError> {
    let vector_keys: Vec<u64> = vector_ids.iter().map(|&id| key_of(id)).collect();
    let text_keys: Vec<u64> = text_ids.iter().map(|&id| key_of(id)).collect();

    // First occurrence is the best rank, matching how fusion treats a repeat.
    let ranks = |keys: &[u64]| -> HashMap<u64, i32> {
        let mut map = HashMap::with_capacity(keys.len());
        for (pos, &key) in keys.iter().enumerate() {
            map.entry(key).or_insert(pos as i32 + 1);
        }
        map
    };
    let vector_rank = ranks(&vector_keys);
    let text_rank = ranks(&text_keys);

    let fused = fusion::rrf(&[&vector_keys, &text_keys], rrf_k).map_err(HybridError::Fusion)?;

    Ok(fused
        .into_iter()
        .take(k as usize)
        .map(|(key, score)| {
            (
                id_of(key),
                score,
                vector_rank.get(&key).copied(),
                text_rank.get(&key).copied(),
            )
        })
        .collect())
}

/// A text SPI argument, NULL when absent.
fn text_arg(value: Option<&str>) -> (PgOid, Option<pg_sys::Datum>) {
    (
        PgOid::from(pg_sys::TEXTOID),
        value.and_then(|v| v.into_datum()),
    )
}

/// A `bigint` SPI argument.
fn int8_arg(value: i64) -> (PgOid, Option<pg_sys::Datum>) {
    (PgOid::from(pg_sys::INT8OID), value.into_datum())
}

// The pgrx test harness looks every #[pg_test] up in a schema named `tests`.
#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use std::collections::HashMap;

    use pgrx::prelude::*;

    /// One returned row, in the function's output (fused-score) order.
    struct Row {
        id: i32,
        score: f64,
        vector_rank: Option<i32>,
        text_rank: Option<i32>,
    }

    /// A six-row corpus over `opclass`, with an id, a `body` text column, a
    /// materialised `tsv` tsvector, and an indexed `brindle_vector`.
    ///
    /// The embeddings and text are chosen against `query = [1,0,0]` and the
    /// lexical query "wireless headphones" so that:
    ///   id 1 (A) is strong in BOTH  — near the query and matches the words,
    ///   id 2 (B) is strong in VECTOR only — nearest vector, unrelated words,
    ///   id 3 (C) is strong in TEXT only  — far vector, matches the words,
    /// with three filler rows in between. It is the worked example behind the
    /// canonical RRF property: A must outrank both B and C.
    fn create_corpus(table: &str, opclass: &str) {
        Spi::run(&format!(
            "CREATE TABLE {table} (
                 id int PRIMARY KEY,
                 body text,
                 tsv tsvector,
                 embedding brindle_vector
             )"
        ))
        .expect("create table");
        Spi::run(&format!(
            "INSERT INTO {table} (id, body, embedding) VALUES
               (1, 'wireless headphones',                       '[0.9,0.1,0.0]'),
               (2, 'garden hose reel',                          '[1.0,0.0,0.0]'),
               (3, 'wireless bluetooth headphones earbuds noise','[0.0,0.0,1.0]'),
               (4, 'ceramic coffee mug',                        '[0.5,0.5,0.0]'),
               (5, 'ergonomic office chair',                    '[0.2,0.8,0.0]'),
               (6, 'desk lamp',                                 '[0.0,1.0,0.0]')"
        ))
        .expect("insert rows");
        Spi::run(&format!(
            "UPDATE {table} SET tsv = to_tsvector('english', body)"
        ))
        .expect("build tsvector");
        Spi::run(&format!(
            "CREATE INDEX {table}_idx ON {table} USING brindle (embedding {opclass})"
        ))
        .expect("create index");
    }

    /// Call `brindle_hybrid` and collect its rows in output order.
    fn hybrid(call: &str) -> Vec<Row> {
        Spi::connect(|client| {
            client
                .select(call, None, None)
                .expect("hybrid call")
                .map(|row| Row {
                    id: row.get::<i64>(1).expect("id").expect("id not null") as i32,
                    score: row.get::<f64>(2).expect("score").expect("score not null"),
                    vector_rank: row.get::<i32>(3).expect("vector_rank"),
                    text_rank: row.get::<i32>(4).expect("text_rank"),
                })
                .collect()
        })
    }

    /// Index the returned rows by id for property assertions.
    fn by_id(rows: &[Row]) -> HashMap<i32, &Row> {
        rows.iter().map(|r| (r.id, r)).collect()
    }

    /// The canonical RRF property, over a `tsvector` column: a row strong in
    /// both signals beats a row strong in only one — including the row that is
    /// *first* on the vector side.
    #[pg_test]
    fn a_row_strong_in_both_signals_wins() {
        create_corpus("t_both", "brindle_vector_cosine_ops");
        let rows = hybrid(
            "SELECT id, score, vector_rank, text_rank FROM brindle_hybrid(
                 't_both', 'id', 'embedding', 'tsv',
                 'wireless headphones', '[1,0,0]'::brindle_vector, config => 'english')",
        );
        assert!(!rows.is_empty(), "expected fused rows");
        assert_eq!(
            rows[0].id,
            1,
            "the both-signal row must rank first: {:?}",
            rows.iter().map(|r| r.id).collect::<Vec<_>>()
        );

        let map = by_id(&rows);
        let a = map.get(&1).expect("A present");
        let b = map.get(&2).expect("B present (it is the vector #1)");
        let c = map.get(&3).expect("C present");

        // A (both) beats B (vector-only, and the single nearest vector) and C
        // (text-strong) — the whole point of fusing.
        assert!(
            a.score > b.score,
            "A {} should beat vector-#1 B {}",
            a.score,
            b.score
        );
        assert!(
            a.score > c.score,
            "A {} should beat text-strong C {}",
            a.score,
            c.score
        );

        // A hits on both signals; B hits on the vector only.
        assert!(
            a.vector_rank.is_some() && a.text_rank.is_some(),
            "A hits both"
        );
        assert_eq!(b.vector_rank, Some(1), "B is the nearest vector");
        assert_eq!(b.text_rank, None, "B matches no lexical query");
        assert!(c.text_rank.is_some(), "C hits the lexical query");
    }

    /// An empty lexical query degrades to the vector ranking alone: the nearest
    /// vector (B) leads, and nothing carries a text rank.
    #[pg_test]
    fn empty_query_text_is_vector_only() {
        create_corpus("t_vec_only", "brindle_vector_cosine_ops");
        let rows = hybrid(
            "SELECT id, score, vector_rank, text_rank FROM brindle_hybrid(
                 't_vec_only', 'id', 'embedding', 'tsv',
                 '', '[1,0,0]'::brindle_vector, config => 'english')",
        );
        assert_eq!(rows[0].id, 2, "nearest vector leads a vector-only fusion");
        assert!(
            rows.iter().all(|r| r.text_rank.is_none()),
            "no row may carry a text rank when the lexical query is empty"
        );
        assert!(
            rows.iter().all(|r| r.vector_rank.is_some()),
            "every row came from the vector side"
        );
    }

    /// A lexical query that matches nothing also degrades to vector-only,
    /// without error.
    #[pg_test]
    fn no_lexical_match_degrades_to_vector() {
        create_corpus("t_nomatch", "brindle_vector_cosine_ops");
        let rows = hybrid(
            "SELECT id, score, vector_rank, text_rank FROM brindle_hybrid(
                 't_nomatch', 'id', 'embedding', 'tsv',
                 'quicksilver zeppelin', '[1,0,0]'::brindle_vector, config => 'english')",
        );
        assert_eq!(
            rows[0].id, 2,
            "vector #1 leads when the words match nothing"
        );
        assert!(rows.iter().all(|r| r.text_rank.is_none()));
    }

    /// An empty index (and so no vector matches, with no lexical query) fuses to
    /// nothing rather than erroring.
    #[pg_test]
    fn empty_index_returns_no_rows() {
        Spi::run(
            "CREATE TABLE t_empty (id int PRIMARY KEY, tsv tsvector, embedding brindle_vector)",
        )
        .expect("create");
        Spi::run("CREATE INDEX t_empty_idx ON t_empty USING brindle (embedding)").expect("index");
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM brindle_hybrid(
                 't_empty', 'id', 'embedding', 'tsv',
                 'anything', '[1,0,0]'::brindle_vector, config => 'english')",
        )
        .expect("spi")
        .expect("non-null");
        assert_eq!(count, 0);
    }

    /// The lexical side also works against a raw `text` column, lexing it in the
    /// query rather than reading a stored tsvector — the canonical property must
    /// hold there too.
    #[pg_test]
    fn fuses_against_a_plain_text_column() {
        create_corpus("t_text", "brindle_vector_cosine_ops");
        let rows = hybrid(
            "SELECT id, score, vector_rank, text_rank FROM brindle_hybrid(
                 't_text', 'id', 'embedding', 'body',
                 'wireless headphones', '[1,0,0]'::brindle_vector, config => 'english')",
        );
        assert_eq!(
            rows[0].id, 1,
            "the both-signal row wins over a text column too"
        );
        let map = by_id(&rows);
        assert!(
            map.get(&1).unwrap().text_rank.is_some(),
            "A matched the text column"
        );
    }

    /// The distance operator is read from the index's operator class, not
    /// hardcoded: an L2 index fuses correctly through the `<->` operator.
    #[pg_test]
    fn infers_the_metric_from_an_l2_index() {
        create_corpus("t_l2", "brindle_vector_l2_ops");
        let rows = hybrid(
            "SELECT id, score, vector_rank, text_rank FROM brindle_hybrid(
                 't_l2', 'id', 'embedding', 'tsv',
                 'wireless headphones', '[1,0,0]'::brindle_vector, config => 'english')",
        );
        // The nearest by L2 to [1,0,0] is still id 2 ([1,0,0]); id 1 is second
        // and, hitting both signals, still fuses to the top.
        assert_eq!(
            rows[0].id,
            1,
            "both-signal row wins under an L2 index: {:?}",
            rows.iter()
                .map(|r| (r.id, r.vector_rank))
                .collect::<Vec<_>>()
        );
        assert_eq!(by_id(&rows).get(&2).unwrap().vector_rank, Some(1));
    }

    /// `k` bounds the number of fused rows returned.
    #[pg_test]
    fn k_limits_the_result_count() {
        create_corpus("t_k", "brindle_vector_cosine_ops");
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM brindle_hybrid(
                 't_k', 'id', 'embedding', 'tsv',
                 'wireless headphones', '[1,0,0]'::brindle_vector, k => 3, config => 'english')",
        )
        .expect("spi")
        .expect("non-null");
        assert_eq!(count, 3);
    }

    #[pg_test(error = "brindle: relation \"nope\" does not exist")]
    fn rejects_an_unknown_relation() {
        Spi::run(
            "SELECT * FROM brindle_hybrid(
                 'nope', 'id', 'embedding', 'tsv', 'x', '[1,0,0]'::brindle_vector)",
        )
        .expect("call");
    }

    #[pg_test(
        error = "brindle: hybrid search needs a brindle index on vector column \"embedding\" (its operator class selects the distance metric)"
    )]
    fn requires_a_brindle_index() {
        Spi::run(
            "CREATE TABLE t_noidx (id int PRIMARY KEY, tsv tsvector, embedding brindle_vector)",
        )
        .expect("create");
        Spi::run(
            "SELECT * FROM brindle_hybrid(
                 't_noidx', 'id', 'embedding', 'tsv', 'x', '[1,0,0]'::brindle_vector)",
        )
        .expect("call");
    }

    #[pg_test(
        error = "brindle: id column \"id\" is text, but must be an integer type (smallint, integer, or bigint)"
    )]
    fn rejects_a_non_integer_id() {
        Spi::run(
            "CREATE TABLE t_txtid (id text PRIMARY KEY, tsv tsvector, embedding brindle_vector)",
        )
        .expect("create");
        Spi::run("CREATE INDEX t_txtid_idx ON t_txtid USING brindle (embedding)").expect("index");
        Spi::run(
            "SELECT * FROM brindle_hybrid(
                 't_txtid', 'id', 'embedding', 'tsv', 'x', '[1,0,0]'::brindle_vector)",
        )
        .expect("call");
    }

    #[pg_test(
        error = "brindle: text column \"embedding\" is brindle_vector, but must be tsvector or a text type (text, varchar, char)"
    )]
    fn rejects_an_unsearchable_text_column() {
        create_corpus("t_badtext", "brindle_vector_cosine_ops");
        Spi::run(
            "SELECT * FROM brindle_hybrid(
                 't_badtext', 'id', 'embedding', 'embedding', 'x', '[1,0,0]'::brindle_vector)",
        )
        .expect("call");
    }

    /// The lexical side also works with the *database default* text-search
    /// config (the `config => NULL` branch, which uses the single-argument
    /// `websearch_to_tsquery`/`to_tsvector` forms).
    #[pg_test]
    fn fuses_with_the_default_text_config() {
        create_corpus("t_defcfg", "brindle_vector_cosine_ops");
        Spi::run("SET LOCAL default_text_search_config = 'english'").expect("set config");
        let rows = hybrid(
            "SELECT id, score, vector_rank, text_rank FROM brindle_hybrid(
                 't_defcfg', 'id', 'embedding', 'tsv',
                 'wireless headphones', '[1,0,0]'::brindle_vector)",
        );
        assert_eq!(
            rows[0].id, 1,
            "both-signal row wins under the default config too"
        );
        assert!(
            by_id(&rows).get(&1).unwrap().text_rank.is_some(),
            "A matched via the default config"
        );
    }

    /// A lexical hit that falls outside the (narrow) vector depth window comes
    /// back with a text rank but no vector rank — the explainability combination
    /// the wider corpus never produces.
    #[pg_test]
    fn a_text_only_hit_has_no_vector_rank() {
        create_corpus("t_window", "brindle_vector_cosine_ops");
        // Depth 1: only the single nearest vector (id 2) is in the vector list.
        let rows = hybrid(
            "SELECT id, score, vector_rank, text_rank FROM brindle_hybrid(
                 't_window', 'id', 'embedding', 'tsv',
                 'wireless headphones', '[1,0,0]'::brindle_vector, n => 1, config => 'english')",
        );
        let map = by_id(&rows);
        let a = map.get(&1).expect("the top text hit is present");
        assert!(
            a.vector_rank.is_none(),
            "A is outside the depth-1 vector window"
        );
        assert!(a.text_rank.is_some(), "A is a lexical hit");
        let b = map.get(&2).expect("the nearest vector is present");
        assert_eq!(b.vector_rank, Some(1));
        assert_eq!(b.text_rank, None);
    }

    /// Tied fused scores order by *signed* id: the sign-flip key mapping must put
    /// a negative id before a positive one. Reinterpreting the id as `u64`
    /// without the flip would reverse this (a negative id becomes a huge `u64`).
    #[pg_test]
    fn tied_scores_order_by_signed_id() {
        Spi::run("CREATE TABLE t_neg (id bigint PRIMARY KEY, body text, embedding brindle_vector)")
            .expect("create");
        // id -100 matches the vector only; id 100 matches the text only (its
        // embedding is NULL, so it is not in the graph). Each is rank 1 in
        // exactly one list, so the two tie on fused score.
        Spi::run(
            "INSERT INTO t_neg (id, body, embedding) VALUES
               (-100, 'garden hose reel',   '[1,0,0]'),
               (100,  'wireless headphones', NULL)",
        )
        .expect("insert");
        Spi::run(
            "CREATE INDEX t_neg_idx ON t_neg USING brindle (embedding brindle_vector_cosine_ops)",
        )
        .expect("index");
        let rows = hybrid(
            "SELECT id, score, vector_rank, text_rank FROM brindle_hybrid(
                 't_neg', 'id', 'embedding', 'body',
                 'wireless headphones', '[1,0,0]'::brindle_vector, config => 'english')",
        );
        assert_eq!(
            rows.iter().map(|r| r.id).collect::<Vec<_>>(),
            vec![-100, 100],
            "tied rows must order by signed id, negative first"
        );
    }

    /// A caller-supplied `rrf_k` that fusion rejects surfaces as a clean error at
    /// the boundary rather than a silently inverted (or empty) ranking.
    #[pg_test(error = "brindle: invalid RRF k: -1 (must be finite and >= 0)")]
    fn rejects_an_invalid_rrf_k() {
        create_corpus("t_badrrfk", "brindle_vector_cosine_ops");
        Spi::run(
            "SELECT * FROM brindle_hybrid(
                 't_badrrfk', 'id', 'embedding', 'tsv',
                 'wireless headphones', '[1,0,0]'::brindle_vector, rrf_k => -1, config => 'english')",
        )
        .expect("call");
    }

    /// `k < 1` is a clear error, not a query that quietly returns nothing.
    #[pg_test(error = "brindle: k must be at least 1, got 0")]
    fn rejects_a_non_positive_k() {
        create_corpus("t_zerok", "brindle_vector_cosine_ops");
        Spi::run(
            "SELECT * FROM brindle_hybrid(
                 't_zerok', 'id', 'embedding', 'tsv',
                 'wireless headphones', '[1,0,0]'::brindle_vector, k => 0, config => 'english')",
        )
        .expect("call");
    }

    /// Two brindle indexes with *different* metrics on the same column make the
    /// metric ambiguous — the function refuses rather than picking one silently.
    /// Asserted through a PL/pgSQL handler so the check is robust to the schema
    /// and ordering in the (schema-qualified) operator list.
    #[pg_test]
    fn rejects_ambiguous_metric_indexes() {
        Spi::run(
            "CREATE TABLE t_ambig (id int PRIMARY KEY, tsv tsvector, embedding brindle_vector)",
        )
        .expect("create");
        Spi::run("INSERT INTO t_ambig (id, embedding) VALUES (1, '[1,0,0]'), (2, '[0,1,0]')")
            .expect("insert");
        Spi::run(
            "CREATE INDEX t_ambig_cos ON t_ambig USING brindle (embedding brindle_vector_cosine_ops)",
        )
        .expect("cosine index");
        Spi::run(
            "CREATE INDEX t_ambig_l2 ON t_ambig USING brindle (embedding brindle_vector_l2_ops)",
        )
        .expect("l2 index");
        Spi::run(
            "DO $$ BEGIN
                 PERFORM * FROM brindle_hybrid(
                     't_ambig', 'id', 'embedding', 'tsv',
                     'x', '[1,0,0]'::brindle_vector);
                 RAISE EXCEPTION 'expected an ambiguous-metric error, got none';
             EXCEPTION WHEN OTHERS THEN
                 IF SQLERRM NOT LIKE '%different metrics%' THEN RAISE; END IF;
             END $$;",
        )
        .expect("ambiguous metric must raise");
    }
}
