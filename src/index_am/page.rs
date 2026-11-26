//! The durable page byte-layout for a Brindle index — the on-disk format from
//! `docs/STORAGE.md` §3–§5.
//!
//! Pure byte (de)serialization: the 8-byte special space, the version-4
//! metapage, element tuples and neighbor chunks, plus node addressing and the
//! sizing limits. There is no buffer manager and no `unsafe` here — the index
//! access method wires these onto real buffer pages in a following change; this
//! module is just the plain, unit-testable format. Little-endian throughout, to
//! match every other layer of the interim format and the graph codec.

use std::mem::size_of;

use pgrx::pg_sys;

use super::storage::TidPair;
use crate::filter::AttrValue;
use crate::vector::Metric;

/// `'BR'` — marks a page as Brindle's, the way pgvector tags its pages, so a
/// read of the wrong relation or fork fails loudly instead of misparsing.
pub(crate) const PAGE_ID: u16 = 0x4252;

// Page kinds, stored in the special space. Never 0, so a zeroed (newly extended
// but not yet initialized) page cannot decode as a valid kind.
pub(crate) const PAGE_KIND_META: u16 = 1;
pub(crate) const PAGE_KIND_ELEMENT: u16 = 2;
pub(crate) const PAGE_KIND_NEIGHBOR: u16 = 3;

// Item kinds, stored as the first byte of each line-pointer item. Never 0.
pub(crate) const ITEM_KIND_ELEMENT: u8 = 1;
pub(crate) const ITEM_KIND_NEIGHBOR: u8 = 2;

// Attribute-slot tags (first byte of a 16-byte slot).
const ATTR_NULL: u8 = 0;
const ATTR_INT: u8 = 1;
const ATTR_FLOAT: u8 = 2;

/// Mirrors `pg_sys::InvalidBlockNumber`: "no node" / "no free page" / "no entry
/// point". Spelled directly so the pure codec does not depend on the binding
/// exposing it as a constant.
pub(crate) const INVALID_BLOCK: u32 = u32::MAX;

/// Round `n` up to an 8-byte (`MAXALIGN`) boundary. The sizing in
/// `docs/STORAGE.md` assumes `MAXIMUM_ALIGNOF == 8`, which holds on every
/// platform Brindle targets.
#[inline]
pub(crate) const fn maxalign(n: usize) -> usize {
    (n + 7) & !7
}

// ---- node addressing (§4) ------------------------------------------------

/// A node's identity is the address of its element tuple, packed as
/// `block << 16 | offset`. There is no dense id space on disk.
#[inline]
pub(crate) fn pack_node(block: u32, offset: u16) -> u64 {
    ((block as u64) << 16) | offset as u64
}

/// Inverse of [`pack_node`].
#[inline]
pub(crate) fn unpack_node(node: u64) -> (u32, u16) {
    ((node >> 16) as u32, (node & 0xffff) as u16)
}

// ---- errors --------------------------------------------------------------

/// Why a page or item failed to decode. Converted to a Postgres `ERROR` at the
/// access-method boundary; an [`PageError::UnsupportedVersion`] is the one a
/// user acts on — it means `REINDEX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageError {
    BadMagic(u32),
    UnsupportedVersion(u32),
    BadPageId(u16),
    BadPageKind(u16),
    BadItemKind(u8),
    Corrupt(&'static str),
    Truncated,
}

impl std::fmt::Display for PageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PageError::BadMagic(m) => write!(f, "not a brindle index (magic {m:#010x})"),
            PageError::UnsupportedVersion(v) => write!(
                f,
                "brindle index is an older on-disk format (version {v}); REINDEX to rebuild it"
            ),
            PageError::BadPageId(id) => write!(f, "not a brindle page (page_id {id:#06x})"),
            PageError::BadPageKind(k) => write!(f, "unexpected brindle page kind {k}"),
            PageError::BadItemKind(k) => write!(f, "unexpected brindle item kind {k}"),
            PageError::Corrupt(what) => write!(f, "corrupt brindle page: {what}"),
            PageError::Truncated => write!(f, "brindle page item is truncated"),
        }
    }
}

impl std::error::Error for PageError {}

// ---- special space (§3.1) ------------------------------------------------

pub(crate) const SPECIAL_SIZE: usize = 8;

/// The 8 bytes every page ends with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Special {
    /// Free-list link; [`INVALID_BLOCK`] when the page is not free.
    pub next_free: u32,
    pub kind: u16,
    pub page_id: u16,
}

impl Special {
    /// A fresh, not-free special space for a page of `kind`.
    pub(crate) fn new(kind: u16) -> Self {
        Self {
            next_free: INVALID_BLOCK,
            kind,
            page_id: PAGE_ID,
        }
    }

    pub(crate) fn encode(&self) -> [u8; SPECIAL_SIZE] {
        let mut b = [0u8; SPECIAL_SIZE];
        b[0..4].copy_from_slice(&self.next_free.to_le_bytes());
        b[4..6].copy_from_slice(&self.kind.to_le_bytes());
        b[6..8].copy_from_slice(&self.page_id.to_le_bytes());
        b
    }

    pub(crate) fn decode(b: &[u8; SPECIAL_SIZE]) -> Self {
        Self {
            next_free: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            kind: u16::from_le_bytes([b[4], b[5]]),
            page_id: u16::from_le_bytes([b[6], b[7]]),
        }
    }

    /// Validate page identity, then that the kind is the one expected — `page_id`
    /// first, so a page from the wrong relation fails as [`PageError::BadPageId`]
    /// rather than a confusing kind mismatch.
    pub(crate) fn check(&self, expected_kind: u16) -> Result<(), PageError> {
        if self.page_id != PAGE_ID {
            return Err(PageError::BadPageId(self.page_id));
        }
        if self.kind != expected_kind {
            return Err(PageError::BadPageKind(self.kind));
        }
        Ok(())
    }
}

// ---- metapage (§3.2) -----------------------------------------------------

const META_MAGIC: u32 = 0x4252_4E44; // "BRND"
/// Version 4: the first paged format. The interim blob format reached 3, and a
/// paged reader claiming 2 or 3 would misparse a blob metapage — so the number
/// moves with either format and only one layout ever holds a given value.
pub(crate) const META_VERSION: u32 = 4;
pub(crate) const META_SIZE: usize = 80;

/// The version-4 metapage, written into the page content area (offset 24, right
/// after the standard header) so `magic` and `version` land at the same page
/// offsets the interim format and pgvector use — which is what lets the version
/// check reject an older index cleanly rather than misparse it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Meta {
    pub metric: Metric,
    pub flags: u8,
    /// Fixed by the first indexed vector; 0 while the index is empty.
    pub dim: u32,
    pub m: u32,
    pub ef_construction: u32,
    pub gamma: f32,
    /// Entry point; `entry_block == INVALID_BLOCK` while the graph is empty.
    pub entry_block: u32,
    pub entry_offset: u16,
    /// The entry point's layer = the graph's top layer. No separate `max_layer`
    /// exists: the two could only ever disagree.
    pub entry_level: u16,
    pub element_insert_page: u32,
    pub neighbor_insert_page: u32,
    pub free_head: u32,
    /// Planner/compaction hints — never consulted for a query's result (§3.2).
    pub node_count: u64,
    pub deleted_count: u64,
    pub seed: u64,
    /// Live PRNG state, advanced by each insert so level assignment survives a
    /// restart.
    pub rng_state: u64,
}

impl Meta {
    /// Whether the graph has no entry point yet.
    pub(crate) fn is_empty(&self) -> bool {
        self.entry_block == INVALID_BLOCK
    }

    pub(crate) fn encode(&self) -> [u8; META_SIZE] {
        let mut b = [0u8; META_SIZE];
        b[0..4].copy_from_slice(&META_MAGIC.to_le_bytes());
        b[4..8].copy_from_slice(&META_VERSION.to_le_bytes());
        b[8] = self.metric.code();
        b[9] = self.flags;
        // [10..12] reserved, zero.
        b[12..16].copy_from_slice(&self.dim.to_le_bytes());
        b[16..20].copy_from_slice(&self.m.to_le_bytes());
        b[20..24].copy_from_slice(&self.ef_construction.to_le_bytes());
        b[24..28].copy_from_slice(&self.gamma.to_bits().to_le_bytes());
        b[28..32].copy_from_slice(&self.entry_block.to_le_bytes());
        b[32..34].copy_from_slice(&self.entry_offset.to_le_bytes());
        b[34..36].copy_from_slice(&self.entry_level.to_le_bytes());
        b[36..40].copy_from_slice(&self.element_insert_page.to_le_bytes());
        b[40..44].copy_from_slice(&self.neighbor_insert_page.to_le_bytes());
        b[44..48].copy_from_slice(&self.free_head.to_le_bytes());
        b[48..56].copy_from_slice(&self.node_count.to_le_bytes());
        b[56..64].copy_from_slice(&self.deleted_count.to_le_bytes());
        b[64..72].copy_from_slice(&self.seed.to_le_bytes());
        b[72..80].copy_from_slice(&self.rng_state.to_le_bytes());
        b
    }

    /// Decode and validate a metapage: magic, then version, then the metric code
    /// — the order `docs/STORAGE.md` invariant I2b requires so an older format
    /// yields a clean "REINDEX" error rather than a misparse.
    pub(crate) fn decode(b: &[u8; META_SIZE]) -> Result<Self, PageError> {
        let magic = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        if magic != META_MAGIC {
            return Err(PageError::BadMagic(magic));
        }
        let version = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        if version != META_VERSION {
            return Err(PageError::UnsupportedVersion(version));
        }
        let metric = Metric::from_code(b[8]).ok_or(PageError::Corrupt("unknown metric code"))?;
        Ok(Self {
            metric,
            flags: b[9],
            dim: u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
            m: u32::from_le_bytes([b[16], b[17], b[18], b[19]]),
            ef_construction: u32::from_le_bytes([b[20], b[21], b[22], b[23]]),
            gamma: f32::from_bits(u32::from_le_bytes([b[24], b[25], b[26], b[27]])),
            entry_block: u32::from_le_bytes([b[28], b[29], b[30], b[31]]),
            entry_offset: u16::from_le_bytes([b[32], b[33]]),
            entry_level: u16::from_le_bytes([b[34], b[35]]),
            element_insert_page: u32::from_le_bytes([b[36], b[37], b[38], b[39]]),
            neighbor_insert_page: u32::from_le_bytes([b[40], b[41], b[42], b[43]]),
            free_head: u32::from_le_bytes([b[44], b[45], b[46], b[47]]),
            node_count: u64::from_le_bytes([
                b[48], b[49], b[50], b[51], b[52], b[53], b[54], b[55],
            ]),
            deleted_count: u64::from_le_bytes([
                b[56], b[57], b[58], b[59], b[60], b[61], b[62], b[63],
            ]),
            seed: u64::from_le_bytes([b[64], b[65], b[66], b[67], b[68], b[69], b[70], b[71]]),
            rng_state: u64::from_le_bytes([b[72], b[73], b[74], b[75], b[76], b[77], b[78], b[79]]),
        })
    }
}

// ---- element tuple (§3.3) ------------------------------------------------

pub(crate) const ELEMENT_HEADER_LEN: usize = 16;
pub(crate) const NEIGHBOR_PTR_LEN: usize = 6;
pub(crate) const ATTR_SLOT_LEN: usize = 16;
const ELEMENT_FLAG_DELETED: u8 = 0b0000_0001;

/// Byte length of an element tuple for a node at `level` with `attr_count`
/// attributes and `dim` dimensions (§3.3): header and neighbor pointers padded
/// to an 8-byte boundary, then the fixed-stride attribute row, then the vector.
pub(crate) fn element_len(level: usize, attr_count: usize, dim: usize) -> usize {
    element_attrs_offset(level) + ATTR_SLOT_LEN * attr_count + 4 * dim
}

/// Offset of the attribute row within an element tuple at `level`.
#[inline]
fn element_attrs_offset(level: usize) -> usize {
    maxalign(ELEMENT_HEADER_LEN + NEIGHBOR_PTR_LEN * (level + 1))
}

fn write_attr_slot(slot: &mut [u8], v: &AttrValue) {
    slot[..ATTR_SLOT_LEN].fill(0);
    match v {
        AttrValue::Null => slot[0] = ATTR_NULL,
        AttrValue::Int(i) => {
            slot[0] = ATTR_INT;
            slot[8..16].copy_from_slice(&i.to_le_bytes());
        }
        AttrValue::Float(f) => {
            slot[0] = ATTR_FLOAT;
            slot[8..16].copy_from_slice(&f.to_bits().to_le_bytes());
        }
    }
}

fn read_attr_slot(slot: &[u8]) -> AttrValue {
    let payload = u64::from_le_bytes([
        slot[8], slot[9], slot[10], slot[11], slot[12], slot[13], slot[14], slot[15],
    ]);
    match slot[0] {
        ATTR_INT => AttrValue::Int(payload as i64),
        ATTR_FLOAT => AttrValue::Float(f64::from_bits(payload)),
        // ATTR_NULL and any unknown tag read as Null: predicate evaluation stays
        // a subscript, and a missing/absent value matches no atom anyway.
        _ => AttrValue::Null,
    }
}

/// Serialize an element tuple onto the end of `out` (§3.3). `neighbor_ptrs` holds
/// `level + 1` entries, layer 0 first; the vector's length fixes `dim`.
pub(crate) fn write_element(
    out: &mut Vec<u8>,
    deleted: bool,
    level: usize,
    heap: TidPair,
    neighbor_ptrs: &[TidPair],
    attrs: &[AttrValue],
    vector: &[f32],
) {
    debug_assert_eq!(neighbor_ptrs.len(), level + 1);
    let total = element_len(level, attrs.len(), vector.len());
    let base = out.len();
    out.resize(base + total, 0);
    let buf = &mut out[base..base + total];

    buf[0] = ITEM_KIND_ELEMENT;
    buf[1] = if deleted { ELEMENT_FLAG_DELETED } else { 0 };
    buf[2..4].copy_from_slice(&(level as u16).to_le_bytes());
    buf[4..6].copy_from_slice(&(vector.len() as u16).to_le_bytes());
    buf[6..8].copy_from_slice(&(attrs.len() as u16).to_le_bytes());
    buf[8..12].copy_from_slice(&heap.0.to_le_bytes());
    buf[12..14].copy_from_slice(&heap.1.to_le_bytes());
    // [14..16] reserved, zero.

    let mut off = ELEMENT_HEADER_LEN;
    for &(block, offset) in neighbor_ptrs {
        buf[off..off + 4].copy_from_slice(&block.to_le_bytes());
        buf[off + 4..off + 6].copy_from_slice(&offset.to_le_bytes());
        off += NEIGHBOR_PTR_LEN;
    }

    let mut aoff = element_attrs_offset(level);
    for a in attrs {
        write_attr_slot(&mut buf[aoff..aoff + ATTR_SLOT_LEN], a);
        aoff += ATTR_SLOT_LEN;
    }

    for (i, &c) in vector.iter().enumerate() {
        let voff = aoff + 4 * i;
        buf[voff..voff + 4].copy_from_slice(&c.to_le_bytes());
    }
}

/// A read view over an element tuple's bytes (§3.3). Accessors are subscripts,
/// not a parse, so predicate-aware traversal rejects a non-matching node cheaply.
#[derive(Clone, Copy)]
pub(crate) struct ElementRef<'a> {
    bytes: &'a [u8],
}

impl<'a> ElementRef<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Result<Self, PageError> {
        if bytes.len() < ELEMENT_HEADER_LEN {
            return Err(PageError::Truncated);
        }
        if bytes[0] != ITEM_KIND_ELEMENT {
            return Err(PageError::BadItemKind(bytes[0]));
        }
        Ok(Self { bytes })
    }

    #[inline]
    fn u16_at(&self, off: usize) -> u16 {
        u16::from_le_bytes([self.bytes[off], self.bytes[off + 1]])
    }

    #[inline]
    fn u32_at(&self, off: usize) -> u32 {
        u32::from_le_bytes([
            self.bytes[off],
            self.bytes[off + 1],
            self.bytes[off + 2],
            self.bytes[off + 3],
        ])
    }

    pub(crate) fn is_deleted(&self) -> bool {
        self.bytes[1] & ELEMENT_FLAG_DELETED != 0
    }
    pub(crate) fn level(&self) -> usize {
        self.u16_at(2) as usize
    }
    pub(crate) fn dim(&self) -> usize {
        self.u16_at(4) as usize
    }
    pub(crate) fn attr_count(&self) -> usize {
        self.u16_at(6) as usize
    }
    pub(crate) fn heap_tid(&self) -> TidPair {
        (self.u32_at(8), self.u16_at(12))
    }
    /// Address of this node's neighbor chunk for `layer`.
    pub(crate) fn neighbor_ptr(&self, layer: usize) -> TidPair {
        let off = ELEMENT_HEADER_LEN + NEIGHBOR_PTR_LEN * layer;
        (self.u32_at(off), self.u16_at(off + 4))
    }
    pub(crate) fn attr(&self, i: usize) -> AttrValue {
        let off = element_attrs_offset(self.level()) + ATTR_SLOT_LEN * i;
        read_attr_slot(&self.bytes[off..off + ATTR_SLOT_LEN])
    }
    pub(crate) fn read_attrs(&self, out: &mut Vec<AttrValue>) {
        out.clear();
        for i in 0..self.attr_count() {
            out.push(self.attr(i));
        }
    }
    fn vector_offset(&self) -> usize {
        element_attrs_offset(self.level()) + ATTR_SLOT_LEN * self.attr_count()
    }
    /// The vector's raw little-endian bytes, 8-byte aligned within the tuple.
    pub(crate) fn vector_bytes(&self) -> &'a [u8] {
        let base = self.vector_offset();
        &self.bytes[base..base + 4 * self.dim()]
    }
    pub(crate) fn read_vector(&self, out: &mut Vec<f32>) {
        out.clear();
        let base = self.vector_offset();
        for i in 0..self.dim() {
            let o = base + 4 * i;
            out.push(f32::from_le_bytes([
                self.bytes[o],
                self.bytes[o + 1],
                self.bytes[o + 2],
                self.bytes[o + 3],
            ]));
        }
    }
}

// ---- neighbor chunk (§3.4) -----------------------------------------------

pub(crate) const NEIGHBOR_HEADER_LEN: usize = 8;

/// Byte length of a neighbor chunk holding `capacity` slots (§3.4).
pub(crate) fn neighbor_chunk_len(capacity: usize) -> usize {
    NEIGHBOR_HEADER_LEN + NEIGHBOR_PTR_LEN * capacity
}

/// Serialize a neighbor chunk onto the end of `out` with `capacity` slots — the
/// first `entries.len()` filled, the rest left zero (§3.4). The chunk is written
/// at full capacity and never grows, so a later back-edge is a `count += 1` plus
/// one 6-byte write inside the page.
pub(crate) fn write_neighbor_chunk(
    out: &mut Vec<u8>,
    layer: usize,
    capacity: usize,
    entries: &[TidPair],
) {
    debug_assert!(entries.len() <= capacity);
    let total = neighbor_chunk_len(capacity);
    let base = out.len();
    out.resize(base + total, 0);
    let buf = &mut out[base..base + total];
    buf[0] = ITEM_KIND_NEIGHBOR;
    // [1] flags, reserved zero.
    buf[2..4].copy_from_slice(&(layer as u16).to_le_bytes());
    buf[4..6].copy_from_slice(&(capacity as u16).to_le_bytes());
    buf[6..8].copy_from_slice(&(entries.len() as u16).to_le_bytes());
    for (i, &(block, offset)) in entries.iter().enumerate() {
        let off = NEIGHBOR_HEADER_LEN + NEIGHBOR_PTR_LEN * i;
        buf[off..off + 4].copy_from_slice(&block.to_le_bytes());
        buf[off + 4..off + 6].copy_from_slice(&offset.to_le_bytes());
    }
}

/// A read view over a neighbor chunk's bytes (§3.4).
#[derive(Clone, Copy)]
pub(crate) struct NeighborChunkRef<'a> {
    bytes: &'a [u8],
}

impl<'a> NeighborChunkRef<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Result<Self, PageError> {
        if bytes.len() < NEIGHBOR_HEADER_LEN {
            return Err(PageError::Truncated);
        }
        if bytes[0] != ITEM_KIND_NEIGHBOR {
            return Err(PageError::BadItemKind(bytes[0]));
        }
        Ok(Self { bytes })
    }
    pub(crate) fn layer(&self) -> usize {
        u16::from_le_bytes([self.bytes[2], self.bytes[3]]) as usize
    }
    pub(crate) fn capacity(&self) -> usize {
        u16::from_le_bytes([self.bytes[4], self.bytes[5]]) as usize
    }
    pub(crate) fn count(&self) -> usize {
        u16::from_le_bytes([self.bytes[6], self.bytes[7]]) as usize
    }
    pub(crate) fn entry(&self, i: usize) -> TidPair {
        let off = NEIGHBOR_HEADER_LEN + NEIGHBOR_PTR_LEN * i;
        (
            u32::from_le_bytes([
                self.bytes[off],
                self.bytes[off + 1],
                self.bytes[off + 2],
                self.bytes[off + 3],
            ]),
            u16::from_le_bytes([self.bytes[off + 4], self.bytes[off + 5]]),
        )
    }
    /// Replace `out` with the live neighbors, packed as [`pack_node`] refs.
    pub(crate) fn read_into(&self, out: &mut Vec<u64>) {
        out.clear();
        for i in 0..self.count() {
            let (block, offset) = self.entry(i);
            out.push(pack_node(block, offset));
        }
    }
}

/// Set a live-slot count in a neighbor chunk in place — the back-edge primitive
/// (§7.2 step 4), paired with [`neighbor_set_entry`].
pub(crate) fn neighbor_set_count(bytes: &mut [u8], count: usize) {
    bytes[6..8].copy_from_slice(&(count as u16).to_le_bytes());
}

/// Write one neighbor entry in place at slot `i`.
pub(crate) fn neighbor_set_entry(bytes: &mut [u8], i: usize, entry: TidPair) {
    let off = NEIGHBOR_HEADER_LEN + NEIGHBOR_PTR_LEN * i;
    bytes[off..off + 4].copy_from_slice(&entry.0.to_le_bytes());
    bytes[off + 4..off + 6].copy_from_slice(&entry.1.to_le_bytes());
}

// ---- sizing and limits (§5) ----------------------------------------------

/// Largest dimension count indexing accepts. The `brindle_vector` type still
/// takes up to 16000; this cap applies to the index, where a tuple must fit one
/// page.
pub(crate) const MAX_INDEXABLE_DIM: usize = 2000;

/// Largest layer-0 degree (`2·m·γ`) a build may request. Admits `m = 128, γ = 4`
/// and `m = 64, γ = 8` — far past any useful setting — while keeping a neighbor
/// chunk on one page.
pub(crate) const MAX_LAYER0_DEGREE: usize = 1024;

/// Largest single item a page can hold: `BLCKSZ` minus the page header, the
/// special space and one line pointer, rounded down to `MAXALIGN` (§5).
pub(crate) fn max_item_len() -> usize {
    let usable = pg_sys::BLCKSZ as usize
        - size_of::<pg_sys::PageHeaderData>()
        - SPECIAL_SIZE
        - size_of::<pg_sys::ItemIdData>();
    usable & !7
}

/// Whether an element tuple for these dimensions fits one page (§5, invariant I7).
pub(crate) fn element_fits(level: usize, attr_count: usize, dim: usize) -> bool {
    maxalign(element_len(level, attr_count, dim)) <= max_item_len()
}

/// Whether a neighbor chunk of `capacity` slots fits one page (§5, invariant I7).
pub(crate) fn neighbor_chunk_fits(capacity: usize) -> bool {
    maxalign(neighbor_chunk_len(capacity)) <= max_item_len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_ref_round_trips() {
        for (block, offset) in [(0u32, 0u16), (1, 2), (123_456, 7), (INVALID_BLOCK, 0xffff)] {
            assert_eq!(unpack_node(pack_node(block, offset)), (block, offset));
        }
    }

    #[test]
    fn special_round_trips_and_checks() {
        let s = Special::new(PAGE_KIND_ELEMENT);
        let got = Special::decode(&s.encode());
        assert_eq!(got, s);
        assert_eq!(got.page_id, PAGE_ID);
        assert_eq!(got.next_free, INVALID_BLOCK);
        assert!(got.check(PAGE_KIND_ELEMENT).is_ok());
        assert_eq!(
            got.check(PAGE_KIND_NEIGHBOR),
            Err(PageError::BadPageKind(PAGE_KIND_ELEMENT))
        );
        let mut bytes = s.encode();
        bytes[6..8].copy_from_slice(&0u16.to_le_bytes()); // corrupt page_id
        assert_eq!(
            Special::decode(&bytes).check(PAGE_KIND_ELEMENT),
            Err(PageError::BadPageId(0))
        );
    }

    fn sample_meta() -> Meta {
        Meta {
            metric: Metric::Cosine,
            flags: 0,
            dim: 128,
            m: 16,
            ef_construction: 200,
            gamma: 1.5,
            entry_block: 42,
            entry_offset: 3,
            entry_level: 2,
            element_insert_page: 1,
            neighbor_insert_page: 7,
            free_head: INVALID_BLOCK,
            node_count: 9_000,
            deleted_count: 12,
            seed: 0x9E37_79B9_7F4A_7C15,
            rng_state: 0xDEAD_BEEF_CAFE_F00D,
        }
    }

    #[test]
    fn meta_round_trips_every_field() {
        let m = sample_meta();
        assert_eq!(Meta::decode(&m.encode()).unwrap(), m);
    }

    #[test]
    fn meta_offsets_are_fixed() {
        // magic at 24 / version at 28 of the *page* means 0 / 4 of the metapage
        // struct (which starts at the page content offset) — the agreement with
        // the interim format that makes the version check work.
        let b = sample_meta().encode();
        assert_eq!(&b[0..4], &0x4252_4E44u32.to_le_bytes());
        assert_eq!(&b[4..8], &4u32.to_le_bytes());
        assert_eq!(b.len(), 80);
    }

    #[test]
    fn meta_empty_is_detected() {
        let mut m = sample_meta();
        m.entry_block = INVALID_BLOCK;
        assert!(m.is_empty());
        assert!(!sample_meta().is_empty());
    }

    #[test]
    fn meta_rejects_bad_magic() {
        let mut b = sample_meta().encode();
        b[0] ^= 0xff;
        assert!(matches!(Meta::decode(&b), Err(PageError::BadMagic(_))));
    }

    #[test]
    fn meta_rejects_an_older_version() {
        // A version-3 (interim blob) index must get the REINDEX error, not a
        // misparse — and magic is checked before version.
        let mut b = sample_meta().encode();
        b[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(Meta::decode(&b), Err(PageError::UnsupportedVersion(3)));
    }

    #[test]
    fn element_round_trips_at_layer_zero() {
        let mut out = vec![0xAAu8; 3]; // leading bytes must be untouched
        let attrs = [AttrValue::Int(-5), AttrValue::Float(2.5), AttrValue::Null];
        let vector = [1.0f32, -2.0, 3.5, 4.25];
        write_element(&mut out, false, 0, (11, 2), &[(99, 1)], &attrs, &vector);
        assert_eq!(&out[..3], &[0xAA; 3]);

        let e = ElementRef::new(&out[3..]).unwrap();
        assert!(!e.is_deleted());
        assert_eq!(e.level(), 0);
        assert_eq!(e.dim(), 4);
        assert_eq!(e.attr_count(), 3);
        assert_eq!(e.heap_tid(), (11, 2));
        assert_eq!(e.neighbor_ptr(0), (99, 1));
        let mut got_attrs = Vec::new();
        e.read_attrs(&mut got_attrs);
        assert_eq!(got_attrs, attrs);
        let mut got_vec = Vec::new();
        e.read_vector(&mut got_vec);
        assert_eq!(got_vec, vector);
        assert_eq!(out.len() - 3, element_len(0, 3, 4));
    }

    #[test]
    fn element_round_trips_multi_layer_and_tombstone() {
        let mut out = Vec::new();
        let ptrs = [(1u32, 1u16), (2, 2), (3, 3)]; // level 2 => 3 pointers
        let vector = [0.5f32; 7];
        write_element(&mut out, true, 2, (500, 9), &ptrs, &[], &vector);
        let e = ElementRef::new(&out).unwrap();
        assert!(e.is_deleted());
        assert_eq!(e.level(), 2);
        assert_eq!(e.attr_count(), 0);
        for (layer, &p) in ptrs.iter().enumerate() {
            assert_eq!(e.neighbor_ptr(layer), p);
        }
        let mut v = Vec::new();
        e.read_vector(&mut v);
        assert_eq!(v, vector);
    }

    #[test]
    fn element_vector_is_eight_byte_aligned_within_the_tuple() {
        let mut out = Vec::new();
        write_element(
            &mut out,
            false,
            0,
            (1, 1),
            &[(1, 1)],
            &[AttrValue::Int(1)],
            &[1.0],
        );
        let e = ElementRef::new(&out).unwrap();
        // vector_offset is private; reconstruct it from the public shape.
        let voff = element_attrs_offset(e.level()) + ATTR_SLOT_LEN * e.attr_count();
        assert_eq!(voff % 8, 0);
    }

    #[test]
    fn bad_item_kind_is_rejected() {
        let mut out = Vec::new();
        write_element(&mut out, false, 0, (1, 1), &[(0, 0)], &[], &[1.0]);
        out[0] = 0; // a zeroed page byte must not read as an element
        assert_eq!(ElementRef::new(&out).err(), Some(PageError::BadItemKind(0)));
    }

    #[test]
    fn neighbor_chunk_round_trips_with_spare_capacity() {
        let entries = [(10u32, 1u16), (20, 2)];
        let mut out = Vec::new();
        write_neighbor_chunk(&mut out, 1, 5, &entries);
        assert_eq!(out.len(), neighbor_chunk_len(5));
        let c = NeighborChunkRef::new(&out).unwrap();
        assert_eq!(c.layer(), 1);
        assert_eq!(c.capacity(), 5);
        assert_eq!(c.count(), 2);
        assert_eq!(c.entry(0), (10, 1));
        assert_eq!(c.entry(1), (20, 2));
        let mut refs = Vec::new();
        c.read_into(&mut refs);
        assert_eq!(refs, vec![pack_node(10, 1), pack_node(20, 2)]);
    }

    #[test]
    fn neighbor_chunk_in_place_edits() {
        let mut out = Vec::new();
        write_neighbor_chunk(&mut out, 0, 4, &[(1, 1)]);
        // Add a back-edge in place: write slot 1, bump count.
        neighbor_set_entry(&mut out, 1, (2, 2));
        neighbor_set_count(&mut out, 2);
        let c = NeighborChunkRef::new(&out).unwrap();
        assert_eq!(c.count(), 2);
        assert_eq!(c.entry(1), (2, 2));
    }

    #[test]
    fn sizing_limits_match_the_spec() {
        // On the default 8 kB build, the largest item is 8152 bytes.
        if pg_sys::BLCKSZ == 8192 {
            assert_eq!(max_item_len(), 8152);
        }
        // 2000-dim vector at layer 0 with no attributes fits; 2048 would not.
        assert!(element_fits(0, 0, MAX_INDEXABLE_DIM));
        assert!(!element_fits(0, 0, 2048));
        // The max layer-0 degree fits a chunk; a chunk past the page does not.
        assert!(neighbor_chunk_fits(MAX_LAYER0_DEGREE));
        assert!(!neighbor_chunk_fits(max_item_len())); // 6 bytes/slot => far too big
    }
}
