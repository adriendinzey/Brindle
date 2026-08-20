//! Interim on-disk format for a brindle index: one versioned blob spanning
//! standard pages.
//!
//! Block 0 is a metapage (`magic`, `version`, blob length); blocks `1..N` carry
//! the blob split into per-page chunks. Chunks live in the page content area
//! directly after the standard header, with `pd_lower` marking the used bytes —
//! the same convention Postgres metapages use, so tools like `pageinspect` see
//! ordinary initialized pages.
//!
//! The blob itself is [`encode_index`] framing: the serialized graph
//! (see `Hnsw::to_bytes`) followed by the node-id → heap-TID table.
//!
//! TODO: replace this rebuild-everything blob with durable page-structured
//! storage — per-node pages read through the buffer manager on demand,
//! WAL-logged incremental updates, and vacuum integration.

use core::ffi::c_char;

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::hnsw::Hnsw;

/// Errors from decoding the blob payload framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadError {
    /// The payload ended before the encoding said it would.
    Truncated,
    /// Extra bytes follow the encoded payload.
    TrailingBytes,
}

impl std::fmt::Display for PayloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PayloadError::Truncated => write!(f, "index payload is truncated"),
            PayloadError::TrailingBytes => write!(f, "index payload has trailing bytes"),
        }
    }
}

impl std::error::Error for PayloadError {}

/// A heap tuple address as stored in the blob: `(block number, offset number)`.
/// Plain integers rather than `ItemPointerData` so the framing stays pure Rust
/// and unit-testable without a server.
pub type TidPair = (u32, u16);

/// Frame a graph and its node-id → TID table into one blob, serializing the
/// graph directly into the payload buffer so the build never holds a second
/// full copy of it. `tids[i]` is the heap address of graph node `i` (ids are
/// dense, in insertion order).
pub fn encode_index(hnsw: &Hnsw, tids: &[TidPair]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + hnsw.serialized_len_hint() + tids.len() * 6);
    let len_pos = out.len();
    out.extend_from_slice(&0u64.to_le_bytes()); // graph-length placeholder
    let graph_start = out.len();
    hnsw.to_bytes_into(&mut out);
    let graph_len = (out.len() - graph_start) as u64;
    out[len_pos..len_pos + 8].copy_from_slice(&graph_len.to_le_bytes());
    append_tid_table(&mut out, tids);
    out
}

/// Frame an already-serialized graph and the TID table into one blob. The
/// byte-based counterpart to [`encode_index`], kept for round-trip tests that
/// exercise the framing without building a real graph.
#[cfg(test)]
pub fn encode_index_payload(graph: &[u8], tids: &[TidPair]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + graph.len() + tids.len() * 6);
    out.extend_from_slice(&(graph.len() as u64).to_le_bytes());
    out.extend_from_slice(graph);
    append_tid_table(&mut out, tids);
    out
}

fn append_tid_table(out: &mut Vec<u8>, tids: &[TidPair]) {
    out.extend_from_slice(&(tids.len() as u64).to_le_bytes());
    for &(block, offset) in tids {
        out.extend_from_slice(&block.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }
}

/// Split a blob produced by [`encode_index`] back into the serialized graph
/// and the node-id → TID table.
pub fn decode_index_payload(blob: &[u8]) -> Result<(&[u8], Vec<TidPair>), PayloadError> {
    let (graph_len, rest) = read_u64(blob)?;
    let graph_len = usize::try_from(graph_len).map_err(|_| PayloadError::Truncated)?;
    if rest.len() < graph_len {
        return Err(PayloadError::Truncated);
    }
    let (graph, rest) = rest.split_at(graph_len);

    let (count, mut rest) = read_u64(rest)?;
    let count = usize::try_from(count).map_err(|_| PayloadError::Truncated)?;
    if rest.len() / 6 < count {
        return Err(PayloadError::Truncated);
    }
    let mut tids = Vec::with_capacity(count);
    for _ in 0..count {
        let block = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
        let offset = u16::from_le_bytes([rest[4], rest[5]]);
        tids.push((block, offset));
        rest = &rest[6..];
    }
    if !rest.is_empty() {
        return Err(PayloadError::TrailingBytes);
    }
    Ok((graph, tids))
}

fn read_u64(buf: &[u8]) -> Result<(u64, &[u8]), PayloadError> {
    if buf.len() < 8 {
        return Err(PayloadError::Truncated);
    }
    let (head, rest) = buf.split_at(8);
    let value = u64::from_le_bytes([
        head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7],
    ]);
    Ok((value, rest))
}

// --- page IO -------------------------------------------------------------

const META_MAGIC: u32 = 0x4252_4E44; // "BRND"
const STORAGE_VERSION: u32 = 1;

/// Metapage payload at block 0: magic, version, blob length — little-endian,
/// like every other layer of the blob format.
const META_SIZE: usize = 16;

fn encode_meta(blob_len: u64) -> [u8; META_SIZE] {
    let mut meta = [0u8; META_SIZE];
    meta[0..4].copy_from_slice(&META_MAGIC.to_le_bytes());
    meta[4..8].copy_from_slice(&STORAGE_VERSION.to_le_bytes());
    meta[8..16].copy_from_slice(&blob_len.to_le_bytes());
    meta
}

/// `(magic, version, blob_len)` from metapage contents.
fn decode_meta(bytes: &[u8; META_SIZE]) -> (u32, u32, u64) {
    (
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]),
    )
}

/// First usable byte in a page: right after the (maxaligned) standard header,
/// where the line-pointer array would normally start. We store no line
/// pointers; `pd_lower` marks the end of the chunk instead.
// SAFETY: MAXALIGN is a pure const computation (unsafe only by inheritance
// from its cfg-dependent constant).
const PAGE_CONTENTS_OFFSET: usize =
    unsafe { pg_sys::MAXALIGN(core::mem::size_of::<pg_sys::PageHeaderData>()) };

/// Usable bytes per data page.
const PAGE_CHUNK_CAPACITY: usize = pg_sys::BLCKSZ as usize - PAGE_CONTENTS_OFFSET;

/// Append one initialized page holding `contents` to `forknum`.
///
/// # Safety
/// `index` must be an open index relation this backend may extend; `contents`
/// must fit in [`PAGE_CHUNK_CAPACITY`].
unsafe fn append_page(index: pg_sys::Relation, forknum: pg_sys::ForkNumber::Type, contents: &[u8]) {
    if contents.len() > PAGE_CHUNK_CAPACITY {
        error!(
            "brindle: page chunk of {} bytes exceeds page capacity",
            contents.len()
        );
    }
    // SAFETY: P_NEW (InvalidBlockNumber) extends the fork by one page; the
    // relation is ours to extend (build holds an exclusive lock), and the
    // buffer stays pinned+locked until the copy below is done.
    let buffer = pg_sys::ReadBufferExtended(
        index,
        forknum,
        pg_sys::InvalidBlockNumber,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        core::ptr::null_mut(),
    );
    pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let page = pg_sys::BufferGetPage(buffer);
    pg_sys::PageInit(page, pg_sys::BLCKSZ as usize, 0);
    // SAFETY: PageInit zeroed a full BLCKSZ page, so offset..offset+len stays
    // in bounds per the capacity check above.
    core::ptr::copy_nonoverlapping(
        contents.as_ptr(),
        page.cast::<u8>().add(PAGE_CONTENTS_OFFSET),
        contents.len(),
    );
    (*page.cast::<pg_sys::PageHeaderData>()).pd_lower =
        (PAGE_CONTENTS_OFFSET + contents.len()) as u16;
    pg_sys::MarkBufferDirty(buffer);
    pg_sys::UnlockReleaseBuffer(buffer);
}

/// Write the whole index blob: a metapage at block 0 plus chunked data pages,
/// WAL-logging the result when the relation needs it.
///
/// # Safety
/// `index` must be an open, exclusively-locked index relation whose `forknum`
/// fork is empty — i.e. a freshly created relfilenode during an index build.
pub unsafe fn write_index_blob(
    index: pg_sys::Relation,
    blob: &[u8],
    forknum: pg_sys::ForkNumber::Type,
) {
    if pg_sys::RelationGetNumberOfBlocksInFork(index, forknum) != 0 {
        error!("brindle: refusing to overwrite a non-empty index fork");
    }

    append_page(index, forknum, &encode_meta(blob.len() as u64));
    for chunk in blob.chunks(PAGE_CHUNK_CAPACITY) {
        append_page(index, forknum, chunk);
    }

    // WAL-log full page images so the build survives a crash and reaches
    // replicas. The init fork of an unlogged index must always be logged (it
    // seeds the main fork after recovery); the main fork only when the
    // relation is WAL-logged at all.
    // SAFETY: rd_rel is always a valid Form_pg_class for an open relation.
    let needs_wal = forknum == pg_sys::ForkNumber::INIT_FORKNUM
        || (*(*index).rd_rel).relpersistence == pg_sys::RELPERSISTENCE_PERMANENT as c_char;
    if needs_wal {
        pg_sys::log_newpage_range(
            index,
            forknum,
            0,
            pg_sys::RelationGetNumberOfBlocksInFork(index, forknum),
            true,
        );
    }
}

/// Copy the content area (header..`pd_lower`) of `blkno` into `out`.
///
/// # Safety
/// `index` must be an open index relation locked at least AccessShare, and
/// `blkno` must exist in its main fork.
unsafe fn read_page_contents(
    index: pg_sys::Relation,
    blkno: pg_sys::BlockNumber,
    out: &mut Vec<u8>,
) {
    let buffer = pg_sys::ReadBufferExtended(
        index,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        blkno,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        core::ptr::null_mut(),
    );
    pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buffer);
    let pd_lower = (*page.cast::<pg_sys::PageHeaderData>()).pd_lower as usize;
    if pd_lower < PAGE_CONTENTS_OFFSET || pd_lower > pg_sys::BLCKSZ as usize {
        pg_sys::UnlockReleaseBuffer(buffer);
        error!("brindle: corrupted index page {blkno} (pd_lower {pd_lower})");
    }
    // SAFETY: pd_lower was just validated to lie inside the BLCKSZ page.
    let contents = core::slice::from_raw_parts(page.cast::<u8>(), pd_lower);
    out.extend_from_slice(&contents[PAGE_CONTENTS_OFFSET..]);
    pg_sys::UnlockReleaseBuffer(buffer);
}

/// Read back the blob written by [`write_index_blob`] from the main fork.
///
/// # Safety
/// `index` must be an open brindle index relation locked at least AccessShare.
pub unsafe fn read_index_blob(index: pg_sys::Relation) -> Vec<u8> {
    let nblocks = pg_sys::RelationGetNumberOfBlocksInFork(index, pg_sys::ForkNumber::MAIN_FORKNUM);
    if nblocks == 0 {
        error!("brindle: index has no pages (never built?)");
    }

    let mut meta_bytes = Vec::new();
    read_page_contents(index, 0, &mut meta_bytes);
    let meta: &[u8; META_SIZE] = meta_bytes
        .first_chunk::<META_SIZE>()
        .unwrap_or_else(|| error!("brindle: metapage too short ({} bytes)", meta_bytes.len()));
    let (magic, version, blob_len) = decode_meta(meta);
    if magic != META_MAGIC {
        error!("brindle: bad metapage magic {magic:#x}");
    }
    if version != STORAGE_VERSION {
        error!("brindle: unsupported storage version {version}");
    }
    let blob_len = usize::try_from(blob_len)
        .unwrap_or_else(|_| error!("brindle: metapage blob length out of range"));
    // Never trust a length the relation can't physically hold — it would turn
    // one corrupt metapage into a giant allocation.
    if blob_len > (nblocks as usize - 1) * PAGE_CHUNK_CAPACITY {
        error!(
            "brindle: metapage declares {blob_len} blob bytes but the index has {nblocks} pages"
        );
    }

    let mut blob = Vec::with_capacity(blob_len);
    for blkno in 1..nblocks {
        read_page_contents(index, blkno, &mut blob);
    }
    if blob.len() != blob_len {
        error!(
            "brindle: index blob is {} bytes but metapage declared {}",
            blob.len(),
            blob_len
        );
    }
    blob
}

/// The one way to load a persisted index: read the blob, decode both halves,
/// and enforce the invariant that ties them together — `tids[i]` addresses
/// graph node `i`, so the table must cover every node. All future readers
/// (scans, incremental inserts, vacuum) go through here rather than
/// re-deriving that check.
///
/// # Safety
/// `index` must be an open brindle index relation locked at least AccessShare.
pub unsafe fn load_index(index: pg_sys::Relation) -> (Hnsw, Vec<TidPair>) {
    let blob = read_index_blob(index);
    let (graph_bytes, tids) =
        decode_index_payload(&blob).unwrap_or_else(|e| error!("brindle: {e}"));
    let hnsw = Hnsw::from_bytes(graph_bytes).unwrap_or_else(|e| error!("brindle: {e}"));
    if hnsw.len() != tids.len() {
        error!(
            "brindle: corrupted index: {} graph nodes but {} heap pointers",
            hnsw.len(),
            tids.len()
        );
    }
    (hnsw, tids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_round_trip() {
        let graph = vec![7u8; 1000];
        let tids: Vec<TidPair> = (0..50)
            .map(|i| (i as u32 * 3, (i % 7) as u16 + 1))
            .collect();
        let blob = encode_index_payload(&graph, &tids);
        let (graph2, tids2) = decode_index_payload(&blob).expect("decode");
        assert_eq!(graph2, &graph[..]);
        assert_eq!(tids2, tids);
    }

    #[test]
    fn encode_index_matches_byte_framing() {
        use crate::hnsw::{Hnsw, HnswParams};

        // The streaming encoder (graph serialized in place) must produce the
        // exact bytes the byte-based framing does over the same graph.
        let mut hnsw = Hnsw::new(HnswParams::default());
        let mut rng = 0x51EDu64;
        for _ in 0..64 {
            let v: Vec<f32> = (0..8)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (rng >> 33) as f32 / u32::MAX as f32
                })
                .collect();
            hnsw.insert(v).unwrap();
        }
        let tids: Vec<TidPair> = (0..hnsw.len()).map(|i| (i as u32, i as u16 + 1)).collect();

        let streamed = encode_index(&hnsw, &tids);
        let framed = encode_index_payload(&hnsw.to_bytes(), &tids);
        assert_eq!(streamed, framed);

        let (graph_bytes, tids2) = decode_index_payload(&streamed).expect("decode");
        assert_eq!(Hnsw::from_bytes(graph_bytes).unwrap().len(), hnsw.len());
        assert_eq!(tids2, tids);
    }

    #[test]
    fn encode_index_empty_graph() {
        use crate::hnsw::{Hnsw, HnswParams};

        let hnsw = Hnsw::new(HnswParams::default());
        let blob = encode_index(&hnsw, &[]);
        let (graph_bytes, tids) = decode_index_payload(&blob).expect("decode");
        assert!(tids.is_empty());
        assert!(Hnsw::from_bytes(graph_bytes).unwrap().is_empty());
    }

    #[test]
    fn payload_round_trip_empty() {
        let blob = encode_index_payload(&[], &[]);
        let (graph, tids) = decode_index_payload(&blob).expect("decode");
        assert!(graph.is_empty());
        assert!(tids.is_empty());
    }

    #[test]
    fn payload_rejects_any_truncation() {
        let blob = encode_index_payload(&[1, 2, 3], &[(9, 1), (10, 2)]);
        for cut in 0..blob.len() {
            assert!(
                decode_index_payload(&blob[..cut]).is_err(),
                "prefix of {cut} bytes decoded successfully"
            );
        }
    }

    #[test]
    fn payload_rejects_trailing_bytes() {
        let mut blob = encode_index_payload(&[1, 2, 3], &[(9, 1)]);
        blob.push(0);
        assert_eq!(
            decode_index_payload(&blob),
            Err(PayloadError::TrailingBytes)
        );
    }
}
