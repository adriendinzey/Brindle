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

use crate::hnsw::{GraphBytes, Hnsw, HnswDecodeError};

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
    let mut src: &[u8] = blob;
    let graph_len = read_graph_len(&mut src)?;
    let prefix = blob.len() - src.len();
    let (graph, mut rest) = (
        &blob[prefix..prefix + graph_len],
        &blob[prefix + graph_len..],
    );
    let tids = read_tid_table(&mut rest)?;
    Ok((graph, tids))
}

/// Read the graph's declared byte length and check the source can supply it.
///
/// Shared for the same reason [`read_tid_table`] is: `load_index` walks pages
/// while the framing tests hand in a slice, and a second copy of this would let
/// the tested path and the shipped one drift apart.
fn read_graph_len<S: GraphBytes + ?Sized>(src: &mut S) -> Result<usize, PayloadError> {
    let len = src.read_len().map_err(|_| PayloadError::Truncated)?;
    if len > src.remaining() {
        return Err(PayloadError::Truncated);
    }
    Ok(len)
}

/// Read the node-id → heap-TID table that follows the graph, and require the
/// payload to end there.
///
/// Shared deliberately: `load_index` walks pages while the framing tests hand in
/// a slice, and a second copy of this would let the tested path and the shipped
/// one drift apart.
fn read_tid_table<S: GraphBytes + ?Sized>(src: &mut S) -> Result<Vec<TidPair>, PayloadError> {
    let count = src.read_len().map_err(|_| PayloadError::Truncated)?;
    // Bound the reservation by what the source can actually supply, so a corrupt
    // count cannot turn into a huge allocation.
    if src.remaining() / 6 < count {
        return Err(PayloadError::Truncated);
    }
    let mut tids = Vec::with_capacity(count);
    let mut entry = [0u8; 6];
    for _ in 0..count {
        src.read_exact(&mut entry)
            .map_err(|_| PayloadError::Truncated)?;
        let block = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
        let offset = u16::from_le_bytes([entry[4], entry[5]]);
        tids.push((block, offset));
    }
    if src.remaining() != 0 {
        return Err(PayloadError::TrailingBytes);
    }
    Ok(tids)
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

/// Write `contents` as a freshly initialized page at `blkno`, or as a newly
/// appended page when `blkno` is [`pg_sys::InvalidBlockNumber`] (Postgres'
/// `P_NEW`).
///
/// # Safety
/// `index` must be an open index relation this backend may write and extend,
/// with every other writer excluded; `contents` must fit in
/// [`PAGE_CHUNK_CAPACITY`].
unsafe fn write_page(
    index: pg_sys::Relation,
    forknum: pg_sys::ForkNumber::Type,
    blkno: pg_sys::BlockNumber,
    contents: &[u8],
) {
    if contents.len() > PAGE_CHUNK_CAPACITY {
        error!(
            "brindle: page chunk of {} bytes exceeds page capacity",
            contents.len()
        );
    }
    // SAFETY: an existing block is re-read in place, while P_NEW
    // (InvalidBlockNumber) extends the fork by one page. Either way the
    // relation is ours to write (callers exclude other writers) and the buffer
    // stays pinned+locked until the copy below is done.
    let buffer = pg_sys::ReadBufferExtended(
        index,
        forknum,
        blkno,
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

    write_page(
        index,
        forknum,
        pg_sys::InvalidBlockNumber,
        &encode_meta(blob.len() as u64),
    );
    for chunk in blob.chunks(PAGE_CHUNK_CAPACITY) {
        write_page(index, forknum, pg_sys::InvalidBlockNumber, chunk);
    }
    log_fork_pages(index, forknum);
}

/// Replace the main fork's contents with `blob`, reusing the pages already
/// allocated and extending only when the blob outgrows them.
///
/// No caller shrinks the blob yet — inserting appends a node and tombstoning
/// flips a byte the encoding already carries — so the tail-clearing pass below
/// exists for the compaction that will reclaim tombstoned nodes. Leftover pages
/// are re-initialized empty rather than truncated away: shrinking a relation
/// needs a lock this path doesn't hold, and the read path consumes only each
/// page's used bytes and stops at the length the metapage declares, so an empty
/// tail page contributes nothing.
///
/// # Safety
/// `index` must be an open index relation this backend may write, and the
/// caller must hold [`IMAGE_LOCK_BLOCK`] exclusively. That excludes writers —
/// the rewrite is not atomic, so a second one would interleave with it — and
/// readers, who would otherwise splice together halves of two images.
pub unsafe fn rewrite_index_blob(index: pg_sys::Relation, blob: &[u8]) {
    let forknum = pg_sys::ForkNumber::MAIN_FORKNUM;
    let existing = pg_sys::RelationGetNumberOfBlocksInFork(index, forknum);

    let meta = encode_meta(blob.len() as u64);
    let mut blkno: pg_sys::BlockNumber = 0;
    for contents in core::iter::once(&meta[..]).chain(blob.chunks(PAGE_CHUNK_CAPACITY)) {
        let target = if blkno < existing {
            blkno
        } else {
            pg_sys::InvalidBlockNumber
        };
        write_page(index, forknum, target, contents);
        blkno += 1;
    }
    while blkno < existing {
        write_page(index, forknum, blkno, &[]);
        blkno += 1;
    }

    log_fork_pages(index, forknum);
}

/// WAL-log every page of `forknum` as a full page image, so the contents
/// survive a crash and reach replicas. The init fork of an unlogged index must
/// always be logged (it seeds the main fork after recovery); the main fork only
/// when the relation is WAL-logged at all.
///
/// # Safety
/// `index` must be an open index relation this backend may read.
unsafe fn log_fork_pages(index: pg_sys::Relation, forknum: pg_sys::ForkNumber::Type) {
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

/// The stored blob, delivered a page at a time.
///
/// A graph worth measuring spans thousands of pages, which is far more than a
/// backend may pin simultaneously, so the bytes cannot be borrowed all at once.
/// Reading them page by page into the decoder — rather than concatenating them
/// into one buffer first — is what keeps the load from copying the whole index
/// twice: once into the blob, once out of it.
struct PageBytes {
    index: pg_sys::Relation,
    /// Next block to fault in; `nblocks` means the pages are exhausted.
    blkno: pg_sys::BlockNumber,
    nblocks: pg_sys::BlockNumber,
    /// The pinned, share-locked buffer whose contents are being served, or
    /// `InvalidBuffer` before the first page and after the last is released.
    buffer: pg_sys::Buffer,
    /// Content area of that buffer: borrowed from the buffer pool, valid only
    /// while `buffer` stays pinned.
    page: *const u8,
    len: usize,
    pos: usize,
    /// Payload bytes not yet handed out, from the metapage's declared length.
    left: usize,
}

impl PageBytes {
    /// # Safety
    /// `index` must be an open brindle index relation locked at least
    /// AccessShare, with `nblocks` blocks in its main fork.
    unsafe fn new(index: pg_sys::Relation, nblocks: pg_sys::BlockNumber, blob_len: usize) -> Self {
        Self {
            index,
            blkno: 1, // block 0 is the metapage, already consumed by the caller
            nblocks,
            buffer: pg_sys::InvalidBuffer as pg_sys::Buffer,
            page: core::ptr::null(),
            len: 0,
            pos: 0,
            left: blob_len,
        }
    }

    fn release(&mut self) {
        if self.buffer != pg_sys::InvalidBuffer as pg_sys::Buffer {
            // SAFETY: the buffer was pinned and share-locked by `advance`, and
            // is released exactly once because this clears the handle.
            unsafe { pg_sys::UnlockReleaseBuffer(self.buffer) };
            self.buffer = pg_sys::InvalidBuffer as pg_sys::Buffer;
            self.page = core::ptr::null();
            self.len = 0;
            self.pos = 0;
        }
    }

    /// Pin and lock the next page, serving its bytes in place. Returns false
    /// once none are left.
    ///
    /// Reads are answered straight out of the buffer pool rather than from a
    /// copy, which is the point: a copy per page is a second pass over the whole
    /// index on every scan. Holding the content lock while the decoder consumes
    /// the page is what index access methods normally do, and the work between
    /// pages is bounded by the page.
    fn advance(&mut self) -> bool {
        self.release();
        if self.blkno >= self.nblocks {
            return false;
        }
        // SAFETY: the relation is locked by the caller for the whole decode and
        // `blkno` is below `nblocks`, so the block exists in the main fork.
        unsafe {
            let buffer = pg_sys::ReadBufferExtended(
                self.index,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                self.blkno,
                pg_sys::ReadBufferMode::RBM_NORMAL,
                core::ptr::null_mut(),
            );
            pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buffer);
            let pd_lower = (*page.cast::<pg_sys::PageHeaderData>()).pd_lower as usize;
            if pd_lower < PAGE_CONTENTS_OFFSET || pd_lower > pg_sys::BLCKSZ as usize {
                pg_sys::UnlockReleaseBuffer(buffer);
                error!(
                    "brindle: corrupted index page {} (pd_lower {pd_lower})",
                    self.blkno
                );
            }
            self.buffer = buffer;
            self.page = page.cast::<u8>().add(PAGE_CONTENTS_OFFSET);
            self.len = pd_lower - PAGE_CONTENTS_OFFSET;
        }
        self.pos = 0;
        self.blkno += 1;
        true
    }

    /// Restrict how many further bytes this source will hand out. Used to fence
    /// the graph off from the pointer table that follows it.
    fn limit(&mut self, bytes: usize) {
        self.left = bytes;
    }
}

/// Releasing on drop covers the ordinary paths. An `ERROR` inside the decode
/// longjmps past this, which is safe but not tidy: the transaction's own
/// end-of-xact cleanup releases the pin and the lock, exactly as it does for
/// every other backend that errors mid-scan.
impl Drop for PageBytes {
    fn drop(&mut self) {
        self.release();
    }
}

impl GraphBytes for PageBytes {
    fn read_exact(&mut self, out: &mut [u8]) -> Result<(), HnswDecodeError> {
        if out.len() > self.left {
            return Err(HnswDecodeError::Truncated);
        }
        let mut filled = 0;
        while filled < out.len() {
            if self.pos == self.len {
                // Skipping rather than stopping is defensive: the writer
                // re-initializes leftover pages instead of truncating, so a
                // shrunken image leaves empty ones behind. Today `left` stops
                // the read at the length the metapage declares, before any of
                // them — this branch is what keeps that a bound rather than the
                // only thing standing between the reader and stale tail bytes.
                if !self.advance() {
                    return Err(HnswDecodeError::Truncated);
                }
                continue;
            }
            // A value may straddle a page boundary, so take what this page has
            // and come back for the rest.
            let take = (out.len() - filled).min(self.len - self.pos);
            // SAFETY: `page` points at the pinned buffer's content area and
            // `pos + take <= len`, which was derived from that page's pd_lower.
            let src = unsafe { core::slice::from_raw_parts(self.page.add(self.pos), take) };
            out[filled..filled + take].copy_from_slice(src);
            self.pos += take;
            filled += take;
        }
        self.left -= out.len();
        Ok(())
    }

    fn remaining(&self) -> usize {
        self.left
    }
}

/// Validate the metapage and return `(block count, declared payload length)`.
///
/// # Safety
/// `index` must be an open brindle index relation locked at least AccessShare.
unsafe fn read_meta(index: pg_sys::Relation) -> (pg_sys::BlockNumber, usize) {
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

    (nblocks, blob_len)
}

/// Block whose heavyweight page lock arbitrates the stored image: share to read
/// it whole, exclusive to replace it. Nothing about the metapage itself matters
/// — it is simply the one block every index has, so it names the whole image in
/// the lock manager.
///
/// The image spans pages that a rewrite cannot update atomically, and buffer
/// locks are too short-lived to span the whole traversal, so this is what keeps
/// a reader from splicing together halves of two different images. It also
/// serializes writers against each other, which a lost update would otherwise
/// need. Both jobs go away with page-structured storage, where an update
/// touches only the pages it changes.
pub const IMAGE_LOCK_BLOCK: pg_sys::BlockNumber = 0;

/// The one way to load a persisted index: read the blob under the image lock,
/// decode both halves, and enforce the invariant that ties them together —
/// `tids[i]` addresses graph node `i`, so the table must cover every node. All
/// readers (scans, incremental inserts, vacuum) go through here rather than
/// re-deriving that check or the locking.
///
/// # Safety
/// `index` must be an open brindle index relation locked at least AccessShare.
pub unsafe fn load_index(index: pg_sys::Relation) -> (Hnsw, Vec<TidPair>) {
    // A writer already holding the exclusive lock re-enters here freely: the
    // lock manager never conflicts a request with the requester's own locks.
    pg_sys::LockPage(index, IMAGE_LOCK_BLOCK, pg_sys::ShareLock as i32);
    let (nblocks, blob_len) = read_meta(index);
    let mut src = PageBytes::new(index, nblocks, blob_len);

    // Same framing the slice path performs, driven off the page walk so the
    // whole blob never lands in memory at once.
    let graph_len = read_graph_len(&mut src).unwrap_or_else(|e| error!("brindle: {e}"));
    // Hand the decoder exactly the graph's bytes: it bounds its own allocations
    // by what the source says is left, and the trailing pointer table is not
    // part of that budget.
    let after_graph = src.remaining() - graph_len;
    src.limit(graph_len);
    let hnsw = Hnsw::from_graph_bytes(&mut src).unwrap_or_else(|e| error!("brindle: {e}"));
    if src.remaining() != 0 {
        error!("brindle: graph is shorter than its declared length");
    }
    src.limit(after_graph);

    let tids = read_tid_table(&mut src).unwrap_or_else(|e| error!("brindle: {e}"));
    pg_sys::UnlockPage(index, IMAGE_LOCK_BLOCK, pg_sys::ShareLock as i32);
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
