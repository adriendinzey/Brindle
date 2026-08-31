//! Criterion benchmark for loading a serialized graph back into memory.
//!
//! Every index scan currently decodes the whole graph before it can walk it, so
//! this cost is paid per query, not per build. It is the dominant term in query
//! latency and the number the storage work has to move — measured here so the
//! change is provable rather than asserted.
//!
//! Building the fixture dominates the run: the graph is constructed once and
//! only `from_bytes` is timed. `GRAPH_ROWS`/`GRAPH_DIMS` override the size; the
//! default is small enough to run routinely, while the figures quoted in the
//! performance write-up come from a 100k x 128 run:
//!
//! ```bash
//! cargo bench --bench graph_decode --features pg17
//! GRAPH_ROWS=100000 GRAPH_DIMS=128 cargo bench --bench graph_decode --features pg17
//! ```
//!
//! Because Brindle is a single pgrx crate, this bench links the extension crate
//! and therefore needs a `pg*` feature to compile.

use brindle::hnsw::{Hnsw, HnswParams};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic pseudo-random vector, so the fixture is identical run to run
/// and two measurements are comparable.
fn sample_vector(dim: usize, state: &mut u64) -> Vec<f32> {
    (0..dim)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench_decode(c: &mut Criterion) {
    let rows = env_usize("GRAPH_ROWS", 20_000);
    let dims = env_usize("GRAPH_DIMS", 128);

    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut graph = Hnsw::new(HnswParams::default());
    for _ in 0..rows {
        let v = sample_vector(dims, &mut state);
        graph.insert(v).expect("insert");
    }
    let bytes = graph.to_bytes();

    let mut group = c.benchmark_group("graph_decode");
    // Bytes decoded, so a change in the serialized size shows up as throughput
    // rather than hiding inside the wall-clock number.
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function(format!("from_bytes/{rows}x{dims}"), |b| {
        b.iter(|| {
            let g = Hnsw::from_bytes(black_box(&bytes)).expect("decode");
            black_box(g.len())
        })
    });
    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
