//! Criterion micro-benchmarks for the distance kernels.
//!
//! Establishes a repeatable baseline for the distance hot loop so the Phase 5
//! SIMD work (T-060+) can be *measured*, not guessed (see `docs/ARCHITECTURE.md`
//! § Testing & validation strategy). We bench L2², cosine, and inner product at
//! embedding dimensions that match real workloads:
//!
//! - **128** — classic SIFT / older sentence encoders
//! - **768** — BERT / MiniLM-family text embeddings
//! - **1536** — OpenAI `text-embedding-3-small`
//!
//! Both operands are wrapped in `black_box` so the optimizer can't hoist the
//! load or fold the computation away; the returned `Result` is handed back to
//! criterion (which `black_box`es it) for the same reason.
//!
//! Because Brindle is a single pgrx crate, this bench links the extension crate
//! and therefore needs a `pg*` feature to compile. Run it inside WSL with the
//! toolchain from T-003:
//!
//! ```bash
//! cargo bench --no-default-features --features pg17
//! ```

use brindle::distance::{cosine_distance, inner_product, l2_squared};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Embedding dimensions representative of common models.
const DIMS: [usize; 3] = [128, 768, 1536];

/// Deterministic pseudo-random vector in roughly `[-1, 1)`, so runs are
/// comparable without pulling in a `rand` dependency. A 64-bit xorshift is more
/// than enough entropy to keep the kernels from hitting trivial values.
fn sample_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1; // never zero, or xorshift collapses
    (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");

    for dim in DIMS {
        let a = sample_vector(dim, 0x1234_5678_9abc_def0);
        let b = sample_vector(dim, 0x0fed_cba9_8765_4321);

        // Per-element throughput makes the cost-per-dimension comparable across
        // sizes and is the natural unit a SIMD change should move.
        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("l2_squared", dim), &dim, |bn, _| {
            bn.iter(|| l2_squared(black_box(&a), black_box(&b)))
        });
        group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bn, _| {
            bn.iter(|| cosine_distance(black_box(&a), black_box(&b)))
        });
        group.bench_with_input(BenchmarkId::new("inner_product", dim), &dim, |bn, _| {
            bn.iter(|| inner_product(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
