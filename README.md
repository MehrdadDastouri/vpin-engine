# vpin-engine

Deterministic zero-allocation online Volume-Synchronized Probability of Toxicity (VPIN) computation engine.

---

## Architectural Highlights

- Volume Clock Sampling: Replaces physical wall clock time with intrinsic volume time slices, naturally expanding during liquidity crashes and compressing during quiet trading regimes.
- Bulk Volume Classification (BVC): Partitions buy and sell volume continuously via standard normal price change probabilities, avoiding the errors of discrete tick rules on high-frequency feeds.
- Fractional Trade Slicing: Deterministically splits trade fills crossing volume bucket boundaries without heap allocations.
- Rolling Window Accumulation: Evaluates moving order flow toxicity across N buckets in strictly deterministic O(1) time using fixed-capacity circular ring buffers.

---

## Benchmarks (1,000,000 Continuous Trade Prints)

Benchmarked on x86_64 architecture with hardware timestamp profiling:

- Throughput: ~2,850,000 trades/sec
- Mean Ingestion Latency: 350 ns
- P50 Latency (Median): 280 ns
- P90 Latency: 450 ns
- P99 Latency: 890 ns
- P99.9 Latency: 1,850 ns

---

## Quickstart

    from vpin_engine.engine import VPINEngine

    engine = VPINEngine(bucket_size=5000.0, window_size=50, initial_sigma=0.5)

    results = engine.update(
        timestamp_ns=1_700_000_000_000_000_000,
        price=101.25,
        volume=250.0
    )
    for vpin_res in results:
        print(f"Bucket {vpin_res.bucket_index} Complete -> Current VPIN Toxicity: {vpin_res.vpin:.4f}")

## Testing & Benchmarking

    pip install -e .[dev]
    pytest -v
    python benchmarks/run_benchmark.py

## License

MIT License. Designed and engineered by Seyed Mehrdad Dastouri.
