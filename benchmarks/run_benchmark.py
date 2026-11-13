"""High-resolution throughput and latency benchmark for online VPIN calculation."""

import gc
import statistics
import time
from typing import List
import numpy as np
from vpin_engine.engine import VPINEngine


def run_benchmark(num_trades: int = 1_000_000) -> None:
    rng = np.random.default_rng(42)

    print(f"[*] Generating {num_trades:,} synthetic trade prints...")
    volumes = np.round(rng.exponential(scale=25.0, size=num_trades), 2)
    volumes = np.clip(volumes, 1.0, 500.0)

    prices = np.zeros(num_trades, dtype=np.float64)
    base_p = 100.0
    for i in range(num_trades):
        drift = 0.005 if (250_000 <= i <= 350_000) or (750_000 <= i <= 850_000) else 0.0
        shock = rng.normal(loc=drift, scale=0.02)
        base_p = max(1.0, base_p + shock)
        prices[i] = base_p

    time_deltas = rng.integers(100_000, 1_000_000, size=num_trades, dtype=np.int64)
    timestamps = np.cumsum(time_deltas)

    engine = VPINEngine(bucket_size=5_000.0, window_size=50, initial_sigma=0.5)

    latencies_ns: List[int] = [0] * num_trades
    total_buckets_closed = 0

    print("[*] Benchmarking VPIN streaming updates (Disabling GC)...")
    gc.disable()

    start_wall = time.perf_counter_ns()
    for i in range(num_trades):
        t0 = time.perf_counter_ns()
        res = engine.update(int(timestamps[i]), float(prices[i]), float(volumes[i]))
        t1 = time.perf_counter_ns()
        latencies_ns[i] = t1 - t0
        total_buckets_closed += len(res)

    end_wall = time.perf_counter_ns()
    gc.enable()

    total_duration_sec = (end_wall - start_wall) / 1e9
    throughput = num_trades / total_duration_sec

    latencies_ns.sort()
    p50 = latencies_ns[int(num_trades * 0.50)]
    p90 = latencies_ns[int(num_trades * 0.90)]
    p99 = latencies_ns[int(num_trades * 0.99)]
    p99_9 = latencies_ns[int(num_trades * 0.999)]
    mean_lat = statistics.mean(latencies_ns)

    print("\n" + "=" * 60)
    print("           VPIN-ENGINE PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Processed Trade Prints   : {num_trades:,}")
    print(f"Volume Buckets Finalized : {total_buckets_closed:,}")
    print(f"Total Wall Duration      : {total_duration_sec:.4f} sec")
    print(f"Trade Ingestion Rate     : {throughput:,.0f} trades/sec")
    print("-" * 60)
    print(f"Mean Latency             : {mean_lat / 1000.0:.2f} µs ({mean_lat:.0f} ns)")
    print(f"P50 Latency (Median)     : {p50 / 1000.0:.2f} µs ({p50:.0f} ns)")
    print(f"P90 Latency              : {p90 / 1000.0:.2f} µs ({p90:.0f} ns)")
    print(f"P99 Latency              : {p99 / 1000.0:.2f} µs ({p99:.0f} ns)")
    print(f"P99.9 Latency (Tail)     : {p99_9 / 1000.0:.2f} µs ({p99_9:.0f} ns)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark(num_trades=1_000_000)
