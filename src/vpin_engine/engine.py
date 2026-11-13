"""High-performance streaming engine for Volume-Synchronized Probability of Toxicity (VPIN)."""

from typing import List, Optional
import numpy as np
from vpin_engine.math_utils import fast_normal_cdf
from vpin_engine.models import VPINResult, VolumeBucket


class VPINEngine:
    """
    Online VPIN calculator utilizing Bulk Volume Classification (BVC)
    and zero-allocation rolling window ring buffers.
    """

    def __init__(
        self,
        bucket_size: float = 1000.0,
        window_size: int = 50,
        initial_sigma: float = 0.5,
    ) -> None:
        if bucket_size <= 0.0:
            raise ValueError(f"bucket_size must be positive, got {bucket_size}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.bucket_size = bucket_size
        self.window_size = window_size
        self.sigma = initial_sigma

        self._current_bucket_idx: int = 0
        self._bucket_start_ts: int = 0
        self._bucket_start_price: float = 0.0
        self._accumulated_volume: float = 0.0
        self._last_price: float = 0.0

        self._imbalances: np.ndarray = np.zeros(window_size, dtype=np.float64)
        self._buy_volumes: np.ndarray = np.zeros(window_size, dtype=np.float64)
        self._sell_volumes: np.ndarray = np.zeros(window_size, dtype=np.float64)
        self._ring_head: int = 0
        self._completed_buckets_count: int = 0

        self._sum_imbalance: float = 0.0
        self._sum_buys: float = 0.0
        self._sum_sells: float = 0.0

    def update(self, timestamp_ns: int, price: float, volume: float) -> List[VPINResult]:
        """
        Ingest a new trade print.
        Handles trade splits across bucket volume boundaries.
        Returns a list of completed VPINResult instances (if any buckets closed).
        """
        results: List[VPINResult] = []

        if self._bucket_start_ts == 0:
            self._bucket_start_ts = timestamp_ns
            self._bucket_start_price = price
            self._last_price = price

        remaining_trade_vol = volume

        while remaining_trade_vol > 0.0:
            space_in_bucket = self.bucket_size - self._accumulated_volume

            if remaining_trade_vol < space_in_bucket:
                self._accumulated_volume += remaining_trade_vol
                self._last_price = price
                remaining_trade_vol = 0.0
            else:
                self._accumulated_volume += space_in_bucket
                self._last_price = price
                remaining_trade_vol -= space_in_bucket

                completed_bucket = self._seal_bucket(end_timestamp_ns=timestamp_ns, final_price=price)
                vpin_res = self._push_bucket_to_window(completed_bucket)
                if vpin_res is not None:
                    results.append(vpin_res)

                self._current_bucket_idx += 1
                self._bucket_start_ts = timestamp_ns
                self._bucket_start_price = price
                self._accumulated_volume = 0.0

        return results

    def _seal_bucket(self, end_timestamp_ns: int, final_price: float) -> VolumeBucket:
        """Apply Bulk Volume Classification (BVC) to determine buy/sell proportion."""
        delta_p = final_price - self._bucket_start_price
        z_score = delta_p / max(1e-6, self.sigma)

        prob_buy = fast_normal_cdf(z_score)
        v_buy = self.bucket_size * prob_buy
        v_sell = self.bucket_size * (1.0 - prob_buy)
        imbalance = abs(v_buy - v_sell)

        return VolumeBucket(
            bucket_index=self._current_bucket_idx,
            start_timestamp_ns=self._bucket_start_ts,
            end_timestamp_ns=end_timestamp_ns,
            total_volume=self.bucket_size,
            buy_volume=v_buy,
            sell_volume=v_sell,
            price_change=delta_p,
            imbalance=imbalance,
        )

    def _push_bucket_to_window(self, bucket: VolumeBucket) -> Optional[VPINResult]:
        """Update rolling ring buffer in O(1) and compute current VPIN."""
        if self._completed_buckets_count >= self.window_size:
            old_imbalance = self._imbalances[self._ring_head]
            old_buy = self._buy_volumes[self._ring_head]
            old_sell = self._sell_volumes[self._ring_head]

            self._sum_imbalance -= old_imbalance
            self._sum_buys -= old_buy
            self._sum_sells -= old_sell

        self._imbalances[self._ring_head] = bucket.imbalance
        self._buy_volumes[self._ring_head] = bucket.buy_volume
        self._sell_volumes[self._ring_head] = bucket.sell_volume

        self._sum_imbalance += bucket.imbalance
        self._sum_buys += bucket.buy_volume
        self._sum_sells += bucket.sell_volume

        self._ring_head = (self._ring_head + 1) % self.window_size
        self._completed_buckets_count += 1

        if self._completed_buckets_count < self.window_size:
            return None

        vpin = self._sum_imbalance / (self.window_size * self.bucket_size)
        clamped_vpin = max(0.0, min(1.0, float(vpin)))

        return VPINResult(
            bucket_index=bucket.bucket_index,
            timestamp_ns=bucket.end_timestamp_ns,
            vpin=clamped_vpin,
            rolling_buy_volume=float(self._sum_buys),
            rolling_sell_volume=float(self._sum_sells),
            rolling_imbalance_sum=float(self._sum_imbalance),
            window_size=self.window_size,
        )

    def set_volatility_sigma(self, new_sigma: float) -> None:
        """Update baseline price variance scale parameter."""
        if new_sigma > 0:
            self.sigma = new_sigma
