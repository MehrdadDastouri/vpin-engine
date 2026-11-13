"""Domain models for volume buckets and calculated VPIN toxicity metrics."""

from dataclasses import dataclass


@dataclass(slots=True)
class VolumeBucket:
    """Fixed-capacity volume slice aggregating trades under volume time."""
    bucket_index: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    total_volume: float
    buy_volume: float
    sell_volume: float
    price_change: float
    imbalance: float


@dataclass(slots=True)
class VPINResult:
    """Calculated VPIN toxicity metric over a rolling window of N buckets."""
    bucket_index: int
    timestamp_ns: int
    vpin: float
    rolling_buy_volume: float
    rolling_sell_volume: float
    rolling_imbalance_sum: float
    window_size: int
