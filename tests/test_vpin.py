import pytest
from vpin_engine.engine import VPINEngine
from vpin_engine.math_utils import fast_normal_cdf


def test_fast_normal_cdf_accuracy():
    assert fast_normal_cdf(0.0) == 0.5
    assert pytest.approx(fast_normal_cdf(1.96), 1e-4) == 0.9750
    assert pytest.approx(fast_normal_cdf(-1.96), 1e-4) == 0.0250
    assert fast_normal_cdf(10.0) == 1.0
    assert fast_normal_cdf(-10.0) == 0.0


def test_bucket_volume_splitting_across_boundaries():
    engine = VPINEngine(bucket_size=100.0, window_size=2, initial_sigma=1.0)

    res1 = engine.update(100, 10.0, 60.0)
    assert len(res1) == 0
    assert engine._accumulated_volume == 60.0

    res2 = engine.update(200, 10.0, 70.0)
    assert len(res2) == 0
    assert engine._current_bucket_idx == 1
    assert engine._accumulated_volume == 30.0

    res3 = engine.update(300, 10.0, 70.0)
    assert len(res3) == 1
    assert res3[0].bucket_index == 1
    assert res3[0].window_size == 2


def test_pure_neutral_flow_yields_zero_vpin():
    engine = VPINEngine(bucket_size=100.0, window_size=5, initial_sigma=1.0)
    results = []
    for i in range(10):
        out = engine.update(timestamp_ns=(i + 1) * 1_000_000, price=100.0, volume=100.0)
        results.extend(out)

    assert len(results) == 6
    for res in results:
        assert pytest.approx(res.vpin, abs=1e-7) == 0.0


def test_extreme_toxic_flow_yields_unit_vpin():
    engine = VPINEngine(bucket_size=100.0, window_size=3, initial_sigma=0.1)
    results = []
    for i in range(6):
        out = engine.update(timestamp_ns=i * 1000, price=100.0 + (i * 5.0), volume=100.0)
        results.extend(out)

    assert len(results) > 0
    latest = results[-1]
    assert pytest.approx(latest.vpin, 0.01) == 1.0
