"""Unit tests for per-eye openness normalization + calibration loading (E1, Phase A).

Pure functions only — no ONNX, no hardware, no sockets.
Run: python -m pytest scripts/ml/tests -q
"""

import json

import numpy as np

from predict_live_multitask import load_per_eye_openness_calibration, normalize_per_eye


def test_identity_when_uncalibrated_refs():
    x = np.array([0.13, 0.78], dtype=np.float32)
    out = normalize_per_eye(x, np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    assert np.allclose(out, x, atol=1e-6)


def test_linear_map_midpoint():
    x = np.array([0.5, 0.5], dtype=np.float32)
    out = normalize_per_eye(x, np.array([0.8, 0.8]), np.array([0.2, 0.2]))
    # (0.5 - 0.2) / (0.8 - 0.2) = 0.5
    assert np.allclose(out, [0.5, 0.5], atol=1e-6)


def test_clamps_below_and_above():
    x = np.array([0.1, 0.9], dtype=np.float32)
    out = normalize_per_eye(x, np.array([0.8, 0.8]), np.array([0.2, 0.2]))
    assert out[0] == 0.0  # below closed_p05 -> 0
    assert out[1] == 1.0  # above open_p95 -> 1


def test_degenerate_range_is_identity():
    x = np.array([0.4, 0.6], dtype=np.float32)
    # per-eye range 0.02 < min_range 0.05 -> that eye passes through unchanged
    out = normalize_per_eye(x, np.array([0.51, 0.51]), np.array([0.49, 0.49]), min_range=0.05)
    assert np.allclose(out, x, atol=1e-6)


def test_nan_inputs_stay_finite_and_bounded():
    x = np.array([np.nan, 0.5], dtype=np.float32)
    out = normalize_per_eye(x, np.array([0.8, np.nan]), np.array([0.2, 0.2]))
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_per_eye_asymmetry():
    x = np.array([0.5, 0.5], dtype=np.float32)
    # left: (0.5-0.1)/(0.6-0.1)=0.8 ; right: (0.5-0.4)/(0.9-0.4)=0.2
    out = normalize_per_eye(x, np.array([0.6, 0.9]), np.array([0.1, 0.4]))
    assert np.allclose(out, [0.8, 0.2], atol=1e-6)


def test_loader_reads_v2(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(
        json.dumps(
            {
                "schema": "dreamair.openness_calibration.v2",
                "left": {"open_p95": 0.83, "closed_p05": 0.07},
                "right": {"open_p95": 0.79, "closed_p05": 0.06},
            }
        ),
        encoding="utf-8",
    )
    res = load_per_eye_openness_calibration(p)
    assert res is not None
    open_p95, closed_p05, half_p50 = res
    assert np.allclose(open_p95, [0.83, 0.79])
    assert np.allclose(closed_p05, [0.07, 0.06])
    assert half_p50 is None  # older v2 files without the S2 half anchor


def test_loader_reads_v2_with_half_anchor(tmp_path):
    p = tmp_path / "cal_half.json"
    p.write_text(
        json.dumps(
            {
                "schema": "dreamair.openness_calibration.v2",
                "left": {"open_p95": 0.83, "closed_p05": 0.07, "half_p50": 0.44},
                "right": {"open_p95": 0.79, "closed_p05": 0.06, "half_p50": 0.41},
            }
        ),
        encoding="utf-8",
    )
    res = load_per_eye_openness_calibration(p)
    assert res is not None
    _, _, half_p50 = res
    assert half_p50 is not None
    assert np.allclose(half_p50, [0.44, 0.41])


def test_loader_none_when_missing(tmp_path):
    assert load_per_eye_openness_calibration(None) is None
    assert load_per_eye_openness_calibration(tmp_path / "does_not_exist.json") is None


def test_loader_none_for_v1_only(tmp_path):
    p = tmp_path / "v1.json"
    p.write_text(
        json.dumps(
            {
                "schema": "dreamair.openness_calibration.v1",
                "left": {"open_score_median": 0.5, "closed_score_median": 0.1},
                "right": {"open_score_median": 0.5, "closed_score_median": 0.1},
            }
        ),
        encoding="utf-8",
    )
    # v1-only file lacks open_p95/closed_p05 -> None (runtime keeps raw openness).
    assert load_per_eye_openness_calibration(p) is None


def test_loader_and_normalize_roundtrip(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(
        json.dumps(
            {
                "left": {"open_p95": 0.8, "closed_p05": 0.2},
                "right": {"open_p95": 0.8, "closed_p05": 0.2},
            }
        ),
        encoding="utf-8",
    )
    res = load_per_eye_openness_calibration(p)
    assert res is not None
    out = normalize_per_eye(np.array([0.5, 0.8], dtype=np.float32), res[0], res[1])
    assert np.allclose(out, [0.5, 1.0], atol=1e-6)


# ---- S2: piecewise half-anchor normalize ----

def test_piecewise_half_anchor_pins_half_to_05():
    import numpy as np
    from predict_live_multitask import normalize_per_eye
    hi = np.array([1.0, 1.0], dtype=np.float32)
    lo = np.array([0.1, 0.68], dtype=np.float32)   # right eye = the real collapsed-range case
    half = np.array([0.7, 0.9], dtype=np.float32)
    out = normalize_per_eye(np.array([0.7, 0.9], dtype=np.float32), hi, lo, half_p50=half)
    assert abs(float(out[0]) - 0.5) < 1e-5
    assert abs(float(out[1]) - 0.5) < 1e-5
    # anchors still map to extremes
    out_lo = normalize_per_eye(lo.copy(), hi, lo, half_p50=half)
    out_hi = normalize_per_eye(hi.copy(), hi, lo, half_p50=half)
    assert float(out_lo[0]) < 1e-5 and float(out_hi[0]) > 1.0 - 1e-5


def test_piecewise_degenerate_segment_falls_back_linear():
    import numpy as np
    from predict_live_multitask import normalize_per_eye
    hi = np.array([1.0, 1.0], dtype=np.float32)
    lo = np.array([0.1, 0.1], dtype=np.float32)
    bad_half = np.array([0.11, 0.999], dtype=np.float32)  # both segments degenerate
    x = np.array([0.55, 0.55], dtype=np.float32)
    got = normalize_per_eye(x, hi, lo, half_p50=bad_half)
    linear = normalize_per_eye(x, hi, lo)
    assert np.allclose(got, linear, atol=1e-6)


def test_no_half_matches_legacy_linear_exactly():
    import numpy as np
    from predict_live_multitask import normalize_per_eye
    rng = np.random.default_rng(3)
    hi = np.array([0.98, 0.95], dtype=np.float32)
    lo = np.array([0.05, 0.30], dtype=np.float32)
    for _ in range(50):
        x = rng.random(2).astype(np.float32)
        assert np.array_equal(normalize_per_eye(x, hi, lo), normalize_per_eye(x, hi, lo, half_p50=None))


# ---- S1: dual-path hover curve ----

def test_dual_path_hover_tracks_slow_ramp_linearly():
    import numpy as np
    from predict_live_multitask import OpennessDualPath
    dp = OpennessDualPath()
    outs = []
    # slow descent 1.0 -> 0.4 in 0.01 steps (well under enter velocity)
    for v in np.arange(1.0, 0.4, -0.01):
        out = dp.apply(np.array([v, v], dtype=np.float32), 0.90, 0.28, 1.25)
        outs.append(float(out[0]))
    # hover path = v/0.9: at input 0.5 expect ~0.556, NOT the s-curve's collapsed value
    idx = int((1.0 - 0.5) / 0.01)
    assert abs(outs[idx] - 0.5 / 0.9) < 0.03
    # monotone descent, no snapping jumps
    diffs = np.diff(outs)
    assert float(np.max(np.abs(diffs))) < 0.05


def test_dual_path_blink_still_snaps():
    import numpy as np
    from predict_live_multitask import OpennessDualPath, apply_openness_curve
    dp = OpennessDualPath()
    for _ in range(10):
        dp.apply(np.array([1.0, 1.0], dtype=np.float32), 0.90, 0.28, 1.25)
    seq = [0.6, 0.15, 0.02, 0.02, 0.02]
    last = None
    for v in seq:
        last = dp.apply(np.array([v, v], dtype=np.float32), 0.90, 0.28, 1.25)
    # after a fast blink the output must be near fully closed (blink path active)
    assert float(last[0]) < 0.05
    # and reopening is fast too
    for v in (0.5, 0.95, 1.0):
        last = dp.apply(np.array([v, v], dtype=np.float32), 0.90, 0.28, 1.25)
    assert float(last[0]) > 0.9


def test_dual_path_hold_steady_at_half():
    import numpy as np
    from predict_live_multitask import OpennessDualPath
    dp = OpennessDualPath()
    # settle at 0.5 slowly
    for v in np.arange(1.0, 0.5, -0.01):
        dp.apply(np.array([v, v], dtype=np.float32), 0.90, 0.28, 1.25)
    outs = []
    for _ in range(30):  # tiny jitter around 0.5 stays on hover path
        v = 0.5 + np.random.default_rng(1).normal(0, 0.004)
        outs.append(float(dp.apply(np.array([v, v], dtype=np.float32), 0.90, 0.28, 1.25)[0]))
    assert max(outs) - min(outs) < 0.05
