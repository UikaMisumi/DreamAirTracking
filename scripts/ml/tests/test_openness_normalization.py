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
    open_p95, closed_p05 = res
    assert np.allclose(open_p95, [0.83, 0.79])
    assert np.allclose(closed_p05, [0.07, 0.06])


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
