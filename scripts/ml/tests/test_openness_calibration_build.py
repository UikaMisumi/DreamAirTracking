"""Unit tests for build_calibration v2 model-openness p95/p05 (E1, Phase D)."""

from types import SimpleNamespace

from live_openness_calibration import build_calibration

OPTS = SimpleNamespace(roi_x0=0.1, roi_y0=0.1, roi_x1=0.9, roi_y1=0.9)


def _row(stage, lm=None, rm=None):
    row = {
        "stage": stage,
        "left_score": 1.0, "right_score": 1.0,
        "left_aperture_height": 10, "right_aperture_height": 10,
        "left_peak_dark_fraction": 0.3, "right_peak_dark_fraction": 0.3,
        "left_openness": 1.0, "right_openness": 1.0,
    }
    if lm is not None:
        row["left_model_openness"] = lm
    if rm is not None:
        row["right_model_openness"] = rm
    return row


def test_v2_model_percentiles(tmp_path):
    rows = [_row("open", 0.9, 0.85) for _ in range(20)]
    rows += [_row("closed", 0.05, 0.08) for _ in range(20)]
    cal = build_calibration(rows, tmp_path / "x.csv", OPTS)
    assert cal["schema"].endswith("v2")
    assert cal["source"] == "guided_capture_model_openness"
    assert abs(cal["left"]["open_p95"] - 0.9) < 1e-6
    assert cal["left"]["closed_p05"] <= 0.1
    assert cal["left"]["model_open_samples"] == 20
    assert cal["right"]["open_p95"] <= 0.85 + 1e-6
    # v1 fields preserved for back-compat
    assert "open_score_median" in cal["left"]


def test_v1_without_model(tmp_path):
    rows = [_row("open") for _ in range(5)] + [_row("closed") for _ in range(5)]
    cal = build_calibration(rows, tmp_path / "x.csv", OPTS)
    assert cal["schema"].endswith("v1")
    assert "open_p95" not in cal["left"]
    assert "open_p95" not in cal["right"]
