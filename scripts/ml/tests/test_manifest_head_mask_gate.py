"""Unit tests for the head-mask training gate (E2)."""

import csv

from manifest_head_mask_gate import gate_manifest

CLEAN_HEADER = [
    "sample_id", "left_file", "right_file", "target_x", "target_y",
    "gaze_weight", "source", "split", "session_id",
    "openness_valid_left", "openness_valid_right", "pupil_valid_left", "pupil_valid_right",
]
LEGACY_HEADER = [
    "sample_id", "left_file", "right_file", "target_x", "target_y",
    "gaze_weight", "source", "split", "session_id",
]


def _write(path, rows, header=CLEAN_HEADER):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _gaze_row(i, split="train", session="s1", ol=0, pv=0):
    return {
        "sample_id": f"g{i}", "left_file": f"L{i}.jpg", "right_file": f"R{i}.jpg",
        "target_x": "0.7", "target_y": "0.0", "gaze_weight": "1.0", "source": "live",
        "split": split, "session_id": session,
        "openness_valid_left": str(ol), "openness_valid_right": str(ol),
        "pupil_valid_left": str(pv), "pupil_valid_right": str(pv),
    }


def _eyelid_row(i, split="train", session="s2"):
    return {
        "sample_id": f"e{i}", "left_file": f"EL{i}.jpg", "right_file": f"ER{i}.jpg",
        "target_x": "0", "target_y": "0", "gaze_weight": "0", "source": "live_openness_training",
        "split": split, "session_id": session,
        "openness_valid_left": "1", "openness_valid_right": "1",
        "pupil_valid_left": "0", "pupil_valid_right": "0",
    }


def _legacy_gaze(i, session="s1", split="train"):
    return {
        "sample_id": f"g{i}", "left_file": f"L{i}.jpg", "right_file": f"R{i}.jpg",
        "target_x": "0.7", "target_y": "0", "gaze_weight": "1.0", "source": "live",
        "split": split, "session_id": session,
    }


def test_clean_manifest_passes(tmp_path):
    path = tmp_path / "clean.csv"
    _write(path, [_gaze_row(i) for i in range(5)] + [_eyelid_row(i) for i in range(5)])
    report = gate_manifest(path)
    assert report["passed"], report["violations"]
    assert report["gaze_head_conflicts"] == 0


def test_missing_valid_columns_fails(tmp_path):
    path = tmp_path / "legacy.csv"
    _write(path, [_legacy_gaze(i) for i in range(3)], header=LEGACY_HEADER)
    report = gate_manifest(path)
    assert not report["passed"]
    assert any("missing head-mask columns" in v for v in report["violations"])


def test_missing_columns_allowed_with_flag(tmp_path):
    path = tmp_path / "legacy.csv"
    _write(path, [_legacy_gaze(i) for i in range(3)], header=LEGACY_HEADER)
    report = gate_manifest(path, allow_weak_fallback=True)
    assert report["passed"]
    assert report["warnings"]


def test_gaze_row_double_supervision_fails(tmp_path):
    path = tmp_path / "conflict.csv"
    _write(path, [_gaze_row(0, ol=1)] + [_gaze_row(i) for i in range(1, 4)])
    report = gate_manifest(path)
    assert not report["passed"]
    assert report["gaze_head_conflicts"] >= 1


def test_gaze_cosupervision_allowed_with_flag(tmp_path):
    path = tmp_path / "conflict.csv"
    _write(path, [_gaze_row(0, ol=1)] + [_gaze_row(i) for i in range(1, 4)])
    report = gate_manifest(path, allow_gaze_head_cosupervision=True)
    assert report["passed"]
    assert report["warnings"]
    assert report["gaze_head_conflicts"] >= 1


def test_split_leakage_same_pair_fails(tmp_path):
    path = tmp_path / "leak.csv"
    # same image pair (same files) in both train and val
    _write(path, [_gaze_row(0, split="train"), _gaze_row(0, split="val")])
    report = gate_manifest(path)
    assert not report["passed"]
    assert report["pair_split_conflicts"] >= 1


def test_real_session_spanning_splits_fails(tmp_path):
    path = tmp_path / "sess.csv"
    # distinct pairs, same real session split across train/val
    _write(path, [_gaze_row(0, split="train", session="s1"), _gaze_row(1, split="val", session="s1")])
    report = gate_manifest(path)
    assert not report["passed"]
    assert report["real_sessions_multi_split"] >= 1
