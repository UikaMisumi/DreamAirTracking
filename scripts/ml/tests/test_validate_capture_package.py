"""Unit tests for the capture package validator (E10)."""

import json
import zipfile

from validate_capture_package import validate


def _make_pkg(path, *, with_manifest=True, device_family="Dream Air", stage="open", frames=2, training_manifest=False):
    with zipfile.ZipFile(path, "w") as archive:
        if with_manifest:
            archive.writestr("manifest.json", json.dumps({
                "schema": "dream_air_tracking.capture_package.v1",
                "deviceFamily": device_family, "subjectId": "subject_x",
                "wearId": "wear_x", "captureProtocol": "eyelid_calibration", "pairCount": frames,
            }))
        archive.writestr("device.json", "{}")
        archive.writestr("capture_protocol.json", "{}")
        archive.writestr("session.json", "{}")
        archive.writestr("labels.jsonl", "")
        lines = ["sequence,stage,left_file,right_file,delta_ms"]
        for i in range(frames):
            lines.append(f"{i},{stage},frames/{stage}_{i}_left.jpg,frames/{stage}_{i}_right.jpg,5")
        archive.writestr("pairs.csv", "\n".join(lines) + "\n")
        archive.writestr("metrics.json", "{}")
        for i in range(frames):
            archive.writestr(f"frames/{stage}_{i}_left.jpg", b"\xff\xd8\xff\xd9")
            archive.writestr(f"frames/{stage}_{i}_right.jpg", b"\xff\xd8\xff\xd9")
        if training_manifest:
            header = ("sample_id,left_file,right_file,gaze_weight,source,split,session_id,"
                      "openness_valid_left,openness_valid_right,pupil_valid_left,pupil_valid_right,target_x,target_y")
            tm = [header]
            for i in range(frames):
                tm.append(f"e{i},frames/x{i}_l.jpg,frames/x{i}_r.jpg,0,live_openness_training,train,s1,1,1,0,0,0,0")
            archive.writestr("training_manifest_v3.csv", "\n".join(tm) + "\n")


def test_valid_package_passes(tmp_path):
    path = tmp_path / "pkg.zip"
    _make_pkg(path)
    report = validate(path)
    assert report["passed"], report["violations"]
    assert report["info"]["frame_files"] == 4


def test_missing_manifest_fails(tmp_path):
    path = tmp_path / "bad.zip"
    _make_pkg(path, with_manifest=False)
    report = validate(path)
    assert not report["passed"]
    assert any("missing required files" in v for v in report["violations"])


def test_wrong_device_family_fails(tmp_path):
    path = tmp_path / "wrong.zip"
    _make_pkg(path, device_family="Vive")
    report = validate(path)
    assert not report["passed"]


def test_no_stage_labels_fails(tmp_path):
    path = tmp_path / "nostage.zip"
    _make_pkg(path, stage="")
    report = validate(path)
    assert not report["passed"]
    assert any("no stage labels" in v for v in report["violations"])


def test_bundled_training_manifest_gate_runs(tmp_path):
    path = tmp_path / "tm.zip"
    _make_pkg(path, training_manifest=True)
    report = validate(path)
    assert "head_mask_gate" in report["info"]
    assert report["passed"], report["violations"]
