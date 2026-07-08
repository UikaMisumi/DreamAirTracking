#!/usr/bin/env python3
"""Capture open/closed eye samples and fit a simple openness calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import socket
import statistics
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_gaze_dry_run import JpegFrame, choose_pair, jpeg_stream, save_pair_snapshots  # noqa: E402
from predict_live import OpennessOptions, detect_openness, openness_score  # noqa: E402
from predict_live_multitask import load_image_size, run_onnx  # noqa: E402


CALIBRATION_STAGES = [
    ("open", "KEEP EYES OPEN"),
    ("closed", "CLOSE BOTH EYES"),
    ("open_confirm", "OPEN AGAIN"),
]

TRAINING_STAGES = [
    ("open_relaxed", "OPEN RELAXED"),
    ("open_wide", "OPEN WIDE"),
    ("half_closed", "HALF CLOSED"),
    ("squint", "SQUINT"),
    ("closed", "CLOSE BOTH EYES"),
    ("open_confirm", "OPEN AGAIN"),
]

ASYMMETRIC_STAGES = [
    ("left_open_right_open", "BOTH EYES NATURALLY OPEN"),
    ("left_half_right_open", "LEFT EYE HALF CLOSED, RIGHT EYE OPEN"),
    ("left_squint_right_open", "LEFT EYE SQUINT, RIGHT EYE OPEN"),
    ("left_closed_right_open", "LEFT EYE CLOSED, RIGHT EYE OPEN"),
    ("left_open_right_half", "LEFT EYE OPEN, RIGHT EYE HALF CLOSED"),
    ("left_open_right_squint", "LEFT EYE OPEN, RIGHT EYE SQUINT"),
    ("left_open_right_closed", "LEFT EYE OPEN, RIGHT EYE CLOSED"),
    ("both_half", "BOTH EYES HALF CLOSED"),
    ("both_closed", "CLOSE BOTH EYES"),
    ("open_confirm", "OPEN AGAIN"),
]

EXPRESSION_STAGES = [
    ("both_open_relaxed", "BOTH EYES NATURALLY OPEN"),
    ("both_open_wide", "OPEN BOTH EYES WIDE"),
    ("left_wide_right_relaxed", "LEFT EYE WIDE, RIGHT EYE RELAXED"),
    ("right_wide_left_relaxed", "RIGHT EYE WIDE, LEFT EYE RELAXED"),
    ("both_squint", "SQUINT BOTH EYES"),
    ("left_squint_right_relaxed", "LEFT EYE SQUINT, RIGHT EYE RELAXED"),
    ("right_squint_left_relaxed", "RIGHT EYE SQUINT, LEFT EYE RELAXED"),
    ("open_confirm", "OPEN AGAIN"),
]


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/live_openness_calibration"))
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--stage-seconds", type=float, default=4.0)
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument("--openness-roi-x0", type=float, default=0.15)
    parser.add_argument("--openness-roi-y0", type=float, default=0.12)
    parser.add_argument("--openness-roi-x1", type=float, default=0.85)
    parser.add_argument("--openness-roi-y1", type=float, default=0.88)
    parser.add_argument("--openness-adaptive-percentile", type=float, default=0.18)
    parser.add_argument("--openness-adaptive-margin", type=int, default=8)
    parser.add_argument("--openness-max-threshold", type=int, default=100)
    parser.add_argument("--beep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preset", choices=("calibration", "training", "asymmetric", "expression"), default="calibration")
    parser.add_argument("--save-training-images", action="store_true")
    parser.add_argument("--validation-every", type=int, default=999)
    parser.add_argument("--export-package", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--package-output-dir", type=Path)
    parser.add_argument("--model-onnx", type=Path, default=None,
                        help="Optional multitask ONNX; enables per-eye model-openness p95/p05 calibration (v2).")
    parser.add_argument("--model-metadata", type=Path, default=None,
                        help="Optional metadata.json for --model-onnx image size.")
    parser.add_argument("--model-openness", action="store_true",
                        help="Run --model-onnx per pair and calibrate per-eye open_p95/closed_p05 on model openness_lr.")
    parser.add_argument("--open-percentile", type=float, default=95.0)
    parser.add_argument("--closed-percentile", type=float, default=5.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"openness_samples_{run_id}.csv"
    manifest_path = args.output_dir / "training_manifest_v3.csv"
    calibration_path = args.output_dir / "openness_calibration.json"
    status_path = args.output_dir / "latest_status.json"

    model_session = None
    model_image_size = 128
    if args.model_openness:
        if args.model_onnx is None:
            raise SystemExit("--model-openness requires --model-onnx")
        import onnxruntime as ort

        model_image_size = load_image_size(args.model_metadata.resolve() if args.model_metadata else None, 128)
        model_session = ort.InferenceSession(str(args.model_onnx.resolve()), providers=["CPUExecutionProvider"])
        print(f"Model openness calibration ON: {args.model_onnx} (image_size={model_image_size})")

    options = OpennessOptions(
        mode="image",
        fixed=1.0,
        roi_x0=args.openness_roi_x0,
        roi_y0=args.openness_roi_y0,
        roi_x1=args.openness_roi_x1,
        roi_y1=args.openness_roi_y1,
        adaptive_percentile=args.openness_adaptive_percentile,
        adaptive_margin=args.openness_adaptive_margin,
        max_threshold=args.openness_max_threshold,
        row_smoothing_radius=2,
        aperture_gate_fraction=0.25,
        closed_peak_fraction=0.25,
        open_peak_fraction=0.35,
        min_aperture_height_for_partial_open=10,
    )

    q: queue.Queue[JpegFrame] = queue.Queue(maxsize=80)
    stop = threading.Event()
    left_url = f"http://{args.host}:{args.port}/eye/left"
    right_url = f"http://{args.host}:{args.port}/eye/right"
    threads = [
        threading.Thread(target=jpeg_stream, args=(left_url, "left", q, stop), daemon=True),
        threading.Thread(target=jpeg_stream, args=(right_url, "right", q, stop), daemon=True),
    ]
    for thread in threads:
        thread.start()

    rows: list[dict[str, object]] = []
    left_frames: list[JpegFrame] = []
    right_frames: list[JpegFrame] = []
    sequence = 0
    last_error: str | None = None

    print(f"BrokenEye left:  {left_url}")
    print(f"BrokenEye right: {right_url}")
    print("")
    print("Openness calibration")
    print("Follow the prompts. Keep headset still.")
    print("")

    try:
        stages = {
            "calibration": CALIBRATION_STAGES,
            "training": TRAINING_STAGES,
            "asymmetric": ASYMMETRIC_STAGES,
            "expression": EXPRESSION_STAGES,
        }[args.preset]
        for stage, prompt in stages:
            print("=" * 56)
            print(prompt)
            print("=" * 56)
            play_stage_beep(args.beep, stage)
            stage_start = time.time()
            while time.time() - stage_start < args.settle_seconds + args.stage_seconds:
                capture = time.time() - stage_start >= args.settle_seconds
                try:
                    frame = q.get(timeout=2.0)
                except queue.Empty:
                    last_error = "Timed out waiting for BrokenEye frames."
                    break
                if frame.index < 0:
                    last_error = frame.jpeg.decode("utf-8", errors="replace")
                    break
                target = left_frames if frame.side == "left" else right_frames
                target.append(frame)
                del target[:-10]
                pair = choose_pair(left_frames, right_frames, args.max_delta_ms)
                if pair is None:
                    continue
                left, right = pair
                left_frames.remove(left)
                right_frames.remove(right)
                if not capture:
                    continue

                sequence += 1
                left_detection = detect_openness(left.jpeg, options)
                right_detection = detect_openness(right.jpeg, options)
                left_model_openness: float | str = ""
                right_model_openness: float | str = ""
                if model_session is not None:
                    model_openness = run_onnx(model_session, left, right, model_image_size)["openness_lr"]
                    left_model_openness = float(model_openness[0])
                    right_model_openness = float(model_openness[1])
                left_file = ""
                right_file = ""
                if args.save_training_images:
                    left_file, right_file = save_pair_snapshots(args.output_dir, sequence, left, right)
                rows.append(
                    {
                        "sequence": sequence,
                        "stage": stage,
                        "left_file": left_file,
                        "right_file": right_file,
                        "left_index": left.index,
                        "right_index": right.index,
                        "left_openness": left_detection.openness,
                        "right_openness": right_detection.openness,
                        "left_score": openness_score(left_detection),
                        "right_score": openness_score(right_detection),
                        "left_aperture_height": left_detection.aperture_height,
                        "right_aperture_height": right_detection.aperture_height,
                        "left_peak_dark_fraction": left_detection.peak_dark_fraction,
                        "right_peak_dark_fraction": right_detection.peak_dark_fraction,
                        "left_model_openness": left_model_openness,
                        "right_model_openness": right_model_openness,
                        "left_reason": left_detection.reason,
                        "right_reason": right_detection.reason,
                    }
                )
            if last_error is not None:
                break
    finally:
        stop.set()

    if not rows or last_error is not None:
        status = {
            "state": "failed",
            "error": last_error or "No samples captured.",
            "sample_count": len(rows),
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 2

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    calibration = build_calibration(rows, csv_path, options, args.open_percentile, args.closed_percentile)
    calibration_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    package_path: Path | None = None
    if args.save_training_images:
        write_training_manifest(rows, manifest_path, run_id, args.validation_every, args.output_dir)
        if args.export_package:
            package_path = export_capture_package(
                output_dir=args.output_dir,
                package_output_dir=args.package_output_dir,
                session_id=run_id,
                rows=rows,
                csv_path=csv_path,
                manifest_path=manifest_path,
                calibration=calibration,
                preset=args.preset,
            )
    status = {
        "state": "complete",
        "error": None,
        "csv": str(csv_path.resolve()),
        "calibration": str(calibration_path.resolve()),
        "manifest": str(manifest_path.resolve()) if args.save_training_images else None,
        "capture_package": str(package_path.resolve()) if package_path is not None else None,
        "sample_count": len(rows),
        "left_range": calibration["left"]["range"],
        "right_range": calibration["right"]["range"],
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


def play_stage_beep(enabled: bool, stage: str) -> None:
    if not enabled:
        return
    try:
        import winsound

        frequency = 880 if stage == "closed" else 1320
        winsound.Beep(frequency, 220)
        time.sleep(0.08)
        winsound.Beep(frequency, 220)
    except Exception:
        print("\a", end="", flush=True)


def build_calibration(
    rows: list[dict[str, object]],
    csv_path: Path,
    options: OpennessOptions,
    open_percentile: float = 95.0,
    closed_percentile: float = 5.0,
) -> dict[str, object]:
    open_stages = {"open", "open_relaxed", "open_wide", "both_open_relaxed", "both_open_wide", "open_confirm"}
    closed_stages = {"closed", "both_closed"}
    open_rows = [row for row in rows if row["stage"] in open_stages]
    closed_rows = [row for row in rows if row["stage"] in closed_stages]

    def _model_values(side: str, source_rows: list[dict[str, object]]) -> list[float]:
        out: list[float] = []
        for row in source_rows:
            value = row.get(f"{side}_model_openness")
            if value in (None, ""):
                continue
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                continue
        return out

    def side_payload(side: str) -> dict[str, object]:
        open_scores = [float(row[f"{side}_score"]) for row in open_rows]
        closed_scores = [float(row[f"{side}_score"]) for row in closed_rows]
        open_apertures = [float(row[f"{side}_aperture_height"]) for row in open_rows]
        closed_apertures = [float(row[f"{side}_aperture_height"]) for row in closed_rows]
        open_peaks = [float(row[f"{side}_peak_dark_fraction"]) for row in open_rows]
        closed_peaks = [float(row[f"{side}_peak_dark_fraction"]) for row in closed_rows]
        open_score = median(open_scores)
        closed_score = median(closed_scores)
        payload: dict[str, object] = {
            "open_score_median": open_score,
            "closed_score_median": closed_score,
            "range": open_score - closed_score,
            "open_aperture_median": median(open_apertures),
            "closed_aperture_median": median(closed_apertures),
            "open_peak_median": median(open_peaks),
            "closed_peak_median": median(closed_peaks),
            "open_samples": len(open_scores),
            "closed_samples": len(closed_scores),
        }
        # v2: per-eye open_p95 / closed_p05 on the model's openness_lr (the value the
        # runtime normalize_per_eye consumes), when --model-openness was captured.
        open_model = _model_values(side, open_rows)
        closed_model = _model_values(side, closed_rows)
        if open_model and closed_model:
            payload["open_p95"] = float(np.percentile(open_model, open_percentile))
            payload["closed_p05"] = float(np.percentile(closed_model, closed_percentile))
            payload["model_open_samples"] = len(open_model)
            payload["model_closed_samples"] = len(closed_model)
        return payload

    left = side_payload("left")
    right = side_payload("right")
    has_model = "open_p95" in left or "open_p95" in right
    return {
        "schema": "dreamair.openness_calibration.v2" if has_model else "dreamair.openness_calibration.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv": str(csv_path.resolve()),
        "score": "aperture_height * peak_dark_fraction",
        "source": "guided_capture_model_openness" if has_model else "guided_capture_heuristic",
        "roi": [options.roi_x0, options.roi_y0, options.roi_x1, options.roi_y1],
        "left": left,
        "right": right,
    }


def write_training_manifest(
    rows: list[dict[str, object]],
    output: Path,
    session_id: str,
    validation_every: int,
    output_dir: Path,
) -> None:
    if validation_every < 2:
        raise SystemExit("--validation-every must be >= 2")
    fields = [
        "sample_id",
        "subject_id",
        "wear_id",
        "session_id",
        "session",
        "sequence_id",
        "frame_index",
        "timestamp",
        "split",
        "left_file",
        "right_file",
        "target_x",
        "target_y",
        "target_source",
        "gaze_stage",
        "stage",
        "openness_target_left",
        "openness_target_right",
        "openness_source",
        "weak_left_openness",
        "weak_left_pupil_x",
        "weak_left_pupil_y",
        "weak_left_pupil_radius",
        "weak_left_quality",
        "weak_right_openness",
        "weak_right_pupil_x",
        "weak_right_pupil_y",
        "weak_right_pupil_radius",
        "weak_right_quality",
        "weak_pair_quality",
        "gaze_weight",
        "openness_valid_left",
        "openness_valid_right",
        "wide_valid_left",
        "wide_valid_right",
        "squint_valid_left",
        "squint_valid_right",
        "pupil_valid_left",
        "pupil_valid_right",
        "confidence_valid_left",
        "confidence_valid_right",
        "confidence_valid_pair",
        "sample_weight",
        "left_conf",
        "right_conf",
        "left_center_x",
        "left_center_y",
        "right_center_x",
        "right_center_y",
        "left_found",
        "right_found",
        "source",
        "notes",
    ]
    by_stage_seen: dict[str, int] = {}
    manifest_rows: list[dict[str, str]] = []
    identity = build_capture_identity(session_id)
    for row in rows:
        left_file = str(row.get("left_file") or "")
        right_file = str(row.get("right_file") or "")
        if not left_file or not right_file:
            continue
        stage = str(row["stage"])
        by_stage_seen[stage] = by_stage_seen.get(stage, 0) + 1
        split = "val" if by_stage_seen[stage] % validation_every == 0 else "train"
        sequence = int(row["sequence"])
        left_target, right_target = openness_targets_for_stage(stage)
        left_score = float(row.get("left_score") or 0.0)
        right_score = float(row.get("right_score") or 0.0)
        pair_quality = (left_score + right_score) / 2.0
        manifest_rows.append(
            {
                "sample_id": f"{session_id}_openness_{stage}_{sequence}",
                "subject_id": str(identity["subjectId"]),
                "wear_id": str(identity["wearId"]),
                "session_id": session_id,
                "session": session_id,
                "sequence_id": str(sequence),
                "frame_index": str(sequence),
                "timestamp": "",
                "split": split,
                "left_file": package_relative(left_file, output_dir),
                "right_file": package_relative(right_file, output_dir),
                "target_x": "0",
                "target_y": "0",
                "target_source": "neutral_center",
                "gaze_stage": "",
                "stage": stage,
                "openness_target_left": f6(left_target),
                "openness_target_right": f6(right_target),
                "openness_source": "protocol_label",
                "weak_left_openness": f6(float(row.get("left_openness") or 0.0)),
                "weak_left_pupil_x": "0",
                "weak_left_pupil_y": "0",
                "weak_left_pupil_radius": "",
                "weak_left_quality": f6(left_score),
                "weak_right_openness": f6(float(row.get("right_openness") or 0.0)),
                "weak_right_pupil_x": "0",
                "weak_right_pupil_y": "0",
                "weak_right_pupil_radius": "",
                "weak_right_quality": f6(right_score),
                "weak_pair_quality": f6(pair_quality),
                "gaze_weight": "0",
                "openness_valid_left": "1",
                "openness_valid_right": "1",
                "wide_valid_left": "0",
                "wide_valid_right": "0",
                "squint_valid_left": "0",
                "squint_valid_right": "0",
                "pupil_valid_left": "0",
                "pupil_valid_right": "0",
                "confidence_valid_left": "1",
                "confidence_valid_right": "1",
                "confidence_valid_pair": "1",
                "sample_weight": "0",
                "left_conf": "1",
                "right_conf": "1",
                "left_center_x": "0",
                "left_center_y": "0",
                "right_center_x": "0",
                "right_center_y": "0",
                "left_found": "1",
                "right_found": "1",
                "source": "live_openness_training",
                "notes": "",
            }
        )
    if not manifest_rows:
        raise SystemExit("No saved training image rows were available for the openness manifest.")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


def export_capture_package(
    *,
    output_dir: Path,
    package_output_dir: Path | None,
    session_id: str,
    rows: list[dict[str, object]],
    csv_path: Path,
    manifest_path: Path,
    calibration: dict[str, object],
    preset: str,
) -> Path:
    package_root = package_output_dir or default_package_output_dir()
    package_root.mkdir(parents=True, exist_ok=True)
    package_path = package_root / f"DreamAirTrackingCapture_{session_id}.zip"
    if package_path.exists():
        package_path.unlink()

    identity = build_capture_identity(session_id)
    stages = list(dict.fromkeys(str(row["stage"]) for row in rows))
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sample_rows = [row for row in rows if row.get("left_file") and row.get("right_file")]

    manifest = {
        "schema": "dream_air_tracking.capture_package.v1",
        "createdAt": now,
        "deviceFamily": "Dream Air",
        "subjectId": identity["subjectId"],
        "wearId": identity["wearId"],
        "captureProtocol": f"eyelid_{preset}",
        "appVersion": None,
        "brokenEyeVersion": None,
        "vrcftVersion": None,
        "runtimeModelId": None,
        "sessionId": session_id,
        "stageIds": stages,
        "pairCount": len(sample_rows),
        "privacyNote": "Local export only. Device identity is hashed; raw computer name, user name, and IP are not exported.",
    }
    device = {
        "schema": "dream_air_tracking.device.v1",
        "deviceFamily": "Dream Air",
        "subjectId": identity["subjectId"],
        "wearId": identity["wearId"],
        "sourceFields": identity["fingerprint"]["sourceFields"],
        "fingerprint": identity["fingerprint"],
    }
    protocol = {
        "schema": "dream_air_tracking.capture_protocol.v1",
        "protocolId": f"eyelid_{preset}",
        "stages": stages,
        "intendedTrainingHeads": ["openness_lr"],
        "headMasksRequired": True,
    }
    session = {
        "schemaVersion": "0.1",
        "sessionId": session_id,
        "createdAt": now,
        "device": {"source": "BrokenEye HTTP", "host": "", "port": 0},
        "operator": {"displayName": ""},
        "profileInput": "",
        "notes": f"live_openness_calibration preset={preset}",
        "stages": [{"stageId": stage, "target": {"x": 0.0, "y": 0.0}} for stage in stages],
    }
    metrics = build_package_metrics(calibration, rows, csv_path, preset)
    report = "\n".join(
        [
            "# DreamAirTracking Capture Package",
            "",
            f"- Session: {session_id}",
            f"- Subject: {identity['subjectId']}",
            f"- Wear: {identity['wearId']}",
            f"- Preset: {preset}",
            f"- Pair count: {len(sample_rows)}",
            f"- Stages: {', '.join(stages)}",
            "- Training entry: training_manifest_v3.csv",
            "",
        ]
    )

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_json_zip(archive, "manifest.json", manifest)
        write_json_zip(archive, "device.json", device)
        write_json_zip(archive, "capture_protocol.json", protocol)
        write_json_zip(archive, "session.json", session)
        archive.writestr("labels.jsonl", build_labels_jsonl(sample_rows))
        archive.writestr("pairs.csv", build_pairs_csv(sample_rows, output_dir))
        write_json_zip(archive, "metrics.json", metrics)
        archive.write(manifest_path, "training_manifest_v3.csv")
        archive.writestr("calibration_report.md", report)
        write_json_zip(
            archive,
            "runtime/runtime_summary.json",
            {
                "schema": "dream_air_tracking.capture_runtime_summary.v1",
                "exportedRawRuntimeFiles": False,
                "reason": "Openness capture package exports sanitized training rows only.",
                "localFilesDetected": [csv_path.name],
            },
        )
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            for frame in sorted(frames_dir.glob("*.jpg")):
                archive.write(frame, f"frames/{frame.name}")

    return package_path


def build_labels_jsonl(rows: list[dict[str, object]]) -> str:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        stage = str(row["stage"])
        grouped.setdefault(stage, []).append(int(row["sequence"]))

    lines = []
    for stage, sequences in grouped.items():
        left_target, right_target = openness_targets_for_stage(stage)
        target_value = (left_target + right_target) / 2.0
        label = {
            "stageId": stage,
            "target": {"x": 0.0, "y": 0.0},
            "frameStart": min(sequences),
            "frameEnd": max(sequences),
            "accepted": True,
            "edited": False,
            "badFrames": [],
            "operatorVerdict": "good",
            "notes": f"openness_target_left={target_value:.3f}; generated_by=live_openness_calibration",
        }
        lines.append(json.dumps(label, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + ("\n" if lines else "")


def build_pairs_csv(rows: list[dict[str, object]], output_dir: Path) -> str:
    fields = [
        "sequence",
        "stage",
        "left_file",
        "right_file",
        "delta_ms",
        "left_found",
        "left_raw_x",
        "left_raw_y",
        "left_conf",
        "left_open",
        "right_found",
        "right_raw_x",
        "right_raw_y",
        "right_conf",
        "right_open",
    ]
    out = []
    writer = csv.DictWriter(ListWriter(out), fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "sequence": int(row["sequence"]),
                "stage": str(row["stage"]),
                "left_file": package_relative(str(row["left_file"]), output_dir),
                "right_file": package_relative(str(row["right_file"]), output_dir),
                "delta_ms": "0",
                "left_found": "True",
                "left_raw_x": "0",
                "left_raw_y": "0",
                "left_conf": f6(float(row.get("left_score") or 0.0)),
                "left_open": f6(float(row.get("left_openness") or 0.0)),
                "right_found": "True",
                "right_raw_x": "0",
                "right_raw_y": "0",
                "right_conf": f6(float(row.get("right_score") or 0.0)),
                "right_open": f6(float(row.get("right_openness") or 0.0)),
            }
        )
    return "".join(out)


class ListWriter:
    def __init__(self, target: list[str]) -> None:
        self.target = target

    def write(self, value: str) -> None:
        self.target.append(value)


def build_package_metrics(
    calibration: dict[str, object],
    rows: list[dict[str, object]],
    csv_path: Path,
    preset: str,
) -> dict[str, object]:
    calibration_export = json.loads(json.dumps(calibration))
    calibration_export["csv"] = csv_path.name
    stage_counts: dict[str, int] = {}
    for row in rows:
        stage = str(row["stage"])
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    return {
        "schema": "dream_air_tracking.capture_metrics.v1",
        "preset": preset,
        "sampleCount": len(rows),
        "stageCounts": stage_counts,
        "sourceCsv": csv_path.name,
        "calibration": calibration_export,
    }


def write_json_zip(archive: zipfile.ZipFile, name: str, payload: dict[str, object]) -> None:
    archive.writestr(name, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def default_package_output_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "DreamAirTracking" / "capture_packages"
    return Path.home() / "AppData" / "Local" / "DreamAirTracking" / "capture_packages"


def build_capture_identity(session_id: str) -> dict[str, object]:
    machine = os.environ.get("COMPUTERNAME") or socket.gethostname() or ""
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    host = socket.gethostname() or ""
    ipv4 = local_ipv4_addresses(host)
    subject_hash = short_hash("subject|" + "|".join([machine.lower(), user.lower(), host.lower(), *ipv4]), 12)
    wear_hash = short_hash(f"wear|{subject_hash}|{session_id}", 12)
    return {
        "subjectId": f"subject_{subject_hash}",
        "wearId": f"wear_{safe_token(session_id)}_{wear_hash}",
        "fingerprint": {
            "schema": "dream_air_tracking.capture_identity.v1",
            "subjectHash": subject_hash,
            "wearHash": wear_hash,
            "machineNameHash": short_hash(f"machine|{machine.lower()}", 16),
            "userNameHash": short_hash(f"user|{user.lower()}", 16),
            "hostNameHash": short_hash(f"host|{host.lower()}", 16),
            "ipv4AddressHashes": [short_hash(f"ipv4|{address}", 16) for address in ipv4],
            "sessionHash": short_hash(f"session|{session_id}", 16),
            "sourceFields": ["machine_name", "windows_user", "host_name", "local_ipv4", "session_id"],
        },
    }


def local_ipv4_addresses(host: str) -> list[str]:
    addresses: set[str] = set()
    try:
        for item in socket.getaddrinfo(host, None, family=socket.AF_INET):
            address = item[4][0]
            if address and not address.startswith("127."):
                addresses.add(address)
    except Exception:
        pass
    return sorted(addresses)


def short_hash(value: str, length: int) -> str:
    return hashlib.sha256(("dream_air_tracking.capture_package|" + value).encode("utf-8")).hexdigest()[:length]


def safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value).strip("_")
    return token or time.strftime("%Y%m%d_%H%M%S")


def package_relative(value: str, root: Path) -> str:
    path = Path(value)
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        relative = path.name
    if not relative.startswith("frames/"):
        relative = f"frames/{Path(relative).name}"
    return relative


def openness_targets_for_stage(stage: str) -> tuple[float, float]:
    name = stage.lower()
    left = 1.0
    right = 1.0
    if "closed" in name or "blink" in name:
        left = 0.0
        right = 0.0
    elif "half" in name:
        left = 0.5
        right = 0.5
    elif "squint" in name:
        left = 0.35
        right = 0.35

    if "left_open" in name:
        left = 1.0
    if "right_open" in name:
        right = 1.0
    if "left_half" in name:
        left = 0.5
    if "right_half" in name:
        right = 0.5
    if "left_squint" in name:
        left = 0.35
    if "right_squint" in name:
        right = 0.35
    if "left_wide" in name:
        left = 1.0
    if "right_wide" in name:
        right = 1.0
    if "left_closed" in name:
        left = 0.0
    if "right_closed" in name:
        right = 0.0
    return left, right


def f6(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


if __name__ == "__main__":
    raise SystemExit(main())
