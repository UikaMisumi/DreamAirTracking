#!/usr/bin/env python3
"""Capture live model outputs for a quick affine gaze calibration layer."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_gaze_dry_run import JpegFrame, choose_pair, jpeg_stream, preprocess, save_pair_snapshots  # noqa: E402


STAGE_TARGETS = {
    "center": (0.0, 0.0),
    "center_confirm": (0.0, 0.0),
    "micro_left": (-0.25, 0.0),
    "micro_right": (0.25, 0.0),
    "micro_up": (0.0, 0.25),
    "micro_down": (0.0, -0.25),
    "left": (-0.7, 0.0),
    "right": (0.7, 0.0),
    "up": (0.0, 0.7),
    "down": (0.0, -0.7),
    "left_up": (-0.7, 0.7),
    "right_up": (0.7, 0.7),
    "left_down": (-0.7, -0.7),
    "right_down": (0.7, -0.7),
    "far_left": (-1.0, 0.0),
    "far_right": (1.0, 0.0),
    "far_up": (0.0, 1.0),
    "far_down": (0.0, -1.0),
    "far_left_up": (-1.0, 1.0),
    "far_left_down": (-1.0, -1.0),
    "far_right_up": (1.0, 1.0),
    "far_right_down": (1.0, -1.0),
}

STAGE_PRESETS = {
    "five": ["center", "left", "right", "up", "down"],
    "nine": ["center", "left", "right", "up", "down", "left_up", "right_up", "left_down", "right_down"],
    "micro": ["center", "micro_left", "micro_right", "micro_up", "micro_down"],
    "full13": [
        "center",
        "micro_left",
        "micro_right",
        "micro_up",
        "micro_down",
        "left",
        "right",
        "up",
        "down",
        "left_up",
        "right_up",
        "left_down",
        "right_down",
    ],
    "edge9": [
        "center_confirm",
        "far_left",
        "far_right",
        "far_up",
        "far_down",
        "far_left_up",
        "far_left_down",
        "far_right_up",
        "far_right_down",
    ],
}


def fit_affine(raw: np.ndarray, target: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([raw.astype(np.float64), np.ones((raw.shape[0], 1), dtype=np.float64)], axis=1)
    y = target.astype(np.float64)
    penalty = np.eye(3, dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    a = weights[:2, :].T.astype(np.float32)
    b = weights[2, :].astype(np.float32)
    return a, b


def apply_affine(raw: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (raw @ a.T + b).astype(np.float32)


def metric_rows(raw: np.ndarray, target: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    baseline = np.linalg.norm(raw - target, axis=1)
    corrected = np.linalg.norm(apply_affine(raw, a, b) - target, axis=1)
    return {
        "baseline_mean_l2": float(np.mean(baseline)),
        "baseline_p90_l2": float(np.quantile(baseline, 0.9)),
        "corrected_mean_l2": float(np.mean(corrected)),
        "corrected_p90_l2": float(np.quantile(corrected, 0.9)),
        "sample_count": int(raw.shape[0]),
    }


def print_countdown(label: str, seconds: float) -> None:
    whole = max(1, int(round(seconds)))
    for remaining in range(whole, 0, -1):
        print(f"{label} {remaining}...", flush=True)
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("runs/eye_multitask_mobilenetv3_fullparam_round1_twochannel_retarget070/eye_multitask.onnx"))
    parser.add_argument("--metadata", type=Path, default=Path("runs/eye_multitask_mobilenetv3_fullparam_round1_twochannel_retarget070/eye_multitask.metadata.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/live_quick_calibration"))
    parser.add_argument("--preset", choices=tuple(STAGE_PRESETS.keys()), default="five")
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--capture-seconds", type=float, default=2.5)
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--snapshot-every-stage", action="store_true")
    parser.add_argument("--save-training-images", action="store_true", help="Save every captured left/right pair and write a fine-tune manifest.")
    parser.add_argument("--validation-every", type=int, default=5, help="For saved training images, put every Nth sample per stage into val.")
    args = parser.parse_args()

    if not args.metadata.exists():
        raise SystemExit(
            f"Metadata file not found: {args.metadata}\n"
            "Run train_eye_multitask.py and export_eye_multitask_onnx.py first, "
            "or pass --metadata pointing to an existing .metadata.json."
        )
    if not args.onnx.exists():
        raise SystemExit(
            f"ONNX file not found: {args.onnx}\n"
            "Run export_eye_multitask_onnx.py first, or pass --onnx pointing to an existing model."
        )

    metadata = json.load(args.metadata.open("r", encoding="utf-8"))
    image_size = int(metadata.get("image_size", 96))
    session = ort.InferenceSession(str(args.onnx.resolve()), providers=["CPUExecutionProvider"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"quick_calibration_samples_{run_id}.csv"
    manifest_path = args.output_dir / f"fine_tune_manifest_{run_id}.csv"
    layer_path = args.output_dir / f"quick_calibration_layer_{run_id}.json"
    status_path = args.output_dir / "latest_status.json"

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

    print(f"BrokenEye left:  {left_url}")
    print(f"BrokenEye right: {right_url}")
    print(f"ONNX: {args.onnx.resolve()}")
    print("Calibration preset:", args.preset)
    print("Keep your head still. Move only your eyes if possible.")

    rows: list[dict[str, object]] = []
    left_frames: list[JpegFrame] = []
    right_frames: list[JpegFrame] = []
    sequence = 0
    last_error: str | None = None

    try:
        for stage in STAGE_PRESETS[args.preset]:
            tx, ty = STAGE_TARGETS[stage]
            label = stage.upper().replace("_", " ")
            print("")
            print("=" * 48)
            print(f"LOOK {label}   target=({tx:+.2f},{ty:+.2f})")
            print("=" * 48)
            print_countdown("settle", args.settle_seconds)
            end_time = time.time() + args.capture_seconds
            stage_count = 0
            stage_snapshot_saved = False
            while time.time() < end_time:
                try:
                    frame = q.get(timeout=2.0)
                except queue.Empty:
                    last_error = "Timed out waiting for BrokenEye frames."
                    break
                if frame.index < 0:
                    last_error = frame.jpeg.decode("utf-8", errors="replace")
                    break
                target_list = left_frames if frame.side == "left" else right_frames
                target_list.append(frame)
                del target_list[:-10]
                pair = choose_pair(left_frames, right_frames, args.max_delta_ms)
                if pair is None:
                    continue
                left, right = pair
                left_frames.remove(left)
                right_frames.remove(right)
                sequence += 1
                stage_count += 1

                left_img = preprocess(left.jpeg, image_size)
                right_img = preprocess(right.jpeg, image_size)
                image = np.stack([left_img, right_img], axis=0)[None, ...].astype(np.float32)
                raw = session.run(["gaze_xy"], {"image": image})[0][0].astype(np.float32)
                delta_ms = abs(left.received_at - right.received_at) * 1000.0
                left_snapshot = ""
                right_snapshot = ""
                if args.save_training_images:
                    left_snapshot, right_snapshot = save_pair_snapshots(args.output_dir, sequence, left, right)
                elif args.snapshot_every_stage and not stage_snapshot_saved:
                    left_snapshot, right_snapshot = save_pair_snapshots(args.output_dir, sequence, left, right)
                    stage_snapshot_saved = True
                rows.append(
                    {
                        "sequence": sequence,
                        "stage": stage,
                        "target_x": tx,
                        "target_y": ty,
                        "raw_x": float(raw[0]),
                        "raw_y": float(raw[1]),
                        "delta_ms": delta_ms,
                        "left_index": left.index,
                        "right_index": right.index,
                        "left_snapshot": left_snapshot,
                        "right_snapshot": right_snapshot,
                    }
                )
                if stage_count % 20 == 0:
                    print(f"captured {stage_count} samples for {stage}: raw=({raw[0]:+.3f},{raw[1]:+.3f})", flush=True)
            if last_error:
                break
            print(f"stage {stage} samples: {stage_count}")
    finally:
        stop.set()

    if not rows:
        status = {"state": "failed", "error": last_error or "no samples captured"}
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 2

    fields = [
        "sequence",
        "stage",
        "target_x",
        "target_y",
        "raw_x",
        "raw_y",
        "delta_ms",
        "left_index",
        "right_index",
        "left_snapshot",
        "right_snapshot",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.save_training_images:
        write_fine_tune_manifest(rows, manifest_path, run_id, args.validation_every)

    by_stage: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_stage[str(row["stage"])].append(row)
    stage_raw = []
    stage_target = []
    stage_summaries = {}
    for stage, stage_rows in by_stage.items():
        raw = np.array([[float(row["raw_x"]), float(row["raw_y"])] for row in stage_rows], dtype=np.float32)
        target = np.array([STAGE_TARGETS[stage]], dtype=np.float32)
        median = np.median(raw, axis=0).astype(np.float32)
        stage_raw.append(median)
        stage_target.append(target[0])
        stage_summaries[stage] = {
            "sample_count": len(stage_rows),
            "raw_median": [float(median[0]), float(median[1])],
            "target": [float(target[0, 0]), float(target[0, 1])],
        }

    if len(stage_raw) < 3:
        status = {"state": "failed", "error": "need at least three captured stages", "csv": str(csv_path.resolve())}
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 2

    fit_raw = np.stack(stage_raw)
    fit_target = np.stack(stage_target)
    a, b = fit_affine(fit_raw, fit_target, args.alpha)

    all_raw = np.array([[float(row["raw_x"]), float(row["raw_y"])] for row in rows], dtype=np.float32)
    all_target = np.array([[float(row["target_x"]), float(row["target_y"])] for row in rows], dtype=np.float32)
    metrics = metric_rows(all_raw, all_target, a, b)
    layer = {
        "schema": "dream_air_tracking.quick_gaze_calibration_layer.v1",
        "formula": "corrected = A @ raw + b",
        "raw_source": "model_prediction_xy",
        "checkpoint": str(args.onnx.resolve()),
        "manifest": str(csv_path.resolve()),
        "samples_per_stage": None,
        "sessions": {
            run_id: {
                "used": True,
                "calibration_samples": len(rows),
                "stages": list(by_stage.keys()),
                "stage_summaries": stage_summaries,
                "A": a.tolist(),
                "b": b.tolist(),
                "raw_weights": np.vstack([a.T, b]).astype(float).tolist(),
                "all": metrics,
            }
        },
    }
    layer_path.write_text(json.dumps(layer, indent=2, ensure_ascii=False), encoding="utf-8")
    status = {
        "state": "complete" if last_error is None else "failed",
        "error": last_error,
        "csv": str(csv_path.resolve()),
        "manifest": str(manifest_path.resolve()) if args.save_training_images else None,
        "layer": str(layer_path.resolve()),
        "sample_count": len(rows),
        "stage_count": len(by_stage),
        "metrics": metrics,
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["state"] == "complete" else 2


def write_fine_tune_manifest(rows: list[dict[str, object]], output: Path, session_id: str, validation_every: int) -> None:
    if validation_every < 2:
        raise SystemExit("--validation-every must be >= 2")

    fields = [
        "sample_id",
        "session",
        "stage",
        "target_x",
        "target_y",
        "left_file",
        "right_file",
        "left_conf",
        "right_conf",
        "left_center_x",
        "left_center_y",
        "right_center_x",
        "right_center_y",
        "source",
        "sample_weight",
        "split",
    ]
    by_stage_seen: dict[str, int] = defaultdict(int)
    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        stage = str(row["stage"])
        left_file = str(row.get("left_snapshot") or "")
        right_file = str(row.get("right_snapshot") or "")
        if not left_file or not right_file:
            continue
        by_stage_seen[stage] += 1
        split = "val" if by_stage_seen[stage] % validation_every == 0 else "train"
        sequence = int(row["sequence"])
        manifest_rows.append(
            {
                "sample_id": f"{session_id}_live_{stage}_{sequence}",
                "session": session_id,
                "stage": stage,
                "target_x": f"{float(row['target_x']):.6f}",
                "target_y": f"{float(row['target_y']):.6f}",
                "left_file": left_file,
                "right_file": right_file,
                "left_conf": "1.000000",
                "right_conf": "1.000000",
                "left_center_x": "0.000000",
                "left_center_y": "0.000000",
                "right_center_x": "0.000000",
                "right_center_y": "0.000000",
                "source": "live",
                "sample_weight": "1.0",
                "split": split,
            }
        )
    if not manifest_rows:
        raise SystemExit("No saved training image rows were available for the fine-tune manifest.")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    raise SystemExit(main())
