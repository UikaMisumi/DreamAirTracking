#!/usr/bin/env python3
"""Run several ONNX gaze models on the same live BrokenEye frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_gaze_dry_run import JpegFrame, choose_pair, jpeg_stream, preprocess, save_pair_snapshots  # noqa: E402


STAGE_TARGETS = {
    "center": (0.0, 0.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "left_up": (-1.0, 1.0),
    "right_up": (1.0, 1.0),
    "left_down": (-1.0, -1.0),
    "right_down": (1.0, -1.0),
}

STAGE_PRESETS = {
    "five": ["center", "left", "right", "up", "down"],
    "nine": ["center", "left", "right", "up", "down", "left_up", "right_up", "left_down", "right_down"],
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    onnx: Path
    metadata: Path
    image_size: int
    session: ort.InferenceSession


def safe_label(label: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", label.strip())
    if not value:
        raise SystemExit("Model label cannot be empty.")
    if value[0].isdigit():
        value = f"m_{value}"
    return value


def parse_model_spec(raw: str) -> tuple[str, Path, Path | None]:
    if "=" not in raw:
        raise SystemExit("--model must be formatted as label=onnx_path[,metadata_path].")
    label, rest = raw.split("=", 1)
    parts = [part.strip() for part in rest.split(",", 1)]
    onnx = Path(parts[0])
    metadata = Path(parts[1]) if len(parts) > 1 and parts[1] else None
    return safe_label(label), onnx, metadata


def load_models(raw_specs: list[str]) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    seen: set[str] = set()
    for raw in raw_specs:
        label, onnx_path, metadata_path = parse_model_spec(raw)
        if label in seen:
            raise SystemExit(f"Duplicate model label: {label}")
        seen.add(label)
        onnx_path = onnx_path.resolve()
        metadata_path = (metadata_path or onnx_path.with_suffix(".metadata.json")).resolve()
        metadata = json.load(metadata_path.open("r", encoding="utf-8"))
        model_type = str(metadata.get("model_type", "cnn"))
        if model_type == "hybrid":
            raise SystemExit("live_gaze_compare_models.py currently supports image-only ONNX models, not hybrid geometry models.")
        image_size = int(metadata.get("image_size", 96))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        models.append(ModelSpec(label, onnx_path, metadata_path, image_size, session))
    if not models:
        raise SystemExit("Pass at least one --model.")
    return models


def summarize(csv_path: Path, model_labels: list[str], summary_path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig", newline="")))
    summary: dict[str, object] = {"csv": str(csv_path.resolve()), "models": {}}
    for label in model_labels:
        stage_rows: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            stage = row.get("stage", "")
            if stage in STAGE_TARGETS:
                stage_rows.setdefault(stage, []).append(row)
        stage_summary: dict[str, object] = {}
        l2s: list[float] = []
        medians: dict[str, tuple[float, float]] = {}
        for stage in STAGE_PRESETS["nine"]:
            current = stage_rows.get(stage, [])
            if not current:
                continue
            x_key = f"smooth_x_{label}"
            y_key = f"smooth_y_{label}"
            median = (
                statistics.median(float(row[x_key]) for row in current),
                statistics.median(float(row[y_key]) for row in current),
            )
            target = STAGE_TARGETS[stage]
            l2 = math.dist(median, target)
            medians[stage] = median
            l2s.append(l2)
            stage_summary[stage] = {
                "sample_count": len(current),
                "smooth_median": [median[0], median[1]],
                "target": [target[0], target[1]],
                "l2": l2,
            }
        ordering = False
        if all(stage in medians for stage in STAGE_PRESETS["nine"]):
            ordering = (
                medians["left"][0] < medians["center"][0] < medians["right"][0]
                and medians["down"][1] < medians["center"][1] < medians["up"][1]
                and medians["left_up"][0] < medians["right_up"][0]
                and medians["left_down"][0] < medians["right_down"][0]
                and medians["left_down"][1] < medians["left_up"][1]
                and medians["right_down"][1] < medians["right_up"][1]
            )
        summary["models"][label] = {
            "stage_summary": stage_summary,
            "median_stage_l2": statistics.median(l2s) if l2s else None,
            "mean_stage_l2": statistics.mean(l2s) if l2s else None,
            "max_stage_l2": max(l2s) if l2s else None,
            "ordering_ok": ordering,
        }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="Repeat as label=onnx_path[,metadata_path].")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/live_gaze_compare_models"))
    parser.add_argument("--direction-preset", choices=("five", "nine"), default="nine")
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--stage-seconds", type=float, default=3.0)
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--snapshot-every", type=int, default=30)
    args = parser.parse_args()

    models = load_models(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"live_gaze_compare_{run_id}.csv"
    status_path = args.output_dir / "latest_status.json"
    summary_path = args.output_dir / f"comparison_summary_{run_id}.json"

    fields = [
        "sequence",
        "timestamp",
        "stage",
        "stage_elapsed",
        "left_index",
        "right_index",
        "delta_ms",
        "left_snapshot",
        "right_snapshot",
    ]
    for model in models:
        fields.extend([f"raw_x_{model.label}", f"raw_y_{model.label}", f"smooth_x_{model.label}", f"smooth_y_{model.label}"])

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
    print(f"CSV: {csv_path.resolve()}")
    for model in models:
        print(f"MODEL {model.label}: {model.onnx}")
    print("")
    print("Same-frame guided comparison. Keep your head still; move only your eyes if possible.")

    stages = STAGE_PRESETS[args.direction_preset]
    duration_seconds = len(stages) * (args.settle_seconds + args.stage_seconds)
    left_frames: list[JpegFrame] = []
    right_frames: list[JpegFrame] = []
    smooth: dict[str, np.ndarray | None] = {model.label: None for model in models}
    sequence = 0
    started = time.time()
    stage_index = -1
    stage_started = started
    current_stage = ""
    last_error: str | None = None

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while time.time() - started < duration_seconds:
                now = time.time()
                elapsed = now - started
                block = args.settle_seconds + args.stage_seconds
                next_stage_index = min(int(elapsed // block), len(stages) - 1)
                stage_phase = elapsed - next_stage_index * block
                if next_stage_index != stage_index:
                    stage_index = next_stage_index
                    current_stage = stages[stage_index]
                    stage_started = now
                    print("")
                    print("=" * 56)
                    print(f"LOOK {current_stage.upper().replace('_', ' ')}")
                    print("=" * 56)
                capture_enabled = stage_phase >= args.settle_seconds

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
                if not capture_enabled:
                    continue

                sequence += 1
                delta_ms = abs(left.received_at - right.received_at) * 1000.0
                left_snapshot = ""
                right_snapshot = ""
                if args.snapshot_every > 0 and sequence % args.snapshot_every == 0:
                    left_snapshot, right_snapshot = save_pair_snapshots(args.output_dir, sequence, left, right)

                row: dict[str, object] = {
                    "sequence": sequence,
                    "timestamp": f"{time.time():.6f}",
                    "stage": current_stage,
                    "stage_elapsed": f"{time.time() - stage_started:0.3f}",
                    "left_index": left.index,
                    "right_index": right.index,
                    "delta_ms": f"{delta_ms:.3f}",
                    "left_snapshot": left_snapshot,
                    "right_snapshot": right_snapshot,
                }
                print_parts: list[str] = []
                for model in models:
                    left_img = preprocess(left.jpeg, model.image_size)
                    right_img = preprocess(right.jpeg, model.image_size)
                    image = np.stack([left_img, right_img], axis=0)[None, ...].astype(np.float32)
                    raw = model.session.run(["gaze_xy"], {"image": image})[0][0].astype(np.float32)
                    previous = smooth[model.label]
                    current_smooth = raw if previous is None else (args.ema_alpha * raw + (1.0 - args.ema_alpha) * previous)
                    smooth[model.label] = current_smooth
                    row[f"raw_x_{model.label}"] = f"{raw[0]:.6f}"
                    row[f"raw_y_{model.label}"] = f"{raw[1]:.6f}"
                    row[f"smooth_x_{model.label}"] = f"{current_smooth[0]:.6f}"
                    row[f"smooth_y_{model.label}"] = f"{current_smooth[1]:.6f}"
                    print_parts.append(f"{model.label}=({raw[0]:+0.2f},{raw[1]:+0.2f})")
                writer.writerow(row)
                if args.print_every > 0 and sequence % args.print_every == 0:
                    print(f"{sequence:06d} dt={delta_ms:0.1f} " + " ".join(print_parts), flush=True)
    finally:
        stop.set()

    state = "complete" if last_error is None and sequence > 0 else "failed"
    summary = summarize(csv_path, [model.label for model in models], summary_path) if sequence > 0 else None
    status = {
        "state": state,
        "error": last_error,
        "csv": str(csv_path.resolve()),
        "summary": str(summary_path.resolve()) if summary is not None else None,
        "sequence_count": sequence,
        "duration_seconds": time.time() - started,
        "models": {model.label: str(model.onnx.resolve()) for model in models},
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if summary is not None:
        for label, metrics in summary["models"].items():  # type: ignore[union-attr]
            print(
                f"{label}: median_l2={metrics['median_stage_l2']:.3f} "  # type: ignore[index]
                f"mean_l2={metrics['mean_stage_l2']:.3f} "  # type: ignore[index]
                f"ordering_ok={metrics['ordering_ok']}"  # type: ignore[index]
            )
    return 0 if state == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
