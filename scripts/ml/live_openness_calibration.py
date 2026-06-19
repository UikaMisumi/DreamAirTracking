#!/usr/bin/env python3
"""Capture open/closed eye samples and fit a simple openness calibration."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_gaze_dry_run import JpegFrame, choose_pair, jpeg_stream, save_pair_snapshots  # noqa: E402
from predict_live import OpennessOptions, detect_openness, openness_score  # noqa: E402


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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"openness_samples_{run_id}.csv"
    manifest_path = args.output_dir / f"fine_tune_manifest_{run_id}.csv"
    calibration_path = args.output_dir / "openness_calibration.json"
    status_path = args.output_dir / "latest_status.json"

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

    calibration = build_calibration(rows, csv_path, options)
    calibration_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_training_images:
        write_training_manifest(rows, manifest_path, run_id, args.validation_every)
    status = {
        "state": "complete",
        "error": None,
        "csv": str(csv_path.resolve()),
        "calibration": str(calibration_path.resolve()),
        "manifest": str(manifest_path.resolve()) if args.save_training_images else None,
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


def build_calibration(rows: list[dict[str, object]], csv_path: Path, options: OpennessOptions) -> dict[str, object]:
    open_stages = {"open", "open_confirm"}
    closed_stages = {"closed"}

    def side_payload(side: str) -> dict[str, object]:
        open_rows = [row for row in rows if row["stage"] in open_stages]
        closed_rows = [row for row in rows if row["stage"] in closed_stages]
        open_scores = [float(row[f"{side}_score"]) for row in open_rows]
        closed_scores = [float(row[f"{side}_score"]) for row in closed_rows]
        open_apertures = [float(row[f"{side}_aperture_height"]) for row in open_rows]
        closed_apertures = [float(row[f"{side}_aperture_height"]) for row in closed_rows]
        open_peaks = [float(row[f"{side}_peak_dark_fraction"]) for row in open_rows]
        closed_peaks = [float(row[f"{side}_peak_dark_fraction"]) for row in closed_rows]
        open_score = median(open_scores)
        closed_score = median(closed_scores)
        return {
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

    return {
        "schema": "dreamair.openness_calibration.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv": str(csv_path.resolve()),
        "score": "aperture_height * peak_dark_fraction",
        "roi": [options.roi_x0, options.roi_y0, options.roi_x1, options.roi_y1],
        "left": side_payload("left"),
        "right": side_payload("right"),
    }


def write_training_manifest(rows: list[dict[str, object]], output: Path, session_id: str, validation_every: int) -> None:
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
    by_stage_seen: dict[str, int] = {}
    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        left_file = str(row.get("left_file") or "")
        right_file = str(row.get("right_file") or "")
        if not left_file or not right_file:
            continue
        stage = str(row["stage"])
        by_stage_seen[stage] = by_stage_seen.get(stage, 0) + 1
        split = "val" if by_stage_seen[stage] % validation_every == 0 else "train"
        sequence = int(row["sequence"])
        manifest_rows.append(
            {
                "sample_id": f"{session_id}_openness_{stage}_{sequence}",
                "session": session_id,
                "stage": stage,
                "target_x": "0.000000",
                "target_y": "0.000000",
                "left_file": left_file,
                "right_file": right_file,
                "left_conf": "1.000000",
                "right_conf": "1.000000",
                "left_center_x": "0.000000",
                "left_center_y": "0.000000",
                "right_center_x": "0.000000",
                "right_center_y": "0.000000",
                "source": "live_openness_training",
                "sample_weight": "1.0",
                "split": split,
            }
        )
    if not manifest_rows:
        raise SystemExit("No saved training image rows were available for the openness manifest.")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    raise SystemExit(main())
