#!/usr/bin/env python3
"""Fit a quick affine gaze calibration layer from a guided live_gaze CSV."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def fit_affine(raw: np.ndarray, target: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([raw.astype(np.float64), np.ones((raw.shape[0], 1), dtype=np.float64)], axis=1)
    y = target.astype(np.float64)
    penalty = np.eye(3, dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return weights[:2, :].T.astype(np.float32), weights[2, :].astype(np.float32)


def apply_affine(raw: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (raw @ a.T + b).astype(np.float32)


def l2_metrics(raw: np.ndarray, target: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    baseline = np.linalg.norm(raw - target, axis=1)
    corrected = np.linalg.norm(apply_affine(raw, a, b) - target, axis=1)
    return {
        "baseline_mean_l2": float(np.mean(baseline)),
        "baseline_median_l2": float(np.median(baseline)),
        "baseline_p90_l2": float(np.quantile(baseline, 0.9)),
        "corrected_mean_l2": float(np.mean(corrected)),
        "corrected_median_l2": float(np.median(corrected)),
        "corrected_p90_l2": float(np.quantile(corrected, 0.9)),
        "sample_count": int(raw.shape[0]),
    }


def read_rows(path: Path, source: str, min_stage_elapsed: float) -> list[dict[str, object]]:
    x_key = f"{source}_x"
    y_key = f"{source}_y"
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            stage = (raw.get("stage") or "").strip()
            if stage not in STAGE_TARGETS:
                continue
            stage_elapsed = float(raw.get("stage_elapsed") or 0.0)
            if stage_elapsed < min_stage_elapsed:
                continue
            try:
                x = float(raw[x_key])
                y = float(raw[y_key])
            except (KeyError, TypeError, ValueError):
                continue
            tx, ty = STAGE_TARGETS[stage]
            rows.append(
                {
                    "stage": stage,
                    "x": x,
                    "y": y,
                    "target_x": tx,
                    "target_y": ty,
                    "stage_elapsed": stage_elapsed,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Guided live_gaze_*.csv with stage labels.")
    parser.add_argument("--output", type=Path, help="Output quick calibration layer JSON.")
    parser.add_argument("--checkpoint", type=Path, help="ONNX checkpoint path to record in the layer metadata.")
    parser.add_argument("--source", choices=("raw", "corrected", "smooth"), default="raw")
    parser.add_argument("--min-stage-elapsed", type=float, default=0.5, help="Discard early rows after each prompt switch.")
    parser.add_argument("--alpha", type=float, default=1e-3)
    args = parser.parse_args()

    rows = read_rows(args.csv.resolve(), args.source, args.min_stage_elapsed)
    by_stage: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_stage[str(row["stage"])].append(row)

    if len(by_stage) < 3:
        raise SystemExit("Need at least three guided stages to fit affine quick layer.")

    stage_raw: list[np.ndarray] = []
    stage_target: list[np.ndarray] = []
    stage_summaries: dict[str, dict[str, object]] = {}
    for stage in STAGE_TARGETS:
        stage_rows = by_stage.get(stage)
        if not stage_rows:
            continue
        values = np.array([[float(row["x"]), float(row["y"])] for row in stage_rows], dtype=np.float32)
        target = np.array(STAGE_TARGETS[stage], dtype=np.float32)
        median = np.median(values, axis=0).astype(np.float32)
        stage_raw.append(median)
        stage_target.append(target)
        stage_summaries[stage] = {
            "sample_count": len(stage_rows),
            f"{args.source}_median": [float(median[0]), float(median[1])],
            "target": [float(target[0]), float(target[1])],
        }

    fit_raw = np.stack(stage_raw)
    fit_target = np.stack(stage_target)
    a, b = fit_affine(fit_raw, fit_target, args.alpha)

    all_raw = np.array([[float(row["x"]), float(row["y"])] for row in rows], dtype=np.float32)
    all_target = np.array([[float(row["target_x"]), float(row["target_y"])] for row in rows], dtype=np.float32)
    stage_metrics = l2_metrics(fit_raw, fit_target, a, b)
    all_metrics = l2_metrics(all_raw, all_target, a, b)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or args.csv.with_name(f"quick_layer_from_live_{run_id}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = {
        "schema": "dream_air_tracking.quick_gaze_calibration_layer.v1",
        "formula": "corrected = A @ raw + b",
        "raw_source": "model_prediction_xy",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "manifest": str(args.csv.resolve()),
        "fit_source": args.source,
        "min_stage_elapsed": args.min_stage_elapsed,
        "sessions": {
            run_id: {
                "used": True,
                "calibration_samples": len(rows),
                "stages": list(stage_summaries.keys()),
                "stage_summaries": stage_summaries,
                "A": a.tolist(),
                "b": b.tolist(),
                "raw_weights": np.vstack([a.T, b]).astype(float).tolist(),
                "stage_medians": stage_metrics,
                "all": all_metrics,
            }
        },
    }
    output.write_text(json.dumps(layer, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "state": "complete",
                "layer": str(output.resolve()),
                "stage_count": len(stage_summaries),
                "sample_count": len(rows),
                "source": args.source,
                "stage_medians": stage_metrics,
                "all": all_metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
