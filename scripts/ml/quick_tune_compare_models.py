#!/usr/bin/env python3
"""Compare same-frame gaze models before and after affine quick tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
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


def l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - target, axis=1)


def metrics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def ordering_ok(medians: dict[str, tuple[float, float]]) -> bool:
    return bool(
        medians["left"][0] < medians["center"][0] < medians["right"][0]
        and medians["down"][1] < medians["center"][1] < medians["up"][1]
        and medians["left_up"][0] < medians["right_up"][0]
        and medians["left_down"][0] < medians["right_down"][0]
        and medians["left_down"][1] < medians["left_up"][1]
        and medians["right_down"][1] < medians["right_up"][1]
    )


def read_stage_arrays(rows: list[dict[str, str]], label: str, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    stage_raw: list[np.ndarray] = []
    stage_target: list[tuple[float, float]] = []
    all_raw: list[list[float]] = []
    all_target: list[tuple[float, float]] = []
    counts: dict[str, int] = {}
    x_key = f"{source}_x_{label}"
    y_key = f"{source}_y_{label}"
    for stage, target in STAGE_TARGETS.items():
        current = [row for row in rows if row.get("stage") == stage]
        if not current:
            raise SystemExit(f"CSV has no rows for stage {stage}.")
        values = np.array([[float(row[x_key]), float(row[y_key])] for row in current], dtype=np.float64)
        counts[stage] = len(current)
        stage_raw.append(np.median(values, axis=0))
        stage_target.append(target)
        all_raw.extend(values.tolist())
        all_target.extend([target] * len(values))
    return (
        np.array(stage_raw, dtype=np.float64),
        np.array(stage_target, dtype=np.float64),
        np.array(all_raw, dtype=np.float64),
        np.array(all_target, dtype=np.float64),
        counts,
    )


def model_summary(rows: list[dict[str, str]], label: str, source: str, alpha: float) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    stage_raw, stage_target, all_raw, all_target, counts = read_stage_arrays(rows, label, source)
    a, b = fit_affine(stage_raw, stage_target, alpha)
    tuned_stage = stage_raw @ a.T + b
    tuned_all = all_raw @ a.T + b

    loso_errors: list[float] = []
    stages = list(STAGE_TARGETS)
    for index in range(len(stages)):
        mask = np.ones(len(stages), dtype=bool)
        mask[index] = False
        loso_a, loso_b = fit_affine(stage_raw[mask], stage_target[mask], alpha)
        pred = stage_raw[index : index + 1] @ loso_a.T + loso_b
        loso_errors.append(float(l2(pred, stage_target[index : index + 1])[0]))

    raw_medians = {stage: tuple(stage_raw[index]) for index, stage in enumerate(stages)}
    tuned_medians = {stage: tuple(tuned_stage[index]) for index, stage in enumerate(stages)}
    raw_stage_error = l2(stage_raw, stage_target)
    tuned_stage_error = l2(tuned_stage, stage_target)

    per_stage: dict[str, object] = {}
    for index, stage in enumerate(stages):
        per_stage[stage] = {
            "sample_count": counts[stage],
            "raw_median": [float(stage_raw[index, 0]), float(stage_raw[index, 1])],
            "tuned_median": [float(tuned_stage[index, 0]), float(tuned_stage[index, 1])],
            "target": [float(stage_target[index, 0]), float(stage_target[index, 1])],
            "raw_l2": float(raw_stage_error[index]),
            "tuned_l2": float(tuned_stage_error[index]),
            "loso_tuned_l2": float(loso_errors[index]),
        }

    return (
        {
            "raw_stage": metrics(raw_stage_error),
            "tuned_stage_same_fit": metrics(tuned_stage_error),
            "raw_all_frames": metrics(l2(all_raw, all_target)),
            "tuned_all_frames_same_fit": metrics(l2(tuned_all, all_target)),
            "loso_stage_tuned": metrics(np.array(loso_errors, dtype=np.float64)),
            "raw_ordering_ok": ordering_ok(raw_medians),
            "tuned_ordering_ok": ordering_ok(tuned_medians),
            "A": a.tolist(),
            "b": b.tolist(),
            "per_stage": per_stage,
        },
        a,
        b,
    )


def write_layer(output_dir: Path, run_id: str, label: str, csv_path: Path, checkpoint: str | None, summary: dict[str, object]) -> str:
    layer_path = output_dir / f"quick_layer_{label}_{run_id}.json"
    session = {
        "used": True,
        "calibration_samples": int(sum(stage["sample_count"] for stage in summary["per_stage"].values())),  # type: ignore[index,union-attr]
        "stages": list(STAGE_TARGETS),
        "stage_summaries": summary["per_stage"],
        "A": summary["A"],
        "b": summary["b"],
        "stage_medians": summary["tuned_stage_same_fit"],
        "all": summary["tuned_all_frames_same_fit"],
    }
    payload = {
        "schema": "dream_air_tracking.quick_gaze_calibration_layer.v1",
        "formula": "corrected = A @ raw + b",
        "raw_source": "model_prediction_xy",
        "checkpoint": checkpoint,
        "manifest": str(csv_path.resolve()),
        "sessions": {run_id: session},
    }
    layer_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(layer_path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="CSV from live_gaze_compare_models.py.")
    parser.add_argument("--label", action="append", required=True, help="Model label to compare; repeat.")
    parser.add_argument("--source", choices=("raw", "smooth"), default="raw")
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-layers", action="store_true", help="Write compatible quick-layer JSON files for each model.")
    parser.add_argument("--checkpoint", action="append", help="Optional label=onnx_path metadata for layer JSON; repeat.")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.resolve().open("r", encoding="utf-8-sig", newline="")))
    checkpoints: dict[str, str] = {}
    for item in args.checkpoint or []:
        if "=" not in item:
            raise SystemExit("--checkpoint must be formatted as label=onnx_path.")
        label, path = item.split("=", 1)
        checkpoints[label] = str(Path(path).resolve())

    output = args.output or args.csv.with_name(f"quick_tune_compare_{time.strftime('%Y%m%d_%H%M%S')}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    summaries: dict[str, object] = {}
    layers: dict[str, str] = {}
    for label in args.label:
        summary, _, _ = model_summary(rows, label, args.source, args.alpha)
        summaries[label] = summary
        if args.write_layers:
            layers[label] = write_layer(output.parent, run_id, label, args.csv.resolve(), checkpoints.get(label), summary)

    result = {
        "csv": str(args.csv.resolve()),
        "source": args.source,
        "alpha": args.alpha,
        "summary": summaries,
        "layers": layers,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote quick tune comparison: {output.resolve()}")
    for label, summary in summaries.items():
        raw = summary["raw_stage"]  # type: ignore[index]
        tuned = summary["tuned_stage_same_fit"]  # type: ignore[index]
        loso = summary["loso_stage_tuned"]  # type: ignore[index]
        print(
            f"{label}: raw median={raw['median']:.3f} mean={raw['mean']:.3f} "
            f"-> tuned median={tuned['median']:.3f} mean={tuned['mean']:.3f} "
            f"max={tuned['max']:.3f} loso_median={loso['median']:.3f} "
            f"order={summary['tuned_ordering_ok']}"
        )
    if layers:
        print(json.dumps({"layers": layers}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
