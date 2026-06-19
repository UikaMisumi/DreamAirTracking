#!/usr/bin/env python3
"""Fit a Bridge-output quick calibration layer from an accepted calibration session."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_STAGES = (
    "center",
    "up",
    "down",
    "left",
    "right",
    "left_up",
    "right_up",
    "left_down",
    "right_down",
)


def load_accepted_labels(session_dir: Path) -> list[dict[str, object]]:
    path = session_dir / "labels.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing labels.jsonl beside {session_dir}")
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if bool(item.get("accepted", False)):
            labels.append(item)
    return labels


def label_for(row: dict[str, str], labels: list[dict[str, object]]) -> tuple[float, float] | None:
    stage = row.get("stage", "")
    sequence = int(float(row.get("sequence") or row.get("pair") or "0"))
    for label in labels:
        if label.get("stageId") != stage:
            continue
        if int(label.get("frameStart", 0)) <= sequence <= int(label.get("frameEnd", -1)):
            target = label.get("operatorOverrideTarget") or label.get("target")
            return float(target["x"]), float(target["y"])
    return None


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def transform(raw_x: float, raw_y: float, eye: dict[str, object]) -> tuple[float, float]:
    item = dict(eye.get("inputTransform") or {})
    if not bool(item.get("enabled", False)):
        return raw_x, raw_y
    scale_x = float(item.get("scaleX", 1) or 1)
    scale_y = float(item.get("scaleY", 1) or 1)
    if abs(scale_x) < 1e-9:
        scale_x = 1
    if abs(scale_y) < 1e-9:
        scale_y = 1
    return (
        float(item.get("targetCenterX", 0)) + ((raw_x - float(item.get("currentCenterX", 0))) * scale_x),
        float(item.get("targetCenterY", 0)) + ((raw_y - float(item.get("currentCenterY", 0))) * scale_y),
    )


def normalize_axis(value: float, center: float, axis_range: float, sign: float, offset: float) -> float:
    if axis_range <= 0:
        return 0.0
    output_sign = -1.0 if sign < 0 else 1.0
    return max(-1.0, min(1.0, (((value - center) / axis_range) * output_sign) + offset))


def map_eye(row: dict[str, str], side: str, profile: dict[str, object]) -> tuple[bool, float, float, float]:
    found = parse_bool(row.get(f"{side}_found", ""))
    if not found:
        return False, 0.0, 0.0, 0.0
    eye = dict(profile[side])
    center_x = row.get(f"{side}_center_x") or row.get(f"{side}_raw_x")
    center_y = row.get(f"{side}_center_y") or row.get(f"{side}_raw_y")
    if center_x is None or center_y is None:
        return False, 0.0, 0.0, 0.0
    raw_x, raw_y = transform(float(center_x), float(center_y), eye)
    calibration = dict(eye.get("calibration") or {})
    x = normalize_axis(
        raw_x,
        float(calibration.get("centerX", 100)),
        float(calibration.get("horizontalRange", 40)),
        float(calibration.get("horizontalOutputSign", 1)),
        float(calibration.get("horizontalOutputOffset", 0)),
    )
    y = normalize_axis(
        raw_y,
        float(calibration.get("centerY", 100)),
        float(calibration.get("verticalRange", 30)),
        float(calibration.get("verticalOutputSign", 1)),
        float(calibration.get("verticalOutputOffset", 0)),
    )
    return True, x, y, float(row.get(f"{side}_conf") or row.get(f"{side}_confidence") or 0)


def apply_output_axis(value: float, deadzone: float, gain: float, offset: float) -> float:
    abs_value = abs(value)
    sign = -1.0 if value < 0 else 1.0
    normalized = 0.0 if abs_value <= deadzone else (abs_value - deadzone) / max(0.001, 1 - deadzone)
    return max(-1.0, min(1.0, (normalized * sign * gain) + offset))


def bridge_output_xy(row: dict[str, str], profile: dict[str, object]) -> tuple[float, float] | None:
    left_found, left_x, left_y, left_conf = map_eye(row, "left", profile)
    right_found, right_x, right_y, right_conf = map_eye(row, "right", profile)
    if not left_found and not right_found:
        return None
    if left_found and right_found:
        left_weight = max(left_conf, 0.001)
        right_weight = max(right_conf, 0.001)
        total = left_weight + right_weight
        x = ((left_x * left_weight) + (right_x * right_weight)) / total
        y = ((left_y * left_weight) + (right_y * right_weight)) / total
    elif left_found:
        x, y = left_x, left_y
    else:
        x, y = right_x, right_y
    output = dict(profile.get("output") or {})
    return (
        apply_output_axis(x, float(output.get("centerDeadzoneX", 0.12)), float(output.get("outputGainX", 1)), float(output.get("outputOffsetX", 0))),
        apply_output_axis(y, float(output.get("centerDeadzoneY", 0.22)), float(output.get("outputGainY", 1)), float(output.get("outputOffsetY", 0))),
    )


def fit_affine(source: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    x = np.concatenate([source.astype(np.float64), np.ones((source.shape[0], 1), dtype=np.float64)], axis=1)
    y = target.astype(np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def weights_to_layer(weights: np.ndarray) -> tuple[list[list[float]], list[float]]:
    return (
        [
            [float(weights[0, 0]), float(weights[1, 0])],
            [float(weights[0, 1]), float(weights[1, 1])],
        ],
        [float(weights[2, 0]), float(weights[2, 1])],
    )


def apply_layer(source: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.concatenate([source.astype(np.float64), np.ones((source.shape[0], 1), dtype=np.float64)], axis=1)
    return np.clip(x @ weights, -1, 1).astype(np.float32)


def summarize(errors: list[float]) -> dict[str, float]:
    ordered = sorted(errors)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1)) if ordered else 0
    return {
        "count": len(errors),
        "mean_l2": statistics.fmean(errors) if errors else math.nan,
        "p90_l2": ordered[p90_index] if ordered else math.nan,
        "max_l2": max(errors) if errors else math.nan,
    }


def summarize_rows(rows: list[dict[str, object]], corrected: np.ndarray, selected_ids: set[str]) -> dict[str, object]:
    all_source = np.array([row["raw"] for row in rows], dtype=np.float32)
    all_target = np.array([row["target"] for row in rows], dtype=np.float32)
    base_errors = [float(np.linalg.norm(raw - tgt)) for raw, tgt in zip(all_source, all_target)]
    corrected_errors = [float(np.linalg.norm(raw - tgt)) for raw, tgt in zip(corrected, all_target)]
    calibration_indexes = [index for index, row in enumerate(rows) if str(row["sample_id"]) in selected_ids]
    heldout_indexes = [index for index, row in enumerate(rows) if str(row["sample_id"]) not in selected_ids]

    def pick(values: list[float], indexes: list[int]) -> list[float]:
        return [values[index] for index in indexes]

    return {
        "all": {
            "baseline_bridge_output": summarize(base_errors),
            "corrected_bridge_output": summarize(corrected_errors),
        },
        "calibration": {
            "baseline_bridge_output": summarize(pick(base_errors, calibration_indexes)),
            "corrected_bridge_output": summarize(pick(corrected_errors, calibration_indexes)),
        },
        "heldout": {
            "baseline_bridge_output": summarize(pick(base_errors, heldout_indexes)),
            "corrected_bridge_output": summarize(pick(corrected_errors, heldout_indexes)),
        },
    }


def make_runtime_recommendation(
    metrics: dict[str, object],
    min_mean_improvement: float,
    max_corrected_mean: float,
) -> dict[str, object]:
    heldout = dict(metrics.get("heldout") or {})
    baseline = dict(heldout.get("baseline_bridge_output") or {})
    corrected = dict(heldout.get("corrected_bridge_output") or {})
    baseline_mean = baseline.get("mean_l2")
    corrected_mean = corrected.get("mean_l2")
    if not isinstance(baseline_mean, (int, float)) or not isinstance(corrected_mean, (int, float)):
        return {
            "enable": False,
            "reason": "heldout metrics unavailable",
        }
    improvement = float(baseline_mean) - float(corrected_mean)
    if improvement < min_mean_improvement:
        return {
            "enable": False,
            "reason": f"heldout mean improvement {improvement:.4f} is below required {min_mean_improvement:.4f}",
            "heldout_mean_improvement": improvement,
        }
    if float(corrected_mean) > max_corrected_mean:
        return {
            "enable": False,
            "reason": f"heldout corrected mean {float(corrected_mean):.4f} is above allowed {max_corrected_mean:.4f}",
            "heldout_mean_improvement": improvement,
        }
    return {
        "enable": True,
        "reason": "heldout improvement and absolute error pass the runtime gate",
        "heldout_mean_improvement": improvement,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-stage", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--min-heldout-mean-improvement", type=float, default=0.08)
    parser.add_argument("--max-heldout-corrected-mean", type=float, default=0.20)
    args = parser.parse_args()

    profile = json.load(args.profile.open("r", encoding="utf-8"))
    labels = load_accepted_labels(args.pairs.parent)
    rows = []
    with args.pairs.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            target = label_for(row, labels)
            if target is None:
                continue
            raw = bridge_output_xy(row, profile)
            if raw is None:
                continue
            rows.append(
                {
                    "sample_id": f"{args.pairs.parent.name}_{row.get('stage')}_{row.get('sequence')}",
                    "stage": row.get("stage", ""),
                    "raw": raw,
                    "target": target,
                }
            )

    selected = []
    seen = Counter()
    for row in rows:
        if seen[row["stage"]] < args.samples_per_stage:
            selected.append(row)
            seen[row["stage"]] += 1
    if len(selected) < 3:
        raise SystemExit("Need at least 3 accepted bridge-output samples to fit quick calibration.")

    source = np.array([row["raw"] for row in selected], dtype=np.float32)
    target = np.array([row["target"] for row in selected], dtype=np.float32)
    weights = fit_affine(source, target, args.ridge_alpha)
    all_source = np.array([row["raw"] for row in rows], dtype=np.float32)
    all_target = np.array([row["target"] for row in rows], dtype=np.float32)
    corrected = apply_layer(all_source, weights)
    a, b = weights_to_layer(weights)
    stages = sorted({row["stage"] for row in rows})
    selected_ids = {str(row["sample_id"]) for row in selected}
    metrics = summarize_rows(rows, corrected, selected_ids)
    recommendation = make_runtime_recommendation(
        metrics,
        args.min_heldout_mean_improvement,
        args.max_heldout_corrected_mean,
    )
    payload = {
        "schema": "dream_air_tracking.quick_gaze_calibration_layer.v1",
        "formula": "corrected = A @ raw + b",
        "raw_source": "bridge_output_xy",
        "profile": str(args.profile.resolve()),
        "pairs": str(args.pairs.resolve()),
        "samples_per_stage": args.samples_per_stage,
        "sessions": {
            args.pairs.parent.name: {
                "used": True,
                "calibration_samples": len(selected),
                "calibration_sample_ids": [row["sample_id"] for row in selected],
                "stages": stages,
                "missing_required_stages": [stage for stage in REQUIRED_STAGES if stage not in stages],
                "A": a,
                "b": b,
                "raw_weights": weights.tolist(),
                **metrics,
                "runtime_recommendation": recommendation,
            }
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload["sessions"][args.pairs.parent.name], indent=2, ensure_ascii=False))
    print(f"Wrote bridge quick calibration layer: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
