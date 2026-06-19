#!/usr/bin/env python3
"""Apply a quick gaze calibration layer to prediction CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def l2(x0: float, y0: float, x1: float, y1: float) -> float:
    dx = x0 - x1
    dy = y0 - y1
    return math.sqrt((dx * dx) + (dy * dy))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def apply_layer(raw_x: float, raw_y: float, layer: dict[str, object]) -> tuple[float, float]:
    a = layer["A"]
    b = layer["b"]
    return (
        float(a[0][0]) * raw_x + float(a[0][1]) * raw_y + float(b[0]),
        float(a[1][0]) * raw_x + float(a[1][1]) * raw_y + float(b[1]),
    )


def summarize(rows: list[dict[str, str]], prefix: str) -> dict[str, float]:
    errors = [float(row[f"{prefix}_l2"]) for row in rows]
    return {
        "count": len(errors),
        "mean_l2": statistics.fmean(errors) if errors else math.nan,
        "p90_l2": percentile(errors, 90),
        "max_l2": max(errors) if errors else math.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    payload = json.load(args.layer.open("r", encoding="utf-8"))
    layers = dict(payload.get("sessions", {}))
    corrected_rows: list[dict[str, str]] = []
    for row in rows:
        session_layer = layers.get(row["session"])
        raw_x = float(row["model_pred_x"])
        raw_y = float(row["model_pred_y"])
        if isinstance(session_layer, dict) and session_layer.get("used") and "A" in session_layer and "b" in session_layer:
            out_x, out_y = apply_layer(raw_x, raw_y, session_layer)
            layer_used = "true"
        else:
            out_x, out_y = raw_x, raw_y
            layer_used = "false"
        target_x = float(row["target_x"])
        target_y = float(row["target_y"])
        corrected = {
            **row,
            "quick_layer_used": layer_used,
            "quick_layer_pred_x": f"{out_x:.9f}",
            "quick_layer_pred_y": f"{out_y:.9f}",
            "quick_layer_l2": f"{l2(out_x, out_y, target_x, target_y):.9f}",
        }
        corrected_rows.append(corrected)

    extra_fields = ["quick_layer_used", "quick_layer_pred_x", "quick_layer_pred_y", "quick_layer_l2"]
    output_fields = fieldnames + [field for field in extra_fields if field not in fieldnames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(corrected_rows)

    metrics_path = args.metrics or args.output.with_suffix(".metrics.json")
    applied = [row for row in corrected_rows if row["quick_layer_used"] == "true"]
    heldout = [
        row for row in applied
        if str(row.get("used_for_quick_calibration", "")).lower() not in {"true", "1", "yes"}
    ]
    metrics = {
        "predictions": str(args.predictions.resolve()),
        "layer": str(args.layer.resolve()),
        "output": str(args.output.resolve()),
        "all_with_layer": summarize(applied, "quick_layer"),
        "heldout_with_layer": summarize(heldout, "quick_layer"),
        "baseline_model_on_same_heldout": summarize(heldout, "model") if heldout else {"count": 0},
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
