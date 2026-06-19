#!/usr/bin/env python3
"""Estimate simple center-offset and x/y gain from a guided gaze CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


REQUIRED_STAGES = ["center", "left", "right", "up", "down"]


def median_xy(rows: list[dict[str, str]], x_key: str, y_key: str) -> tuple[float, float]:
    return (
        statistics.median(float(row[x_key]) for row in rows),
        statistics.median(float(row[y_key]) for row in rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--label", help="Model label from live_gaze_compare_models.py, e.g. tiny or mobilenet.")
    parser.add_argument("--source", choices=("raw", "smooth", "corrected"), default="raw")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--offset-mode",
        choices=("axis-midpoint", "center"),
        default="axis-midpoint",
        help="Use left/right and up/down midpoints by default; center can be noisy in guided checks.",
    )
    parser.add_argument("--min-gain-denominator", type=float, default=0.15)
    args = parser.parse_args()

    x_key = f"{args.source}_x_{args.label}" if args.label else f"{args.source}_x"
    y_key = f"{args.source}_y_{args.label}" if args.label else f"{args.source}_y"
    rows = list(csv.DictReader(args.csv.resolve().open("r", encoding="utf-8-sig", newline="")))
    by_stage = {stage: [row for row in rows if row.get("stage") == stage] for stage in REQUIRED_STAGES}
    missing = [stage for stage, stage_rows in by_stage.items() if not stage_rows]
    if missing:
        raise SystemExit(f"Missing required stages: {', '.join(missing)}")

    med = {stage: median_xy(stage_rows, x_key, y_key) for stage, stage_rows in by_stage.items()}
    x_span = med["right"][0] - med["left"][0]
    y_span = med["up"][1] - med["down"][1]
    if args.offset_mode == "center":
        offset_x, offset_y = med["center"]
    else:
        offset_x = (med["left"][0] + med["right"][0]) * 0.5
        offset_y = (med["up"][1] + med["down"][1]) * 0.5
    x_gain = 1.0 if abs(x_span) < args.min_gain_denominator else 2.0 / x_span
    y_gain = 1.0 if abs(y_span) < args.min_gain_denominator else 2.0 / y_span
    result = {
        "csv": str(args.csv.resolve()),
        "label": args.label,
        "source": args.source,
        "offset_mode": args.offset_mode,
        "stage_medians": {stage: [float(x), float(y)] for stage, (x, y) in med.items()},
        "center_offset_x": float(offset_x),
        "center_offset_y": float(offset_y),
        "observed_center_x": float(med["center"][0]),
        "observed_center_y": float(med["center"][1]),
        "x_span": float(x_span),
        "y_span": float(y_span),
        "x_gain": float(x_gain),
        "y_gain": float(y_gain),
        "dry_run_args": (
            f"--center-offset-x {offset_x:.6f} --center-offset-y {offset_y:.6f} "
            f"--x-gain {x_gain:.6f} --y-gain {y_gain:.6f}"
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
