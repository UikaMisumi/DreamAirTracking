#!/usr/bin/env python3
"""Summarize live multitask runtime CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


NUMERIC_FIELDS = [
    "timestamp",
    "stage_elapsed",
    "delta_ms",
    "raw_x",
    "raw_y",
    "mapped_x",
    "mapped_y",
    "smooth_x",
    "smooth_y",
    "left_model_openness",
    "right_model_openness",
    "left_openness",
    "right_openness",
    "left_model_wide",
    "right_model_wide",
    "left_model_squint",
    "right_model_squint",
    "left_wide",
    "right_wide",
    "left_squint",
    "right_squint",
    "left_pupil_expression",
    "right_pupil_expression",
    "left_pupil_x",
    "left_pupil_y",
    "left_pupil_radius",
    "right_pupil_x",
    "right_pupil_y",
    "right_pupil_radius",
    "left_confidence",
    "right_confidence",
    "pair_confidence",
]


def parse_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_vector(rows: list[dict[str, str]], x_key: str, y_key: str) -> dict[str, Any]:
    xs = [value for row in rows if (value := parse_float(row, x_key)) is not None]
    ys = [value for row in rows if (value := parse_float(row, y_key)) is not None]
    if not xs or not ys:
        return {"count": 0}
    return {
        "count": min(len(xs), len(ys)),
        "x": summarize(xs),
        "y": summarize(ys),
        "median": [float(np.median(xs)), float(np.median(ys))],
        "mean": [float(np.mean(xs)), float(np.mean(ys))],
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_csv(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    timestamps = [value for row in rows if (value := parse_float(row, "timestamp")) is not None]
    duration = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    by_field = {
        field: summarize([value for row in rows if (value := parse_float(row, field)) is not None])
        for field in NUMERIC_FIELDS
    }
    by_stage: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("stage", "") or "-"].append(row)
    for stage, stage_rows in sorted(grouped.items()):
        by_stage[stage] = {
            "count": len(stage_rows),
            "raw": summarize_vector(stage_rows, "raw_x", "raw_y"),
            "smooth": summarize_vector(stage_rows, "smooth_x", "smooth_y"),
            "openness_min": summarize(
                [
                    min(left, right)
                    for row in stage_rows
                    if (left := parse_float(row, "left_openness")) is not None
                    and (right := parse_float(row, "right_openness")) is not None
                ]
            ),
            "wide_max": summarize(
                [
                    max(left, right)
                    for row in stage_rows
                    if (left := parse_float(row, "left_wide")) is not None
                    and (right := parse_float(row, "right_wide")) is not None
                ]
            ),
            "squint_max": summarize(
                [
                    max(left, right)
                    for row in stage_rows
                    if (left := parse_float(row, "left_squint")) is not None
                    and (right := parse_float(row, "right_squint")) is not None
                ]
            ),
            "pair_confidence": summarize(
                [value for row in stage_rows if (value := parse_float(row, "pair_confidence")) is not None]
            ),
        }
    return {
        "csv": str(path.resolve()),
        "row_count": len(rows),
        "duration_seconds": duration,
        "fps": len(rows) / duration if duration > 0 else 0.0,
        "fields": by_field,
        "by_stage": by_stage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True, type=Path, help="CSV file to summarize; repeatable.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "files": [summarize_csv(path.resolve()) for path in args.csv],
    }
    report["total_rows"] = sum(item["row_count"] for item in report["files"])
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
