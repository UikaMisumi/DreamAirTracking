#!/usr/bin/env python3
"""Analyze predict_live CSV output for MobileNetV3 runtime acceptance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wear_templates import load_store, match_template, signature_from_csv  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_report(csv_path: Path, templates_path: Path | None) -> dict[str, Any]:
    rows = read_rows(csv_path)
    report: dict[str, Any] = {
        "csv": str(csv_path.resolve()),
        "row_count": len(rows),
        "duration_seconds": duration_seconds(rows),
        "gaze": gaze_metrics(rows),
        "openness": openness_metrics(rows),
        "normalization": normalization_metrics(rows),
    }

    if templates_path is not None:
        try:
            payload = signature_from_csv(csv_path)
            store = load_store(templates_path)
            report["wear_template"] = {
                "templates": str(templates_path.resolve()),
                "quality": payload["quality"],
                "match": match_template(payload["signature"], store),
            }
        except Exception as exc:
            report["wear_template"] = {"error": str(exc)}

    report["acceptance_hints"] = acceptance_hints(report)
    return report


def duration_seconds(rows: list[dict[str, str]]) -> float | None:
    values = [number(row.get("timestamp")) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    return max(values) - min(values)


def gaze_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    raw_x = series(rows, "raw_x")
    raw_y = series(rows, "raw_y")
    mapped_x = series(rows, "mapped_x")
    mapped_y = series(rows, "mapped_y")
    smooth_x = series(rows, "smooth_x")
    smooth_y = series(rows, "smooth_y")
    smooth_mag = [norm(x, y) for x, y in zip(smooth_x, smooth_y)]
    mapped_mag = [norm(x, y) for x, y in zip(mapped_x, mapped_y)]
    smooth_step = [norm(smooth_x[i] - smooth_x[i - 1], smooth_y[i] - smooth_y[i - 1]) for i in range(1, min(len(smooth_x), len(smooth_y)))]
    edge_count = sum(1 for value in mapped_mag if value >= 0.93)
    return {
        "raw_x": summary(raw_x),
        "raw_y": summary(raw_y),
        "mapped_x": summary(mapped_x),
        "mapped_y": summary(mapped_y),
        "smooth_x": summary(smooth_x),
        "smooth_y": summary(smooth_y),
        "smooth_magnitude": summary(smooth_mag),
        "smooth_step": summary(smooth_step),
        "edge_clamp_rate": edge_count / len(mapped_mag) if mapped_mag else None,
    }


def openness_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "left_raw": summary(series(rows, "left_openness_raw")),
        "right_raw": summary(series(rows, "right_openness_raw")),
        "left_calibrated": summary(series(rows, "left_openness_calibrated")),
        "right_calibrated": summary(series(rows, "right_openness_calibrated")),
        "left_output": summary(series(rows, "left_openness")),
        "right_output": summary(series(rows, "right_openness")),
        "left_aperture_height": summary(series(rows, "left_aperture_height")),
        "right_aperture_height": summary(series(rows, "right_aperture_height")),
    }


def normalization_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    modes = Counter(row.get("normalization_mode", "") for row in rows)
    reasons = Counter()
    for row in rows:
        reason = row.get("normalization_drop_reason", "")
        if reason:
            reasons.update(part.strip() for part in reason.split(";") if part.strip())
    left_found = boolean_rate(rows, "left_norm_found")
    right_found = boolean_rate(rows, "right_norm_found")
    return {
        "mode_counts": dict(modes),
        "left_found_rate": left_found,
        "right_found_rate": right_found,
        "left_confidence": summary(series(rows, "left_norm_confidence")),
        "right_confidence": summary(series(rows, "right_norm_confidence")),
        "left_shift_x": summary(series(rows, "left_norm_shift_x")),
        "left_shift_y": summary(series(rows, "left_norm_shift_y")),
        "right_shift_x": summary(series(rows, "right_norm_shift_x")),
        "right_shift_y": summary(series(rows, "right_norm_shift_y")),
        "left_scale": summary(series(rows, "left_norm_scale")),
        "right_scale": summary(series(rows, "right_norm_scale")),
        "top_drop_reasons": reasons.most_common(8),
    }


def acceptance_hints(report: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    row_count = int(report.get("row_count") or 0)
    if row_count < 60:
        hints.append(f"Need at least 60 rows for wear-template probe; got {row_count}.")

    normalization = report.get("normalization", {})
    left_conf = value_at(normalization, "left_confidence", "median")
    right_conf = value_at(normalization, "right_confidence", "median")
    modes = normalization.get("mode_counts")
    normalization_enabled = isinstance(modes, dict) and any(key != "off" and value for key, value in modes.items())
    if normalization_enabled and left_conf is not None and right_conf is not None and min(left_conf, right_conf) < 0.10:
        hints.append(f"Normalization median confidence is below 0.10 on at least one eye: L={left_conf:.3f}, R={right_conf:.3f}.")
    left_shift_x = value_at(normalization, "left_shift_x", "median")
    right_shift_x = value_at(normalization, "right_shift_x", "median")
    if normalization_enabled and left_shift_x is not None and right_shift_x is not None:
        if abs(left_shift_x) >= 0.175 or abs(right_shift_x) >= 0.175:
            hints.append(f"Normalization x-shift is near the configured limit: L={left_shift_x:.3f}, R={right_shift_x:.3f}.")
    left_scale = value_at(normalization, "left_scale", "median")
    right_scale = value_at(normalization, "right_scale", "median")
    if normalization_enabled and left_scale is not None and right_scale is not None:
        if min(left_scale, right_scale) <= 0.865:
            hints.append(f"Normalization scale is near the minimum limit: L={left_scale:.3f}, R={right_scale:.3f}.")

    gaze = report.get("gaze", {})
    step_p95 = value_at(gaze, "smooth_step", "p95")
    if step_p95 is not None and step_p95 > 0.18:
        hints.append(f"Smooth gaze p95 step is high: {step_p95:.3f}.")
    edge_rate = gaze.get("edge_clamp_rate")
    if isinstance(edge_rate, float) and edge_rate > 0.10:
        hints.append(f"Mapped gaze is near clamp edge often: {edge_rate:.1%}.")

    wear = report.get("wear_template", {})
    quality = wear.get("quality") if isinstance(wear, dict) else None
    if isinstance(quality, dict) and not quality.get("requirements_met"):
        hints.append("Wear-template probe requirements are not met.")

    if not hints:
        hints.append("No automatic red flags in this CSV. Review live behavior and contact sheets next.")
    return hints


def value_at(payload: dict[str, Any], key: str, metric: str) -> float | None:
    value = payload.get(key)
    if not isinstance(value, dict):
        return None
    output = value.get(metric)
    return output if isinstance(output, float | int) else None


def series(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = number(row.get(key))
        if value is not None:
            values.append(value)
    return values


def number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def boolean_rate(rows: list[dict[str, str]], key: str) -> float | None:
    if not rows:
        return None
    truthy = {"1", "true", "True", "yes", "Y"}
    return sum(1 for row in rows if str(row.get(key, "")).strip() in truthy) / len(rows)


def summary(values: list[float]) -> dict[str, float | int | None]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return {"count": 0, "min": None, "p50": None, "median": None, "p90": None, "p95": None, "max": None}
    clean.sort()
    return {
        "count": len(clean),
        "min": clean[0],
        "p50": percentile(clean, 50),
        "median": median(clean),
        "p90": percentile(clean, 90),
        "p95": percentile(clean, 95),
        "max": clean[-1],
    }


def percentile(values: list[float], percent: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (percent / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def norm(x: float, y: float) -> float:
    return math.sqrt((x * x) + (y * y))


def print_summary(report: dict[str, Any]) -> None:
    print(f"CSV: {report['csv']}")
    print(f"Rows: {report['row_count']} duration={report.get('duration_seconds')}")
    gaze = report["gaze"]
    print(f"Gaze smooth step p50/p95: {fmt(value_at(gaze, 'smooth_step', 'p50'))} / {fmt(value_at(gaze, 'smooth_step', 'p95'))}")
    print(f"Gaze edge clamp rate: {fmt_percent(gaze.get('edge_clamp_rate'))}")
    norm_report = report["normalization"]
    print(f"Normalization found L/R: {fmt_percent(norm_report.get('left_found_rate'))} / {fmt_percent(norm_report.get('right_found_rate'))}")
    print(f"Normalization confidence median L/R: {fmt(value_at(norm_report, 'left_confidence', 'median'))} / {fmt(value_at(norm_report, 'right_confidence', 'median'))}")
    wear = report.get("wear_template")
    if isinstance(wear, dict) and "match" in wear:
        match = wear["match"]
        print(f"Wear template: {match.get('status')} action={match.get('action')} distance={fmt(match.get('distance'))}")
    print("Hints:")
    for hint in report["acceptance_hints"]:
        print(f"- {hint}")


def fmt(value: Any) -> str:
    return "n/a" if not isinstance(value, float | int) else f"{value:.3f}"


def fmt_percent(value: Any) -> str:
    return "n/a" if not isinstance(value, float | int) else f"{value:.1%}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--templates", type=Path, default=Path("runs/wear_templates/wear_templates.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args.csv, args.templates)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
