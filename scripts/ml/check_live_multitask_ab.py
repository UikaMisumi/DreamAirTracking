#!/usr/bin/env python3
"""Check live multitask A/B reports for runtime-quality gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


STAGE_PRESETS = {
    "five": ["center", "left", "right", "up", "down"],
    "nine": ["center", "left", "right", "up", "down", "left_up", "right_up", "left_down", "right_down"],
}


def stat(summary: dict[str, Any], field: str, key: str, default: float | None = None) -> float | None:
    value = summary.get("fields", {}).get(field, {}).get(key, default)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def add_check(checks: list[dict[str, Any]], name: str, value: Any, passed: bool, severity: str, detail: str = "") -> None:
    checks.append(
        {
            "name": name,
            "value": value,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
        }
    )


def check_run(run: dict[str, Any], args: argparse.Namespace, required_stages: list[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    summary = run.get("summary") or {}
    row_count = int(summary.get("row_count") or 0)
    fps = float(summary.get("fps") or 0.0)
    delta_p95 = stat(summary, "delta_ms", "p95")
    pair_conf_median = stat(summary, "pair_confidence", "median")
    left_conf_median = stat(summary, "left_confidence", "median")
    right_conf_median = stat(summary, "right_confidence", "median")
    openness_min_median = None
    if summary.get("by_stage"):
        stage_openness = [
            float(stage_summary.get("openness_min", {}).get("median"))
            for stage_summary in summary["by_stage"].values()
            if stage_summary.get("openness_min", {}).get("median") is not None
        ]
        if stage_openness:
            openness_min_median = min(stage_openness)

    add_check(checks, "returncode_zero", run.get("returncode"), run.get("returncode") == 0, "hard")
    add_check(checks, "csv_present", run.get("csv"), bool(run.get("csv")), "hard")
    add_check(checks, "min_rows", row_count, row_count >= args.min_rows, "hard", f">= {args.min_rows}")
    add_check(checks, "min_fps", fps, fps >= args.min_fps, "hard", f">= {args.min_fps}")
    add_check(
        checks,
        "delta_ms_p95",
        delta_p95,
        delta_p95 is not None and delta_p95 <= args.max_delta_p95_ms,
        "hard",
        f"<= {args.max_delta_p95_ms}",
    )
    add_check(
        checks,
        "pair_confidence_median",
        pair_conf_median,
        pair_conf_median is not None and pair_conf_median >= args.min_pair_confidence_median,
        "warning",
        f">= {args.min_pair_confidence_median}",
    )
    add_check(
        checks,
        "left_confidence_median",
        left_conf_median,
        left_conf_median is not None and left_conf_median >= args.min_eye_confidence_median,
        "warning",
        f">= {args.min_eye_confidence_median}",
    )
    add_check(
        checks,
        "right_confidence_median",
        right_conf_median,
        right_conf_median is not None and right_conf_median >= args.min_eye_confidence_median,
        "warning",
        f">= {args.min_eye_confidence_median}",
    )
    if openness_min_median is not None:
        add_check(
            checks,
            "openness_min_stage_median",
            openness_min_median,
            openness_min_median >= args.min_openness_stage_median,
            "warning",
            f">= {args.min_openness_stage_median}",
        )

    if required_stages:
        by_stage = summary.get("by_stage", {})
        for stage in required_stages:
            stage_count = int(by_stage.get(stage, {}).get("count") or 0)
            add_check(
                checks,
                f"stage_{stage}_min_rows",
                stage_count,
                stage_count >= args.min_stage_rows,
                "hard",
                f">= {args.min_stage_rows}",
            )

    hard_passed = all(check["passed"] for check in checks if check["severity"] == "hard")
    warnings = [check for check in checks if check["severity"] == "warning" and not check["passed"]]
    return {
        "variant": run.get("variant"),
        "label": run.get("label"),
        "passed": hard_passed,
        "warning_count": len(warnings),
        "checks": checks,
    }


def stage_median(summary: dict[str, Any], stage: str, key: str) -> tuple[float, float] | None:
    value = summary.get("by_stage", {}).get(stage, {}).get(key, {}).get("median")
    if not isinstance(value, list) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def compare_runs(runs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if len(runs) < 2:
        return []
    comparisons: list[dict[str, Any]] = []
    baseline = runs[0]
    baseline_summary = baseline.get("summary") or {}
    baseline_stages = set((baseline_summary.get("by_stage") or {}).keys())
    for run in runs[1:]:
        summary = run.get("summary") or {}
        shared_stages = sorted(stage for stage in baseline_stages.intersection((summary.get("by_stage") or {}).keys()) if stage != "-")
        max_delta = 0.0
        max_stage = None
        for stage in shared_stages:
            left = stage_median(baseline_summary, stage, "smooth")
            right = stage_median(summary, stage, "smooth")
            if left is None or right is None:
                continue
            delta = math.hypot(left[0] - right[0], left[1] - right[1])
            if delta > max_delta:
                max_delta = delta
                max_stage = stage
        comparisons.append(
            {
                "baseline": baseline.get("label"),
                "candidate": run.get("label"),
                "shared_stage_count": len(shared_stages),
                "max_smooth_stage_median_delta": max_delta,
                "max_delta_stage": max_stage,
                "warning": bool(shared_stages and max_delta > args.max_ab_smooth_delta_warning),
                "warning_threshold": args.max_ab_smooth_delta_warning,
            }
        )
    return comparisons


def check_report(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    runs = payload.get("runs") or []
    required_stages = STAGE_PRESETS.get(str(payload.get("direction_preset") or ""), [])
    run_reports = [check_run(run, args, required_stages) for run in runs]
    report = {
        "source_state": payload.get("state"),
        "passed": payload.get("state") == "complete" and bool(run_reports) and all(run["passed"] for run in run_reports),
        "direction_preset": payload.get("direction_preset"),
        "run_count": len(run_reports),
        "runs": run_reports,
        "comparisons": compare_runs(runs, args),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--min-stage-rows", type=int, default=5)
    parser.add_argument("--min-fps", type=float, default=12.0)
    parser.add_argument("--max-delta-p95-ms", type=float, default=30.0)
    parser.add_argument("--min-pair-confidence-median", type=float, default=0.50)
    parser.add_argument("--min-eye-confidence-median", type=float, default=0.50)
    parser.add_argument("--min-openness-stage-median", type=float, default=0.05)
    parser.add_argument("--max-ab-smooth-delta-warning", type=float, default=0.35)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    checked = check_report(payload, args)
    text = json.dumps(checked, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if checked["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
