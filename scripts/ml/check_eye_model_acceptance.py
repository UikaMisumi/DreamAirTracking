#!/usr/bin/env python3
"""Check eye model artifacts against task-specific offline acceptance gates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


REACH_GATES = {
    "left": 0.765,
    "right": 0.730,
    "up": 0.690,
    "down": 0.540,
}


def stage_reach(predictions: list[dict[str, Any]], split: str, stage: str) -> dict[str, float | int]:
    items = [
        item
        for item in predictions
        if item.get("split") == split
        and item.get("stage") == stage
        and float(item.get("gaze_weight", 1.0)) > 0.0
    ]
    if not items:
        return {"count": 0}
    target_x = float(items[0]["target_x"])
    target_y = float(items[0]["target_y"])
    pred_x = [float(item["gaze_x"]) for item in items]
    pred_y = [float(item["gaze_y"]) for item in items]
    l2 = [float(item["l2"]) for item in items]
    med_x = float(statistics.median(pred_x))
    med_y = float(statistics.median(pred_y))
    if abs(target_x) >= abs(target_y) and abs(target_x) > 1e-6:
        reach = med_x / target_x
    elif abs(target_y) > 1e-6:
        reach = med_y / target_y
    else:
        reach = float(statistics.median(math.hypot(x, y) for x, y in zip(pred_x, pred_y)))
    return {
        "count": len(items),
        "target_x": target_x,
        "target_y": target_y,
        "median_x": med_x,
        "median_y": med_y,
        "reach": reach,
        "l2_mean": float(statistics.fmean(l2)),
    }


def expression_median(audit: dict[str, Any], stage: str, key: str) -> float | None:
    value = audit.get("expression", {}).get("by_stage", {}).get(stage, {}).get(key, {})
    if not isinstance(value, dict):
        return None
    median = value.get("median")
    return None if median is None else float(median)


def mean_pair(values: tuple[float | None, float | None]) -> float | None:
    if values[0] is None or values[1] is None:
        return None
    return (values[0] + values[1]) * 0.5


def add_gate(gates: list[dict[str, Any]], name: str, value: float | None, op: str, threshold: float) -> None:
    if value is None:
        passed = False
    elif op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    else:
        raise ValueError(op)
    gates.append(
        {
            "name": name,
            "value": value,
            "op": op,
            "threshold": threshold,
            "passed": passed,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args()

    eval_payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    audit_payload = json.loads(args.audit_json.read_text(encoding="utf-8"))
    model = eval_payload["models"]["multitask"]
    predictions = model["predictions"]

    reaches = {stage: stage_reach(predictions, args.split, stage) for stage in ["center", *REACH_GATES]}
    gates: list[dict[str, Any]] = []
    add_gate(gates, "center_median_radius", float(reaches["center"].get("reach", math.inf)), "<=", 0.055)
    for stage, threshold in REACH_GATES.items():
        add_gate(gates, f"{stage}_reach", float(reaches[stage].get("reach", -math.inf)), ">=", threshold)

    pupil_left_mean = audit_payload["pupil"]["left"]["center_error"].get("mean")
    pupil_right_mean = audit_payload["pupil"]["right"]["center_error"].get("mean")
    add_gate(gates, "pupil_left_center_error_mean", pupil_left_mean, "<=", 0.020)
    add_gate(gates, "pupil_right_center_error_mean", pupil_right_mean, "<=", 0.020)

    open_wide = mean_pair(
        (
            expression_median(audit_payload, "open_wide", "left_wide_pred"),
            expression_median(audit_payload, "open_wide", "right_wide_pred"),
        )
    )
    squint = mean_pair(
        (
            expression_median(audit_payload, "squint", "left_squint_pred"),
            expression_median(audit_payload, "squint", "right_squint_pred"),
        )
    )
    relaxed_wide = mean_pair(
        (
            expression_median(audit_payload, "open_relaxed", "left_wide_pred"),
            expression_median(audit_payload, "open_relaxed", "right_wide_pred"),
        )
    )
    relaxed_squint = mean_pair(
        (
            expression_median(audit_payload, "open_relaxed", "left_squint_pred"),
            expression_median(audit_payload, "open_relaxed", "right_squint_pred"),
        )
    )
    add_gate(gates, "open_wide_wide_median", open_wide, ">=", 0.750)
    add_gate(gates, "squint_squint_median", squint, ">=", 0.750)
    add_gate(gates, "open_relaxed_false_wide_median", relaxed_wide, "<=", 0.100)
    add_gate(gates, "open_relaxed_false_squint_median", relaxed_squint, "<=", 0.100)

    passed = all(gate["passed"] for gate in gates)
    report = {
        "label": args.label,
        "passed": passed,
        "eval_json": str(args.eval_json.resolve()),
        "audit_json": str(args.audit_json.resolve()),
        "split": args.split,
        "reaches": reaches,
        "gates": gates,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
