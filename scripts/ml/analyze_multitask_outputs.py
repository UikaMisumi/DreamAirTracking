#!/usr/bin/env python3
"""Analyze multitask eye-model outputs against a v3 manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def mask_value(row: dict[str, str], key: str, fallback: float = 1.0) -> float:
    value = row.get(key, "")
    return fallback if value == "" else float(value)


def gaze_weight(row: dict[str, str]) -> float:
    return mask_value(row, "gaze_weight", mask_value(row, "sample_weight", 1.0))


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def stage_target_axis(stage: str, target_x: float, target_y: float) -> str:
    if abs(target_x) >= abs(target_y) and abs(target_x) > 1e-6:
        return "x"
    if abs(target_y) > 1e-6:
        return "y"
    return "center"


def analyze_gaze(rows: list[dict[str, str]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    row_by_id = {row["sample_id"]: row for row in rows}
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        row = row_by_id[item["sample_id"]]
        if gaze_weight(row) <= 0.0:
            continue
        by_stage[str(item.get("stage", ""))].append(item)
    stages: dict[str, Any] = {}
    for stage, items in sorted(by_stage.items()):
        target_x = float(items[0]["target_x"])
        target_y = float(items[0]["target_y"])
        pred_x = [float(item["gaze_x"]) for item in items]
        pred_y = [float(item["gaze_y"]) for item in items]
        l2 = [float(item["l2"]) for item in items]
        axis = stage_target_axis(stage, target_x, target_y)
        if axis == "x":
            target = target_x
            pred_axis = float(np.median(pred_x))
        elif axis == "y":
            target = target_y
            pred_axis = float(np.median(pred_y))
        else:
            target = 0.0
            pred_axis = float(np.median(np.linalg.norm(np.asarray([pred_x, pred_y], dtype=np.float32).T, axis=1)))
        stages[stage] = {
            "count": len(items),
            "target": [target_x, target_y],
            "pred_mean": [float(np.mean(pred_x)), float(np.mean(pred_y))],
            "pred_median": [float(np.median(pred_x)), float(np.median(pred_y))],
            "l2_mean": float(np.mean(l2)),
            "l2_p90": float(np.percentile(l2, 90)),
            "axis_reach_ratio": None if abs(target) < 1e-6 else float(pred_axis / target),
        }
    return {"valid_count": sum(len(items) for items in by_stage.values()), "by_stage": stages}


def analyze_openness(rows: list[dict[str, str]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    row_by_id = {row["sample_id"]: row for row in rows}
    left_pred: list[float] = []
    right_pred: list[float] = []
    left_target: list[float] = []
    right_target: list[float] = []
    by_stage: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in predictions:
        row = row_by_id[item["sample_id"]]
        pred = item.get("openness_lr") or [1.0, 1.0]
        lp = float(pred[0])
        rp = float(pred[1])
        lt = parse_float(row, "weak_left_openness", 1.0)
        rt = parse_float(row, "weak_right_openness", 1.0)
        stage = str(item.get("stage", ""))
        left_mask = mask_value(row, "openness_valid_left", 1.0)
        right_mask = mask_value(row, "openness_valid_right", 1.0)
        if left_mask > 0.0:
            left_pred.append(lp)
            left_target.append(lt)
            by_stage[stage]["left_abs_error"].append(abs(lp - lt))
            by_stage[stage]["left_pred"].append(lp)
        if right_mask > 0.0:
            right_pred.append(rp)
            right_target.append(rt)
            by_stage[stage]["right_abs_error"].append(abs(rp - rt))
            by_stage[stage]["right_pred"].append(rp)
        if left_mask > 0.0 or right_mask > 0.0:
            by_stage[stage]["target"].append((lt * left_mask + rt * right_mask) / max(left_mask + right_mask, 1e-6))

    diff = [abs(l - r) for l, r in zip(left_pred, right_pred)]
    corr = float(np.corrcoef(left_pred, right_pred)[0, 1]) if len(left_pred) > 2 else 0.0
    stage_report = {}
    for stage, values in sorted(by_stage.items()):
        stage_report[stage] = {
            "target_mean": float(np.mean(values["target"])),
            "left_pred": summary(values["left_pred"]),
            "right_pred": summary(values["right_pred"]),
            "left_mae": summary(values["left_abs_error"]),
            "right_mae": summary(values["right_abs_error"]),
        }
    return {
        "left_pred": summary(left_pred),
        "right_pred": summary(right_pred),
        "left_right_abs_diff": summary(diff),
        "left_right_correlation": corr,
        "left_mae": summary([abs(p - t) for p, t in zip(left_pred, left_target)]),
        "right_mae": summary([abs(p - t) for p, t in zip(right_pred, right_target)]),
        "by_stage": stage_report,
    }


def analyze_pupil(rows: list[dict[str, str]], predictions: list[dict[str, Any]], min_quality: float) -> dict[str, Any]:
    row_by_id = {row["sample_id"]: row for row in rows}
    report: dict[str, Any] = {"left": {}, "right": {}, "by_stage": {}}
    accum = {
        "left": defaultdict(list),
        "right": defaultdict(list),
    }
    by_stage: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "left_quality": 0, "right_quality": 0, "left_center_error": [], "right_center_error": [], "left_radius_error": [], "right_radius_error": []})

    for item in predictions:
        row = row_by_id[item["sample_id"]]
        stage = str(item.get("stage", ""))
        pupil = item.get("pupil_lr") or [0.5, 0.54, 0.08, 0.5, 0.54, 0.08]
        by_stage[stage]["count"] += 1
        for side, offset in (("left", 0), ("right", 3)):
            side_mask = mask_value(row, f"pupil_valid_{side}", 1.0)
            if side_mask <= 0.0:
                continue
            quality = parse_float(row, f"weak_{side}_quality", 0.0)
            by_stage[stage][f"{side}_quality"] += int(quality >= min_quality)
            if quality < min_quality:
                continue
            tx = parse_float(row, f"weak_{side}_pupil_x", 0.5)
            ty = parse_float(row, f"weak_{side}_pupil_y", 0.54)
            tr = parse_float(row, f"weak_{side}_pupil_radius", 0.08)
            px = float(pupil[offset + 0])
            py = float(pupil[offset + 1])
            pr = float(pupil[offset + 2])
            center_error = math.hypot(px - tx, py - ty)
            radius_error = abs(pr - tr)
            accum[side]["center_error"].append(center_error)
            accum[side]["x_error"].append(abs(px - tx))
            accum[side]["y_error"].append(abs(py - ty))
            accum[side]["radius_error"].append(radius_error)
            accum[side]["pred_radius"].append(pr)
            accum[side]["target_radius"].append(tr)
            by_stage[stage][f"{side}_center_error"].append(center_error)
            by_stage[stage][f"{side}_radius_error"].append(radius_error)

    for side in ("left", "right"):
        report[side] = {
            "center_error": summary(accum[side]["center_error"]),
            "x_error": summary(accum[side]["x_error"]),
            "y_error": summary(accum[side]["y_error"]),
            "radius_error": summary(accum[side]["radius_error"]),
            "pred_radius": summary(accum[side]["pred_radius"]),
            "target_radius": summary(accum[side]["target_radius"]),
        }

    for stage, values in sorted(by_stage.items()):
        count = max(int(values["count"]), 1)
        report["by_stage"][stage] = {
            "count": int(values["count"]),
            "left_quality_ratio": float(values["left_quality"] / count),
            "right_quality_ratio": float(values["right_quality"] / count),
            "left_center_error": summary(values["left_center_error"]),
            "right_center_error": summary(values["right_center_error"]),
            "left_radius_error": summary(values["left_radius_error"]),
            "right_radius_error": summary(values["right_radius_error"]),
        }
    return report


def expression_targets_for_stage(stage: str) -> tuple[tuple[float, float], tuple[float, float], bool]:
    normalized = stage.strip().lower()
    wide = (0.0, 0.0)
    squint = (0.0, 0.0)
    has_target = True
    if normalized in {"open_wide", "both_open_wide"}:
        wide = (1.0, 1.0)
    elif normalized == "left_wide_right_relaxed":
        wide = (1.0, 0.0)
    elif normalized == "right_wide_left_relaxed":
        wide = (0.0, 1.0)
    elif normalized in {"squint", "both_squint"}:
        squint = (1.0, 1.0)
    elif normalized in {"left_squint_right_open", "left_squint_right_relaxed"}:
        squint = (1.0, 0.0)
    elif normalized in {"right_squint_left_open", "right_squint_left_relaxed", "left_open_right_squint"}:
        squint = (0.0, 1.0)
    elif normalized == "":
        has_target = False
    return wide, squint, has_target


def analyze_expression(rows: list[dict[str, str]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    row_by_id = {row["sample_id"]: row for row in rows}
    by_stage: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in predictions:
        row = row_by_id[item["sample_id"]]
        stage = str(item.get("stage", ""))
        wide_target, squint_target, has_target = expression_targets_for_stage(stage)
        if not has_target:
            continue
        wide_left_mask = mask_value(row, "wide_valid_left", 1.0)
        wide_right_mask = mask_value(row, "wide_valid_right", 1.0)
        squint_left_mask = mask_value(row, "squint_valid_left", 1.0)
        squint_right_mask = mask_value(row, "squint_valid_right", 1.0)
        if max(wide_left_mask, wide_right_mask, squint_left_mask, squint_right_mask) <= 0.0:
            continue
        wide = item.get("wide_lr") or [0.0, 0.0]
        squint = item.get("squint_lr") or [0.0, 0.0]
        if wide_left_mask > 0.0:
            by_stage[stage]["left_wide_pred"].append(float(wide[0]))
            by_stage[stage]["left_wide_error"].append(abs(float(wide[0]) - wide_target[0]))
        if wide_right_mask > 0.0:
            by_stage[stage]["right_wide_pred"].append(float(wide[1]))
            by_stage[stage]["right_wide_error"].append(abs(float(wide[1]) - wide_target[1]))
        if squint_left_mask > 0.0:
            by_stage[stage]["left_squint_pred"].append(float(squint[0]))
            by_stage[stage]["left_squint_error"].append(abs(float(squint[0]) - squint_target[0]))
        if squint_right_mask > 0.0:
            by_stage[stage]["right_squint_pred"].append(float(squint[1]))
            by_stage[stage]["right_squint_error"].append(abs(float(squint[1]) - squint_target[1]))
        by_stage[stage]["wide_target"].append((wide_target[0] + wide_target[1]) * 0.5)
        by_stage[stage]["squint_target"].append((squint_target[0] + squint_target[1]) * 0.5)

    stage_report = {}
    for stage, values in sorted(by_stage.items()):
        stage_report[stage] = {
            "wide_target_mean": float(np.mean(values["wide_target"])),
            "squint_target_mean": float(np.mean(values["squint_target"])),
            "left_wide_pred": summary(values["left_wide_pred"]),
            "right_wide_pred": summary(values["right_wide_pred"]),
            "left_squint_pred": summary(values["left_squint_pred"]),
            "right_squint_pred": summary(values["right_squint_pred"]),
            "left_wide_mae": summary(values["left_wide_error"]),
            "right_wide_mae": summary(values["right_wide_error"]),
            "left_squint_mae": summary(values["left_squint_error"]),
            "right_squint_mae": summary(values["right_squint_error"]),
        }
    return {"by_stage": stage_report}


def create_pupil_contact_sheet(
    rows: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    output: Path,
    min_quality: float,
    limit: int = 48,
) -> None:
    row_by_id = {row["sample_id"]: row for row in rows}
    candidates = []
    for item in predictions:
        row = row_by_id[item["sample_id"]]
        pupil = item.get("pupil_lr") or [0.5, 0.54, 0.08, 0.5, 0.54, 0.08]
        errors = []
        for side, offset in (("left", 0), ("right", 3)):
            quality = parse_float(row, f"weak_{side}_quality", 0.0)
            if quality < min_quality:
                continue
            tx = parse_float(row, f"weak_{side}_pupil_x", 0.5)
            ty = parse_float(row, f"weak_{side}_pupil_y", 0.54)
            px = float(pupil[offset + 0])
            py = float(pupil[offset + 1])
            errors.append(math.hypot(px - tx, py - ty))
        if errors:
            candidates.append((max(errors), item, row))
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[:limit]
    if not chosen:
        return

    cell_w, cell_h = 260, 150
    cols = 4
    sheet = Image.new("RGB", (cols * cell_w, math.ceil(len(chosen) / cols) * cell_h), (28, 30, 38))
    draw = ImageDraw.Draw(sheet)
    for index, (err, item, row) in enumerate(chosen):
        x0 = (index % cols) * cell_w
        y0 = (index // cols) * cell_h
        for eye_index, side in enumerate(("left", "right")):
            path = Path(row[f"{side}_file"])
            image = Image.open(path).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
            ox = x0 + 4 + eye_index * 104
            oy = y0 + 4
            sheet.paste(image, (ox, oy))
            pupil = item.get("pupil_lr") or [0.5, 0.54, 0.08, 0.5, 0.54, 0.08]
            offset = 0 if side == "left" else 3
            if parse_float(row, f"weak_{side}_quality", 0.0) >= min_quality:
                tx = parse_float(row, f"weak_{side}_pupil_x", 0.5)
                ty = parse_float(row, f"weak_{side}_pupil_y", 0.54)
                px = float(pupil[offset + 0])
                py = float(pupil[offset + 1])
                draw.ellipse((ox + tx * 96 - 3, oy + ty * 96 - 3, ox + tx * 96 + 3, oy + ty * 96 + 3), outline=(0, 255, 120), width=2)
                draw.ellipse((ox + px * 96 - 3, oy + py * 96 - 3, ox + px * 96 + 3, oy + py * 96 + 3), outline=(255, 80, 80), width=2)
        draw.text((x0 + 4, y0 + 106), f"{item['stage']} {item['session'][:24]}", fill=(235, 235, 235))
        draw.text((x0 + 4, y0 + 122), f"max pupil center err={err:.3f}", fill=(255, 210, 120))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--eval-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-quality", type=float, default=0.12)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    predictions = payload["models"]["multitask"]["predictions"]
    report = {
        "manifest": str(args.manifest.resolve()),
        "eval_json": str(args.eval_json.resolve()),
        "model_onnx": payload["models"]["multitask"].get("onnx"),
        "sample_count": len(predictions),
        "gaze": analyze_gaze(rows, predictions),
        "openness": analyze_openness(rows, predictions),
        "expression": analyze_expression(rows, predictions),
        "pupil": analyze_pupil(rows, predictions, args.min_quality),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_output_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    create_pupil_contact_sheet(rows, predictions, args.output_dir / "pupil_worst_contact_sheet.jpg", args.min_quality)
    print(json.dumps({
        "output": str((args.output_dir / "model_output_audit.json").resolve()),
        "pupil_contact_sheet": str((args.output_dir / "pupil_worst_contact_sheet.jpg").resolve()),
        "sample_count": len(predictions),
        "openness_left_right_diff": report["openness"]["left_right_abs_diff"],
        "expression_stage_count": len(report["expression"]["by_stage"]),
        "pupil_left_center_error": report["pupil"]["left"]["center_error"],
        "pupil_right_center_error": report["pupil"]["right"]["center_error"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
