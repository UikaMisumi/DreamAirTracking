#!/usr/bin/env python3
"""Evaluate gaze-only and multitask ONNX models on a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_eye_multitask import METADATA_FEATURE_COUNT  # noqa: E402


def load_pair(row: dict[str, str], image_size: int) -> np.ndarray:
    images = []
    for key in ("left_file", "right_file"):
        image = Image.open(row[key]).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(images, axis=0)[None].astype(np.float32)


def parse_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    return default if value == "" else float(value)


def row_gaze_weight(row: dict[str, str]) -> float:
    return parse_float(row, "gaze_weight", parse_float(row, "sample_weight", 1.0))


def row_pupil(row: dict[str, str]) -> tuple[float, float, float, float, float, float]:
    return (
        parse_float(row, "weak_left_pupil_x", 0.5),
        parse_float(row, "weak_left_pupil_y", 0.54),
        parse_float(row, "weak_left_pupil_radius", 0.08),
        parse_float(row, "weak_right_pupil_x", 0.5),
        parse_float(row, "weak_right_pupil_y", 0.54),
        parse_float(row, "weak_right_pupil_radius", 0.08),
    )


def build_session_center_anchors(
    rows: list[dict[str, str]],
    min_weak_quality: float,
) -> dict[str, tuple[float, float, float, float, float, float]]:
    grouped: dict[str, list[tuple[float, float, float, float, float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("stage") != "center":
            continue
        left_quality = parse_float(row, "weak_left_quality", 0.0)
        right_quality = parse_float(row, "weak_right_quality", 0.0)
        if left_quality < min_weak_quality or right_quality < min_weak_quality:
            continue
        grouped[row.get("session", "")].append(row_pupil(row))
    return {
        session: tuple(float(x) for x in np.median(np.asarray(values, dtype=np.float32), axis=0))
        for session, values in grouped.items()
        if values
    }


def metadata_features(
    row: dict[str, str],
    anchors: dict[str, tuple[float, float, float, float, float, float]],
    min_weak_quality: float,
) -> np.ndarray:
    neutral = (0.5, 0.54, 0.08, 0.5, 0.54, 0.08)
    anchor = anchors.get(row.get("session", ""), neutral)
    left_quality = parse_float(row, "weak_left_quality", 0.0)
    right_quality = parse_float(row, "weak_right_quality", 0.0)
    pair_quality = parse_float(row, "weak_pair_quality", 0.0)
    left_visible = 1.0 if left_quality >= min_weak_quality else 0.0
    right_visible = 1.0 if right_quality >= min_weak_quality else 0.0
    left_x, left_y, left_radius, right_x, right_y, right_radius = row_pupil(row)
    center_left_x, center_left_y, center_left_radius, center_right_x, center_right_y, center_right_radius = anchor
    if not left_visible:
        left_x, left_y, left_radius = center_left_x, center_left_y, center_left_radius
    if not right_visible:
        right_x, right_y, right_radius = center_right_x, center_right_y, center_right_radius
    return np.asarray(
        [
            left_x,
            left_y,
            left_radius,
            left_quality,
            right_x,
            right_y,
            right_radius,
            right_quality,
            left_x - center_left_x,
            left_y - center_left_y,
            left_radius - center_left_radius,
            right_x - center_right_x,
            right_y - center_right_y,
            right_radius - center_right_radius,
            pair_quality,
            0.5 * (left_visible + right_visible),
        ],
        dtype=np.float32,
    )[None]


def l2(pred: np.ndarray, row: dict[str, str]) -> float:
    target = np.array([float(row["target_x"]), float(row["target_y"])], dtype=np.float32)
    return float(np.linalg.norm(pred.astype(np.float32) - target))


def summarize(values: list[np.ndarray], distances: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {"count": 0}
    d = np.asarray(distances, dtype=np.float32)
    return {
        "count": int(array.shape[0]),
        "median": [float(x) for x in np.median(array, axis=0)],
        "mean": [float(x) for x in np.mean(array, axis=0)],
        "l2_mean": float(np.mean(d)),
        "l2_p90": float(np.percentile(d, 90)),
    }


def evaluate_model(
    rows: list[dict[str, str]],
    onnx: Path,
    image_size: int,
    multitask: bool,
    min_weak_quality: float,
    expression_onnx: Path | None = None,
    expression_image_size: int = 128,
) -> dict[str, object]:
    session = ort.InferenceSession(str(onnx.resolve()), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    needs_metadata = "metadata" in input_names
    if needs_metadata:
        expected = next(item for item in session.get_inputs() if item.name == "metadata").shape[-1]
        if expected not in (METADATA_FEATURE_COUNT, "metadata_features"):
            raise SystemExit(f"Unexpected metadata input shape in {onnx}: {expected}")
    anchors = build_session_center_anchors(rows, min_weak_quality) if needs_metadata else {}
    available_outputs = {item.name for item in session.get_outputs()}
    if multitask:
        output_names = ["gaze_xy", "openness_lr"]
        if "wide_lr" in available_outputs:
            output_names.append("wide_lr")
        if "squint_lr" in available_outputs:
            output_names.append("squint_lr")
        output_names.extend(["pupil_lr", "confidence"])
    else:
        output_names = ["gaze_xy"]
    expression_session = None
    expression_output_names: list[str] = []
    expression_needs_metadata = False
    expression_anchors: dict[str, tuple[float, float, float, float, float, float]] = {}
    if multitask and expression_onnx is not None:
        expression_session = ort.InferenceSession(str(expression_onnx.resolve()), providers=["CPUExecutionProvider"])
        expression_input_names = {item.name for item in expression_session.get_inputs()}
        expression_needs_metadata = "metadata" in expression_input_names
        if expression_needs_metadata:
            expected = next(item for item in expression_session.get_inputs() if item.name == "metadata").shape[-1]
            if expected not in (METADATA_FEATURE_COUNT, "metadata_features"):
                raise SystemExit(f"Unexpected metadata input shape in {expression_onnx}: {expected}")
            expression_anchors = build_session_center_anchors(rows, min_weak_quality)
        expression_available_outputs = {item.name for item in expression_session.get_outputs()}
        expression_output_names = [name for name in ("wide_lr", "squint_lr") if name in expression_available_outputs]
        if not expression_output_names:
            raise SystemExit(f"Expression ONNX has no wide_lr/squint_lr outputs: {expression_onnx}")
    by_stage: dict[str, list[np.ndarray]] = defaultdict(list)
    by_stage_l2: dict[str, list[float]] = defaultdict(list)
    by_split: dict[str, list[np.ndarray]] = defaultdict(list)
    by_split_l2: dict[str, list[float]] = defaultdict(list)
    by_session: dict[str, list[np.ndarray]] = defaultdict(list)
    by_session_l2: dict[str, list[float]] = defaultdict(list)
    by_split_stage: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    by_split_stage_l2: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    valid_by_stage: dict[str, list[np.ndarray]] = defaultdict(list)
    valid_by_stage_l2: dict[str, list[float]] = defaultdict(list)
    valid_by_split: dict[str, list[np.ndarray]] = defaultdict(list)
    valid_by_split_l2: dict[str, list[float]] = defaultdict(list)
    valid_by_split_stage: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    valid_by_split_stage_l2: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    predictions = []
    for row in rows:
        feeds = {"image": load_pair(row, image_size)}
        if needs_metadata:
            feeds["metadata"] = metadata_features(row, anchors, min_weak_quality)
        result_values = session.run(output_names, feeds)
        result = {name: value[0] for name, value in zip(output_names, result_values)}
        if expression_session is not None:
            expression_feeds = {"image": load_pair(row, expression_image_size)}
            if expression_needs_metadata:
                expression_feeds["metadata"] = metadata_features(row, expression_anchors, min_weak_quality)
            expression_values = expression_session.run(expression_output_names, expression_feeds)
            result.update({name: value[0] for name, value in zip(expression_output_names, expression_values)})
        gaze = result["gaze_xy"].astype(np.float32)
        stage = row.get("stage", "")
        split = row.get("split", "")
        row_session = row.get("session", "")
        by_stage[stage].append(gaze)
        distance = l2(gaze, row)
        by_stage_l2[stage].append(distance)
        by_split[split].append(gaze)
        by_split_l2[split].append(distance)
        by_session[row_session].append(gaze)
        by_session_l2[row_session].append(distance)
        by_split_stage[split][stage].append(gaze)
        by_split_stage_l2[split][stage].append(distance)
        if row_gaze_weight(row) > 0.0:
            valid_by_stage[stage].append(gaze)
            valid_by_stage_l2[stage].append(distance)
            valid_by_split[split].append(gaze)
            valid_by_split_l2[split].append(distance)
            valid_by_split_stage[split][stage].append(gaze)
            valid_by_split_stage_l2[split][stage].append(distance)
        item: dict[str, object] = {
            "sample_id": row.get("sample_id", ""),
            "session": row_session,
            "split": split,
            "stage": stage,
            "target_x": float(row["target_x"]),
            "target_y": float(row["target_y"]),
            "gaze_weight": row_gaze_weight(row),
            "gaze_x": float(gaze[0]),
            "gaze_y": float(gaze[1]),
            "l2": distance,
        }
        if multitask:
            item["openness_lr"] = [float(x) for x in result["openness_lr"]]
            item["wide_lr"] = [float(x) for x in result.get("wide_lr", np.zeros(2, dtype=np.float32))]
            item["squint_lr"] = [float(x) for x in result.get("squint_lr", np.zeros(2, dtype=np.float32))]
            item["pupil_lr"] = [float(x) for x in result["pupil_lr"]]
            item["confidence"] = [float(x) for x in result["confidence"]]
        predictions.append(item)

    all_l2 = [float(item["l2"]) for item in predictions]
    return {
        "onnx": str(onnx.resolve()),
        "image_size": image_size,
        "multitask": multitask,
        "uses_metadata": needs_metadata,
        "expression_onnx": str(expression_onnx.resolve()) if expression_onnx is not None else None,
        "expression_image_size": expression_image_size if expression_onnx is not None else None,
        "expression_uses_metadata": expression_needs_metadata,
        "count": len(predictions),
        "overall": {
            "l2_mean": float(np.mean(all_l2)) if all_l2 else 0.0,
            "l2_p90": float(np.percentile(all_l2, 90)) if all_l2 else 0.0,
        },
        "by_stage": {
            stage: summarize(by_stage[stage], by_stage_l2[stage])
            for stage in sorted(by_stage)
        },
        "by_split": {
            split: summarize(by_split[split], by_split_l2[split])
            for split in sorted(by_split)
        },
        "by_session": {
            session: summarize(by_session[session], by_session_l2[session])
            for session in sorted(by_session)
        },
        "by_split_stage": {
            split: {
                stage: summarize(by_split_stage[split][stage], by_split_stage_l2[split][stage])
                for stage in sorted(by_split_stage[split])
            }
            for split in sorted(by_split_stage)
        },
        "gaze_valid_by_stage": {
            stage: summarize(valid_by_stage[stage], valid_by_stage_l2[stage])
            for stage in sorted(valid_by_stage)
        },
        "gaze_valid_by_split": {
            split: summarize(valid_by_split[split], valid_by_split_l2[split])
            for split in sorted(valid_by_split)
        },
        "gaze_valid_by_split_stage": {
            split: {
                stage: summarize(valid_by_split_stage[split][stage], valid_by_split_stage_l2[split][stage])
                for stage in sorted(valid_by_split_stage[split])
            }
            for split in sorted(valid_by_split_stage)
        },
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session", action="append", help="Optional session id filter; repeatable.")
    parser.add_argument("--old-gaze-onnx", type=Path)
    parser.add_argument("--old-gaze-image-size", type=int, default=96)
    parser.add_argument("--multitask-onnx", required=True, type=Path)
    parser.add_argument("--multitask-image-size", type=int, default=128)
    parser.add_argument("--expression-onnx", type=Path, help="Optional secondary multitask ONNX used only for wide_lr/squint_lr.")
    parser.add_argument("--expression-image-size", type=int, default=128)
    parser.add_argument("--min-weak-quality", type=float, default=0.12)
    args = parser.parse_args()

    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.session:
        sessions = set(args.session)
        rows = [row for row in rows if row.get("session") in sessions]
    if not rows:
        raise SystemExit("No rows matched the requested filters.")

    report: dict[str, object] = {
        "manifest": str(args.manifest.resolve()),
        "session_filter": args.session,
        "row_count": len(rows),
        "models": {},
    }
    if args.old_gaze_onnx:
        report["models"]["old_gaze"] = evaluate_model(
            rows,
            args.old_gaze_onnx.resolve(),
            args.old_gaze_image_size,
            multitask=False,
            min_weak_quality=args.min_weak_quality,
        )
    report["models"]["multitask"] = evaluate_model(
        rows,
        args.multitask_onnx.resolve(),
        args.multitask_image_size,
        multitask=True,
        min_weak_quality=args.min_weak_quality,
        expression_onnx=args.expression_onnx.resolve() if args.expression_onnx else None,
        expression_image_size=args.expression_image_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=2, ensure_ascii=False))
    for name, model in report["models"].items():
        print(name, json.dumps(model["overall"], ensure_ascii=False))
        if model.get("expression_onnx"):
            print(name, "expression_onnx", model["expression_onnx"])
        print(name, "by_split", json.dumps(model["by_split"], ensure_ascii=False))
        print(name, "gaze_valid_by_split", json.dumps(model["gaze_valid_by_split"], ensure_ascii=False))
        center = model["by_stage"].get("center")
        if center:
            print(name, "center", json.dumps(center, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
