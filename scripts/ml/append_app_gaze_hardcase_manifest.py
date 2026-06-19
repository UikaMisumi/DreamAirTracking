#!/usr/bin/env python3
"""Append app calibration sessions as gaze-only hardcases to a head-masked manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


V5_EXTRA_COLUMNS = [
    "hardcase_source_session",
    "hardcase_stage_verdict",
    "hardcase_notes",
]

STAGE_WEIGHTS = {
    "center": 1.0,
    "up": 1.0,
    "left": 1.0,
    "left_up": 1.0,
    "right_up": 2.0,
    "left_down": 2.0,
    "right": 4.0,
    "down": 4.0,
    "right_down": 4.0,
}


def fmt(value: float | int | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    value = float(value)
    if not math.isfinite(value):
        value = 0.0
    return f"{value:.6f}"


def boolish(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_session_targets(session_dir: Path) -> dict[str, tuple[float, float]]:
    session_path = session_dir / "session.json"
    payload = json.loads(session_path.read_text(encoding="utf-8-sig"))
    result: dict[str, tuple[float, float]] = {}
    for stage in payload.get("stages", []):
        target = stage.get("target", {})
        result[str(stage.get("stageId", ""))] = (float(target.get("x", 0.0)), float(target.get("y", 0.0)))
    return result


def load_accepted_labels(session_dir: Path) -> dict[str, dict[str, Any]]:
    labels = {}
    for item in load_jsonl(session_dir / "labels.jsonl"):
        stage = str(item.get("stageId", ""))
        if item.get("accepted", False):
            labels[stage] = item
    return labels


def parse_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def pair_key(row: dict[str, str]) -> str:
    return f"{Path(row.get('left_file', '')).as_posix().lower()}|{Path(row.get('right_file', '')).as_posix().lower()}"


def app_pairs(session_dir: Path) -> list[dict[str, str]]:
    pairs_path = session_dir / "pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs.csv not found: {pairs_path}")
    with pairs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_row(
    base_fields: list[str],
    session_dir: Path,
    session_id: str,
    pair: dict[str, str],
    targets: dict[str, tuple[float, float]],
    labels: dict[str, dict[str, Any]],
    split: str,
    default_weight: float,
) -> dict[str, str]:
    stage = pair.get("stage", "").strip()
    if stage not in labels or stage not in targets:
        raise ValueError(f"Stage is not accepted or missing target: {stage}")

    sequence = int(float(pair.get("sequence", "0") or "0"))
    target_x, target_y = targets[stage]
    label = labels[stage]
    left_file = (session_dir / pair["left_file"]).resolve()
    right_file = (session_dir / pair["right_file"]).resolve()
    weight = STAGE_WEIGHTS.get(stage, default_weight)
    left_found = boolish(pair.get("left_found", ""))
    right_found = boolish(pair.get("right_found", ""))
    left_conf = parse_float(pair, "left_conf", 0.0)
    right_conf = parse_float(pair, "right_conf", 0.0)
    pair_quality = min(left_conf if left_found else 0.0, right_conf if right_found else 0.0)
    left_open = max(0.0, min(1.0, parse_float(pair, "left_open", 1.0)))
    right_open = max(0.0, min(1.0, parse_float(pair, "right_open", 1.0)))

    row = {field: "" for field in base_fields}
    row.update(
        {
            "sample_id": f"{session_id}_app9_gaze_{stage}_{sequence:06d}",
            "wear_id": f"app_calibration_{session_id}",
            "session_id": session_id,
            "session": session_id,
            "sequence_id": str(sequence),
            "frame_index": str(sequence),
            "timestamp": "",
            "split": split,
            "left_file": str(left_file),
            "right_file": str(right_file),
            "target_x": fmt(target_x),
            "target_y": fmt(target_y),
            "target_source": "app_9point_hardcase",
            "gaze_stage": stage,
            "stage": stage,
            "openness_target_left": fmt(left_open),
            "openness_target_right": fmt(right_open),
            "openness_source": "app_runtime_observed_not_supervised",
            "pupil_left_x": "0.500000",
            "pupil_left_y": "0.540000",
            "pupil_left_radius": "0.080000",
            "pupil_right_x": "0.500000",
            "pupil_right_y": "0.540000",
            "pupil_right_radius": "0.080000",
            "pupil_source": "not_supervised",
            "left_quality": fmt(left_conf),
            "right_quality": fmt(right_conf),
            "pair_quality": fmt(pair_quality),
            "left_found": fmt(left_found),
            "right_found": fmt(right_found),
            "no_eye": "false",
            "blink": "false",
            "closed": "false",
            "occluded": "false",
            "reflection": "",
            "normalization_left_pupil_x": "0.500000",
            "normalization_left_pupil_y": "0.540000",
            "normalization_right_pupil_x": "0.500000",
            "normalization_right_pupil_y": "0.540000",
            "normalization_confidence": fmt(pair_quality),
            "sample_weight": fmt(weight),
            "notes": "app_9point_hardcase_gaze_only",
            "weak_left_openness": fmt(left_open),
            "weak_left_wide": "0.000000",
            "weak_left_squint": "0.000000",
            "weak_left_pupil_x": "0.500000",
            "weak_left_pupil_y": "0.540000",
            "weak_left_pupil_radius": "0.080000",
            "weak_left_quality": fmt(left_conf),
            "weak_right_openness": fmt(right_open),
            "weak_right_wide": "0.000000",
            "weak_right_squint": "0.000000",
            "weak_right_pupil_x": "0.500000",
            "weak_right_pupil_y": "0.540000",
            "weak_right_pupil_radius": "0.080000",
            "weak_right_quality": fmt(right_conf),
            "weak_pair_quality": fmt(pair_quality),
            "weak_blink": "0.000000",
            "weak_expression_mask": "0.000000",
            "weak_label_source": "app_9point_session_target_only",
            "source": "live",
            "left_conf": fmt(left_conf),
            "right_conf": fmt(right_conf),
            "left_center_x": fmt(parse_float(pair, "left_raw_x", 0.0)),
            "left_center_y": fmt(parse_float(pair, "left_raw_y", 0.0)),
            "right_center_x": fmt(parse_float(pair, "right_raw_x", 0.0)),
            "right_center_y": fmt(parse_float(pair, "right_raw_y", 0.0)),
            "manifest_version": "v5_app9_gaze_hardcase",
            "row_repeat_count": "1",
            "gaze_valid": "1.000000",
            "gaze_weight": fmt(weight),
            "openness_valid_left": "0.000000",
            "openness_valid_right": "0.000000",
            "wide_valid_left": "0.000000",
            "wide_valid_right": "0.000000",
            "squint_valid_left": "0.000000",
            "squint_valid_right": "0.000000",
            "pupil_valid_left": "0.000000",
            "pupil_valid_right": "0.000000",
            "confidence_valid_left": "0.000000",
            "confidence_valid_right": "0.000000",
            "confidence_valid_pair": "0.000000",
            "v4_notes": "app_9point_hardcase_gaze_only;no_expression_or_pupil_supervision",
            "hardcase_source_session": str(session_dir.resolve()),
            "hardcase_stage_verdict": str(label.get("operatorVerdict", "")),
            "hardcase_notes": str(label.get("notes", "")),
        }
    )
    row["pair_key"] = pair_key(row)
    return row


def build_report(base_rows: list[dict[str, str]], hardcase_rows: list[dict[str, str]], output_rows: list[dict[str, str]]) -> dict[str, Any]:
    def gaze_weight(row: dict[str, str]) -> float:
        return parse_float(row, "gaze_weight", parse_float(row, "sample_weight", 0.0))

    return {
        "base_rows": len(base_rows),
        "hardcase_rows": len(hardcase_rows),
        "output_rows": len(output_rows),
        "hardcase_by_session": dict(Counter(row.get("session", "") for row in hardcase_rows)),
        "hardcase_by_stage": dict(Counter(row.get("stage", "") for row in hardcase_rows)),
        "hardcase_weight_by_stage": {
            stage: sum(gaze_weight(row) for row in hardcase_rows if row.get("stage", "") == stage)
            for stage in sorted({row.get("stage", "") for row in hardcase_rows})
        },
        "output_by_split": dict(Counter(row.get("split", "") for row in output_rows)),
        "output_gaze_weight_by_split": {
            split: sum(gaze_weight(row) for row in output_rows if row.get("split", "") == split)
            for split in sorted({row.get("split", "") for row in output_rows})
        },
        "hardcase_policy": {
            "split": hardcase_rows[0].get("split", "") if hardcase_rows else "",
            "stage_weights": STAGE_WEIGHTS,
            "head_masks": "gaze only; openness/expression/pupil/confidence masks are zero",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--session-dir", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--hardcase-split", default="train")
    parser.add_argument("--default-hardcase-weight", type=float, default=1.0)
    args = parser.parse_args()

    with args.base_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        base_rows = [dict(row) for row in reader]
        base_fields = list(reader.fieldnames or [])
    fields = base_fields + [field for field in V5_EXTRA_COLUMNS if field not in base_fields]

    hardcase_rows: list[dict[str, str]] = []
    for session_dir in args.session_dir:
        session_dir = session_dir.resolve()
        session_id = session_dir.name
        targets = load_session_targets(session_dir)
        labels = load_accepted_labels(session_dir)
        for pair in app_pairs(session_dir):
            stage = pair.get("stage", "")
            if stage not in labels:
                continue
            row = make_row(
                fields,
                session_dir,
                session_id,
                pair,
                targets,
                labels,
                args.hardcase_split,
                args.default_hardcase_weight,
            )
            hardcase_rows.append(row)

    output_rows = base_rows + hardcase_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    report = build_report(base_rows, hardcase_rows, output_rows)
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {args.output.resolve()}")
    print(f"Wrote report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
