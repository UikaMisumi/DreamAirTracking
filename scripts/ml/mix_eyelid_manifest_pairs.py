#!/usr/bin/env python3
"""Create train-only asymmetric eyelid rows by recombining real left/right eye crops."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


OPEN_STAGES = ("open_relaxed", "open_confirm")
HALF_STAGES = ("half_closed",)
SQUINT_STAGES = ("squint",)
CLOSED_STAGES = ("closed",)

COMBOS = (
    ("left_open_right_half", OPEN_STAGES, HALF_STAGES, 1.0, 0.5),
    ("left_half_right_open", HALF_STAGES, OPEN_STAGES, 0.5, 1.0),
    ("left_open_right_squint", OPEN_STAGES, SQUINT_STAGES, 1.0, 0.7),
    ("left_squint_right_open", SQUINT_STAGES, OPEN_STAGES, 0.7, 1.0),
    ("left_open_right_closed", OPEN_STAGES, CLOSED_STAGES, 1.0, 0.0),
    ("left_closed_right_open", CLOSED_STAGES, OPEN_STAGES, 0.0, 1.0),
    ("left_half_right_closed", HALF_STAGES, CLOSED_STAGES, 0.5, 0.0),
    ("left_closed_right_half", CLOSED_STAGES, HALF_STAGES, 0.0, 0.5),
)

LEFT_FIELDS = (
    "left_file",
    "openness_target_left",
    "pupil_left_x",
    "pupil_left_y",
    "pupil_left_radius",
    "left_quality",
    "left_found",
    "normalization_left_pupil_x",
    "normalization_left_pupil_y",
    "weak_left_openness",
    "weak_left_wide",
    "weak_left_squint",
    "weak_left_pupil_x",
    "weak_left_pupil_y",
    "weak_left_pupil_radius",
    "weak_left_quality",
    "left_conf",
    "left_center_x",
    "left_center_y",
)

RIGHT_FIELDS = (
    "right_file",
    "openness_target_right",
    "pupil_right_x",
    "pupil_right_y",
    "pupil_right_radius",
    "right_quality",
    "right_found",
    "normalization_right_pupil_x",
    "normalization_right_pupil_y",
    "weak_right_openness",
    "weak_right_wide",
    "weak_right_squint",
    "weak_right_pupil_x",
    "weak_right_pupil_y",
    "weak_right_pupil_radius",
    "weak_right_quality",
    "right_conf",
    "right_center_x",
    "right_center_y",
)


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def stage(row: dict[str, str]) -> str:
    return (row.get("stage") or row.get("gaze_stage") or "").strip()


def rows_for_stages(grouped: dict[str, list[dict[str, str]]], stages: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in stages:
        rows.extend(grouped.get(item, []))
    return rows


def copy_fields(target: dict[str, str], source: dict[str, str], fields: Iterable[str]) -> None:
    for field in fields:
        if field in target and field in source:
            target[field] = source[field]


def quality(row: dict[str, str], side: str) -> float:
    return parse_float(row.get(f"weak_{side}_quality") or row.get(f"{side}_quality") or row.get(f"{side}_conf"), 0.0)


def expression_for_stage(stage_name: str) -> tuple[float, float, float, float]:
    normalized = stage_name.strip().lower()
    left_wide = right_wide = left_squint = right_squint = 0.0
    if normalized in {"open_wide", "both_open_wide"}:
        left_wide = right_wide = 1.0
    elif normalized == "left_wide_right_relaxed":
        left_wide = 1.0
    elif normalized == "right_wide_left_relaxed":
        right_wide = 1.0
    elif normalized in {"squint", "both_squint"}:
        left_squint = right_squint = 1.0
    elif normalized in {"left_squint_right_open", "left_squint_right_relaxed"}:
        left_squint = 1.0
    elif normalized in {"right_squint_left_open", "right_squint_left_relaxed", "left_open_right_squint"}:
        right_squint = 1.0
    return left_wide, right_wide, left_squint, right_squint


def make_synthetic_row(
    fieldnames: list[str],
    combo_name: str,
    left_row: dict[str, str],
    right_row: dict[str, str],
    left_open: float,
    right_open: float,
    index: int,
) -> dict[str, str]:
    row = {field: "" for field in fieldnames}
    row.update(left_row)
    copy_fields(row, left_row, LEFT_FIELDS)
    copy_fields(row, right_row, RIGHT_FIELDS)

    left_quality = quality(left_row, "left")
    right_quality = quality(right_row, "right")
    pair_quality = min(left_quality, right_quality)
    sample_id = f"synthetic_eyelid_mix_{combo_name}_{index:06d}"
    left_wide, _, left_squint, _ = expression_for_stage(combo_name)
    _, right_wide, _, right_squint = expression_for_stage(combo_name)

    updates = {
        "sample_id": sample_id,
        "wear_id": "synthetic_eyelid_mix",
        "session_id": "synthetic_eyelid_mix",
        "session": "synthetic_eyelid_mix",
        "sequence_id": str(index),
        "frame_index": str(index),
        "timestamp": "",
        "split": "train",
        "target_x": "0",
        "target_y": "0",
        "target_source": "synthetic_eye_mix_gaze_disabled",
        "gaze_stage": combo_name,
        "stage": combo_name,
        "openness_target_left": f"{left_open:.6f}",
        "openness_target_right": f"{right_open:.6f}",
        "openness_source": "synthetic_eye_mix_protocol_v1",
        "pair_quality": f"{pair_quality:.6f}",
        "normalization_confidence": f"{pair_quality:.6f}",
        "sample_weight": "0",
        "weak_left_openness": f"{left_open:.6f}",
        "weak_left_wide": f"{left_wide:.6f}",
        "weak_left_squint": f"{left_squint:.6f}",
        "weak_right_openness": f"{right_open:.6f}",
        "weak_right_wide": f"{right_wide:.6f}",
        "weak_right_squint": f"{right_squint:.6f}",
        "weak_pair_quality": f"{pair_quality:.6f}",
        "weak_blink": "false",
        "weak_expression_mask": "1",
        "weak_label_source": "synthetic_eye_mix_protocol_v1",
        "source": "synthetic_eyelid_mix",
        "no_eye": "false",
        "blink": "false",
        "closed": "false",
        "occluded": "false",
        "reflection": "",
        "notes": f"left_stage={stage(left_row)}; right_stage={stage(right_row)}; gaze_loss_disabled",
    }
    for key, value in updates.items():
        if key in row:
            row[key] = value
    return row


def build(args: argparse.Namespace) -> dict[str, object]:
    fieldnames, rows = read_rows(args.input.resolve())
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split", "train") != "train":
            continue
        grouped[stage(row)].append(row)

    rng = random.Random(args.seed)
    synthetic: list[dict[str, str]] = []
    combo_counts: dict[str, int] = {}
    index = 1
    for combo_name, left_stages, right_stages, left_open, right_open in COMBOS:
        left_pool = rows_for_stages(grouped, left_stages)
        right_pool = rows_for_stages(grouped, right_stages)
        if not left_pool or not right_pool:
            combo_counts[combo_name] = 0
            continue
        count = min(args.max_per_combo, len(left_pool), len(right_pool))
        for _ in range(count):
            left_row = rng.choice(left_pool)
            right_row = rng.choice(right_pool)
            synthetic.append(make_synthetic_row(fieldnames, combo_name, left_row, right_row, left_open, right_open, index))
            index += 1
        combo_counts[combo_name] = count

    output_rows = rows + synthetic
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "original_rows": len(rows),
        "synthetic_rows": len(synthetic),
        "output_rows": len(output_rows),
        "max_per_combo": args.max_per_combo,
        "combo_counts": combo_counts,
        "gaze_loss_for_synthetic": "disabled_by_sample_weight_0",
    }
    summary_path = args.output.with_suffix(".synthetic_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-combo", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
