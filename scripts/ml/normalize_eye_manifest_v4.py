#!/usr/bin/env python3
"""Normalize v3 eye manifests into a deduped v4 head-specific manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPRESSION_VALID_STAGES = {
    "open_relaxed",
    "open_confirm",
    "both_open_relaxed",
    "open_wide",
    "both_open_wide",
    "left_wide_right_relaxed",
    "right_wide_left_relaxed",
    "squint",
    "both_squint",
    "left_squint_right_open",
    "left_squint_right_relaxed",
    "right_squint_left_open",
    "right_squint_left_relaxed",
    "left_open_right_squint",
    "half_closed",
    "closed",
    "left_open_right_half",
    "left_half_right_open",
    "left_open_right_closed",
    "left_closed_right_open",
    "left_half_right_closed",
    "left_closed_right_half",
}

EXPRESSION_VALID_SOURCES = {
    "live_openness_training",
    "runtime_low_openness_hardcase",
    "synthetic_eyelid_mix",
}

GAZE_VALID_SOURCES = {"live"}

V4_COLUMNS = [
    "manifest_version",
    "pair_key",
    "row_repeat_count",
    "gaze_valid",
    "gaze_weight",
    "openness_valid_left",
    "openness_valid_right",
    "wide_valid_left",
    "wide_valid_right",
    "squint_valid_left",
    "squint_valid_right",
    "pupil_valid_left",
    "pupil_valid_right",
    "confidence_valid_left",
    "confidence_valid_right",
    "confidence_valid_pair",
    "v4_notes",
]


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def has_value(row: dict[str, str], key: str) -> bool:
    return row.get(key, "") != ""


def fmt(value: float) -> str:
    if not math.isfinite(value):
        value = 0.0
    return f"{value:.6f}"


def normalize_stage(stage: str) -> str:
    return stage.strip().lower()


def path_key(value: str) -> str:
    return str(Path(value)).lower()


def pair_key(row: dict[str, str]) -> str:
    return f"{path_key(row.get('left_file', ''))}|{path_key(row.get('right_file', ''))}"


def expression_targets_for_stage(stage: str) -> tuple[float, float, float, float]:
    normalized = normalize_stage(stage)
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


def target_abs_max(row: dict[str, str]) -> float:
    return max(abs(as_float(row, "target_x")), abs(as_float(row, "target_y")))


def gaze_bucket(row: dict[str, str]) -> str:
    value = target_abs_max(row)
    if value < 1e-6:
        return "center"
    if value < 0.45:
        return "micro"
    return "edge"


def is_gaze_valid(row: dict[str, str], args: argparse.Namespace) -> bool:
    source = row.get("source", "").strip().lower()
    if source not in set(args.gaze_source):
        return False
    if as_float(row, "sample_weight", 1.0) <= 0.0:
        return False
    return has_value(row, "target_x") and has_value(row, "target_y")


def is_expression_valid(row: dict[str, str]) -> bool:
    source = row.get("source", "").strip().lower()
    stage = normalize_stage(row.get("stage", ""))
    if source in EXPRESSION_VALID_SOURCES:
        return True
    return stage in EXPRESSION_VALID_STAGES


def openness_valid(row: dict[str, str], side: str) -> bool:
    return has_value(row, f"weak_{side}_openness") or has_value(row, f"openness_target_{side}")


def openness_target(row: dict[str, str], side: str) -> float:
    return as_float(row, f"weak_{side}_openness", as_float(row, f"openness_target_{side}", 1.0))


def pupil_valid(row: dict[str, str], side: str, args: argparse.Namespace) -> bool:
    source = row.get("source", "").strip().lower()
    if source == "synthetic_eyelid_mix":
        return False
    quality = as_float(row, f"weak_{side}_quality", as_float(row, f"{side}_quality", 0.0))
    if quality < args.min_pupil_quality:
        return False
    return openness_target(row, side) >= args.min_pupil_openness


def confidence_valid(row: dict[str, str], side: str | None) -> bool:
    source = row.get("source", "").strip().lower()
    if source == "synthetic_eyelid_mix":
        return False
    if side is None:
        return has_value(row, "weak_pair_quality") or has_value(row, "pair_quality")
    return has_value(row, f"weak_{side}_quality") or has_value(row, f"{side}_quality")


def row_score(row: dict[str, str], args: argparse.Namespace) -> tuple[int, int, int, int, str]:
    sample_id = row.get("sample_id", "")
    return (
        1 if is_gaze_valid(row, args) else 0,
        1 if row.get("source", "").strip().lower() == "live" else 0,
        0 if "__bal" in sample_id else 1,
        1 if as_float(row, "sample_weight", 1.0) > 0.0 else 0,
        sample_id,
    )


def ensure_expression_columns(row: dict[str, str]) -> None:
    left_wide, right_wide, left_squint, right_squint = expression_targets_for_stage(row.get("stage", ""))
    row["weak_left_wide"] = row.get("weak_left_wide", "") or fmt(left_wide)
    row["weak_right_wide"] = row.get("weak_right_wide", "") or fmt(right_wide)
    row["weak_left_squint"] = row.get("weak_left_squint", "") or fmt(left_squint)
    row["weak_right_squint"] = row.get("weak_right_squint", "") or fmt(right_squint)
    row["weak_expression_mask"] = row.get("weak_expression_mask", "") or fmt(1.0 if is_expression_valid(row) else 0.0)


def merge_group(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, str]:
    base = max(rows, key=lambda row: row_score(row, args)).copy()
    ensure_expression_columns(base)

    gaze_weight = max((as_float(row, "sample_weight", 1.0) for row in rows if is_gaze_valid(row, args)), default=0.0)
    expression_valid = any(is_expression_valid(row) for row in rows)
    notes: list[str] = []
    if len(rows) > 1:
        notes.append("deduped_repeated_rows")
    if not gaze_weight and target_abs_max(base) < 1e-6 and base.get("source", "").strip().lower() != "live":
        notes.append("center_target_not_used_for_gaze")

    for side in ("left", "right"):
        if not has_value(base, f"weak_{side}_openness") and has_value(base, f"openness_target_{side}"):
            base[f"weak_{side}_openness"] = fmt(as_float(base, f"openness_target_{side}", 1.0))

    base["manifest_version"] = "v4_head_specific_deduped"
    base["pair_key"] = pair_key(base)
    base["row_repeat_count"] = str(len(rows))
    base["gaze_valid"] = fmt(1.0 if gaze_weight > 0.0 else 0.0)
    base["gaze_weight"] = fmt(gaze_weight)
    base["sample_weight"] = fmt(gaze_weight)
    base["openness_valid_left"] = fmt(1.0 if any(openness_valid(row, "left") for row in rows) else 0.0)
    base["openness_valid_right"] = fmt(1.0 if any(openness_valid(row, "right") for row in rows) else 0.0)
    base["wide_valid_left"] = fmt(1.0 if expression_valid else 0.0)
    base["wide_valid_right"] = fmt(1.0 if expression_valid else 0.0)
    base["squint_valid_left"] = fmt(1.0 if expression_valid else 0.0)
    base["squint_valid_right"] = fmt(1.0 if expression_valid else 0.0)
    base["pupil_valid_left"] = fmt(1.0 if any(pupil_valid(row, "left", args) for row in rows) else 0.0)
    base["pupil_valid_right"] = fmt(1.0 if any(pupil_valid(row, "right", args) for row in rows) else 0.0)
    base["confidence_valid_left"] = fmt(1.0 if any(confidence_valid(row, "left") for row in rows) else 0.0)
    base["confidence_valid_right"] = fmt(1.0 if any(confidence_valid(row, "right") for row in rows) else 0.0)
    base["confidence_valid_pair"] = fmt(1.0 if any(confidence_valid(row, None) for row in rows) else 0.0)
    base["v4_notes"] = ";".join(notes)
    return base


def count_valid(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for split in sorted({row.get("split", "") for row in rows}):
        items = [row for row in rows if row.get("split", "") == split]
        report[split] = {
            "rows": len(items),
            "gaze_rows": sum(as_float(row, "gaze_weight") > 0.0 for row in items),
            "gaze_center": sum(as_float(row, "gaze_weight") > 0.0 and gaze_bucket(row) == "center" for row in items),
            "gaze_micro": sum(as_float(row, "gaze_weight") > 0.0 and gaze_bucket(row) == "micro" for row in items),
            "gaze_edge": sum(as_float(row, "gaze_weight") > 0.0 and gaze_bucket(row) == "edge" for row in items),
            "openness_left": sum(as_float(row, "openness_valid_left") > 0.0 for row in items),
            "openness_right": sum(as_float(row, "openness_valid_right") > 0.0 for row in items),
            "wide_left": sum(as_float(row, "wide_valid_left") > 0.0 for row in items),
            "wide_right": sum(as_float(row, "wide_valid_right") > 0.0 for row in items),
            "squint_left": sum(as_float(row, "squint_valid_left") > 0.0 for row in items),
            "squint_right": sum(as_float(row, "squint_valid_right") > 0.0 for row in items),
            "pupil_left": sum(as_float(row, "pupil_valid_left") > 0.0 for row in items),
            "pupil_right": sum(as_float(row, "pupil_valid_right") > 0.0 for row in items),
        }
    return report


def build_report(input_rows: list[dict[str, str]], output_rows: list[dict[str, str]], groups: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    split_conflicts = []
    session_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in output_rows:
        session_split[row.get("session", "")][row.get("split", "")] += 1
    for key, rows in groups.items():
        splits = sorted({row.get("split", "") for row in rows})
        if len(splits) > 1:
            split_conflicts.append({"pair_key": key, "splits": splits, "sample_ids": [row.get("sample_id", "") for row in rows[:8]]})
    return {
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "unique_pairs": len(groups),
        "duplicate_factor": len(input_rows) / max(len(groups), 1),
        "deduped_groups": sum(1 for rows in groups.values() if len(rows) > 1),
        "max_repeat_count": max((len(rows) for rows in groups.values()), default=0),
        "input_by_split": dict(Counter(row.get("split", "") for row in input_rows)),
        "output_by_split": dict(Counter(row.get("split", "") for row in output_rows)),
        "output_by_source": dict(Counter(row.get("source", "") for row in output_rows)),
        "output_by_stage_top40": dict(Counter(row.get("stage", "") for row in output_rows).most_common(40)),
        "head_valid_counts": count_valid(output_rows),
        "split_conflict_groups": split_conflicts[:50],
        "split_conflict_group_count": len(split_conflicts),
        "sessions_with_multiple_splits": {
            session: dict(counts)
            for session, counts in session_split.items()
            if sum(1 for count in counts.values() if count > 0) > 1
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-pupil-quality", type=float, default=0.12)
    parser.add_argument("--min-pupil-openness", type=float, default=0.75)
    parser.add_argument("--gaze-source", action="append", default=["live"], help="Source values allowed to supervise gaze; can be repeated.")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        input_rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in input_rows:
        groups[pair_key(row)].append(row)

    output_rows = [merge_group(rows, args) for _, rows in sorted(groups.items(), key=lambda item: item[0])]
    output_fields = [field for field in fieldnames if field not in V4_COLUMNS] + V4_COLUMNS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    report = build_report(input_rows, output_rows, groups)
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("input_rows", "output_rows", "duplicate_factor", "head_valid_counts")}, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {args.output.resolve()}")
    print(f"Wrote report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
