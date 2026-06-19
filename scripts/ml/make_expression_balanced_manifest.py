from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPRESSION_STAGES = {
    "open_wide",
    "both_open_wide",
    "left_wide_right_relaxed",
    "right_wide_left_relaxed",
    "squint",
    "both_squint",
    "left_squint_right_open",
    "right_squint_left_open",
    "left_open_right_squint",
    "left_squint_right_relaxed",
    "right_squint_left_relaxed",
    "open_relaxed",
    "both_open_relaxed",
    "open_confirm",
}

EXPRESSION_VAL_SESSIONS = {
    "expression_02_problem_pose",
    "eyelid_04_slight_shift",
}

OPENNESS_GUARD_STAGES = {
    "half_closed",
    "squint",
    "both_squint",
    "closed",
    "left_open_right_half",
    "left_half_right_open",
    "left_open_right_closed",
    "left_closed_right_open",
    "left_half_right_closed",
    "left_closed_right_half",
    "left_open_right_squint",
    "left_squint_right_open",
    "right_open_left_squint",
    "right_squint_left_open",
    "left_squint_right_relaxed",
    "right_squint_left_relaxed",
}

GAZE_EDGE_STAGES = {
    "left",
    "right",
    "up",
    "down",
    "left_up",
    "left_down",
    "right_up",
    "right_down",
    "micro_left",
    "micro_right",
    "micro_up",
    "micro_down",
}

GAZE_STAGES = GAZE_EDGE_STAGES | {"center"}

RIGHT_DOWN_STAGES = {
    "right",
    "down",
    "right_down",
    "micro_right",
    "micro_down",
}


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) % max(1, buckets)


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except ValueError:
        return 0.0


def has_positive_expression(row: dict[str, str]) -> bool:
    return any(
        as_float(row, key) > 0.5
        for key in (
            "weak_left_wide",
            "weak_right_wide",
            "weak_left_squint",
            "weak_right_squint",
        )
    )


def is_expression_relaxed(row: dict[str, str]) -> bool:
    stage = (row.get("stage") or "").strip().lower()
    return stage in {"open_relaxed", "both_open_relaxed", "open_confirm"}


def output_split(row: dict[str, str]) -> str:
    session = row.get("session") or ""
    stage = (row.get("stage") or "").strip().lower()
    guarded_stages = EXPRESSION_STAGES | OPENNESS_GUARD_STAGES
    if session in EXPRESSION_VAL_SESSIONS and stage in guarded_stages:
        return "val"
    if session == "synthetic_eyelid_mix" and stage in OPENNESS_GUARD_STAGES:
        sample_id = row.get("sample_id") or f"{session}:{stage}:{row.get('left_image','')}:{row.get('right_image','')}"
        if stable_bucket(sample_id, 5) == 0:
            return "val"
        return "train"
    if stage in EXPRESSION_STAGES:
        return "train"
    if stage in OPENNESS_GUARD_STAGES:
        return "train"
    return row.get("split") or "train"


def repeat_count(row: dict[str, str], args: argparse.Namespace) -> int:
    split = row.get("split") or "train"
    if split != "train":
        return 1
    stage = (row.get("stage") or "").strip().lower()
    repeats = 1
    if has_positive_expression(row):
        repeats = max(repeats, args.expression_positive_repeat)
    if is_expression_relaxed(row):
        repeats = max(repeats, args.expression_relaxed_repeat)
    if stage in OPENNESS_GUARD_STAGES:
        repeats = max(repeats, args.openness_guard_repeat)
    if stage in RIGHT_DOWN_STAGES:
        repeats = max(repeats, args.right_down_repeat)
    if stage in GAZE_EDGE_STAGES:
        repeats = max(repeats, args.gaze_edge_repeat)
    return repeats


def should_zero_duplicate_gaze_weight(row: dict[str, str], dup_index: int, args: argparse.Namespace) -> bool:
    if dup_index == 0 or not args.zero_non_gaze_duplicate_weight:
        return False
    stage = (row.get("stage") or "").strip().lower()
    if stage in GAZE_EDGE_STAGES:
        return False
    if stage in EXPRESSION_STAGES or stage in OPENNESS_GUARD_STAGES:
        return True
    return False


def duplicate_row(row: dict[str, str], dup_index: int, args: argparse.Namespace) -> dict[str, str]:
    if dup_index == 0:
        return row
    copied = dict(row)
    copied["sample_id"] = f"{row.get('sample_id', 'sample')}__bal{dup_index}"
    if should_zero_duplicate_gaze_weight(row, dup_index, args):
        copied["sample_weight"] = "0.000000"
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expression-positive-repeat", type=int, default=6)
    parser.add_argument("--expression-relaxed-repeat", type=int, default=4)
    parser.add_argument("--openness-guard-repeat", type=int, default=1)
    parser.add_argument("--gaze-edge-repeat", type=int, default=2)
    parser.add_argument("--right-down-repeat", type=int, default=2)
    parser.add_argument(
        "--zero-non-gaze-duplicate-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When expression/openness rows are repeated, keep duplicated rows out of gaze loss by setting sample_weight=0.",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("Input manifest has no header.")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]

    output_rows: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    stage_split_counts: Counter[tuple[str, str]] = Counter()
    positive_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for row in rows:
        row["split"] = output_split(row)
        source_counts[row["split"]] += 1
        for dup_index in range(repeat_count(row, args)):
            copied = duplicate_row(row, dup_index, args)
            output_rows.append(copied)
            split = copied.get("split") or "train"
            stage = copied.get("stage") or ""
            split_counts[split] += 1
            stage_split_counts[(split, stage)] += 1
            if has_positive_expression(copied):
                positive_counts[f"{split}_positive_expression"] += 1
            if as_float(copied, "weak_left_wide") > 0.5 or as_float(copied, "weak_right_wide") > 0.5:
                positive_counts[f"{split}_wide"] += 1
            if as_float(copied, "weak_left_squint") > 0.5 or as_float(copied, "weak_right_squint") > 0.5:
                positive_counts[f"{split}_squint"] += 1
            if as_float(copied, "sample_weight") <= 0.0:
                positive_counts[f"{split}_gaze_weight_zero"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "source_split_counts": dict(source_counts),
        "balanced_split_counts": dict(split_counts),
        "positive_counts": dict(positive_counts),
        "expression_val_sessions": sorted(EXPRESSION_VAL_SESSIONS),
        "openness_guard_stages": sorted(OPENNESS_GUARD_STAGES),
        "repeat_policy": {
            "expression_positive": args.expression_positive_repeat,
            "expression_relaxed": args.expression_relaxed_repeat,
            "openness_guard": args.openness_guard_repeat,
            "gaze_edge": args.gaze_edge_repeat,
            "right_down": args.right_down_repeat,
            "zero_non_gaze_duplicate_weight": args.zero_non_gaze_duplicate_weight,
        },
        "top_stage_split_counts": {
            f"{split}:{stage}": count
            for (split, stage), count in stage_split_counts.most_common(80)
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
