#!/usr/bin/env python3
"""Build a gaze-training manifest from Dream Air calibration captures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


STAGE_TARGETS: dict[str, tuple[float, float]] = {
    "center": (0.0, 0.0),
    "center_confirm": (0.0, 0.0),
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "left_up": (-1.0, 1.0),
    "right_up": (1.0, 1.0),
    "left_down": (-1.0, -1.0),
    "right_down": (1.0, -1.0),
}


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def parse_float(row: dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in row and row[name] != "":
            return float(row[name])
    return default


def normalize_file(root: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def load_accepted_ranges(session_root: Path) -> list[dict[str, object]]:
    path = session_root / "labels.jsonl"
    if not path.exists():
        return []

    ranges: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            label = json.loads(line)
            if not bool(label.get("accepted", False)):
                continue
            target = label.get("operatorOverrideTarget") or label.get("target") or {}
            ranges.append(
                {
                    "stage": str(label.get("stageId", "")),
                    "start": int(label.get("frameStart", 0)),
                    "end": int(label.get("frameEnd", -1)),
                    "target_x": float(target.get("x", STAGE_TARGETS.get(str(label.get("stageId", "")), (0.0, 0.0))[0])),
                    "target_y": float(target.get("y", STAGE_TARGETS.get(str(label.get("stageId", "")), (0.0, 0.0))[1])),
                }
            )
    return ranges


def load_session_exclude(session_root: Path) -> dict[str, object] | None:
    path = session_root / "session_exclude.json"
    if not path.exists():
        return None
    try:
        data = json.load(path.open("r", encoding="utf-8"))
    except json.JSONDecodeError:
        return {"exclude": True, "reason": "invalid session_exclude.json"}
    return data if bool(data.get("exclude", True)) else None


def accepted_label_for(row: dict[str, str], accepted_ranges: list[dict[str, object]]) -> tuple[float, float] | None:
    if not accepted_ranges:
        stage = (row.get("stage") or row.get("Stage") or "").strip()
        return STAGE_TARGETS.get(stage)

    stage = (row.get("stage") or row.get("Stage") or "").strip()
    sequence = int(float(row.get("sequence") or row.get("Sequence") or row.get("pair") or "0"))
    for item in accepted_ranges:
        if item["stage"] == stage and int(item["start"]) <= sequence <= int(item["end"]):
            return float(item["target_x"]), float(item["target_y"])
    return None


def read_pairs(
    csv_path: Path,
    min_confidence: float,
    min_openness: float,
    include_stages: set[str] | None,
) -> tuple[list[dict[str, str]], Counter[str], list[str]]:
    root = csv_path.parent
    session_exclude = load_session_exclude(root)
    if session_exclude is not None:
        reason = str(session_exclude.get("reason", "session excluded"))
        return [], Counter({"excluded_session": 1}), [f"Session excluded: {reason}"]

    accepted_ranges = load_accepted_ranges(root)
    rows: list[dict[str, str]] = []
    skipped = Counter()
    warnings: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            stage = (raw.get("stage") or raw.get("Stage") or "").strip()
            if stage not in STAGE_TARGETS:
                skipped["non_gaze_stage"] += 1
                continue
            accepted_target = accepted_label_for(raw, accepted_ranges)
            if accepted_target is None:
                skipped["not_accepted_label"] += 1
                continue
            if include_stages and stage not in include_stages:
                skipped["stage_filter"] += 1
                continue

            left_found = parse_bool(raw.get("left_found", ""))
            right_found = parse_bool(raw.get("right_found", ""))
            left_conf = parse_float(raw, "left_conf", "left_confidence")
            right_conf = parse_float(raw, "right_conf", "right_confidence")
            left_open = parse_float(raw, "left_open", "left_openness", default=1.0)
            right_open = parse_float(raw, "right_open", "right_openness", default=1.0)
            if not left_found or not right_found:
                skipped["missing_eye"] += 1
                continue
            if left_conf < min_confidence or right_conf < min_confidence:
                skipped["low_confidence"] += 1
                continue
            if left_open < min_openness or right_open < min_openness:
                skipped["low_openness"] += 1
                continue

            left_file = normalize_file(root, raw.get("left_file", ""))
            right_file = normalize_file(root, raw.get("right_file", ""))
            if not Path(left_file).exists() or not Path(right_file).exists():
                skipped["missing_file"] += 1
                continue

            target_x, target_y = accepted_target
            pair_id = raw.get("pair", raw.get("Sequence", str(len(rows) + 1)))
            session = csv_path.parent.name
            rows.append(
                {
                    "sample_id": f"{session}_{csv_path.stem}_{stage}_{pair_id}",
                    "session": session,
                    "stage": stage,
                    "target_x": f"{target_x:.6f}",
                    "target_y": f"{target_y:.6f}",
                    "left_file": left_file,
                    "right_file": right_file,
                    "left_conf": f"{left_conf:.6f}",
                    "right_conf": f"{right_conf:.6f}",
                    "left_center_x": f"{parse_float(raw, 'left_x', 'left_center_x', 'left_raw_x'):.6f}",
                    "left_center_y": f"{parse_float(raw, 'left_y', 'left_center_y', 'left_raw_y'):.6f}",
                    "right_center_x": f"{parse_float(raw, 'right_x', 'right_center_x', 'right_raw_x'):.6f}",
                    "right_center_y": f"{parse_float(raw, 'right_y', 'right_center_y', 'right_raw_y'):.6f}",
                    "source": "real",
                    "sample_weight": "1.0",
                }
            )

    if len(rows) < 20:
        warnings.append("Very few gaze samples survived filtering; collect more real calibration data before trusting model metrics.")
    return rows, skipped, warnings


def split_interleaved(rows: list[dict[str, str]], validation_every: int) -> None:
    by_stage_seen = Counter()
    for row in rows:
        by_stage_seen[row["stage"]] += 1
        row["split"] = "val" if by_stage_seen[row["stage"]] % validation_every == 0 else "train"


def split_by_session(
    rows: list[dict[str, str]],
    val_sessions: set[str] | None,
    train_sessions: set[str] | None,
) -> list[str]:
    warnings: list[str] = []
    sessions = list(dict.fromkeys(row["session"] for row in rows))
    if train_sessions:
        rows[:] = [row for row in rows if row["session"] in train_sessions or (val_sessions and row["session"] in val_sessions)]
        sessions = list(dict.fromkeys(row["session"] for row in rows))

    if not val_sessions:
        if len(sessions) < 2:
            raise SystemExit("Session split needs at least two sessions, or pass --split-mode interleaved for smoke tests.")
        val_sessions = {sessions[-1]}
        warnings.append(f"No --val-session supplied; using {sessions[-1]} as validation session.")

    for row in rows:
        row["split"] = "val" if row["session"] in val_sessions else "train"

    train_count = sum(1 for row in rows if row["split"] == "train")
    val_count = sum(1 for row in rows if row["split"] == "val")
    if train_count == 0 or val_count == 0:
        raise SystemExit(f"Invalid split: train={train_count}, val={val_count}. Check --val-session/--train-session.")

    train_stages = {row["stage"] for row in rows if row["split"] == "train"}
    val_stages = {row["stage"] for row in rows if row["split"] == "val"}
    missing_in_train = sorted(val_stages - train_stages)
    missing_in_val = sorted(train_stages - val_stages)
    if missing_in_train:
        warnings.append(f"Validation has stages absent from train: {', '.join(missing_in_train)}.")
    if missing_in_val:
        warnings.append(f"Train has stages absent from validation: {', '.join(missing_in_val)}.")
    return warnings


def write_manifest(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "session",
        "stage",
        "target_x",
        "target_y",
        "left_file",
        "right_file",
        "left_conf",
        "right_conf",
        "left_center_x",
        "left_center_y",
        "right_center_x",
        "right_center_y",
        "source",
        "sample_weight",
        "split",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]], skipped: Counter[str], warnings: list[str], output: Path) -> None:
    summary = {
        "total_samples": len(rows),
        "split_counts": Counter(row["split"] for row in rows),
        "stage_counts": Counter(row["stage"] for row in rows),
        "session_counts": Counter(row["session"] for row in rows),
        "split_session_counts": Counter(f"{row['split']}:{row['session']}" for row in rows),
        "source_counts": Counter(row["source"] for row in rows),
        "skipped_counts": skipped,
        "warnings": warnings,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, action="append", type=Path, help="Path to calibration_pairs.csv; repeat for multiple sessions.")
    parser.add_argument("--output", required=True, type=Path, help="Output training manifest CSV.")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--min-confidence", type=float, default=0.03)
    parser.add_argument("--min-openness", type=float, default=0.08)
    parser.add_argument("--split-mode", choices=("session", "interleaved"), default="session")
    parser.add_argument("--val-session", action="append", help="Session directory name to reserve for validation; repeat for multiple.")
    parser.add_argument("--train-session", action="append", help="Optional session directory name to keep for training.")
    parser.add_argument("--validation-every", type=int, default=5, help="Use every Nth sample per stage for validation.")
    parser.add_argument("--include-stage", action="append", help="Restrict to one stage id; repeat for multiple stages.")
    args = parser.parse_args()

    if args.validation_every < 2:
        raise SystemExit("--validation-every must be >= 2")

    include_stages = set(args.include_stage) if args.include_stage else None
    rows: list[dict[str, str]] = []
    skipped = Counter()
    warnings: list[str] = []
    for pairs_path in args.pairs:
        session_rows, session_skipped, session_warnings = read_pairs(
            pairs_path.resolve(),
            args.min_confidence,
            args.min_openness,
            include_stages,
        )
        rows.extend(session_rows)
        skipped.update(session_skipped)
        warnings.extend(f"{pairs_path.parent.name}: {warning}" for warning in session_warnings)

    if not rows:
        raise SystemExit("No gaze samples survived filtering.")

    if args.split_mode == "interleaved":
        split_interleaved(rows, args.validation_every)
        warnings.append("Using interleaved split; this is for smoke tests, not real robustness evaluation.")
    else:
        warnings.extend(split_by_session(rows, set(args.val_session) if args.val_session else None, set(args.train_session) if args.train_session else None))

    write_manifest(rows, args.output.resolve())

    summary_path = args.summary.resolve() if args.summary else args.output.resolve().with_suffix(".summary.json")
    write_summary(rows, skipped, warnings, summary_path)

    print(f"Wrote {len(rows)} samples to {args.output}")
    print(f"Wrote summary to {summary_path}")
    if skipped:
        print("Skipped:", dict(skipped))
    for warning in warnings:
        print("Warning:", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
