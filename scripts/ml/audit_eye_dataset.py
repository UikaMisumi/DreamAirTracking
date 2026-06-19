#!/usr/bin/env python3
"""Audit eye tracking training manifests before model training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class AuditRow:
    raw: dict[str, str]
    index: int
    sample_id: str
    split: str
    session: str
    wear_id: str
    stage: str
    left_file: Path | None
    right_file: Path | None


def get_first(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def parse_float(row: dict[str, str], *names: str) -> float | None:
    value = get_first(row, *names)
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def resolve_path(value: str, manifest_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_dir / path


def read_manifest(path: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            session = get_first(row, "session_id", "session")
            wear_id = get_first(row, "wear_id", default=session)
            stage = get_first(row, "gaze_stage", "stage")
            rows.append(
                AuditRow(
                    raw=row,
                    index=index,
                    sample_id=get_first(row, "sample_id", default=str(index)),
                    split=get_first(row, "split", default=""),
                    session=session,
                    wear_id=wear_id,
                    stage=stage,
                    left_file=resolve_path(get_first(row, "left_file"), path.parent),
                    right_file=resolve_path(get_first(row, "right_file"), path.parent),
                )
            )
    return rows


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)

    def q(p: float) -> float:
        i = (len(values) - 1) * p
        low = int(math.floor(i))
        high = int(math.ceil(i))
        if low == high:
            return values[low]
        return values[low] * (high - i) + values[high] * (i - low)

    return {
        "min": values[0],
        "p05": q(0.05),
        "p50": q(0.50),
        "p95": q(0.95),
        "max": values[-1],
    }


def count_existing_files(rows: Iterable[AuditRow]) -> tuple[int, list[str]]:
    missing: list[str] = []
    checked = 0
    for row in rows:
        for path in (row.left_file, row.right_file):
            if path is None:
                missing.append(f"{row.sample_id}: empty file path")
                continue
            checked += 1
            if not path.exists():
                missing.append(str(path))
    return checked, missing


def field_stats(rows: list[AuditRow], field_names: list[str]) -> dict[str, object]:
    values: list[float] = []
    for row in rows:
        value = parse_float(row.raw, *field_names)
        if value is not None:
            values.append(value)
    return {"count": len(values), **quantiles(values)}


def pick_rows(rows: list[AuditRow], limit: int, seed: int) -> list[AuditRow]:
    if len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def make_contact_sheet(rows: list[AuditRow], output: Path, title: str, image_size: int = 96, columns: int = 4) -> None:
    valid = [row for row in rows if row.left_file and row.right_file and row.left_file.exists() and row.right_file.exists()]
    if not valid:
        return
    tile_w = image_size * 2
    label_h = 28
    rows_count = math.ceil(len(valid) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows_count * (image_size + label_h)), (24, 26, 32))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), title[:80], fill=(235, 235, 235))
    for i, row in enumerate(valid):
        x = (i % columns) * tile_w
        y = (i // columns) * (image_size + label_h)
        try:
            left = Image.open(row.left_file).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
            right = Image.open(row.right_file).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
        except OSError:
            continue
        sheet.paste(Image.merge("RGB", (left, left, left)), (x, y + label_h))
        sheet.paste(Image.merge("RGB", (right, right, right)), (x + image_size, y + label_h))
        label = f"{row.split} {row.session} {row.stage}"
        draw.text((x + 4, y + 8), label[:34], fill=(245, 245, 245))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def fail_if(condition: bool, failures: list[str], message: str) -> None:
    if condition:
        failures.append(message)


def audit(args: argparse.Namespace) -> dict[str, object]:
    manifest = args.manifest.resolve()
    rows = read_manifest(manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts = Counter(row.split for row in rows)
    stage_counts = Counter(row.stage for row in rows)
    split_sessions: dict[str, set[str]] = defaultdict(set)
    split_wears: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_sessions[row.split].add(row.session)
        split_wears[row.split].add(row.wear_id)

    failures: list[str] = []
    checked_files, missing_files = count_existing_files(rows)
    fail_if(len(rows) == 0, failures, "manifest has no rows")
    fail_if(bool(missing_files), failures, f"missing image files: {len(missing_files)}")

    train_wears = split_wears.get("train", set())
    val_wears = split_wears.get("val", set())
    test_wears = split_wears.get("test", set())
    fail_if(bool(train_wears & val_wears), failures, "train and val share wear_id/session")
    fail_if(bool(train_wears & test_wears), failures, "train and test share wear_id/session")
    fail_if(bool(val_wears & test_wears), failures, "val and test share wear_id/session")
    fail_if(len(val_wears) < args.min_val_sessions, failures, f"val wear/session count < {args.min_val_sessions}")
    fail_if(len(test_wears) < args.min_test_sessions, failures, f"test wear/session count < {args.min_test_sessions}")

    for stage in args.required_stage:
        fail_if(stage_counts.get(stage, 0) < args.min_stage_samples, failures, f"stage {stage} samples < {args.min_stage_samples}")

    if args.require_micro:
        for stage in ("micro_left", "micro_right"):
            fail_if(stage_counts.get(stage, 0) == 0, failures, f"required micro stage missing: {stage}")

    quality_values = [
        parse_float(row.raw, "weak_pair_quality", "pair_quality")
        for row in rows
    ]
    quality_values = [value for value in quality_values if value is not None]
    high_quality = sum(1 for value in quality_values if value >= args.min_pair_quality)
    if rows:
        fail_if(high_quality / len(rows) < args.min_high_quality_ratio, failures, "high-quality pupil/confidence labels below threshold")

    report = {
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "sample_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "split_session_counts": {split: len(values) for split, values in sorted(split_sessions.items())},
        "split_wear_counts": {split: len(values) for split, values in sorted(split_wears.items())},
        "checked_files": checked_files,
        "missing_file_count": len(missing_files),
        "missing_files_preview": missing_files[:50],
        "quality": {
            "pair": field_stats(rows, ["weak_pair_quality", "pair_quality"]),
            "left": field_stats(rows, ["weak_left_quality", "left_quality", "left_conf"]),
            "right": field_stats(rows, ["weak_right_quality", "right_quality", "right_conf"]),
        },
        "openness": {
            "left": field_stats(rows, ["weak_left_openness", "openness_target_left", "left_openness"]),
            "right": field_stats(rows, ["weak_right_openness", "openness_target_right", "right_openness"]),
        },
        "pupil": {
            "left_x": field_stats(rows, ["weak_left_pupil_x", "pupil_left_x"]),
            "right_x": field_stats(rows, ["weak_right_pupil_x", "pupil_right_x"]),
            "left_radius": field_stats(rows, ["weak_left_pupil_radius", "pupil_left_radius"]),
            "right_radius": field_stats(rows, ["weak_right_pupil_radius", "pupil_right_radius"]),
        },
        "failures": failures,
        "passed": not failures,
    }

    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(output_dir / "audit_report.md", report)

    make_contact_sheet(pick_rows(rows, args.contact_sheet_samples, args.seed), output_dir / "contact_sheet_random.jpg", "random samples")
    low_quality_rows = [
        row for row in rows
        if (parse_float(row.raw, "weak_pair_quality", "pair_quality") or 0.0) < args.min_pair_quality
    ]
    make_contact_sheet(pick_rows(low_quality_rows, args.contact_sheet_samples, args.seed + 1), output_dir / "contact_sheet_low_quality.jpg", "low quality samples")
    per_stage: list[AuditRow] = []
    for stage in sorted(stage_counts):
        per_stage.extend(pick_rows([row for row in rows if row.stage == stage], max(1, args.contact_sheet_samples // max(len(stage_counts), 1)), args.seed + len(stage)))
    make_contact_sheet(per_stage[: args.contact_sheet_samples], output_dir / "contact_sheet_per_stage.jpg", "per-stage samples")
    return report


def markdown_table(mapping: dict[str, object]) -> str:
    lines = ["| key | value |", "| --- | ---: |"]
    for key, value in mapping.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    failures = report["failures"]
    lines = [
        "# Eye Dataset Audit",
        "",
        f"Manifest: `{report['manifest']}`",
        "",
        f"Status: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Summary",
        "",
        markdown_table(
            {
                "sample_count": report["sample_count"],
                "checked_files": report["checked_files"],
                "missing_file_count": report["missing_file_count"],
            }
        ),
        "",
        "## Splits",
        "",
        "### Sample Counts",
        "",
        markdown_table(report["split_counts"]),
        "",
        "### Wear Counts",
        "",
        markdown_table(report["split_wear_counts"]),
        "",
        "## Stages",
        "",
        markdown_table(report["stage_counts"]),
        "",
        "## Failures",
        "",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Contact Sheets",
            "",
            "- `contact_sheet_random.jpg`",
            "- `contact_sheet_low_quality.jpg`",
            "- `contact_sheet_per_stage.jpg`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-val-sessions", type=int, default=2)
    parser.add_argument("--min-test-sessions", type=int, default=2)
    parser.add_argument("--min-stage-samples", type=int, default=300)
    parser.add_argument("--required-stage", action="append", default=["center", "left", "right", "up", "down"])
    parser.add_argument("--require-micro", action="store_true")
    parser.add_argument("--min-pair-quality", type=float, default=0.50)
    parser.add_argument("--min-high-quality-ratio", type=float, default=0.50)
    parser.add_argument("--contact-sheet-samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    report = audit(args)
    print(f"Audit {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Report: {Path(report['output_dir']) / 'audit_report.md'}")
    if report["failures"]:
        for failure in report["failures"]:
            print(f"- {failure}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
