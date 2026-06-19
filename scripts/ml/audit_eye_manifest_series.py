#!/usr/bin/env python3
"""Compare eye-training manifests with head-specific supervision diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFESTS = [
    (
        "B_fullparam_round1_retarget070",
        REPO_ROOT / "runs/eye_manifest_v3_fullparam_round1_retarget070/manifest_v3.csv",
        "v3_fullparam",
    ),
    (
        "C_round2_openness_tune_mixed",
        REPO_ROOT / "runs/eye_manifest_v3_openness_tune_round2_20260617/manifest_v3_mixed.csv",
        "v3_mixed",
    ),
    (
        "C_round5_eye_control_guard",
        REPO_ROOT / "runs/eye_manifest_v3_round5_eye_control_guard_20260617/manifest_v3_eye_control_guard.csv",
        "v3_reweighted",
    ),
    (
        "C_round6_gaze_preserve",
        REPO_ROOT / "runs/eye_manifest_v3_round6_gaze_preserve_20260618/manifest_v3_gaze_preserve.csv",
        "v3_reweighted",
    ),
    (
        "D_v4_round6_clean_headmask",
        REPO_ROOT / "runs/eye_manifest_v4_round6_clean_20260618/manifest_v4_clean.csv",
        "v4_headmask",
    ),
]

DEFAULT_METADATA_ONLY = [
    (
        "A_gaze_only_mobilenetv3_round1_round2",
        REPO_ROOT / "runs/gaze_mobilenetv3_small_live_round1_round2/training_report.json",
        REPO_ROOT / "runs/gaze_mobilenetv3_small_live_round1_round2/gaze_baseline.metadata.json",
    )
]

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
    "left_squint_right_relaxed",
    "right_squint_left_relaxed",
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


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def has_value(row: dict[str, str], key: str) -> bool:
    return row.get(key, "") != ""


def norm(value: str | None) -> str:
    return (value or "").strip().lower()


def path_key(value: str) -> str:
    return str(Path(value)).lower()


def pair_key(row: dict[str, str]) -> str:
    if row.get("pair_key"):
        return norm(row["pair_key"])
    return f"{path_key(row.get('left_file', ''))}|{path_key(row.get('right_file', ''))}"


def target_abs_max(row: dict[str, str]) -> float:
    return max(abs(parse_float(row.get("target_x"))), abs(parse_float(row.get("target_y"))))


def gaze_bucket(row: dict[str, str]) -> str:
    value = target_abs_max(row)
    if value < 1e-6:
        return "center"
    if value < 0.45:
        return "micro"
    return "edge"


def legacy_gaze_weight(row: dict[str, str]) -> float:
    if not has_value(row, "target_x") or not has_value(row, "target_y"):
        return 0.0
    return max(0.0, parse_float(row.get("sample_weight"), 1.0))


def strict_gaze_weight(row: dict[str, str]) -> float:
    if has_value(row, "gaze_weight"):
        return max(0.0, parse_float(row.get("gaze_weight")))
    if norm(row.get("source")) != "live":
        return 0.0
    return legacy_gaze_weight(row)


def expression_valid(row: dict[str, str]) -> bool:
    if has_value(row, "wide_valid_left") or has_value(row, "wide_valid_right") or has_value(row, "squint_valid_left") or has_value(row, "squint_valid_right"):
        return (
            parse_float(row.get("wide_valid_left")) > 0.0
            or parse_float(row.get("wide_valid_right")) > 0.0
            or parse_float(row.get("squint_valid_left")) > 0.0
            or parse_float(row.get("squint_valid_right")) > 0.0
        )
    source = norm(row.get("source"))
    stage = norm(row.get("stage") or row.get("gaze_stage"))
    return source in EXPRESSION_VALID_SOURCES or stage in EXPRESSION_VALID_STAGES or parse_float(row.get("weak_expression_mask")) > 0.0


def openness_valid(row: dict[str, str], side: str) -> bool:
    key = f"openness_valid_{side}"
    if has_value(row, key):
        return parse_float(row.get(key)) > 0.0
    return has_value(row, f"weak_{side}_openness") or has_value(row, f"openness_target_{side}")


def openness_target(row: dict[str, str], side: str) -> float:
    if has_value(row, f"weak_{side}_openness"):
        return parse_float(row.get(f"weak_{side}_openness"), 1.0)
    return parse_float(row.get(f"openness_target_{side}"), 1.0)


def pupil_valid(row: dict[str, str], side: str, min_quality: float, min_openness: float) -> bool:
    key = f"pupil_valid_{side}"
    if has_value(row, key):
        return parse_float(row.get(key)) > 0.0
    if norm(row.get("source")) == "synthetic_eyelid_mix":
        return False
    quality = parse_float(row.get(f"weak_{side}_quality"), parse_float(row.get(f"{side}_quality")))
    return quality >= min_quality and openness_target(row, side) >= min_openness


def quantiles(values: list[float]) -> dict[str, float | int]:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return {"count": 0}

    def q(p: float) -> float:
        pos = (len(values) - 1) * p
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return values[low]
        return values[low] * (high - pos) + values[high] * (pos - low)

    return {
        "count": len(values),
        "min": values[0],
        "p05": q(0.05),
        "p50": q(0.50),
        "p95": q(0.95),
        "max": values[-1],
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def split_counts(rows: list[dict[str, str]], predicate) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if predicate(row):
            counts[row.get("split", "")] += 1
    return dict(sorted(counts.items()))


def bucket_counts(rows: list[dict[str, str]], weight_fn) -> dict[str, dict[str, int]]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if weight_fn(row) > 0.0:
            by_split[row.get("split", "")][gaze_bucket(row)] += 1
    return {split: dict(sorted(counts.items())) for split, counts in sorted(by_split.items())}


def supervised_counts(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "legacy_gaze_rows": split_counts(rows, lambda row: legacy_gaze_weight(row) > 0.0),
        "strict_gaze_rows": split_counts(rows, lambda row: strict_gaze_weight(row) > 0.0),
        "legacy_gaze_buckets": bucket_counts(rows, legacy_gaze_weight),
        "strict_gaze_buckets": bucket_counts(rows, strict_gaze_weight),
        "openness_left": split_counts(rows, lambda row: openness_valid(row, "left")),
        "openness_right": split_counts(rows, lambda row: openness_valid(row, "right")),
        "expression_rows": split_counts(rows, expression_valid),
        "pupil_left": split_counts(rows, lambda row: pupil_valid(row, "left", args.min_pupil_quality, args.min_pupil_openness)),
        "pupil_right": split_counts(rows, lambda row: pupil_valid(row, "right", args.min_pupil_quality, args.min_pupil_openness)),
    }


def sample_quality(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "pair": quantiles([parse_float(row.get("weak_pair_quality"), parse_float(row.get("pair_quality"))) for row in rows if has_value(row, "weak_pair_quality") or has_value(row, "pair_quality")]),
        "left": quantiles([parse_float(row.get("weak_left_quality"), parse_float(row.get("left_quality"), parse_float(row.get("left_conf")))) for row in rows if has_value(row, "weak_left_quality") or has_value(row, "left_quality") or has_value(row, "left_conf")]),
        "right": quantiles([parse_float(row.get("weak_right_quality"), parse_float(row.get("right_quality"), parse_float(row.get("right_conf")))) for row in rows if has_value(row, "weak_right_quality") or has_value(row, "right_quality") or has_value(row, "right_conf")]),
    }


def check_files(rows: list[dict[str, str]], manifest_dir: Path, limit: int) -> dict[str, Any]:
    missing: list[str] = []
    checked = 0
    unique_paths: set[str] = set()
    for row in rows:
        for key in ("left_file", "right_file"):
            value = row.get(key, "")
            if not value:
                missing.append(f"row missing {key}: {row.get('sample_id', '')}")
                continue
            path = Path(value)
            if not path.is_absolute():
                path = manifest_dir / path
            path_string = str(path)
            if path_string in unique_paths:
                continue
            unique_paths.add(path_string)
            checked += 1
            if not path.exists():
                missing.append(path_string)
            if limit > 0 and checked >= limit:
                return {"checked_unique_files": checked, "missing_file_count": len(missing), "missing_preview": missing[:20], "limited": True}
    return {"checked_unique_files": checked, "missing_file_count": len(missing), "missing_preview": missing[:20], "limited": False}


def audit_manifest(label: str, path: Path, kind: str, args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(path)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[pair_key(row)].append(row)

    split_conflicts = [
        {
            "pair_key": key,
            "splits": sorted({row.get("split", "") for row in group}),
            "sample_ids": [row.get("sample_id", "") for row in group[:6]],
        }
        for key, group in groups.items()
        if len({row.get("split", "") for row in group}) > 1
    ]
    session_splits: dict[str, Counter[str]] = defaultdict(Counter)
    session_sources: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        session = row.get("session_id") or row.get("session") or row.get("wear_id", "")
        session_splits[session][row.get("split", "")] += 1
        session_sources[session][norm(row.get("source")) or "-"] += 1
    sessions_with_multiple_splits = {
        session: dict(counter)
        for session, counter in session_splits.items()
        if session and sum(1 for value in counter.values() if value > 0) > 1
    }
    real_sessions_with_multiple_splits = {
        session: counts
        for session, counts in sessions_with_multiple_splits.items()
        if set(session_sources.get(session, {})) - {"synthetic_eyelid_mix"}
    }
    pseudo_sessions_with_multiple_splits = {
        session: counts
        for session, counts in sessions_with_multiple_splits.items()
        if session not in real_sessions_with_multiple_splits
    }

    legacy_total = sum(1 for row in rows if legacy_gaze_weight(row) > 0.0)
    strict_total = sum(1 for row in rows if strict_gaze_weight(row) > 0.0)
    legacy_nonlive_center = sum(
        1
        for row in rows
        if legacy_gaze_weight(row) > 0.0 and strict_gaze_weight(row) <= 0.0 and gaze_bucket(row) == "center"
    )

    file_report = check_files(rows, path.parent, args.max_file_checks) if args.check_files else {"checked_unique_files": 0, "missing_file_count": None, "missing_preview": [], "limited": False}
    risks = []
    duplicate_factor = len(rows) / max(len(groups), 1)
    if duplicate_factor > args.max_duplicate_factor:
        risks.append(f"duplicate_factor {duplicate_factor:.2f} > {args.max_duplicate_factor:.2f}")
    if strict_total and legacy_total / max(strict_total, 1) > args.max_legacy_to_strict_gaze_ratio:
        risks.append(f"legacy/strict gaze ratio {legacy_total / strict_total:.2f} > {args.max_legacy_to_strict_gaze_ratio:.2f}")
    if legacy_nonlive_center:
        risks.append(f"legacy non-gaze center rows would supervise gaze: {legacy_nonlive_center}")
    if split_conflicts:
        risks.append(f"same image pair appears in multiple splits: {len(split_conflicts)} groups")
    if real_sessions_with_multiple_splits:
        risks.append(f"real sessions span multiple splits: {len(real_sessions_with_multiple_splits)} sessions")
    if file_report["missing_file_count"]:
        risks.append(f"missing image files: {file_report['missing_file_count']}")

    return {
        "label": label,
        "kind": kind,
        "manifest": str(path.resolve()),
        "row_count": len(rows),
        "unique_pairs": len(groups),
        "duplicate_factor": duplicate_factor,
        "duplicate_group_count": sum(1 for group in groups.values() if len(group) > 1),
        "max_repeat_count": max((len(group) for group in groups.values()), default=0),
        "split_counts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "source_counts": dict(Counter(norm(row.get("source")) or "-" for row in rows).most_common()),
        "stage_counts_top40": dict(Counter(norm(row.get("stage") or row.get("gaze_stage")) for row in rows).most_common(40)),
        "supervision": supervised_counts(rows, args),
        "legacy_gaze_rows_total": legacy_total,
        "strict_gaze_rows_total": strict_total,
        "legacy_nonlive_center_gaze_rows": legacy_nonlive_center,
        "quality": sample_quality(rows),
        "file_check": file_report,
        "split_conflict_group_count": len(split_conflicts),
        "split_conflict_preview": split_conflicts[:20],
        "sessions_with_multiple_splits_count": len(sessions_with_multiple_splits),
        "sessions_with_multiple_splits_preview": dict(list(sessions_with_multiple_splits.items())[:20]),
        "real_sessions_with_multiple_splits_count": len(real_sessions_with_multiple_splits),
        "real_sessions_with_multiple_splits_preview": dict(list(real_sessions_with_multiple_splits.items())[:20]),
        "pseudo_sessions_with_multiple_splits_count": len(pseudo_sessions_with_multiple_splits),
        "pseudo_sessions_with_multiple_splits_preview": dict(list(pseudo_sessions_with_multiple_splits.items())[:20]),
        "risks": risks,
        "passed": not risks,
    }


def load_metadata_only(label: str, report_path: Path, metadata_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {
        "label": label,
        "training_report": str(report_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "model_type": report.get("model_type") or metadata.get("model_type"),
        "train_samples": report.get("train_samples"),
        "val_samples": report.get("val_samples"),
        "best_val_loss": report.get("best_val_loss"),
        "best_epoch": report.get("best_epoch"),
        "image_size": metadata.get("image_size"),
        "input_image_shape": metadata.get("input_image_shape"),
        "output": metadata.get("output"),
        "audit_note": "No manifest CSV is recorded here, so duplicate/split/head-specific supervision cannot be audited from this run alone.",
    }


def fmt_int_map(mapping: dict[str, int]) -> str:
    if not mapping:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in mapping.items())


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Eye Manifest Series Audit",
        "",
        "This report compares row count with effective head-specific supervision. `legacy_gaze_rows` is what an old target/sample_weight-only trainer would consume; `strict_gaze_rows` is what the current head-isolated rule allows.",
        "",
        "## Metadata-Only Baselines",
        "",
        "| label | model | train | val | note |",
        "|---|---|---:|---:|---|",
    ]
    for item in report["metadata_only"]:
        lines.append(
            f"| {item['label']} | {item.get('model_type') or '-'} | {item.get('train_samples') or '-'} | {item.get('val_samples') or '-'} | manifest missing |"
        )
    lines.extend(
        [
            "",
            "## Manifest Summary",
            "",
            "| label | rows | unique pairs | dup factor | strict gaze | legacy gaze | nonlive center legacy gaze | expression | risks |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["manifests"]:
        expression_total = sum(item["supervision"]["expression_rows"].values())
        risks = "<br>".join(item["risks"]) if item["risks"] else "none"
        lines.append(
            f"| {item['label']} | {item['row_count']} | {item['unique_pairs']} | {item['duplicate_factor']:.2f} | "
            f"{item['strict_gaze_rows_total']} | {item['legacy_gaze_rows_total']} | {item['legacy_nonlive_center_gaze_rows']} | "
            f"{expression_total} | {risks} |"
        )

    lines.extend(["", "## Head Supervision By Split", ""])
    for item in report["manifests"]:
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- strict gaze: {fmt_int_map(item['supervision']['strict_gaze_rows'])}",
                f"- legacy gaze: {fmt_int_map(item['supervision']['legacy_gaze_rows'])}",
                f"- expression: {fmt_int_map(item['supervision']['expression_rows'])}",
                f"- openness L/R: {fmt_int_map(item['supervision']['openness_left'])} / {fmt_int_map(item['supervision']['openness_right'])}",
                f"- pupil L/R: {fmt_int_map(item['supervision']['pupil_left'])} / {fmt_int_map(item['supervision']['pupil_right'])}",
                f"- strict gaze buckets: `{json.dumps(item['supervision']['strict_gaze_buckets'], ensure_ascii=False)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Engineering Reading",
            "",
            "- Treat CSV row count as a sampling weight, not as new information. Unique image pairs and strict per-head counts are the decision numbers.",
            "- If `legacy_gaze_rows` is much larger than `strict_gaze_rows`, the old trainer would train gaze on rows whose purpose was eyelid/expression/pupil.",
            "- Any manifest with high duplicate factor should not be used to claim data scaling unless new unique image pairs also increased.",
            "- New captures should be accepted only if this audit shows they add strict gaze edge coverage or real expression/openness coverage without split leakage.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_manifest_arg(value: str) -> tuple[str, Path, str]:
    parts = value.split("=", 1)
    if len(parts) == 1:
        path = Path(parts[0])
        return path.stem, path, "custom"
    label, path = parts
    return label, Path(path), "custom"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", help="Optional manifest path or label=path. Defaults to the current key series.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/eye_manifest_series_audit_20260618")
    parser.add_argument("--check-files", action="store_true", help="Check image file existence.")
    parser.add_argument("--max-file-checks", type=int, default=0, help="Maximum unique files to check; 0 means all.")
    parser.add_argument("--min-pupil-quality", type=float, default=0.12)
    parser.add_argument("--min-pupil-openness", type=float, default=0.75)
    parser.add_argument("--max-duplicate-factor", type=float, default=1.50)
    parser.add_argument("--max-legacy-to-strict-gaze-ratio", type=float, default=1.25)
    args = parser.parse_args()

    manifests = [parse_manifest_arg(value) for value in args.manifest] if args.manifest else DEFAULT_MANIFESTS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_reports = []
    for label, path, kind in manifests:
        resolved = path if path.is_absolute() else REPO_ROOT / path
        if not resolved.exists():
            manifest_reports.append({"label": label, "kind": kind, "manifest": str(resolved), "error": "missing_manifest", "passed": False})
            continue
        manifest_reports.append(audit_manifest(label, resolved, kind, args))

    metadata_only = [
        load_metadata_only(label, report_path, metadata_path)
        for label, report_path, metadata_path in DEFAULT_METADATA_ONLY
        if report_path.exists() or metadata_path.exists()
    ]
    report = {
        "output_dir": str(args.output_dir.resolve()),
        "manifests": manifest_reports,
        "metadata_only": metadata_only,
        "passed": all(item.get("passed") for item in manifest_reports if "error" not in item),
    }
    json_path = args.output_dir / "series_audit.json"
    md_path = args.output_dir / "series_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, report)

    print(json.dumps({"passed": report["passed"], "manifest_count": len(manifest_reports), "json": str(json_path.resolve()), "markdown": str(md_path.resolve())}, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
