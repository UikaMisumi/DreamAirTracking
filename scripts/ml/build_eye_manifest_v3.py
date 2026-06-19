#!/usr/bin/env python3
"""Build a strict v3 eye-model manifest from DreamAirTracking calibration sessions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_eye_weak_labels import detect_eye_features, empty_features  # noqa: E402
from eye_normalization import _detect_pupil_candidate  # noqa: E402


SCHEMA = "dream_air_tracking.eye_manifest.v3"

CORE_FIELDNAMES = [
    "sample_id",
    "wear_id",
    "session_id",
    "session",
    "sequence_id",
    "frame_index",
    "timestamp",
    "split",
    "left_file",
    "right_file",
    "target_x",
    "target_y",
    "target_source",
    "gaze_stage",
    "stage",
    "openness_target_left",
    "openness_target_right",
    "openness_source",
    "pupil_left_x",
    "pupil_left_y",
    "pupil_left_radius",
    "pupil_right_x",
    "pupil_right_y",
    "pupil_right_radius",
    "pupil_source",
    "left_quality",
    "right_quality",
    "pair_quality",
    "left_found",
    "right_found",
    "no_eye",
    "blink",
    "closed",
    "occluded",
    "reflection",
    "normalization_left_pupil_x",
    "normalization_left_pupil_y",
    "normalization_right_pupil_x",
    "normalization_right_pupil_y",
    "normalization_confidence",
    "sample_weight",
    "notes",
]

WEAK_FIELDNAMES = [
    "weak_left_openness",
    "weak_left_wide",
    "weak_left_squint",
    "weak_left_pupil_x",
    "weak_left_pupil_y",
    "weak_left_pupil_radius",
    "weak_left_quality",
    "weak_right_openness",
    "weak_right_wide",
    "weak_right_squint",
    "weak_right_pupil_x",
    "weak_right_pupil_y",
    "weak_right_pupil_radius",
    "weak_right_quality",
    "weak_pair_quality",
    "weak_blink",
    "weak_expression_mask",
    "weak_label_source",
]

LEGACY_FIELDNAMES = [
    "source",
    "left_conf",
    "right_conf",
    "left_center_x",
    "left_center_y",
    "right_center_x",
    "right_center_y",
]

FIELDNAMES = CORE_FIELDNAMES + WEAK_FIELDNAMES + LEGACY_FIELDNAMES

STAGE_TARGETS = {
    "center": (0.0, 0.0),
    "center_confirm": (0.0, 0.0),
    "open": (0.0, 0.0),
    "open_confirm": (0.0, 0.0),
    "closed": (0.0, 0.0),
    "left": (-0.7, 0.0),
    "right": (0.7, 0.0),
    "up": (0.0, 0.7),
    "down": (0.0, -0.7),
    "left_up": (-0.7, 0.7),
    "right_up": (0.7, 0.7),
    "left_down": (-0.7, -0.7),
    "right_down": (0.7, -0.7),
    "micro_left": (-0.25, 0.0),
    "micro_right": (0.25, 0.0),
    "micro_up": (0.0, 0.25),
    "micro_down": (0.0, -0.25),
    "far_left": (-1.0, 0.0),
    "far_right": (1.0, 0.0),
    "far_up": (0.0, 1.0),
    "far_down": (0.0, -1.0),
    "far_left_up": (-1.0, 1.0),
    "far_left_down": (-1.0, -1.0),
    "far_right_up": (1.0, 1.0),
    "far_right_down": (1.0, -1.0),
}

OPENNESS_TARGETS = {
    "open": (1.0, 1.0, "protocol_open"),
    "open_confirm": (1.0, 1.0, "protocol_open"),
    "open_relaxed": (1.0, 1.0, "protocol_open_relaxed"),
    "open_wide": (1.0, 1.0, "protocol_open_wide"),
    "half_closed": (0.5, 0.5, "protocol_half_closed"),
    "squint": (0.7, 0.7, "protocol_squint"),
    "closed": (0.0, 0.0, "protocol_closed"),
    "left_open_right_open": (1.0, 1.0, "protocol_asymmetric_open"),
    "left_half_right_open": (0.5, 1.0, "protocol_asymmetric_half"),
    "left_squint_right_open": (0.75, 1.0, "protocol_asymmetric_squint"),
    "left_closed_right_open": (0.0, 1.0, "protocol_asymmetric_closed"),
    "left_open_right_half": (1.0, 0.5, "protocol_asymmetric_half"),
    "left_open_right_squint": (1.0, 0.75, "protocol_asymmetric_squint"),
    "left_open_right_closed": (1.0, 0.0, "protocol_asymmetric_closed"),
    "both_half": (0.5, 0.5, "protocol_asymmetric_half"),
    "both_closed": (0.0, 0.0, "protocol_asymmetric_closed"),
    "both_open_relaxed": (1.0, 1.0, "protocol_expression_relaxed"),
    "both_open_wide": (1.0, 1.0, "protocol_expression_wide"),
    "left_wide_right_relaxed": (1.0, 1.0, "protocol_expression_wide"),
    "right_wide_left_relaxed": (1.0, 1.0, "protocol_expression_wide"),
    "both_squint": (0.7, 0.7, "protocol_expression_squint"),
    "left_squint_right_relaxed": (0.7, 1.0, "protocol_expression_squint"),
    "right_squint_left_relaxed": (1.0, 0.7, "protocol_expression_squint"),
}


@dataclass(frozen=True)
class Label:
    stage_id: str
    frame_start: int
    frame_end: int
    accepted: bool
    target_x: float
    target_y: float
    operator_verdict: str
    bad_frames: frozenset[int]
    notes: str


def fmt(value: float | int | str | bool | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def calibration_root() -> Path:
    return Path(os.environ.get("USERPROFILE", ".")) / "Documents" / "DreamAirTracking" / "calibration_data"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_session_targets(session_dir: Path) -> dict[str, tuple[float, float]]:
    payload = load_json(session_dir / "session.json")
    targets = dict(STAGE_TARGETS)
    for stage in payload.get("stages", []) or []:
        stage_id = str(stage.get("stageId") or stage.get("stage_id") or "")
        target = stage.get("target") or {}
        if stage_id:
            targets[stage_id] = (float(target.get("x", 0.0)), float(target.get("y", 0.0)))
    return targets


def load_labels(session_dir: Path, targets: dict[str, tuple[float, float]]) -> list[Label]:
    path = session_dir / "labels.jsonl"
    if not path.exists():
        return []
    labels: list[Label] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        stage_id = str(item.get("stageId") or item.get("stage_id") or "")
        target = item.get("operatorOverrideTarget") or item.get("target") or {}
        default_target = targets.get(stage_id, (0.0, 0.0))
        labels.append(
            Label(
                stage_id=stage_id,
                frame_start=int(item.get("frameStart", 0)),
                frame_end=int(item.get("frameEnd", 0)),
                accepted=bool(item.get("accepted", True)),
                target_x=float(target.get("x", default_target[0])),
                target_y=float(target.get("y", default_target[1])),
                operator_verdict=str(item.get("operatorVerdict") or ""),
                bad_frames=frozenset(int(value) for value in item.get("badFrames", []) or []),
                notes=str(item.get("notes") or ""),
            )
        )
    return labels


def label_for_sequence(labels: list[Label], stage: str, sequence: int) -> Label | None:
    for label in labels:
        if label.stage_id == stage and label.frame_start <= sequence <= label.frame_end:
            return label
    for label in labels:
        if label.frame_start <= sequence <= label.frame_end:
            return label
    return None


def read_pairs(session_dir: Path) -> list[dict[str, str]]:
    path = session_dir / "pairs.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_sessions(args: argparse.Namespace) -> list[Path]:
    sessions = [path.resolve() for path in args.session_dir]
    roots = args.session_root
    if not sessions and not roots and not args.source_manifest and not args.source_manifest_root:
        roots = [calibration_root()]
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        sessions.extend(path.resolve() for path in root.iterdir() if path.is_dir() and (path / "pairs.csv").exists())
    unique = sorted({path: None for path in sessions}.keys(), key=lambda path: path.name)
    return unique


def discover_source_manifests(args: argparse.Namespace) -> list[Path]:
    manifests = [path.resolve() for path in args.source_manifest]
    for root in args.source_manifest_root:
        root = root.resolve()
        if not root.exists():
            continue
        fine_tune_by_parent: dict[Path, Path] = {}
        for path in root.glob("**/fine_tune_manifest_*.csv"):
            current = fine_tune_by_parent.get(path.parent)
            if current is None or path.stat().st_mtime > current.stat().st_mtime:
                fine_tune_by_parent[path.parent] = path
        manifests.extend(fine_tune_by_parent.values())
        manifests.extend(root.glob("**/manifest_v3.csv"))
    return sorted({path.resolve(): None for path in manifests}.keys(), key=lambda path: str(path))


def source_manifest_session_id(path: Path) -> str:
    parent = path.parent.name
    if parent and parent not in {"frames", "snapshots"}:
        return parent
    return path.stem


def split_sessions(sessions: list[Path], args: argparse.Namespace, extra_session_ids: list[str] | None = None) -> dict[str, str]:
    explicit: dict[str, str] = {}
    for split, names in (("train", args.train_session), ("val", args.val_session), ("test", args.test_session)):
        for name in names:
            explicit[name] = split

    session_ids = [path.name for path in sessions] + list(extra_session_ids or [])
    session_ids = sorted(dict.fromkeys(session_ids))
    result: dict[str, str] = {}
    for session_id in session_ids:
        if session_id in explicit:
            result[session_id] = explicit[session_id]

    remaining = [session_id for session_id in session_ids if session_id not in result]
    if args.split_policy == "all_train":
        for session_id in remaining:
            result[session_id] = "train"
        return result

    if args.split_policy == "random":
        rng = random.Random(args.seed)
        rng.shuffle(remaining)
    # auto_latest keeps the newest sessions for harder validation and test.
    test_count = min(args.test_count, max(0, len(remaining) - 1))
    val_count = min(args.val_count, max(0, len(remaining) - test_count - 1))
    test_ids = set(remaining[-test_count:] if test_count else [])
    val_pool = remaining[: len(remaining) - test_count]
    val_ids = set(val_pool[-val_count:] if val_count else [])
    for session_id in remaining:
        if session_id in test_ids:
            result[session_id] = "test"
        elif session_id in val_ids:
            result[session_id] = "val"
        else:
            result[session_id] = "train"
    return result


def load_gray_for_weak_labels(image_path: Path, size: int) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    if size > 0:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def weak_features(image_path: Path, side: str, size: int) -> dict[str, float]:
    try:
        gray = load_gray_for_weak_labels(image_path, size)
        legacy = detect_eye_features(gray, side)
        candidate = _detect_pupil_candidate((gray * 255.0).astype(np.uint8))
        if candidate is None:
            return {
                "openness": legacy["openness"],
                "pupil_x": 0.0,
                "pupil_y": 0.0,
                "pupil_radius": 0.0,
                "quality": 0.0,
            }
        return {
            "openness": legacy["openness"],
            "pupil_x": float(np.clip(candidate.x, 0.0, 1.0)),
            "pupil_y": float(np.clip(candidate.y, 0.0, 1.0)),
            "pupil_radius": float(np.clip(candidate.radius, 0.0, 1.0)),
            "quality": float(np.clip(candidate.confidence, 0.0, 1.0)),
        }
    except Exception:
        return empty_features()


def target_for_stage(stage: str, label: Label | None, targets: dict[str, tuple[float, float]]) -> tuple[float, float, str]:
    if label is not None:
        return label.target_x, label.target_y, "label_accepted" if label.accepted else "label_rejected"
    target = targets.get(stage, STAGE_TARGETS.get(stage, (0.0, 0.0)))
    return target[0], target[1], "stage_default"


def openness_for_stage(stage: str, left_open: float, right_open: float, open_gaze_openness: float = 0.85) -> tuple[float, float, str]:
    if stage in OPENNESS_TARGETS:
        left, right, source = OPENNESS_TARGETS[stage]
        return left, right, source
    if stage in STAGE_TARGETS:
        return open_gaze_openness, open_gaze_openness, "protocol_open_gaze"
    return left_open, right_open, "legacy_detector_weak"


def expression_for_stage(stage: str) -> tuple[float, float, float, float, float]:
    normalized = stage.strip().lower()
    left_wide = right_wide = left_squint = right_squint = 0.0
    mask = 1.0
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
    elif normalized == "":
        mask = 0.0
    return left_wide, right_wide, left_squint, right_squint, mask


def openness_label_quality(source: str, fallback_quality: float) -> float:
    return 1.0 if source.startswith("protocol_") else fallback_quality


def suppress_pupil_when_not_visible(stage: str, features: dict[str, float]) -> dict[str, float]:
    if stage not in {"closed", "blink", "blink_sequence"}:
        return features
    updated = dict(features)
    updated["pupil_x"] = 0.0
    updated["pupil_y"] = 0.0
    updated["pupil_radius"] = 0.0
    updated["quality"] = 0.0
    return updated


def stage_flags(stage: str, label: Label | None, left_found: bool, right_found: bool, pair_quality: float) -> dict[str, str]:
    closed = stage in {"closed", "blink", "blink_sequence"}
    no_eye = not left_found and not right_found
    return {
        "no_eye": fmt(no_eye),
        "blink": fmt(stage in {"blink", "blink_sequence"}),
        "closed": fmt(closed),
        "occluded": fmt(no_eye or pair_quality < 0.05),
        "reflection": "",
        "notes": label.notes if label else "",
    }


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    sessions = discover_sessions(args)
    source_manifests = discover_source_manifests(args)
    source_session_ids = [source_manifest_session_id(path) for path in source_manifests]
    split_map = split_sessions(sessions, args, source_session_ids)
    wear_map = load_json(args.wear_map) if args.wear_map else {}
    rows: list[dict[str, str]] = []
    skipped_sessions: list[str] = []
    session_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}

    for session_dir in sessions:
        pairs = read_pairs(session_dir)
        if not pairs:
            skipped_sessions.append(str(session_dir))
            continue
        session_id = session_dir.name
        split = split_map.get(session_id, "train")
        wear_id = str(wear_map.get(session_id) or session_id)
        targets = load_session_targets(session_dir)
        labels = load_labels(session_dir, targets)
        accepted_ranges = [label for label in labels if label.accepted]

        for frame_index, pair in enumerate(pairs, start=1):
            sequence = int(parse_float(pair.get("sequence"), frame_index))
            stage = pair.get("stage", "")
            label = label_for_sequence(labels, stage, sequence)
            if args.accepted_only and accepted_ranges and (label is None or not label.accepted or sequence in label.bad_frames):
                continue

            left_rel = pair.get("left_file", "")
            right_rel = pair.get("right_file", "")
            left_path = session_dir / left_rel
            right_path = session_dir / right_rel
            left_found = parse_bool(pair.get("left_found"))
            right_found = parse_bool(pair.get("right_found"))
            left_open_legacy = parse_float(pair.get("left_open"), 1.0)
            right_open_legacy = parse_float(pair.get("right_open"), 1.0)
            left = weak_features(left_path, "left", args.weak_image_size) if args.weak_labels else empty_features()
            right = weak_features(right_path, "right", args.weak_image_size) if args.weak_labels else empty_features()
            pupil_pair_quality = min(left["quality"], right["quality"]) if args.weak_labels else min(
                parse_float(pair.get("left_conf"), 0.0),
                parse_float(pair.get("right_conf"), 0.0),
            )
            target_x, target_y, target_source = target_for_stage(stage, label, targets)
            openness_left, openness_right, openness_source = openness_for_stage(
                stage,
                left["openness"] if args.weak_labels else left_open_legacy,
                right["openness"] if args.weak_labels else right_open_legacy,
                args.open_gaze_openness,
            )
            left_wide, right_wide, left_squint, right_squint, expression_mask = expression_for_stage(stage)
            left = suppress_pupil_when_not_visible(stage, left)
            right = suppress_pupil_when_not_visible(stage, right)
            pupil_pair_quality = min(left["quality"], right["quality"])
            pair_quality = openness_label_quality(openness_source, pupil_pair_quality)
            flags = stage_flags(stage, label, left_found, right_found, pupil_pair_quality)
            sample_id = f"{session_id}_{stage}_{sequence:06d}"
            sample_weight = 1.0
            if label is not None and not label.accepted:
                sample_weight = 0.0
            elif not left_found or not right_found:
                sample_weight = args.low_quality_weight
            elif pupil_pair_quality < args.min_pair_quality and not openness_source.startswith("protocol_"):
                sample_weight = min(1.0, args.low_quality_weight)

            row = {
                "sample_id": sample_id,
                "wear_id": wear_id,
                "session_id": session_id,
                "session": session_id,
                "sequence_id": fmt(sequence),
                "frame_index": fmt(frame_index),
                "timestamp": "",
                "split": split,
                "left_file": str(left_path.resolve()),
                "right_file": str(right_path.resolve()),
                "target_x": fmt(target_x),
                "target_y": fmt(target_y),
                "target_source": target_source,
                "gaze_stage": stage,
                "stage": stage,
                "openness_target_left": fmt(openness_left),
                "openness_target_right": fmt(openness_right),
                "openness_source": openness_source,
                "pupil_left_x": fmt(left["pupil_x"]),
                "pupil_left_y": fmt(left["pupil_y"]),
                "pupil_left_radius": fmt(left["pupil_radius"]),
                "pupil_right_x": fmt(right["pupil_x"]),
                "pupil_right_y": fmt(right["pupil_y"]),
                "pupil_right_radius": fmt(right["pupil_radius"]),
                "pupil_source": "classical_dark_component_v1" if args.weak_labels else "",
                "left_quality": fmt(left["quality"] if args.weak_labels else parse_float(pair.get("left_conf"), 0.0)),
                "right_quality": fmt(right["quality"] if args.weak_labels else parse_float(pair.get("right_conf"), 0.0)),
                "pair_quality": fmt(pair_quality),
                "left_found": fmt(left_found),
                "right_found": fmt(right_found),
                "normalization_left_pupil_x": fmt(left["pupil_x"]),
                "normalization_left_pupil_y": fmt(left["pupil_y"]),
                "normalization_right_pupil_x": fmt(right["pupil_x"]),
                "normalization_right_pupil_y": fmt(right["pupil_y"]),
                "normalization_confidence": fmt(pupil_pair_quality),
                "sample_weight": fmt(sample_weight),
                "weak_left_openness": fmt(openness_left),
                "weak_left_wide": fmt(left_wide),
                "weak_left_squint": fmt(left_squint),
                "weak_left_pupil_x": fmt(left["pupil_x"]),
                "weak_left_pupil_y": fmt(left["pupil_y"]),
                "weak_left_pupil_radius": fmt(left["pupil_radius"]),
                "weak_left_quality": fmt(left["quality"] if args.weak_labels else parse_float(pair.get("left_conf"), 0.0)),
                "weak_right_openness": fmt(openness_right),
                "weak_right_wide": fmt(right_wide),
                "weak_right_squint": fmt(right_squint),
                "weak_right_pupil_x": fmt(right["pupil_x"]),
                "weak_right_pupil_y": fmt(right["pupil_y"]),
                "weak_right_pupil_radius": fmt(right["pupil_radius"]),
                "weak_right_quality": fmt(right["quality"] if args.weak_labels else parse_float(pair.get("right_conf"), 0.0)),
                "weak_pair_quality": fmt(pair_quality),
                "weak_blink": fmt(flags["blink"] == "true" or (openness_left + openness_right) * 0.5 < 0.18),
                "weak_expression_mask": fmt(expression_mask),
                "weak_label_source": "manifest_v3_protocol_and_classical_dark_component_v1",
                "source": "calibration_session",
                "left_conf": fmt(parse_float(pair.get("left_conf"), 0.0)),
                "right_conf": fmt(parse_float(pair.get("right_conf"), 0.0)),
                "left_center_x": fmt(parse_float(pair.get("left_raw_x"), 0.0)),
                "left_center_y": fmt(parse_float(pair.get("left_raw_y"), 0.0)),
                "right_center_x": fmt(parse_float(pair.get("right_raw_x"), 0.0)),
                "right_center_y": fmt(parse_float(pair.get("right_raw_y"), 0.0)),
                **flags,
            }
            rows.append({field: row.get(field, "") for field in FIELDNAMES})
            session_counts[session_id] = session_counts.get(session_id, 0) + 1
            split_counts[split] = split_counts.get(split, 0) + 1

    for manifest in source_manifests:
        source_rows = read_source_manifest(manifest)
        if not source_rows:
            continue
        source_session = source_manifest_session_id(manifest)
        split = split_map.get(source_session, "train")
        wear_id = str(wear_map.get(source_session) or source_session)
        for index, source_row in enumerate(source_rows, start=1):
            row = build_from_source_manifest_row(
                source_row,
                manifest,
                source_session,
                wear_id,
                split,
                index,
                args,
            )
            rows.append(row)
            session_counts[source_session] = session_counts.get(source_session, 0) + 1
            split_counts[split] = split_counts.get(split, 0) + 1

    summary = {
        "schema": SCHEMA,
        "sessions_discovered": len(sessions),
        "source_manifests_discovered": len(source_manifests),
        "sessions_used": len(session_counts),
        "skipped_sessions": skipped_sessions,
        "row_count": len(rows),
        "session_counts": dict(sorted(session_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_policy": args.split_policy,
        "weak_labels": args.weak_labels,
        "accepted_only": args.accepted_only,
    }
    return rows, summary


def read_source_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_from_source_manifest_row(
    source_row: dict[str, str],
    manifest: Path,
    session_id: str,
    wear_id: str,
    split: str,
    index: int,
    args: argparse.Namespace,
) -> dict[str, str]:
    stage = source_row.get("gaze_stage") or source_row.get("stage") or "center"
    left_path = resolve_manifest_path(manifest, source_row.get("left_file", ""))
    right_path = resolve_manifest_path(manifest, source_row.get("right_file", ""))
    left = weak_features(left_path, "left", args.weak_image_size) if args.weak_labels else empty_features()
    right = weak_features(right_path, "right", args.weak_image_size) if args.weak_labels else empty_features()
    left_found = left_path.exists()
    right_found = right_path.exists()
    pupil_pair_quality = min(left["quality"], right["quality"]) if args.weak_labels else min(
        parse_float(source_row.get("left_conf"), 0.0),
        parse_float(source_row.get("right_conf"), 0.0),
    )
    stage_target = STAGE_TARGETS.get(stage, (0.0, 0.0))
    if args.retarget_standard_gaze and stage in STAGE_TARGETS:
        target_x, target_y = stage_target
        target_source = "stage_retargeted_v3"
    else:
        target_x = parse_float(source_row.get("target_x"), stage_target[0])
        target_y = parse_float(source_row.get("target_y"), stage_target[1])
        target_source = source_row.get("target_source") or "live_prompt"
    openness_left, openness_right, openness_source = openness_for_stage(
        stage,
        left["openness"],
        right["openness"],
        args.open_gaze_openness,
    )
    left_wide, right_wide, left_squint, right_squint, expression_mask = expression_for_stage(stage)
    left = suppress_pupil_when_not_visible(stage, left)
    right = suppress_pupil_when_not_visible(stage, right)
    pupil_pair_quality = min(left["quality"], right["quality"])
    pair_quality = openness_label_quality(openness_source, pupil_pair_quality)
    flags = stage_flags(stage, None, left_found, right_found, pupil_pair_quality)
    sample_id = source_row.get("sample_id") or f"{session_id}_{stage}_{index:06d}"
    sample_weight = parse_float(source_row.get("sample_weight"), 1.0)
    if pupil_pair_quality < args.min_pair_quality and not openness_source.startswith("protocol_"):
        sample_weight = min(sample_weight, args.low_quality_weight)
    row = {
        "sample_id": sample_id,
        "wear_id": wear_id,
        "session_id": session_id,
        "session": session_id,
        "sequence_id": fmt(parse_float(source_row.get("sequence_id"), index)),
        "frame_index": fmt(index),
        "timestamp": source_row.get("timestamp", ""),
        "split": split,
        "left_file": str(left_path.resolve()),
        "right_file": str(right_path.resolve()),
        "target_x": fmt(target_x),
        "target_y": fmt(target_y),
        "target_source": target_source,
        "gaze_stage": stage,
        "stage": stage,
        "openness_target_left": fmt(openness_left),
        "openness_target_right": fmt(openness_right),
        "openness_source": openness_source,
        "pupil_left_x": fmt(left["pupil_x"]),
        "pupil_left_y": fmt(left["pupil_y"]),
        "pupil_left_radius": fmt(left["pupil_radius"]),
        "pupil_right_x": fmt(right["pupil_x"]),
        "pupil_right_y": fmt(right["pupil_y"]),
        "pupil_right_radius": fmt(right["pupil_radius"]),
        "pupil_source": "classical_dark_component_v1" if args.weak_labels else "",
        "left_quality": fmt(left["quality"] if args.weak_labels else parse_float(source_row.get("left_conf"), 0.0)),
        "right_quality": fmt(right["quality"] if args.weak_labels else parse_float(source_row.get("right_conf"), 0.0)),
        "pair_quality": fmt(pair_quality),
        "left_found": fmt(left_found),
        "right_found": fmt(right_found),
        "normalization_left_pupil_x": fmt(left["pupil_x"]),
        "normalization_left_pupil_y": fmt(left["pupil_y"]),
        "normalization_right_pupil_x": fmt(right["pupil_x"]),
        "normalization_right_pupil_y": fmt(right["pupil_y"]),
        "normalization_confidence": fmt(pupil_pair_quality),
        "sample_weight": fmt(sample_weight),
        "weak_left_openness": fmt(openness_left),
        "weak_left_wide": fmt(left_wide),
        "weak_left_squint": fmt(left_squint),
        "weak_left_pupil_x": fmt(left["pupil_x"]),
        "weak_left_pupil_y": fmt(left["pupil_y"]),
        "weak_left_pupil_radius": fmt(left["pupil_radius"]),
        "weak_left_quality": fmt(left["quality"] if args.weak_labels else parse_float(source_row.get("left_conf"), 0.0)),
        "weak_right_openness": fmt(openness_right),
        "weak_right_wide": fmt(right_wide),
        "weak_right_squint": fmt(right_squint),
        "weak_right_pupil_x": fmt(right["pupil_x"]),
        "weak_right_pupil_y": fmt(right["pupil_y"]),
        "weak_right_pupil_radius": fmt(right["pupil_radius"]),
        "weak_right_quality": fmt(right["quality"] if args.weak_labels else parse_float(source_row.get("right_conf"), 0.0)),
        "weak_pair_quality": fmt(pair_quality),
        "weak_blink": fmt(flags["blink"] == "true" or (openness_left + openness_right) * 0.5 < 0.18),
        "weak_expression_mask": fmt(expression_mask),
        "weak_label_source": "manifest_v3_protocol_and_classical_dark_component_v1",
        "source": source_row.get("source") or "live_full_parameter_capture",
        "left_conf": fmt(parse_float(source_row.get("left_conf"), 1.0)),
        "right_conf": fmt(parse_float(source_row.get("right_conf"), 1.0)),
        "left_center_x": source_row.get("left_center_x", "0.000000"),
        "left_center_y": source_row.get("left_center_y", "0.000000"),
        "right_center_x": source_row.get("right_center_x", "0.000000"),
        "right_center_y": source_row.get("right_center_y", "0.000000"),
        **flags,
    }
    return {field: row.get(field, "") for field in FIELDNAMES}


def resolve_manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest.parent / path


def write_outputs(rows: list[dict[str, str]], summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output.with_suffix(".summary.json")
    summary["manifest"] = str(output.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def session_name(value: str) -> str:
    match = re.search(r"(\d{8}_\d{6})", value)
    return match.group(1) if match else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", action="append", type=Path, default=[])
    parser.add_argument("--session-dir", action="append", type=Path, default=[])
    parser.add_argument("--source-manifest-root", action="append", type=Path, default=[])
    parser.add_argument("--source-manifest", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-policy", choices=("auto_latest", "random", "all_train"), default="auto_latest")
    parser.add_argument(
        "--retarget-standard-gaze",
        action="store_true",
        help="Ignore source manifest gaze targets for known stages and use manifest v3 stage targets.",
    )
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=2)
    parser.add_argument("--train-session", action="append", default=[])
    parser.add_argument("--val-session", action="append", default=[])
    parser.add_argument("--test-session", action="append", default=[])
    parser.add_argument("--wear-map", type=Path)
    parser.add_argument("--accepted-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weak-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weak-image-size", type=int, default=128)
    parser.add_argument("--min-pair-quality", type=float, default=0.12)
    parser.add_argument("--low-quality-weight", type=float, default=0.25)
    parser.add_argument(
        "--open-gaze-openness",
        type=float,
        default=0.85,
        help="Protocol openness label for gaze direction stages. Use 1.0 when captured gaze stages are naturally open.",
    )
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    rows, summary = build_rows(args)
    write_outputs(rows, summary, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
