#!/usr/bin/env python3
"""Automatically discover, train, evaluate, and review gaze sessions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
REQUIRED_STAGE_ORDER = (
    "center",
    "up",
    "down",
    "left",
    "right",
    "left_up",
    "right_up",
    "left_down",
    "right_down",
)
REQUIRED_STAGES = set(REQUIRED_STAGE_ORDER)


@dataclass(frozen=True)
class SessionCandidate:
    session: str
    pairs_path: Path
    counts: dict[str, int]
    raw_counts: dict[str, int]
    total_rows: int
    complete: bool
    missing: tuple[str, ...]
    metrics_quality: str
    metrics_average_residual: float | None
    poor_labels: tuple[str, ...]
    high_residual_stages: tuple[str, ...]
    note_causes: dict[str, dict[str, int]]
    excluded: bool = False
    exclude_reason: str = ""


def default_roots() -> list[Path]:
    roots = [WORKSPACE_ROOT]
    docs = Path.home() / "Documents" / "DreamAirTracking" / "calibration_data"
    if docs not in roots:
        roots.append(docs)
    return roots


def read_stage_counts(path: Path, min_openness: float, accepted: list[dict[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not accepted:
        return counts
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stage = (row.get("stage") or row.get("Stage") or row.get("StageId") or "").strip()
            if not stage:
                continue
            sequence = int(float(row.get("sequence") or row.get("Sequence") or row.get("pair") or "0"))
            if accepted and not any(item["stage"] == stage and item["start"] <= sequence <= item["end"] for item in accepted):
                continue
            if not is_usable_pair(row, min_openness):
                continue
            counts[stage] += 1
    return counts


def read_raw_stage_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stage = (row.get("stage") or row.get("Stage") or row.get("StageId") or "").strip()
            if stage:
                counts[stage] += 1
    return counts


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
            ranges.append(
                {
                    "stage": str(label.get("stageId", "")),
                    "start": int(label.get("frameStart", 0)),
                    "end": int(label.get("frameEnd", -1)),
                }
            )
    return ranges


def read_session_exclude(session_root: Path) -> tuple[bool, str]:
    path = session_root / "session_exclude.json"
    if not path.exists():
        return False, ""
    try:
        data = json.load(path.open("r", encoding="utf-8"))
    except json.JSONDecodeError:
        return True, "invalid session_exclude.json"
    if not bool(data.get("exclude", True)):
        return False, ""
    return True, str(data.get("reason", "session excluded"))


def read_poor_labels(session_root: Path) -> tuple[str, ...]:
    path = session_root / "labels.jsonl"
    if not path.exists():
        return ()
    stages: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            label = json.loads(line)
            accepted = bool(label.get("accepted", False))
            verdict = str(label.get("operatorVerdict", "")).lower()
            if not accepted or verdict in {"poor", "bad"}:
                stage = str(label.get("stageId", ""))
                if stage:
                    stages.append(stage)
    return tuple(stages)


def read_label_note_causes(session_root: Path) -> dict[str, dict[str, int]]:
    path = session_root / "labels.jsonl"
    causes = {stage: Counter() for stage in REQUIRED_STAGE_ORDER}
    if not path.exists():
        return {stage: {} for stage in REQUIRED_STAGE_ORDER}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            label = json.loads(line)
            stage = str(label.get("stageId", ""))
            if stage not in causes:
                continue
            causes[stage].update(extract_note_causes(str(label.get("notes", ""))))
    return {stage: dict(causes[stage]) for stage in REQUIRED_STAGE_ORDER}


def extract_note_causes(notes: str) -> Counter[str]:
    result: Counter[str] = Counter()
    if not notes:
        return result
    lowered = notes.lower()
    for part in [item.strip() for item in lowered.replace(",", ";").split(";")]:
        if "=" not in part:
            continue
        key, value = [item.strip() for item in part.split("=", 1)]
        if key not in {"low_openness", "low_confidence", "not_found", "sync_late"}:
            continue
        try:
            count = int(float(value.split("/")[0]))
        except ValueError:
            continue
        if count > 0:
            result[key] += count

    reason_patterns = {
        "low eye openness": "low_openness",
        "pupil detection is missing": "not_found",
        "pupil confidence is too low": "low_confidence",
        "frame sync is too late": "sync_late",
        "too few usable": "not_enough_usable",
    }
    for pattern, cause in reason_patterns.items():
        if pattern in lowered:
            result[cause] += 1
    return result


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def is_usable_pair(row: dict[str, str], min_openness: float) -> bool:
    left_found = parse_bool(row.get("left_found") or row.get("LeftFound") or "")
    right_found = parse_bool(row.get("right_found") or row.get("RightFound") or "")
    left_conf = float(row.get("left_conf") or row.get("left_confidence") or row.get("LeftConfidence") or 0)
    right_conf = float(row.get("right_conf") or row.get("right_confidence") or row.get("RightConfidence") or 0)
    left_open = float(row.get("left_open") or row.get("left_openness") or 1.0)
    right_open = float(row.get("right_open") or row.get("right_openness") or 1.0)
    return left_found and right_found and left_conf >= 0.03 and right_conf >= 0.03 and left_open >= min_openness and right_open >= min_openness


def has_frame_files(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            left = row.get("left_file") or row.get("LeftFile")
            right = row.get("right_file") or row.get("RightFile")
            if not left or not right:
                continue
            left_path = Path(left)
            right_path = Path(right)
            if not left_path.is_absolute():
                left_path = path.parent / left_path
            if not right_path.is_absolute():
                right_path = path.parent / right_path
            return left_path.exists() and right_path.exists()
    return False


def read_metrics(session_root: Path) -> tuple[str, float | None, tuple[str, ...]]:
    path = session_root / "metrics.json"
    if not path.exists():
        return "missing", None, ()
    data = json.load(path.open("r", encoding="utf-8"))
    quality = str(data.get("quality", "unknown"))
    eyes = data.get("eyes", {})
    values = []
    for eye in ("left", "right"):
        value = eyes.get(eye, {}).get("averageResidual")
        if isinstance(value, int | float):
            values.append(float(value))
    average = sum(values) / len(values) if values else None
    residuals = data.get("stageResiduals", {})
    high_residual = tuple(
        stage
        for stage, residual in residuals.items()
        if isinstance(residual, dict) and float(residual.get("distance") or 0) > 0.5
    )
    return quality, average, high_residual


def discover_sessions(roots: list[Path], min_stage_samples: int, min_openness: float) -> list[SessionCandidate]:
    seen: set[Path] = set()
    candidates: list[SessionCandidate] = []
    for root in roots:
        if not root.exists():
            continue
        for name in ("pairs.csv", "calibration_pairs.csv"):
            for path in root.rglob(name):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                if not has_frame_files(path):
                    continue
                excluded, exclude_reason = read_session_exclude(path.parent)
                accepted_ranges = load_accepted_ranges(path.parent)
                counts = read_stage_counts(path, min_openness, accepted_ranges)
                raw_counts = read_raw_stage_counts(path)
                missing = tuple(sorted(stage for stage in REQUIRED_STAGES if counts.get(stage, 0) < min_stage_samples))
                quality, residual, high_residual = read_metrics(path.parent)
                poor_labels = read_poor_labels(path.parent)
                note_causes = read_label_note_causes(path.parent)
                candidates.append(
                    SessionCandidate(
                        session=path.parent.name,
                        pairs_path=resolved,
                        counts=dict(sorted(counts.items())),
                        raw_counts=dict(sorted(raw_counts.items())),
                        total_rows=sum(counts.values()),
                        complete=not missing,
                        missing=missing,
                        metrics_quality=quality,
                        metrics_average_residual=residual,
                        poor_labels=poor_labels,
                        high_residual_stages=high_residual,
                        note_causes=note_causes,
                        excluded=excluded,
                        exclude_reason=exclude_reason,
                    )
                )
    return sorted(candidates, key=lambda item: (item.pairs_path.parent.stat().st_mtime, item.session))


def run_command(args: list[str], cwd: Path) -> None:
    print("+", " ".join(str(arg) for arg in args))
    subprocess.run(args, cwd=str(cwd), check=True)


def strict_complete_quality(candidates: list[SessionCandidate]) -> list[SessionCandidate]:
    return [
        item for item in candidates
        if not item.excluded and item.complete and item.metrics_quality in {"fair", "good"}
    ]


def image_training_sessions(candidates: list[SessionCandidate], allowed_quality: set[str]) -> list[SessionCandidate]:
    return [
        item for item in candidates
        if not item.excluded and item.complete and item.metrics_quality in allowed_quality
    ]


def choose_bridge_layer_source(candidates: list[SessionCandidate]) -> SessionCandidate | None:
    usable = strict_complete_quality(candidates)
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda item: (
            {"good": 2, "fair": 1}.get(item.metrics_quality, 0),
            -(item.metrics_average_residual if item.metrics_average_residual is not None else 999),
            item.pairs_path.parent.stat().st_mtime,
        ),
    )[-1]


def export_bridge_quick_layer(
    candidates: list[SessionCandidate],
    output_dir: Path,
    profile_path: Path,
) -> Path | None:
    source = choose_bridge_layer_source(candidates)
    if source is None or not profile_path.exists():
        return None
    output = output_dir / "bridge_quick_layer" / "quick_calibration_layer.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "export_bridge_quick_calibration_layer.py"),
        "--pairs",
        str(source.pairs_path),
        "--profile",
        str(profile_path),
        "--output",
        str(output),
    ]
    print("+", " ".join(command))
    result = subprocess.run(command, cwd=str(WORKSPACE_ROOT), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"Bridge quick layer export skipped: {result.stderr.strip() or result.returncode}")
        return None
    return output


def bridge_layer_summary(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.load(path.open("r", encoding="utf-8"))
        sessions = payload.get("sessions", {})
        if not isinstance(sessions, dict) or not sessions:
            return None
        session, report = next(iter(sessions.items()))
        heldout = report.get("heldout", {}) if isinstance(report, dict) else {}
        baseline = heldout.get("baseline_bridge_output", {})
        corrected = heldout.get("corrected_bridge_output", {})
        base_mean = baseline.get("mean_l2")
        corrected_mean = corrected.get("mean_l2")
        count = corrected.get("count")
        recommendation = report.get("runtime_recommendation", {}) if isinstance(report, dict) else {}
        enabled = recommendation.get("enable") if isinstance(recommendation, dict) else None
        reason = recommendation.get("reason") if isinstance(recommendation, dict) else None
        if not isinstance(base_mean, (int, float)) or not isinstance(corrected_mean, (int, float)):
            return f"session={session}, heldout metrics unavailable"
        delta = base_mean - corrected_mean
        suffix = f", recommended={enabled}"
        if reason:
            suffix += f", reason={reason}"
        return f"session={session}, heldout_count={count}, heldout_mean_l2 {base_mean:.4f} -> {corrected_mean:.4f} (delta {delta:+.4f}){suffix}"
    except Exception as exc:
        return f"summary unavailable: {exc}"


def write_recording_request(
    candidates: list[SessionCandidate],
    output: Path,
    min_stage_samples: int,
    bridge_layer_path: Path | None = None,
) -> None:
    complete_quality = strict_complete_quality(candidates)
    image_complete = [item for item in candidates if not item.excluded and item.complete]
    excluded = [item for item in candidates if item.excluded]
    good_incomplete = [
        item for item in candidates
        if not item.excluded and not item.complete and item.metrics_quality in {"fair", "good"}
    ]
    complete_poor = [
        item for item in candidates
        if not item.excluded and item.complete and item.metrics_quality == "poor"
    ]
    incomplete_with_poor_labels = [
        item for item in candidates
        if not item.excluded and not item.complete and item.poor_labels
    ]
    latest = sorted(
        candidates,
        key=lambda item: item.pairs_path.parent.stat().st_mtime,
        reverse=True,
    )[:8]
    weak = weak_stage_summary(candidates)
    weak_text = ", ".join(f"{stage}({score})" for stage, score in weak[:4]) or "none"

    lines = [
        "# Recording Request",
        "",
        "The automatic pipeline searched the workspace and Documents calibration store, but did not find enough usable 9-point sessions for independent session validation.",
        "",
        "## Current Evidence",
        "",
        f"- Complete 9-point sessions with fair/good metrics: {len(complete_quality)}",
        f"- Complete 9-point sessions usable for image training: {len(image_complete)}",
        f"- Complete 9-point sessions with poor metrics: {len(complete_poor)}",
        f"- Incomplete sessions with rejected/poor labels: {len(incomplete_with_poor_labels)}",
        f"- Excluded mistaken/non-eye sessions: {len(excluded)}",
        "- Required for train/validation: at least 2",
        f"- Minimum usable samples per stage: {min_stage_samples}",
        f"- Weak stages to watch: {weak_text}",
        "",
    ]
    if complete_quality:
        lines.append("Usable complete sessions found:")
        for item in complete_quality:
            lines.append(f"- {item.session}: quality={item.metrics_quality}, avg_residual={item.metrics_average_residual}")
        lines.append("")
    if bridge_layer_path is not None:
        lines.append("Experimental Bridge quick layer:")
        lines.append(f"- {bridge_layer_path}")
        summary = bridge_layer_summary(bridge_layer_path)
        if summary:
            lines.append(f"- {summary}")
        lines.append("- This artifact uses `raw_source=bridge_output_xy`; it is for Bridge runtime experiments, not proof that V1 passed validation.")
        lines.append("")
    if good_incomplete:
        lines.append("Good/fair but incomplete sessions:")
        for item in good_incomplete[:12]:
            lines.append(f"- {item.session}: missing={', '.join(item.missing)}, quality={item.metrics_quality}")
        lines.append("")
    if complete_poor:
        lines.append("Complete but poor sessions:")
        for item in complete_poor[:12]:
            high = f", high_residual={', '.join(item.high_residual_stages)}" if item.high_residual_stages else ""
            lines.append(f"- {item.session}: quality=poor, avg_residual={item.metrics_average_residual}{high}")
        lines.append("")
    if excluded:
        lines.append("Excluded sessions:")
        for item in excluded[:12]:
            lines.append(f"- {item.session}: reason={item.exclude_reason or 'session excluded'}")
        lines.append("")
    if latest:
        lines.append("Latest session diagnostics:")
        for item in latest:
            missing = ", ".join(item.missing) if item.missing else "none"
            poor = ", ".join(item.poor_labels) if item.poor_labels else "none"
            high = ", ".join(item.high_residual_stages) if item.high_residual_stages else "none"
            excluded_text = f", excluded={item.exclude_reason}" if item.excluded else ""
            lines.append(
                f"- {item.session}: quality={item.metrics_quality}, missing={missing}, poor_labels={poor}, high_residual={high}{excluded_text}"
            )
        lines.append("")

    lines.extend(
        [
            "## Record Next",
            "",
            "Record one new clean 9-point session and let the app save it under `Documents\\DreamAirTracking\\calibration_data`.",
            f"Pay extra attention to weak stages: {weak_text}.",
            "",
            "### Weak Stage Fix Plan",
            "",
            "| Stage | Evidence | Likely cause | Next improvement |",
            "|---|---|---|---|",
            *weak_stage_fix_lines(candidates, weak),
            "",
            "Required stages:",
            "",
            "- center",
            "- up",
            "- down",
            "- left",
            "- right",
            "- left_up",
            "- right_up",
            "- left_down",
            "- right_down",
            "",
            "Acceptance rule:",
            "",
            "- If a point enters manual review, accept only when the eye image and direction look correct; otherwise retry.",
            "- If any point is marked poor, retry that point before finishing.",
            "- Keep the eye open for down/left_down/right_down; do not squeeze the eyelids.",
            "- The new session should produce `metrics.json` with `quality` fair or good.",
            "",
            "After recording, rerun:",
            "",
            "```powershell",
            "python scripts\\ml\\auto_gaze_pipeline.py --output-dir runs\\gaze_auto_v1_9point_after_recording --require-metrics-quality fair",
            "```",
            "",
            "For a quick local audit without training:",
            "",
            "```powershell",
            "python scripts\\ml\\audit_gaze_sessions.py --latest 5",
            "```",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def weak_stage_summary(candidates: list[SessionCandidate]) -> list[tuple[str, int]]:
    scores: Counter[str] = Counter()
    for item in candidates:
        if item.excluded or item.total_rows == 0:
            continue
        for stage in item.missing:
            scores[stage] += 1
        for stage in item.poor_labels:
            scores[stage] += 1
        for stage in item.high_residual_stages:
            scores[stage] += 1
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], REQUIRED_STAGE_ORDER.index(item[0]) if item[0] in REQUIRED_STAGES else 999),
    )


def weak_stage_fix_lines(candidates: list[SessionCandidate], weak: list[tuple[str, int]]) -> list[str]:
    if not weak:
        return ["| none | none | no repeated weak stage | no targeted recording change needed |"]
    lines: list[str] = []
    for stage, score in weak[:6]:
        evidence = stage_evidence(candidates, stage)
        lines.append(
            f"| {stage} | {format_stage_evidence(evidence, score)} | "
            f"{likely_stage_cause(stage, evidence)} | {stage_improvement_plan(stage, evidence)} |"
        )
    return lines


def stage_evidence(candidates: list[SessionCandidate], stage: str) -> Counter[str]:
    evidence: Counter[str] = Counter()
    for item in candidates:
        if item.excluded or item.total_rows == 0:
            continue
        if stage in item.missing:
            evidence["missing_or_low_accepted"] += 1
        if stage in item.poor_labels:
            evidence["poor_or_rejected_label"] += 1
        if stage in item.high_residual_stages:
            evidence["high_residual"] += 1
        for cause, count in (item.note_causes.get(stage) or {}).items():
            evidence[str(cause)] += int(count)
    return evidence


def format_stage_evidence(evidence: Counter[str], score: int) -> str:
    parts = [f"score={score}"]
    for key in (
        "missing_or_low_accepted",
        "poor_or_rejected_label",
        "high_residual",
        "low_openness",
        "not_found",
        "low_confidence",
        "sync_late",
    ):
        if evidence.get(key, 0):
            parts.append(f"{key}={evidence[key]}")
    return ", ".join(parts)


def likely_stage_cause(stage: str, evidence: Counter[str]) -> str:
    note_causes = [
        ("low_openness", "eyelid openness / lower-lid occlusion"),
        ("not_found", "pupil not found or outside the eye crop"),
        ("low_confidence", "glare, reflection, or unclear pupil edge"),
        ("sync_late", "left/right frame synchronization"),
    ]
    for key, label in sorted(note_causes, key=lambda item: evidence.get(item[0], 0), reverse=True):
        if evidence.get(key, 0) > 0:
            return label
    if evidence.get("high_residual", 0) >= evidence.get("missing_or_low_accepted", 0):
        return "geometry/profile mismatch for this gaze direction"
    if "down" in stage:
        return "lower-target capture is fragile, likely eyelid or headset-height related"
    if "left" in stage or "right" in stage:
        return "edge target capture is fragile, likely headset alignment or eye-crop related"
    return "not enough accepted stable samples"


def stage_improvement_plan(stage: str, evidence: Counter[str]) -> str:
    plans: list[str] = []
    if evidence.get("low_openness", 0) or "down" in stage:
        plans.append("keep both eyes open and adjust headset height so the lower eyelid does not cover the pupil")
    if evidence.get("not_found", 0) or "left" in stage or "right" in stage:
        plans.append("check horizontal headset alignment and make sure the pupil stays inside both eye crops")
    if evidence.get("low_confidence", 0):
        plans.append("reduce glare/reflection and hold gaze steady until capture finishes")
    if evidence.get("sync_late", 0):
        plans.append("restart BrokenEye/Bridge or reduce capture load before retrying")
    if evidence.get("high_residual", 0):
        plans.append("keep this stage out of automatic profile decisions until another reseated full session confirms it")
    if not plans:
        plans.append("record a reseated full 9-point session and accept this point only when the eye image direction is visibly correct")
    return "; ".join(dict.fromkeys(plans))


def write_discovery(candidates: list[SessionCandidate], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "session": item.session,
            "pairs_path": str(item.pairs_path),
            "total_rows": item.total_rows,
            "complete": item.complete,
            "missing": list(item.missing),
            "metrics_quality": item.metrics_quality,
            "metrics_average_residual": item.metrics_average_residual,
            "poor_labels": list(item.poor_labels),
            "high_residual_stages": list(item.high_residual_stages),
            "note_causes": item.note_causes,
            "excluded": item.excluded,
            "exclude_reason": item.exclude_reason,
            "raw_counts": item.raw_counts,
            "counts": item.counts,
        }
        for item in candidates
    ]
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, help="Search root; defaults to workspace and Documents calibration_data.")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "runs" / "gaze_auto_v1_9point")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", choices=("cnn", "hybrid"), default="hybrid")
    parser.add_argument("--min-stage-samples", type=int, default=20)
    parser.add_argument("--min-openness", type=float, default=0.08)
    parser.add_argument("--max-train-sessions", type=int, default=8)
    parser.add_argument("--val-session", help="Force a discovered session name as validation.")
    parser.add_argument(
        "--image-session-quality",
        choices=("none", "fair", "good"),
        default="none",
        help="Metrics quality filter for image training sessions. Default accepts complete sessions even when affine/profile metrics are poor.",
    )
    parser.add_argument(
        "--require-metrics-quality",
        choices=("none", "fair", "good"),
        default="fair",
        help="Legacy/profile quality hint; no longer blocks image training.",
    )
    parser.add_argument("--profile", type=Path, default=WORKSPACE_ROOT / "tracking_profile_final.json")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    roots = [path.resolve() for path in args.root] if args.root else default_roots()
    candidates = discover_sessions(roots, args.min_stage_samples, args.min_openness)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_discovery(candidates, args.output_dir / "session_discovery.json")
    bridge_layer_path = export_bridge_quick_layer(candidates, args.output_dir, args.profile.resolve())

    image_allowed_quality = {
        "none": {"missing", "unknown", "poor", "fair", "good"},
        "fair": {"fair", "good"},
        "good": {"good"},
    }[args.image_session_quality]
    complete = image_training_sessions(candidates, image_allowed_quality)
    if len(complete) < 2:
        print(f"Found {len(complete)} complete 9-point image-training sessions. Need at least 2 for session validation.")
        print(f"Discovery written to {args.output_dir / 'session_discovery.json'}")
        write_recording_request(candidates, args.output_dir / "recording_request.md", args.min_stage_samples, bridge_layer_path)
        print(f"Recording request written to {args.output_dir / 'recording_request.md'}")
        return 2

    if args.val_session:
        matches = [item for item in complete if item.session == args.val_session]
        if not matches:
            raise SystemExit(f"--val-session {args.val_session} was not found among complete sessions.")
        val = matches[-1]
    else:
        val = complete[-1]

    train_pool = [item for item in complete if item.pairs_path != val.pairs_path]
    train_pool = sorted(
        train_pool,
        key=lambda item: (
            {"good": 2, "fair": 1}.get(item.metrics_quality, 0),
            -(item.metrics_average_residual if item.metrics_average_residual is not None else 999),
            item.pairs_path.parent.stat().st_mtime,
        ),
    )
    train = train_pool[-args.max_train_sessions :]
    selected = train + [val]

    selection = {
        "train_sessions": [item.session for item in train],
        "val_session": val.session,
        "selected_pairs": [str(item.pairs_path) for item in selected],
        "bridge_quick_layer": str(bridge_layer_path) if bridge_layer_path is not None else None,
        "image_session_quality": args.image_session_quality,
        "note": "Image training sessions are gated by complete accepted image labels; affine/profile metrics are recorded but do not block training.",
    }
    with (args.output_dir / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, ensure_ascii=False)

    manifest = args.output_dir / "manifest.csv"
    summary = args.output_dir / "manifest.summary.json"
    prepare = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_gaze_dataset.py"),
        "--output",
        str(manifest),
        "--summary",
        str(summary),
        "--val-session",
        val.session,
        "--min-openness",
        str(args.min_openness),
    ]
    for item in selected:
        prepare.extend(["--pairs", str(item.pairs_path)])
    run_command(prepare, WORKSPACE_ROOT)

    train_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_gaze_baseline.py"),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--model",
        args.model,
    ]
    if args.cpu:
        train_cmd.append("--cpu")
    run_command(train_cmd, WORKSPACE_ROOT)

    eval_dir = args.output_dir / "eval"
    eval_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_gaze_model.py"),
        "--manifest",
        str(manifest),
        "--checkpoint",
        str(args.output_dir / "gaze_baseline.pt"),
        "--output-dir",
        str(eval_dir),
        "--workspace-root",
        str(WORKSPACE_ROOT),
    ]
    if args.cpu:
        eval_cmd.append("--cpu")
    run_command(eval_cmd, WORKSPACE_ROOT)

    print(f"Auto pipeline complete: {args.output_dir}")
    print(f"Review: {eval_dir / 'auto_review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
