#!/usr/bin/env python3
"""Audit recent gaze calibration sessions before launching training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REQUIRED_STAGES = (
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


@dataclass(frozen=True)
class StageLabel:
    stage: str
    start: int
    end: int
    accepted: bool
    verdict: str
    notes: str


def default_root() -> Path:
    return Path.home() / "Documents" / "DreamAirTracking" / "calibration_data"


def load_labels(session_dir: Path) -> list[StageLabel]:
    path = session_dir / "labels.jsonl"
    if not path.exists():
        return []
    labels: list[StageLabel] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            labels.append(
                StageLabel(
                    stage=str(item.get("stageId", "")),
                    start=int(item.get("frameStart", 0)),
                    end=int(item.get("frameEnd", -1)),
                    accepted=bool(item.get("accepted", False)),
                    verdict=str(item.get("operatorVerdict", "")),
                    notes=str(item.get("notes", "")),
                )
            )
    return labels


def stage_counts(pairs_path: Path, labels: list[StageLabel]) -> tuple[Counter[str], Counter[str]]:
    raw: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    ranges = [item for item in labels if item.accepted]
    with pairs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stage = (row.get("stage") or row.get("Stage") or row.get("StageId") or "").strip()
            if not stage:
                continue
            raw[stage] += 1
            sequence = int(float(row.get("sequence") or row.get("Sequence") or row.get("pair") or "0"))
            if any(item.stage == stage and item.start <= sequence <= item.end for item in ranges):
                accepted[stage] += 1
    return raw, accepted


def load_metrics(session_dir: Path) -> dict[str, object]:
    path = session_dir / "metrics.json"
    if not path.exists():
        return {"quality": "missing", "stageResiduals": {}}
    return json.load(path.open("r", encoding="utf-8"))


def load_session_exclude(session_dir: Path) -> dict[str, object] | None:
    path = session_dir / "session_exclude.json"
    if not path.exists():
        return None
    try:
        data = json.load(path.open("r", encoding="utf-8"))
    except json.JSONDecodeError:
        return {"exclude": True, "reason": "invalid session_exclude.json"}
    return data if bool(data.get("exclude", True)) else None


def discover_sessions(root: Path) -> list[Path]:
    if not root.exists():
        return []
    sessions = [path for path in root.iterdir() if path.is_dir()]
    for pairs_path in root.rglob("pairs.csv"):
        if pairs_path.parent.name == "frames":
            continue
        sessions.append(pairs_path.parent)
    return sorted(set(sessions), key=lambda path: path.stat().st_mtime, reverse=True)


def audit_session(session_dir: Path, min_stage_samples: int) -> dict[str, object]:
    pairs_path = session_dir / "pairs.csv"
    labels = load_labels(session_dir)
    raw, accepted = (Counter(), Counter())
    if pairs_path.exists():
        raw, accepted = stage_counts(pairs_path, labels)
    metrics = load_metrics(session_dir)
    exclude = load_session_exclude(session_dir)
    residuals = metrics.get("stageResiduals") or {}
    poor_labels = [item.stage for item in labels if not item.accepted or item.verdict == "poor"]
    missing = [stage for stage in REQUIRED_STAGES if accepted.get(stage, 0) < min_stage_samples]
    high_residual = [
        stage
        for stage, value in residuals.items()
        if isinstance(value, dict) and float(value.get("distance") or 0) > 0.5
    ]
    note_causes = {stage: Counter() for stage in REQUIRED_STAGES}
    for label in labels:
        if label.stage in note_causes:
            note_causes[label.stage].update(extract_note_causes(label.notes))
    quality = str(metrics.get("quality", "missing"))
    report = {
        "session": session_dir.name,
        "path": str(session_dir),
        "quality": quality,
        "raw_counts": {stage: raw.get(stage, 0) for stage in REQUIRED_STAGES},
        "accepted_counts": {stage: accepted.get(stage, 0) for stage in REQUIRED_STAGES},
        "missing": missing,
        "poor_labels": poor_labels,
        "high_residual": high_residual,
        "note_causes": {stage: dict(note_causes[stage]) for stage in REQUIRED_STAGES},
        "excluded": exclude is not None,
        "exclude_reason": "" if exclude is None else str(exclude.get("reason", "session excluded")),
    }
    category, decision = classify_report(report)
    report["category"] = category
    report["decision"] = decision
    return report


def classify_report(report: dict[str, object]) -> tuple[str, str]:
    quality = str(report["quality"]).lower()
    missing = list(report["missing"])
    poor_labels = list(report["poor_labels"])
    high_residual = list(report["high_residual"])
    accepted_counts = dict(report["accepted_counts"])
    total_accepted = sum(int(value) for value in accepted_counts.values())

    if bool(report.get("excluded", False)):
        return "excluded", f"Excluded from automatic use: {report.get('exclude_reason') or 'session excluded'}."
    if total_accepted == 0:
        return "empty_or_unlabeled", "No accepted calibration ranges; cannot use for calibration or training."
    if not missing and quality in {"fair", "good"} and not poor_labels and not high_residual:
        return "strict_usable", "Usable as a fair held-out session or Bridge-layer source."
    if not missing and (quality == "poor" or high_residual):
        return "complete_but_unstable", "Complete 9-point coverage, but residuals/quality are too weak for automatic use."
    if missing and quality in {"fair", "good"}:
        return "partial_candidate", "Some stages look usable, but missing accepted stages make it incomplete."
    if missing:
        return "incomplete", "Missing accepted stages; use only for debugging weak directions."
    return "review_only", "Keep as review evidence; do not use for automatic model/profile decisions."


def print_text(reports: list[dict[str, object]]) -> None:
    for report in reports:
        print(f"\n== {report['session']} ==")
        print(f"path: {report['path']}")
        print(f"quality: {report['quality']}")
        print(f"category: {report['category']}")
        print(f"decision: {report['decision']}")
        if report.get("excluded"):
            print(f"exclude reason: {report['exclude_reason']}")
        print("accepted counts:")
        counts = report["accepted_counts"]
        for stage in REQUIRED_STAGES:
            print(f"  {stage:10s} {counts[stage]}")
        print(f"missing/low accepted: {', '.join(report['missing']) or 'none'}")
        print(f"poor labels: {', '.join(report['poor_labels']) or 'none'}")
        print(f"high residual > 0.5: {', '.join(report['high_residual']) or 'none'}")


def extract_note_causes(notes: str) -> Counter[str]:
    result: Counter[str] = Counter()
    if not notes:
        return result
    lowered = notes.lower()
    numeric_fields = {
        "low_openness": "low_openness",
        "low_confidence": "low_confidence",
        "not_found": "not_found",
        "sync_late": "sync_late",
    }
    parts = [part.strip() for part in lowered.replace(",", ";").split(";")]
    for part in parts:
        if "=" not in part:
            continue
        key, value = [item.strip() for item in part.split("=", 1)]
        if key not in numeric_fields:
            continue
        try:
            count = int(float(value.split("/")[0]))
        except ValueError:
            continue
        if count > 0:
            result[numeric_fields[key]] += count

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


def render_markdown(reports: list[dict[str, object]]) -> str:
    counts = Counter(str(report["category"]) for report in reports)
    strict = [report for report in reports if report["category"] == "strict_usable"]
    excluded = [report for report in reports if report["category"] == "excluded"]
    weak = weak_stage_summary(reports)
    weak_text = ", ".join(f"{stage}({score})" for stage, score in weak[:4]) or "none"
    need_more = max(0, 2 - len(strict))
    next_plan = (
        "No extra recording is currently requested by the audit."
        if need_more == 0 and not weak
        else f"Record {need_more} new full 9-point session(s) after reseating the headset."
        if need_more > 0
        else "Run another full 9-point session only if live tracking still looks unstable."
    )
    if weak:
        next_plan += f" Pay extra attention to weak stages: {weak_text}."
    lines = [
        "# Gaze Calibration Session Audit",
        "",
        f"- Total sessions audited: {len(reports)}",
        f"- Strict usable sessions: {counts.get('strict_usable', 0)}",
        f"- Excluded mistaken/non-eye sessions: {counts.get('excluded', 0)}",
        f"- Partial candidates: {counts.get('partial_candidate', 0)}",
        f"- Complete but unstable: {counts.get('complete_but_unstable', 0)}",
        f"- Incomplete sessions: {counts.get('incomplete', 0)}",
        f"- Empty/unlabeled sessions: {counts.get('empty_or_unlabeled', 0)}",
        f"- Review-only sessions: {counts.get('review_only', 0)}",
        f"- Weak stages: {weak_text}",
        "",
        "## Bottom Line",
        "",
    ]
    if strict:
        lines.append(
            "The old records are not all useless, but only these sessions pass the strict automatic-use gate:"
        )
        for report in strict:
            lines.append(f"- `{report['session']}`: quality={report['quality']}, path={report['path']}")
    else:
        lines.append("No session currently passes the strict automatic-use gate.")
    if excluded:
        lines.append("")
        lines.append("Excluded mistaken/non-eye sessions:")
        for report in excluded:
            lines.append(f"- `{report['session']}`: {report.get('exclude_reason') or 'session excluded'}, path={report['path']}")
    lines.extend(
        [
            "",
            "A session is strict usable only when all 9 gaze stages have enough accepted samples, metrics are fair/good, no accepted label is poor/bad, and no stage residual exceeds 0.5.",
            "",
            f"Next recording plan: {next_plan}",
            "",
            "## Fairness / Bias Check",
            "",
            *fairness_lines(reports, weak),
            "",
            "### Weak Stage Root Causes",
            "",
            "| Stage | Evidence | Likely cause | Next improvement |",
            "|---|---|---|---|",
            *weak_stage_root_cause_lines(reports, weak),
            "",
            "## Sessions",
            "",
            "| Session | Category | Quality | Missing/low accepted | Poor labels | High residual | Decision |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for report in reports:
        missing = ", ".join(report["missing"]) or "none"
        poor = ", ".join(report["poor_labels"]) or "none"
        high = ", ".join(report["high_residual"]) or "none"
        decision = str(report["decision"]).replace("|", "/")
        lines.append(
            f"| `{report['session']}` | {report['category']} | {report['quality']} | {missing} | {poor} | {high} | {decision} |"
        )
    lines.append("")
    return "\n".join(lines)


def fairness_lines(reports: list[dict[str, object]], weak: list[tuple[str, int]]) -> list[str]:
    reviewed = [report for report in reports if report["category"] not in {"empty_or_unlabeled", "excluded"}]
    if not reviewed:
        return [
            "- Status: not enough reviewed calibration data to check for stage-specific bias.",
            "- Action: record a full 9-point session before trusting fairness conclusions.",
        ]

    if not weak:
        return [
            "- Status: no repeated stage-specific failure pattern detected by the current audit.",
            "- Action: keep session-based validation; do not loosen automatic gates unless live tracking still looks unstable.",
        ]

    top = weak[:4]
    top_text = ", ".join(f"{stage}={score}" for stage, score in top)
    median_like = sorted(score for _, score in weak)[len(weak) // 2]
    top_score = top[0][1]
    pattern_detected = top_score >= 3 and top_score >= max(2, median_like)
    if pattern_detected:
        return [
            f"- Status: stage-specific failure pattern detected across {len(reviewed)} reviewed sessions.",
            f"- Evidence: strongest weak stages are {top_text}.",
            "- Interpretation: this is a calibration-gate bias risk, not proof that the user looked at the wrong point.",
            "- Action: keep these stages out of automatic training/profile decisions, but use them to guide the next recording and manual review.",
            "- Mitigation: reseat the headset and record one full 9-point session; for the flagged edge/down/lateral points, retry only when the eye image is visibly wrong or low-quality.",
        ]

    return [
        f"- Status: weak stages exist, but the pattern is not strong enough to call systematic bias yet ({top_text}).",
        "- Action: keep collecting session-based evidence before changing calibration thresholds.",
    ]


def weak_stage_summary(reports: list[dict[str, object]]) -> list[tuple[str, int]]:
    scores: Counter[str] = Counter()
    for report in reports:
        if report["category"] in {"empty_or_unlabeled", "excluded"}:
            continue
        for stage in report["missing"]:
            scores[str(stage)] += 1
        for stage in report["poor_labels"]:
            scores[str(stage)] += 1
        for stage in report["high_residual"]:
            scores[str(stage)] += 1
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], REQUIRED_STAGES.index(item[0]) if item[0] in REQUIRED_STAGES else 999),
    )


def weak_stage_root_cause_lines(reports: list[dict[str, object]], weak: list[tuple[str, int]]) -> list[str]:
    if not weak:
        return ["| none | none | no repeated weak stage | no targeted recording change needed |"]

    lines: list[str] = []
    for stage, score in weak[:6]:
        evidence = stage_evidence(reports, stage)
        cause = likely_stage_cause(stage, evidence)
        plan = stage_improvement_plan(stage, evidence)
        lines.append(f"| {stage} | {format_evidence(evidence, score)} | {cause} | {plan} |")
    return lines


def stage_evidence(reports: list[dict[str, object]], stage: str) -> Counter[str]:
    evidence: Counter[str] = Counter()
    for report in reports:
        if report["category"] in {"empty_or_unlabeled", "excluded"}:
            continue
        if stage in report["missing"]:
            evidence["missing_or_low_accepted"] += 1
        if stage in report["poor_labels"]:
            evidence["poor_or_rejected_label"] += 1
        if stage in report["high_residual"]:
            evidence["high_residual"] += 1
        note_causes = dict(report.get("note_causes") or {})
        for cause, count in dict(note_causes.get(stage) or {}).items():
            evidence[str(cause)] += int(count)
    return evidence


def format_evidence(evidence: Counter[str], score: int) -> str:
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
        value = evidence.get(key, 0)
        if value:
            parts.append(f"{key}={value}")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--latest", type=int, default=5, help="number of newest sessions to audit; use 0 for all")
    parser.add_argument("--min-stage-samples", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", type=Path, help="write a markdown audit report")
    args = parser.parse_args()

    sessions = discover_sessions(args.root)
    if args.latest > 0:
        sessions = sessions[: args.latest]
    reports = [audit_session(path, args.min_stage_samples) for path in sessions]
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(reports), encoding="utf-8")
    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
    else:
        print_text(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
