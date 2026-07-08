#!/usr/bin/env python3
"""Head-specific-mask training gate for eye manifests (E2).

Turns the head-mask discipline in docs/MODEL_ADAPTATION_DATA_AUDIT_zh.md into a
hard gate at the train_eye_multitask.py entry point, replacing the previous
*silent* fallback-to-weak-label behavior (train_eye_multitask.EyeMultitaskDataset
would supervise openness/pupil from weak quality when valid columns were absent).

Rules enforced (default = FAIL on any violation):
  1. Head-mask columns present: openness_valid_{left,right} and pupil_valid_{left,right}
     must exist. Missing columns are exactly what triggered the silent weak-label
     fallback -> reject unless --allow-weak-label-fallback.
  2. Gaze rows must not double-supervise other heads: a row with strict gaze weight > 0
     must have openness_valid == 0 and pupil_valid == 0 (gaze rows train gaze only).
  3. No split leakage: the same image pair must not appear in multiple splits, and a
     real (non-synthetic) session must not span multiple splits (random-row split).

Standalone:  python manifest_head_mask_gate.py --manifest x.csv [--json]   (exit 0 pass / 2 fail)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_eye_manifest_series import norm, pair_key, read_rows, strict_gaze_weight  # noqa: E402

HEAD_MASK_COLUMNS = (
    "openness_valid_left",
    "openness_valid_right",
    "pupil_valid_left",
    "pupil_valid_right",
)


def _valid_true(row: dict[str, str], key: str) -> bool:
    value = row.get(key, "")
    if value == "":
        return False
    try:
        return float(value) > 0.0
    except ValueError:
        return False


def gate_manifest(
    path: Path | str,
    *,
    allow_weak_fallback: bool = False,
    allow_gaze_head_cosupervision: bool = False,
) -> dict[str, object]:
    rows = read_rows(Path(path))
    header = set(rows[0].keys()) if rows else set()
    present = {col: (col in header) for col in HEAD_MASK_COLUMNS}
    missing_cols = [col for col, ok in present.items() if not ok]

    violations: list[str] = []
    warnings: list[str] = []

    # Rule 1: head-mask columns present (else the silent weak-label fallback would fire).
    if missing_cols and not allow_weak_fallback:
        violations.append(
            f"missing head-mask columns {missing_cols}: rows would silently fall back to weak-label "
            "supervision. Add head masks, or pass --allow-weak-label-fallback to opt in."
        )
    elif missing_cols and allow_weak_fallback:
        warnings.append(
            f"weak-label fallback ALLOWED for {missing_cols}; openness/pupil supervised by weak quality."
        )

    # Rule 2: gaze rows must not double-supervise openness/pupil.
    gaze_rows = 0
    gaze_head_conflicts = 0
    for row in rows:
        if strict_gaze_weight(row) > 0.0:
            gaze_rows += 1
            if (
                _valid_true(row, "openness_valid_left")
                or _valid_true(row, "openness_valid_right")
                or _valid_true(row, "pupil_valid_left")
                or _valid_true(row, "pupil_valid_right")
            ):
                gaze_head_conflicts += 1
    if gaze_head_conflicts > 0:
        message = (
            f"{gaze_head_conflicts}/{gaze_rows} gaze rows also set openness/pupil valid=1 "
            "(head-mask violation: gaze rows must have openness_valid=0 and pupil_valid=0)."
        )
        (warnings if allow_gaze_head_cosupervision else violations).append(message)

    # Rule 3: split leakage (same pair across splits, or a real session spanning splits).
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[pair_key(row)].append(row)
    pair_split_conflicts = [key for key, grp in groups.items() if len({r.get("split", "") for r in grp}) > 1]
    if pair_split_conflicts:
        violations.append(
            f"{len(pair_split_conflicts)} image pairs appear in multiple splits (train/val leakage)."
        )

    session_splits: dict[str, Counter] = defaultdict(Counter)
    session_sources: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        session = row.get("session_id") or row.get("session") or row.get("wear_id", "")
        session_splits[session][row.get("split", "")] += 1
        session_sources[session][norm(row.get("source")) or "-"] += 1
    real_multi = [
        session
        for session, counter in session_splits.items()
        if session
        and sum(1 for value in counter.values() if value > 0) > 1
        and (set(session_sources.get(session, {})) - {"synthetic_eyelid_mix"})
    ]
    if real_multi:
        violations.append(
            f"{len(real_multi)} real sessions span multiple splits (looks like a random-row split; "
            "split by session/wear/subject instead)."
        )

    return {
        "manifest": str(Path(path).resolve()),
        "rows": len(rows),
        "unique_pairs": len(groups),
        "head_mask_columns_present": present,
        "gaze_rows": gaze_rows,
        "gaze_head_conflicts": gaze_head_conflicts,
        "pair_split_conflicts": len(pair_split_conflicts),
        "real_sessions_multi_split": len(real_multi),
        "violations": violations,
        "warnings": warnings,
        "passed": not violations,
    }


def enforce(
    path: Path | str,
    *,
    allow_weak_fallback: bool = False,
    allow_gaze_head_cosupervision: bool = False,
    skip: bool = False,
) -> dict[str, object]:
    """Call from a trainer; prints a summary and raises SystemExit(2) on gate failure."""
    if skip:
        print(f"[manifest-gate] SKIPPED for {path}")
        return {"passed": True, "skipped": True}
    report = gate_manifest(
        path,
        allow_weak_fallback=allow_weak_fallback,
        allow_gaze_head_cosupervision=allow_gaze_head_cosupervision,
    )
    for warning in report["warnings"]:
        print(f"[manifest-gate] WARNING: {warning}")
    if not report["passed"]:
        print(f"[manifest-gate] FAILED for {report['manifest']}:")
        for violation in report["violations"]:
            print(f"  - {violation}")
        print("  Fix the manifest, or re-run with --allow-weak-label-fallback / --skip-manifest-audit to override.")
        raise SystemExit(2)
    print(
        f"[manifest-gate] PASSED: {report['rows']} rows, {report['unique_pairs']} unique pairs, "
        f"{report['gaze_rows']} gaze rows, 0 head-mask conflicts."
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--allow-weak-label-fallback", action="store_true")
    parser.add_argument(
        "--allow-gaze-head-cosupervision",
        action="store_true",
        help="Demote the gaze-row openness/pupil double-supervision check to a warning.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    args = parser.parse_args()

    report = gate_manifest(
        args.manifest,
        allow_weak_fallback=args.allow_weak_label_fallback,
        allow_gaze_head_cosupervision=args.allow_gaze_head_cosupervision,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        status = "PASSED" if report["passed"] else "FAILED"
        print(f"[manifest-gate] {status}: {report['manifest']}")
        for violation in report["violations"]:
            print(f"  VIOLATION: {violation}")
        for warning in report["warnings"]:
            print(f"  warning: {warning}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
