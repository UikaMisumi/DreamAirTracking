"""Build manifest v7: continuous-openness ramp data + fresh gaze sessions on top of v6_clean.

Motivation (verified on-headset + in capture CSVs):
- v6's openness labels were a 4-level staircase {0, 0.5, 0.7, 1} (constants per protocol
  stage) -> the model quantized mid states to attractors; half-open could not hover.
- The wide head fires when looking up, and openness collapses when looking down: the
  gaze<->lid appearance entanglement was never negatively supervised.

What v7 does:
1. Base v6_clean rows pass through, but MID-BAND constant openness supervision is retired:
   any side whose openness target lies strictly inside (0.05, 0.95) gets openness_valid=0.
   Anchors (0 / 1) stay. Mid supervision now comes only from ramp rows (real continuous labels).
2. Ramp sessions (in-app capture, eyelid_ramp_labels.csv): per-frame TIME-progress openness
   labels across the whole [0,1] range; squint frames don't supervise openness; wide/squint
   head labels preserved.
3. Fresh gaze sessions (app quick-cal, pairs.csv): new gaze rows at current wear positions
   (5-point, +-0.7 convention), openness NOT supervised in the manifest (the P1-7 train-time
   flag hard-labels them open=1 -> also treats look-down false-close), and — new in v7 —
   wide=0 / squint=0 WITH valid masks on every gaze row: the user is neutral during gaze
   calibration, so this negatively supervises the wide/squint heads across all gaze
   directions (the model-side fix for look-up falsely triggering EyeWide).

Head-mask discipline (E2): gaze rows carry openness_valid=0 / pupil_valid=0 as required.

Usage:
  python scripts/ml/build_manifest_v7.py --output runs/eye_manifest_v7/manifest_v7.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

BASE_MANIFEST = r"C:\Users\Zhong\Desktop\dream air tracking\runs\eye_manifest_v6_clean_headmask\manifest_v6_clean.csv"
RAMP_GLOB = r"C:\Users\Zhong\Desktop\SumiruiTracking\DreamAirTracking\runs\fullparam_capture\eyelid_ramp_app_*"
GAZE_GLOB = r"C:\Users\Zhong\Documents\DreamAirTracking\calibration_data\2026*"

GAZE_STAGE_TARGETS = {
    "center": (0.0, 0.0),
    "up": (0.0, 0.7),
    "down": (0.0, -0.7),
    "left": (-0.7, 0.0),
    "right": (0.7, 0.0),
}
# pairs.csv "conf" is the legacy classical detector's confidence with a session-dependent scale
# (July sessions sit at ~0.05-0.15) — unusable as an absolute gate; found is the real filter.
MIN_GAZE_CONF = 0.0
MID_LO, MID_HI = 0.05, 0.95


def fmt(v: float) -> str:
    return f"{v:.6f}"


def blank_row(columns: list[str]) -> dict[str, str]:
    return {c: "" for c in columns}


def retire_mid_constants(row: dict[str, str]) -> int:
    """Zero openness_valid for sides whose target is a mid-band constant. Returns #retired."""
    retired = 0
    for side in ("left", "right"):
        valid_col = f"openness_valid_{side}"
        try:
            if float(row.get(valid_col) or 0) <= 0.5:
                continue
            target = float(row.get(f"weak_{side}_openness") or "nan")
        except ValueError:
            continue
        if MID_LO < target < MID_HI:
            row[valid_col] = fmt(0.0)
            retired += 1
    if retired:
        row["v4_notes"] = (row.get("v4_notes") or "") + "|v7_midconst_retired"
    return retired


def ramp_rows(columns: list[str], session_dir: Path) -> list[dict[str, str]]:
    labels = session_dir / "eyelid_ramp_labels.csv"
    if not labels.exists():
        return []
    session = session_dir.name
    out: list[dict[str, str]] = []
    with labels.open(encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            row = blank_row(columns)
            seq = int(r["sequence"])
            row.update({
                "sample_id": f"{session}_{seq:06d}",
                "wear_id": session,
                "session_id": session,
                "session": session,
                "sequence_id": str(seq),
                "frame_index": str(seq),
                "timestamp": r["elapsed_s"],
                "split": "train",
                "left_file": str(session_dir / r["left_file"].replace("/", os.sep)),
                "right_file": str(session_dir / r["right_file"].replace("/", os.sep)),
                "target_x": fmt(0.0),
                "target_y": fmt(0.0),
                "target_source": "ramp_protocol",
                "gaze_stage": "",
                "stage": r["stage"],
                "openness_target_left": r["openness_target_left"],
                "openness_target_right": r["openness_target_right"],
                "openness_source": "ramp_time_progress",
                "weak_left_openness": r["openness_target_left"],
                "weak_right_openness": r["openness_target_right"],
                "weak_left_wide": r["wide_target"],
                "weak_right_wide": r["wide_target"],
                "weak_left_squint": r["squint_target"],
                "weak_right_squint": r["squint_target"],
                "weak_label_source": "ramp_time_progress_v1",
                "source": "eyelid_ramp_app",
                "sample_weight": fmt(0.0),
                "gaze_weight": fmt(0.0),
                "gaze_valid": fmt(0.0),
                "openness_valid_left": fmt(float(r["openness_valid_left"])),
                "openness_valid_right": fmt(float(r["openness_valid_right"])),
                "wide_valid_left": fmt(float(r["wide_valid"])),
                "wide_valid_right": fmt(float(r["wide_valid"])),
                "squint_valid_left": fmt(float(r["squint_valid"])),
                "squint_valid_right": fmt(float(r["squint_valid"])),
                "pupil_valid_left": fmt(0.0),
                "pupil_valid_right": fmt(0.0),
                "confidence_valid_left": fmt(0.0),
                "confidence_valid_right": fmt(0.0),
                "confidence_valid_pair": fmt(0.0),
                "manifest_version": "v7",
                "pair_key": f"{session}_{seq:06d}",
                "row_repeat_count": "1",
            })
            out.append(row)
    return out


def gaze_rows(columns: list[str], session_dir: Path) -> list[dict[str, str]]:
    pairs = session_dir / "pairs.csv"
    frames = session_dir / "frames"
    if not pairs.exists() or not frames.exists():
        return []
    session = session_dir.name
    out: list[dict[str, str]] = []
    with pairs.open(encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            stage = (r.get("stage") or "").strip()
            if stage not in GAZE_STAGE_TARGETS:
                continue
            if (r.get("left_found") or "").lower() not in ("true", "1"):
                continue
            if (r.get("right_found") or "").lower() not in ("true", "1"):
                continue
            try:
                if float(r.get("left_conf") or 0) < MIN_GAZE_CONF or float(r.get("right_conf") or 0) < MIN_GAZE_CONF:
                    continue
            except ValueError:
                continue
            tx, ty = GAZE_STAGE_TARGETS[stage]
            seq = int(r["sequence"])
            left_file = r["left_file"].replace("/", os.sep)
            right_file = r["right_file"].replace("/", os.sep)
            left_path = session_dir / left_file if not os.path.isabs(left_file) else Path(left_file)
            right_path = session_dir / right_file if not os.path.isabs(right_file) else Path(right_file)
            row = blank_row(columns)
            row.update({
                "sample_id": f"gazecal_{session}_{seq:06d}",
                "wear_id": f"gazecal_{session}",
                "session_id": f"gazecal_{session}",
                "session": f"gazecal_{session}",
                "sequence_id": str(seq),
                "frame_index": str(seq),
                "split": "train",
                "left_file": str(left_path),
                "right_file": str(right_path),
                "target_x": fmt(tx),
                "target_y": fmt(ty),
                "target_source": "app_quickcal_5point_v7",
                "gaze_stage": stage,
                "stage": stage,
                "openness_source": "not_supervised",
                # v7: neutral face during gaze calibration -> negative supervision for the
                # expression heads at every gaze direction (kills look-up => wide).
                "weak_left_wide": fmt(0.0),
                "weak_right_wide": fmt(0.0),
                "weak_left_squint": fmt(0.0),
                "weak_right_squint": fmt(0.0),
                "weak_label_source": "gaze_neutral_expression_v7",
                "source": "app_quickcal",
                "sample_weight": fmt(1.0),
                "gaze_weight": fmt(1.0),
                "gaze_valid": fmt(1.0),
                "openness_valid_left": fmt(0.0),
                "openness_valid_right": fmt(0.0),
                "wide_valid_left": fmt(1.0),
                "wide_valid_right": fmt(1.0),
                "squint_valid_left": fmt(1.0),
                "squint_valid_right": fmt(1.0),
                "pupil_valid_left": fmt(0.0),
                "pupil_valid_right": fmt(0.0),
                "confidence_valid_left": fmt(0.0),
                "confidence_valid_right": fmt(0.0),
                "confidence_valid_pair": fmt(0.0),
                "manifest_version": "v7",
                "pair_key": f"gazecal_{session}_{seq:06d}",
                "row_repeat_count": "1",
            })
            out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE_MANIFEST)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ramp-glob", default=RAMP_GLOB)
    parser.add_argument("--gaze-glob", default=GAZE_GLOB)
    args = parser.parse_args()

    with open(args.base, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        base_rows = list(reader)

    base_sessions = {r.get("session_id", "") for r in base_rows}

    retired = 0
    for row in base_rows:
        retired += retire_mid_constants(row)

    added_ramp: list[dict[str, str]] = []
    for d in sorted(glob.glob(args.ramp_glob)):
        rows = ramp_rows(columns, Path(d))
        print(f"ramp {Path(d).name}: +{len(rows)}")
        added_ramp.extend(rows)

    added_gaze: list[dict[str, str]] = []
    for d in sorted(glob.glob(args.gaze_glob)):
        name = Path(d).name
        if name in base_sessions or f"gazecal_{name}" in base_sessions:
            print(f"gaze {name}: already in base, skipped")
            continue
        rows = gaze_rows(columns, Path(d))
        print(f"gaze {name}: +{len(rows)}")
        added_gaze.extend(rows)

    missing = 0
    for row in added_ramp + added_gaze:
        if not Path(row["left_file"]).exists() or not Path(row["right_file"]).exists():
            missing += 1
    if missing:
        raise SystemExit(f"{missing} new rows reference missing frame files — aborting.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in base_rows + added_ramp + added_gaze:
            writer.writerow(row)

    mid = sum(
        1 for r in added_ramp
        if float(r["openness_valid_left"]) > 0.5 and MID_LO < float(r["weak_left_openness"]) < MID_HI
    )
    print(f"\nbase rows: {len(base_rows)} (mid-band constant openness sides retired: {retired})")
    print(f"ramp rows: {len(added_ramp)} (continuous mid-band, valid-left: {mid})")
    print(f"gaze rows: {len(added_gaze)} (wide=0/squint=0 supervised, openness via P1-7)")
    print(f"total: {len(base_rows) + len(added_ramp) + len(added_gaze)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
