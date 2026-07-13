#!/usr/bin/env python3
"""Rewrite an eye manifest so gaze rows stop co-supervising openness/pupil (A / E2 fix).

Enforces the head-specific-mask discipline from docs/MODEL_ADAPTATION_DATA_AUDIT_zh.md:
for every gaze row (strict gaze weight > 0) set openness/wide/squint/pupil valid = 0,
leaving eyelid / pupil / expression rows untouched. Produces a manifest that PASSES
manifest_head_mask_gate.py, so the head-mask fix can be trained and compared in isolation.

  python clean_manifest_head_masks.py --in manifest_v5.csv --out manifest_v6_clean.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_eye_manifest_series import strict_gaze_weight  # noqa: E402

ZERO_ON_GAZE = [
    "openness_valid_left", "openness_valid_right",
    "wide_valid_left", "wide_valid_right",
    "squint_valid_left", "squint_valid_right",
    "pupil_valid_left", "pupil_valid_right",
]
NONZERO = {"", "0", "0.0", "0.000000"}


def clean_rows(rows: list[dict[str, str]]) -> int:
    changed = 0
    for row in rows:
        if strict_gaze_weight(row) <= 0.0:
            continue
        row_changed = False
        for col in ZERO_ON_GAZE:
            if col in row and row[col] not in NONZERO:
                row[col] = "0"
                row_changed = True
        if row_changed:
            changed += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with args.inp.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    changed = clean_rows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[clean-manifest] {len(rows)} rows, zeroed head masks on {changed} gaze rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
