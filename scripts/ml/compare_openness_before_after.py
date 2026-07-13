#!/usr/bin/env python3
"""Compare openness / gaze behavior between two runtime CSVs (E1 before/after).

Metrics over the selected rows (use --stage to restrict to a labeled segment):
  open_false_close_rate : fraction with min(L,R openness) < open_false_th
                          (held-open segment -> expect DOWN after E1)
  closed_success_rate   : fraction with min(L,R openness) < closed_success_th
                          (held-closed segment -> expect UP after E1)
  one_eye_lost_rate     : fraction of frames reported ONE_EYE_LOST
                          (open data -> expect DOWN; spurious single-eye flicker suppressed)
  blinking_rate         : fraction reported BLINKING
  min_openness mean/med : summary of per-frame min(L,R) openness
  gaze_smooth_median    : [median smooth_x, median smooth_y] (gaze non-regression check)

Example:
  python compare_openness_before_after.py --before compat.csv --after e1.csv
  python compare_openness_before_after.py --before compat.csv --after e1.csv --stage closed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_live_multitask_csv import parse_float, read_rows  # noqa: E402


def min_openness(rows: list[dict[str, str]]) -> np.ndarray:
    out = []
    for row in rows:
        left = parse_float(row, "left_openness")
        right = parse_float(row, "right_openness")
        if left is not None and right is not None:
            out.append(min(left, right))
    return np.asarray(out, dtype=np.float64)


def _median(rows: list[dict[str, str]], key: str):
    vals = [v for row in rows if (v := parse_float(row, key)) is not None]
    return float(np.median(vals)) if vals else None


def metrics(rows: list[dict[str, str]], open_false_th: float, closed_success_th: float) -> dict[str, object]:
    mo = min_openness(rows)
    states = [row.get("tracking_state", "") for row in rows]
    n = len(states)

    def state_rate(name: str) -> float:
        return round(states.count(name) / n, 4) if n else 0.0

    return {
        "rows": len(rows),
        "open_false_close_rate": round(float(np.mean(mo < open_false_th)), 4) if mo.size else None,
        "closed_success_rate": round(float(np.mean(mo < closed_success_th)), 4) if mo.size else None,
        "one_eye_lost_rate": state_rate("ONE_EYE_LOST"),
        "blinking_rate": state_rate("BLINKING"),
        "min_openness_mean": round(float(np.mean(mo)), 4) if mo.size else None,
        "min_openness_median": round(float(np.median(mo)), 4) if mo.size else None,
        "gaze_smooth_median": [_median(rows, "smooth_x"), _median(rows, "smooth_y")],
        "state_counts": {s: states.count(s) for s in sorted(set(states)) if s},
    }


def filter_stage(rows: list[dict[str, str]], substr: str) -> list[dict[str, str]]:
    if not substr:
        return rows
    needle = substr.lower()
    return [row for row in rows if needle in str(row.get("stage", "")).lower()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, type=Path, help="Baseline/compat runtime CSV.")
    ap.add_argument("--after", required=True, type=Path, help="E1-mode runtime CSV.")
    ap.add_argument("--stage", default="", help="Restrict to rows whose stage contains this substring.")
    ap.add_argument("--open-false-threshold", type=float, default=0.65)
    ap.add_argument("--closed-success-threshold", type=float, default=0.15)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    before = filter_stage(read_rows(args.before.resolve()), args.stage)
    after = filter_stage(read_rows(args.after.resolve()), args.stage)
    report = {
        "before_csv": str(args.before.resolve()),
        "after_csv": str(args.after.resolve()),
        "stage_filter": args.stage or None,
        "open_false_threshold": args.open_false_threshold,
        "closed_success_threshold": args.closed_success_threshold,
        "before": metrics(before, args.open_false_threshold, args.closed_success_threshold),
        "after": metrics(after, args.open_false_threshold, args.closed_success_threshold),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
