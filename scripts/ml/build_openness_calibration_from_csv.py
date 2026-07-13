#!/usr/bin/env python3
"""Build a per-eye openness calibration (v2) from existing runtime CSV logs.

Zero-new-capture path for E1: pools the raw per-eye model openness columns
(``left_model_openness`` / ``right_model_openness``) across one or more live
multitask runtime CSVs and writes ``open_p95`` / ``closed_p05`` per eye.

A plain runtime CSV has no open/closed stage labels, so this uses whole-column
percentiles: p95 ~ typical wide-open level, p05 ~ blink/closed floor. The
runtime's ``normalize_per_eye`` has a ``min_range`` guard, so a degenerate range
(too few frames, or eyes never closed) safely degrades to identity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_live_multitask_csv import parse_float, read_rows  # noqa: E402


def _column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray(
        [value for row in rows if (value := parse_float(row, key)) is not None],
        dtype=np.float64,
    )


def side_payload(values: np.ndarray, open_pct: float, closed_pct: float) -> dict[str, object]:
    if values.size == 0:
        # No samples for this eye -> identity refs so the file stays valid.
        return {"open_p95": 1.0, "closed_p05": 0.0, "samples": 0, "note": "no_samples_identity"}
    return {
        "open_p95": float(np.percentile(values, open_pct)),
        "closed_p05": float(np.percentile(values, closed_pct)),
        "samples": int(values.size),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True, type=Path, help="Runtime CSV; repeatable.")
    parser.add_argument("--output", type=Path, required=True, help="Destination openness_calibration.json (v2).")
    parser.add_argument("--open-percentile", type=float, default=95.0)
    parser.add_argument("--closed-percentile", type=float, default=5.0)
    args = parser.parse_args()

    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    total_rows = 0
    for path in args.csv:
        rows = read_rows(path.resolve())
        total_rows += len(rows)
        left_parts.append(_column(rows, "left_model_openness"))
        right_parts.append(_column(rows, "right_model_openness"))

    left = np.concatenate(left_parts) if left_parts else np.zeros(0)
    right = np.concatenate(right_parts) if right_parts else np.zeros(0)
    if left.size == 0 and right.size == 0:
        raise SystemExit("No left_model_openness/right_model_openness values found in the given CSV(s).")

    payload = {
        "schema": "dreamair.openness_calibration.v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "runtime_csv_percentile",
        "score": "model_openness_lr",
        "open_percentile": args.open_percentile,
        "closed_percentile": args.closed_percentile,
        "csv_row_count": total_rows,
        "left": side_payload(left, args.open_percentile, args.closed_percentile),
        "right": side_payload(right, args.open_percentile, args.closed_percentile),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
