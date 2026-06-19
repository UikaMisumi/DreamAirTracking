#!/usr/bin/env python3
"""Convert saved low-openness runtime snapshots into a source manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STAGE_TARGETS = {
    "center": (0.0, 0.0),
    "center_confirm": (0.0, 0.0),
    "micro_left": (-0.25, 0.0),
    "micro_right": (0.25, 0.0),
    "micro_up": (0.0, 0.25),
    "micro_down": (0.0, -0.25),
    "left": (-0.7, 0.0),
    "right": (0.7, 0.0),
    "up": (0.0, 0.7),
    "down": (0.0, -0.7),
    "left_up": (-0.7, 0.7),
    "right_up": (0.7, 0.7),
    "left_down": (-0.7, -0.7),
    "right_down": (0.7, -0.7),
}


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--default-stage", default="center")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-weight", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            left_file = row.get("left_snapshot", "")
            right_file = row.get("right_snapshot", "")
            if not left_file or not right_file:
                continue
            if not Path(left_file).exists() or not Path(right_file).exists():
                continue
            stage = (row.get("stage") or args.default_stage).strip() or args.default_stage
            target = STAGE_TARGETS.get(stage, STAGE_TARGETS.get(args.default_stage, (0.0, 0.0)))
            rows.append(
                {
                    "sample_id": f"{args.input.stem}_low_open_{index:06d}",
                    "session": args.input.stem,
                    "stage": stage,
                    "target_x": f"{target[0]:.6f}",
                    "target_y": f"{target[1]:.6f}",
                    "left_file": str(Path(left_file).resolve()),
                    "right_file": str(Path(right_file).resolve()),
                    "left_conf": f"{parse_float(row.get('left_confidence'), 1.0):.6f}",
                    "right_conf": f"{parse_float(row.get('right_confidence'), 1.0):.6f}",
                    "source": "runtime_low_openness_hardcase",
                    "sample_weight": f"{args.sample_weight:.6f}",
                    "split": args.split,
                }
            )
            if args.max_rows > 0 and len(rows) >= args.max_rows:
                break

    if not rows:
        raise SystemExit(f"No rows with saved snapshots found in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
