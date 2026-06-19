#!/usr/bin/env python3
"""Export conservative img2img job metadata from a gaze manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROMPT = (
    "infrared headset eye camera crop, same pupil position, same eyelid contour, "
    "same gaze geometry, slight sensor noise, slight illumination variation"
)
NEGATIVE_PROMPT = (
    "changed pupil position, changed gaze direction, deformed iris, shifted glint, "
    "new eye shape, heavy perspective change, stylized, synthetic looking"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSONL job file for local diffusion tooling.")
    parser.add_argument("--max-per-stage", type=int, default=24)
    parser.add_argument("--strength", type=float, default=0.18, help="Recommended img2img denoise strength.")
    args = parser.parse_args()

    by_stage: dict[str, int] = {}
    jobs = []
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source", "real") != "real" or row.get("split") != "train":
                continue
            stage = row["stage"]
            if by_stage.get(stage, 0) >= args.max_per_stage:
                continue
            by_stage[stage] = by_stage.get(stage, 0) + 1
            for side in ("left", "right"):
                jobs.append(
                    {
                        "sample_id": row["sample_id"],
                        "stage": stage,
                        "side": side,
                        "target_x": float(row["target_x"]),
                        "target_y": float(row["target_y"]),
                        "input_image": row[f"{side}_file"],
                        "suggested_output": f"{row['sample_id']}_{side}_img2img.png",
                        "prompt": PROMPT,
                        "negative_prompt": NEGATIVE_PROMPT,
                        "denoise_strength": args.strength,
                        "label_policy": "Only keep output if pupil, eyelid, and glint geometry remain aligned with the input.",
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    print(f"Wrote {len(jobs)} diffusion jobs to {args.output}")
    print("Per-stage source count:", by_stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
