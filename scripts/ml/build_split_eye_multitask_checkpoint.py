#!/usr/bin/env python3
"""Build a split-branch multitask checkpoint from geometry and expression teachers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from train_eye_multitask import build_model  # noqa: E402


SIAMESE_MODEL_TYPE = "siamese_mobilenetv3_small_multitask"
SPLIT_MODEL_TYPE = "split_siamese_mobilenetv3_small_multitask"


def load_teacher(path: Path, device: torch.device) -> tuple[dict[str, object], dict[str, torch.Tensor], list[str], list[str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_type = str(checkpoint.get("model_type", ""))
    if model_type != SIAMESE_MODEL_TYPE:
        raise SystemExit(f"{path} model_type={model_type}; expected {SIAMESE_MODEL_TYPE}.")
    image_size = int(checkpoint.get("image_size", 128))
    model = build_model(model_type, image_size)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    return checkpoint, model.state_dict(), list(missing), list(unexpected)


def prefixed_state(prefix: str, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"{prefix}.{key}": value for key, value in state.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-checkpoint", required=True, type=Path)
    parser.add_argument("--expression-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    device = torch.device("cpu")
    geometry, geometry_state, geometry_missing, geometry_unexpected = load_teacher(args.geometry_checkpoint.resolve(), device)
    expression, expression_state, expression_missing, expression_unexpected = load_teacher(args.expression_checkpoint.resolve(), device)
    if geometry_unexpected or expression_unexpected:
        raise SystemExit(f"Unexpected teacher keys: geometry={geometry_unexpected}, expression={expression_unexpected}")
    geometry_image_size = int(geometry.get("image_size", 128))
    expression_image_size = int(expression.get("image_size", 128))
    if geometry_image_size != expression_image_size:
        raise SystemExit(f"Image size mismatch: geometry={geometry_image_size}, expression={expression_image_size}.")

    model = build_model(SPLIT_MODEL_TYPE, geometry_image_size)
    state: dict[str, torch.Tensor] = {}
    state.update(prefixed_state("geometry", geometry_state))
    state.update(prefixed_state("expression", expression_state))
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise SystemExit(f"Split load failed; missing={missing}, unexpected={unexpected}")

    payload = {
        "model_state": model.state_dict(),
        "model_type": SPLIT_MODEL_TYPE,
        "architecture": "split_siamese",
        "image_size": geometry_image_size,
        "heads": ["gaze_xy", "openness_lr", "wide_lr", "squint_lr", "pupil_lr", "confidence"],
        "branch_sources": {
            "geometry": str(args.geometry_checkpoint.resolve()),
            "expression": str(args.expression_checkpoint.resolve()),
            "gaze_xy": "geometry",
            "openness_lr": "geometry",
            "pupil_lr": "geometry",
            "confidence": "geometry",
            "wide_lr": "expression",
            "squint_lr": "expression",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    report = {
        "checkpoint": str(args.output.resolve()),
        "model_type": SPLIT_MODEL_TYPE,
        "image_size": geometry_image_size,
        "geometry_checkpoint": str(args.geometry_checkpoint.resolve()),
        "expression_checkpoint": str(args.expression_checkpoint.resolve()),
        "geometry_missing_initialized": geometry_missing,
        "expression_missing_initialized": expression_missing,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "geometry_parameter_count": sum(parameter.numel() for parameter in model.geometry.parameters()),
        "expression_parameter_count": sum(parameter.numel() for parameter in model.expression.parameters()),
    }
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
