#!/usr/bin/env python3
"""Export a MobileNetV3 multitask eye checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from train_eye_multitask import METADATA_FEATURE_COUNT, METADATA_FEATURE_NAMES, build_model, model_uses_metadata  # noqa: E402


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 128))
    model_type = str(checkpoint.get("model_type", "mobilenetv3_small_multitask"))
    try:
        model = build_model(model_type, image_size).to(device)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, image_size, model_type


def export_onnx(checkpoint: Path, output: Path, opset: int, device: torch.device) -> dict[str, object]:
    model, image_size, model_type = load_model(checkpoint, device)
    image = torch.zeros(1, 2, image_size, image_size, dtype=torch.float32, device=device)
    metadata = torch.zeros(1, METADATA_FEATURE_COUNT, dtype=torch.float32, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    uses_metadata = model_uses_metadata(model_type)
    export_args: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    export_args = (image, metadata) if uses_metadata else image
    input_names = ["image", "metadata"] if uses_metadata else ["image"]
    dynamic_axes = {
        "image": {0: "batch"},
        "gaze_xy": {0: "batch"},
        "openness_lr": {0: "batch"},
        "wide_lr": {0: "batch"},
        "squint_lr": {0: "batch"},
        "pupil_lr": {0: "batch"},
        "confidence": {0: "batch"},
    }
    if uses_metadata:
        dynamic_axes["metadata"] = {0: "batch"}
    torch.onnx.export(
        model,
        export_args,
        output,
        input_names=input_names,
        output_names=["gaze_xy", "openness_lr", "wide_lr", "squint_lr", "pupil_lr", "confidence"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(output.resolve()),
        "model_type": model_type,
        "image_size": image_size,
        "input_image_shape": [1, 2, image_size, image_size],
        "input_metadata_shape": [1, METADATA_FEATURE_COUNT] if uses_metadata else None,
        "metadata_features": list(METADATA_FEATURE_NAMES) if uses_metadata else [],
        "input_range": "float32 grayscale normalized to [0, 1]",
        "outputs": {
            "gaze_xy": "[-1, 1]",
            "openness_lr": "[0, 1]",
            "wide_lr": "SteamLink-style EyeWideLeft/Right amount in [0, 1]",
            "squint_lr": "SteamLink-style EyeSquintLeft/Right amount in [0, 1]",
            "pupil_lr": "left_x,left_y,left_radius,right_x,right_y,right_radius in [0, 1]",
            "confidence": "left,right,pair quality in [0, 1]",
        },
        "opset": opset,
    }


def verify_with_onnxruntime(output: Path, metadata: dict[str, object]) -> dict[str, object]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"available": False, "reason": "onnxruntime python package is not installed"}

    image_size = int(metadata["image_size"])
    rng = np.random.default_rng(20260611)
    image = rng.random((1, 2, image_size, image_size), dtype=np.float32)
    feeds = {"image": image}
    if metadata.get("input_metadata_shape"):
        feeds["metadata"] = rng.random(tuple(metadata["input_metadata_shape"]), dtype=np.float32)
    session = ort.InferenceSession(str(output.resolve()), providers=["CPUExecutionProvider"])
    outputs = session.run(["gaze_xy", "openness_lr", "wide_lr", "squint_lr", "pupil_lr", "confidence"], feeds)
    return {
        "available": True,
        "providers": session.get_providers(),
        "output_shapes": {
            "gaze_xy": list(outputs[0].shape),
            "openness_lr": list(outputs[1].shape),
            "wide_lr": list(outputs[2].shape),
            "squint_lr": list(outputs[3].shape),
            "pupil_lr": list(outputs[4].shape),
            "confidence": list(outputs[5].shape),
        },
        "sample_gaze": [float(outputs[0][0, 0]), float(outputs[0][0, 1])],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    metadata = export_onnx(args.checkpoint.resolve(), args.output.resolve(), args.opset, device)
    metadata["export_device"] = str(device)
    metadata["onnxruntime_verify"] = verify_with_onnxruntime(args.output.resolve(), metadata)
    metadata_path = args.metadata.resolve() if args.metadata else args.output.resolve().with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote ONNX: {args.output}")
    print(f"Wrote metadata: {metadata_path}")
    print(json.dumps(metadata["onnxruntime_verify"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
