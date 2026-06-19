#!/usr/bin/env python3
"""Export a Dream Air gaze PyTorch checkpoint to ONNX."""

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

from train_gaze_baseline import HybridGazeNet, create_model  # noqa: E402


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int, str]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 96))
    model_type = str(checkpoint.get("model_type", "cnn"))
    model = create_model(model_type, image_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, image_size, model_type


def export_onnx(
    checkpoint: Path,
    output: Path,
    opset: int,
    device: torch.device,
) -> dict[str, object]:
    model, image_size, model_type = load_model(checkpoint, device)
    image = torch.zeros(1, 2, image_size, image_size, dtype=torch.float32, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "hybrid":
        geometry = torch.zeros(1, 6, dtype=torch.float32, device=device)
        torch.onnx.export(
            model,
            (image, geometry),
            output,
            input_names=["image", "geometry"],
            output_names=["gaze_xy"],
            dynamic_axes={
                "image": {0: "batch"},
                "geometry": {0: "batch"},
                "gaze_xy": {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            model,
            image,
            output,
            input_names=["image"],
            output_names=["gaze_xy"],
            dynamic_axes={"image": {0: "batch"}, "gaze_xy": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )

    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(output.resolve()),
        "model_type": model_type,
        "image_size": image_size,
        "input_image_shape": [1, 2, image_size, image_size],
        "input_range": "float32 grayscale normalized to [0, 1]",
        "output": "gaze_xy in approximately [-1, 1]",
        "opset": opset,
    }
    return metadata


def verify_with_onnxruntime(output: Path, metadata: dict[str, object]) -> dict[str, object]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"available": False, "reason": "onnxruntime python package is not installed"}

    image_size = int(metadata["image_size"])
    model_type = str(metadata["model_type"])
    rng = np.random.default_rng(20260609)
    image = rng.random((1, 2, image_size, image_size), dtype=np.float32)
    feeds = {"image": image}
    if model_type == "hybrid":
        feeds["geometry"] = rng.random((1, 6), dtype=np.float32)
    session = ort.InferenceSession(str(output.resolve()), providers=["CPUExecutionProvider"])
    result = session.run(["gaze_xy"], feeds)[0]
    return {
        "available": True,
        "providers": session.get_providers(),
        "output_shape": list(result.shape),
        "sample_output": [float(result[0, 0]), float(result[0, 1])],
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
