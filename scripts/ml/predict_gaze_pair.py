#!/usr/bin/env python3
"""Run a trained Dream Air binocular gaze checkpoint on eye image pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_gaze_baseline import (  # noqa: E402
    HybridGazeNet,
    Sample,
    create_model,
    geometry_to_tensor,
    image_to_tensor,
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, int, str]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 96))
    model_type = str(checkpoint.get("model_type", "cnn"))
    model = create_model(model_type, image_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, image_size, model_type


def load_quick_layer(path: Path | None, session: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    payload = json.load(path.open("r", encoding="utf-8"))
    if payload.get("raw_source") != "model_prediction_xy":
        raise SystemExit(f"Quick layer raw_source is {payload.get('raw_source')}; expected model_prediction_xy.")

    sessions = payload.get("sessions", {})
    if not isinstance(sessions, dict) or not sessions:
        raise SystemExit("Quick layer has no session entries.")
    if session:
        layer = sessions.get(session)
        if layer is None:
            raise SystemExit(f"Quick layer has no session entry named {session}.")
    else:
        layer = next((item for item in sessions.values() if isinstance(item, dict) and item.get("used")), None)
        if layer is None:
            layer = next(iter(sessions.values()))

    a = np.array(layer["A"], dtype=np.float32)
    b = np.array(layer["b"], dtype=np.float32)
    return a, b


def parse_center(value: str | None, name: str) -> tuple[float, float] | None:
    if not value:
        return None
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"{name} must be formatted as x,y.")
    return float(parts[0]), float(parts[1])


@torch.no_grad()
def predict_pair(
    model: torch.nn.Module,
    model_type: str,
    image_size: int,
    left_file: Path,
    right_file: Path,
    device: torch.device,
    left_center: tuple[float, float] | None = None,
    right_center: tuple[float, float] | None = None,
) -> np.ndarray:
    left = image_to_tensor(left_file, image_size, train=False, rng=None)  # type: ignore[arg-type]
    right = image_to_tensor(right_file, image_size, train=False, rng=None)  # type: ignore[arg-type]
    image = torch.cat([left, right], dim=0).unsqueeze(0).to(device)

    if model_type == "hybrid":
        if left_center is None or right_center is None:
            raise SystemExit("Hybrid checkpoints require --left-center x,y and --right-center x,y.")
        sample = Sample(
            sample_id="live_pair",
            left_file=left_file,
            right_file=right_file,
            target=(0.0, 0.0),
            centers=(left_center[0], left_center[1], right_center[0], right_center[1]),
            weight=1.0,
            split="live",
            source="live",
        )
        pred = model(image, geometry_to_tensor(sample).unsqueeze(0).to(device))
    else:
        pred = model(image)
    return pred.detach().cpu().numpy()[0].astype(np.float32)


def corrected(raw: np.ndarray, layer: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray | None:
    if layer is None:
        return None
    a, b = layer
    return (a @ raw + b).astype(np.float32)


def write_batch(
    args: argparse.Namespace,
    model: torch.nn.Module,
    image_size: int,
    model_type: str,
    device: torch.device,
    layer: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    if args.output_csv is None:
        raise SystemExit("--input-csv requires --output-csv.")
    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0].keys()) if rows else []
    for field in ("raw_x", "raw_y", "corrected_x", "corrected_y"):
        if field not in fields:
            fields.append(field)

    output_rows: list[dict[str, str]] = []
    for row in rows:
        left_file = Path(row.get("left_file") or row.get("left") or "")
        right_file = Path(row.get("right_file") or row.get("right") or "")
        if not left_file.is_absolute():
            left_file = args.input_csv.parent / left_file
        if not right_file.is_absolute():
            right_file = args.input_csv.parent / right_file
        left_center = None
        right_center = None
        if row.get("left_center_x") and row.get("left_center_y"):
            left_center = (float(row["left_center_x"]), float(row["left_center_y"]))
        if row.get("right_center_x") and row.get("right_center_y"):
            right_center = (float(row["right_center_x"]), float(row["right_center_y"]))

        raw = predict_pair(model, model_type, image_size, left_file, right_file, device, left_center, right_center)
        adj = corrected(raw, layer)
        row = dict(row)
        row["raw_x"] = f"{raw[0]:.6f}"
        row["raw_y"] = f"{raw[1]:.6f}"
        row["corrected_x"] = "" if adj is None else f"{adj[0]:.6f}"
        row["corrected_y"] = "" if adj is None else f"{adj[1]:.6f}"
        output_rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote predictions: {args.output_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--left-center", help="Optional hybrid center as x,y.")
    parser.add_argument("--right-center", help="Optional hybrid center as x,y.")
    parser.add_argument("--input-csv", type=Path, help="Batch CSV with left_file/right_file columns.")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--quick-layer", type=Path, help="Optional model_prediction_xy quick calibration layer.")
    parser.add_argument("--quick-layer-session", help="Session key to use from the quick layer.")
    parser.add_argument("--json", action="store_true", help="Print a JSON object for single-pair prediction.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, image_size, model_type = load_checkpoint(args.checkpoint.resolve(), device)
    layer = load_quick_layer(args.quick_layer.resolve() if args.quick_layer else None, args.quick_layer_session)

    if args.input_csv:
        write_batch(args, model, image_size, model_type, device, layer)
        return 0

    if args.left is None or args.right is None:
        raise SystemExit("Pass --left and --right for single-pair prediction, or --input-csv for batch prediction.")

    raw = predict_pair(
        model,
        model_type,
        image_size,
        args.left.resolve(),
        args.right.resolve(),
        device,
        parse_center(args.left_center, "--left-center"),
        parse_center(args.right_center, "--right-center"),
    )
    adj = corrected(raw, layer)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model_type": model_type,
        "device": str(device),
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "raw": {"x": float(raw[0]), "y": float(raw[1])},
        "corrected": None if adj is None else {"x": float(adj[0]), "y": float(adj[1])},
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"raw_x={raw[0]:+.6f} raw_y={raw[1]:+.6f}")
        if adj is not None:
            print(f"corrected_x={adj[0]:+.6f} corrected_y={adj[1]:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
