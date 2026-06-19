#!/usr/bin/env python3
"""Evaluate a gaze checkpoint and write an automatic review report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_gaze_baseline import BinocularGazeNet, HybridGazeNet, geometry_to_tensor, image_to_tensor, Sample  # noqa: E402


REQUIRED_GAZE_STAGES = {
    "center",
    "left",
    "right",
    "up",
    "down",
    "left_up",
    "right_up",
    "left_down",
    "right_down",
}


@dataclass(frozen=True)
class Row:
    data: dict[str, str]

    @property
    def split(self) -> str:
        return self.data.get("split", "train")

    @property
    def stage(self) -> str:
        return self.data["stage"]

    @property
    def session(self) -> str:
        return self.data["session"]

    @property
    def sample_id(self) -> str:
        return self.data["sample_id"]

    @property
    def target(self) -> np.ndarray:
        return np.array([float(self.data["target_x"]), float(self.data["target_y"])], dtype=np.float32)

    @property
    def feature(self) -> np.ndarray:
        return np.array(
            [
                float(self.data["left_center_x"]),
                float(self.data["left_center_y"]),
                float(self.data["right_center_x"]),
                float(self.data["right_center_y"]),
                1.0,
            ],
            dtype=np.float64,
        )


def read_manifest(path: Path) -> list[Row]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [Row(row) for row in csv.DictReader(handle)]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int, str]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("image_size", 96))
    model_type = str(checkpoint.get("model_type", "cnn"))
    model = HybridGazeNet(image_size).to(device) if model_type == "hybrid" else BinocularGazeNet(image_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, image_size, model_type


@torch.no_grad()
def predict_model(model: torch.nn.Module, rows: list[Row], image_size: int, device: torch.device) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for row in rows:
        left = image_to_tensor(Path(row.data["left_file"]), image_size, train=False, rng=None)  # type: ignore[arg-type]
        right = image_to_tensor(Path(row.data["right_file"]), image_size, train=False, rng=None)  # type: ignore[arg-type]
        image = torch.cat([left, right], dim=0).unsqueeze(0).to(device)
        if isinstance(model, HybridGazeNet):
            sample = Sample(
                sample_id=row.sample_id,
                left_file=Path(row.data["left_file"]),
                right_file=Path(row.data["right_file"]),
                target=(float(row.data["target_x"]), float(row.data["target_y"])),
                centers=(
                    float(row.data["left_center_x"]),
                    float(row.data["left_center_y"]),
                    float(row.data["right_center_x"]),
                    float(row.data["right_center_y"]),
                ),
                weight=float(row.data.get("sample_weight", "1.0") or "1.0"),
                split=row.split,
                source=row.data.get("source", "real"),
            )
            pred = model(image, geometry_to_tensor(sample).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        else:
            pred = model(image).detach().cpu().numpy()[0]
        predictions[row.sample_id] = pred.astype(np.float32)
    return predictions


def fit_affine_baseline(rows: list[Row], alpha: float) -> np.ndarray:
    train = [row for row in rows if row.split == "train"]
    if len(train) < 5:
        raise SystemExit("Need at least 5 train samples to fit affine pupil-center baseline.")
    x = np.stack([row.feature for row in train])
    y = np.stack([row.target.astype(np.float64) for row in train])
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def predict_affine(rows: list[Row], weights: np.ndarray) -> dict[str, np.ndarray]:
    return {row.sample_id: (row.feature @ weights).astype(np.float32) for row in rows}


def fit_prediction_affine(source: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    x = np.concatenate([source.astype(np.float64), np.ones((source.shape[0], 1), dtype=np.float64)], axis=1)
    y = target.astype(np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def apply_prediction_affine(source: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.concatenate([source.astype(np.float64), np.ones((source.shape[0], 1), dtype=np.float64)], axis=1)
    return (x @ weights).astype(np.float32)


def simulate_quick_calibration(
    rows: list[Row],
    model_predictions: dict[str, np.ndarray],
    samples_per_stage: int,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    predictions = {sample_id: pred.copy() for sample_id, pred in model_predictions.items()}
    calibration_ids: set[str] = set()
    session_reports: dict[str, object] = {}
    by_session: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.split == "val":
            by_session[row.session].append(row)

    for session, session_rows in by_session.items():
        selected: list[Row] = []
        by_stage_seen = Counter()
        for row in session_rows:
            if by_stage_seen[row.stage] < samples_per_stage:
                selected.append(row)
                by_stage_seen[row.stage] += 1
                calibration_ids.add(row.sample_id)
        if len(selected) < 3:
            session_reports[session] = {
                "calibration_samples": len(selected),
                "used": False,
                "reason": "fewer than 3 calibration samples",
            }
            continue

        source = np.stack([model_predictions[row.sample_id] for row in selected])
        target = np.stack([row.target for row in selected])
        weights = fit_prediction_affine(source, target, alpha)
        all_source = np.stack([model_predictions[row.sample_id] for row in session_rows])
        all_corrected = apply_prediction_affine(all_source, weights)
        for row, pred in zip(session_rows, all_corrected):
            predictions[row.sample_id] = pred
        session_reports[session] = {
            "calibration_samples": len(selected),
            "calibration_sample_ids": [row.sample_id for row in selected],
            "used": True,
            "weights": weights.tolist(),
        }

    return predictions, {
        "samples_per_stage": samples_per_stage,
        "calibration_sample_ids": sorted(calibration_ids),
        "sessions": session_reports,
    }


def weights_to_layer(weights: list[list[float]]) -> dict[str, object]:
    """Convert [raw_x, raw_y, 1] @ W into corrected = A @ raw + b."""
    return {
        "A": [
            [weights[0][0], weights[1][0]],
            [weights[0][1], weights[1][1]],
        ],
        "b": [weights[2][0], weights[2][1]],
        "raw_weights": weights,
    }


def build_quick_calibration_layer(
    quick_calibration: dict[str, object],
    checkpoint: Path,
    manifest: Path,
) -> dict[str, object]:
    sessions = {}
    for session, report in dict(quick_calibration.get("sessions", {})).items():
        if not isinstance(report, dict) or not report.get("used"):
            sessions[session] = report
            continue
        weights = report.get("weights")
        if not isinstance(weights, list) or len(weights) != 3:
            sessions[session] = report
            continue
        sessions[session] = {
            **report,
            **weights_to_layer(weights),
        }
    return {
        "schema": "dream_air_tracking.quick_gaze_calibration_layer.v1",
        "formula": "corrected = A @ raw + b",
        "raw_source": "model_prediction_xy",
        "checkpoint": str(checkpoint.resolve()),
        "manifest": str(manifest.resolve()),
        "samples_per_stage": quick_calibration.get("samples_per_stage"),
        "sessions": sessions,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def summarize(grouped_errors: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, rows in grouped_errors.items():
        errors = [float(row["model_l2"]) for row in rows]
        dx = [float(row["model_pred_x"]) - float(row["target_x"]) for row in rows]
        dy = [float(row["model_pred_y"]) - float(row["target_y"]) for row in rows]
        summary[name] = {
            "count": len(rows),
            "mean_l2": statistics.fmean(errors) if errors else math.nan,
            "median_l2": statistics.median(errors) if errors else math.nan,
            "p90_l2": percentile(errors, 90),
            "max_l2": max(errors) if errors else math.nan,
            "bias_x": statistics.fmean(dx) if dx else math.nan,
            "bias_y": statistics.fmean(dy) if dy else math.nan,
        }
    return summary


def build_records(
    rows: list[Row],
    model_predictions: dict[str, np.ndarray],
    affine_predictions: dict[str, np.ndarray],
    adapted_predictions: dict[str, np.ndarray],
    calibration_sample_ids: set[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in rows:
        target = row.target
        model_pred = model_predictions[row.sample_id]
        affine_pred = affine_predictions[row.sample_id]
        adapted_pred = adapted_predictions[row.sample_id]
        model_delta = model_pred - target
        affine_delta = affine_pred - target
        adapted_delta = adapted_pred - target
        records.append(
            {
                "sample_id": row.sample_id,
                "session": row.session,
                "split": row.split,
                "stage": row.stage,
                "used_for_quick_calibration": row.sample_id in calibration_sample_ids,
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "model_pred_x": float(model_pred[0]),
                "model_pred_y": float(model_pred[1]),
                "model_l2": float(np.linalg.norm(model_delta)),
                "affine_pred_x": float(affine_pred[0]),
                "affine_pred_y": float(affine_pred[1]),
                "affine_l2": float(np.linalg.norm(affine_delta)),
                "adapted_pred_x": float(adapted_pred[0]),
                "adapted_pred_y": float(adapted_pred[1]),
                "adapted_l2": float(np.linalg.norm(adapted_delta)),
                "left_file": row.data["left_file"],
                "right_file": row.data["right_file"],
            }
        )
    return records


def write_predictions(records: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "session",
        "split",
        "stage",
        "used_for_quick_calibration",
        "target_x",
        "target_y",
        "model_pred_x",
        "model_pred_y",
        "model_l2",
        "affine_pred_x",
        "affine_pred_y",
        "affine_l2",
        "adapted_pred_x",
        "adapted_pred_y",
        "adapted_l2",
        "left_file",
        "right_file",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def draw_scatter(records: list[dict[str, object]], output: Path, split: str) -> None:
    selected = [row for row in records if row["split"] == split]
    size = 720
    pad = 70
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    def project(x: float, y: float) -> tuple[int, int]:
        scale = (size - 2 * pad) / 2.4
        return int(size / 2 + x * scale), int(size / 2 - y * scale)

    for value in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        x0, y0 = project(value, -1.1)
        x1, y1 = project(value, 1.1)
        draw.line((x0, y0, x1, y1), fill=(225, 225, 225), width=1)
        x0, y0 = project(-1.1, value)
        x1, y1 = project(1.1, value)
        draw.line((x0, y0, x1, y1), fill=(225, 225, 225), width=1)

    draw.rectangle((*project(-1, 1), *project(1, -1)), outline=(120, 120, 120), width=2)
    for row in selected:
        tx, ty = float(row["target_x"]), float(row["target_y"])
        px, py = float(row["model_pred_x"]), float(row["model_pred_y"])
        ax, ay = float(row["affine_pred_x"]), float(row["affine_pred_y"])
        cx, cy = float(row["adapted_pred_x"]), float(row["adapted_pred_y"])
        t = project(tx, ty)
        p = project(px, py)
        a = project(ax, ay)
        c = project(cx, cy)
        draw.line((*t, *p), fill=(230, 75, 75), width=1)
        draw.line((*t, *a), fill=(75, 120, 220), width=1)
        draw.line((*t, *c), fill=(30, 150, 80), width=1)
        draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=(210, 35, 35))
        draw.rectangle((a[0] - 3, a[1] - 3, a[0] + 3, a[1] + 3), fill=(35, 85, 200))
        draw.ellipse((c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3), fill=(30, 135, 70))
        draw.ellipse((t[0] - 4, t[1] - 4, t[0] + 4, t[1] + 4), outline=(0, 0, 0), width=2)

    draw.text((20, 20), f"{split}: target black, CNN red, affine blue, quick-cal green", fill=(0, 0, 0))
    image.save(output)


def scan_available_sessions(workspace_root: Path) -> list[dict[str, object]]:
    sessions: list[dict[str, object]] = []
    for csv_path in workspace_root.glob("*/calibration_pairs.csv"):
        counts: Counter[str] = Counter()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                stage = row.get("stage", "")
                counts[stage] += 1
        gaze_counts = {stage: counts.get(stage, 0) for stage in sorted(REQUIRED_GAZE_STAGES)}
        sessions.append(
            {
                "session": csv_path.parent.name,
                "path": str(csv_path.resolve()),
                "total_rows": sum(counts.values()),
                "stage_counts": dict(sorted(counts.items())),
                "required_gaze_counts": gaze_counts,
                "missing_required_gaze_stages": sorted(stage for stage in REQUIRED_GAZE_STAGES if counts.get(stage, 0) == 0),
            }
        )
    return sorted(sessions, key=lambda item: item["session"])


def make_review(
    records: list[dict[str, object]],
    session_scan: list[dict[str, object]],
    quick_calibration: dict[str, object],
    mean_gate: float,
    p90_gate: float,
    max_gate: float,
) -> tuple[dict[str, object], str]:
    val = [row for row in records if row["split"] == "val"]
    train = [row for row in records if row["split"] == "train"]
    by_stage: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        by_stage[f"{row['split']}:{row['stage']}"].append(row)
        by_session[f"{row['split']}:{row['session']}"].append(row)
        by_split[str(row["split"])].append(row)

    split_summary = summarize(by_split)
    stage_summary = summarize(by_stage)
    session_summary = summarize(by_session)

    val_errors = [float(row["model_l2"]) for row in val]
    affine_val_errors = [float(row["affine_l2"]) for row in val]
    adapted_eval = [row for row in val if not bool(row["used_for_quick_calibration"])]
    adapted_val_errors = [float(row["adapted_l2"]) for row in adapted_eval]
    model_mean = statistics.fmean(val_errors) if val_errors else math.inf
    model_p90 = percentile(val_errors, 90)
    model_max = max(val_errors) if val_errors else math.inf
    affine_mean = statistics.fmean(affine_val_errors) if affine_val_errors else math.inf
    adapted_mean = statistics.fmean(adapted_val_errors) if adapted_val_errors else math.inf
    adapted_p90 = percentile(adapted_val_errors, 90)
    adapted_max = max(adapted_val_errors) if adapted_val_errors else math.inf
    worst = sorted(val, key=lambda row: float(row["model_l2"]), reverse=True)[:12]

    manifest_stages = {str(row["stage"]) for row in records}
    missing_required = sorted(REQUIRED_GAZE_STAGES - manifest_stages)
    available_stage_counts = Counter()
    for item in session_scan:
        for stage, count in dict(item["stage_counts"]).items():
            available_stage_counts[stage] += int(count)
    found_missing_elsewhere = {
        stage: available_stage_counts[stage]
        for stage in missing_required
        if available_stage_counts[stage] > 0
    }

    decisions: list[str] = []
    if model_mean <= mean_gate and model_p90 <= p90_gate and model_max <= max_gate:
        decisions.append("V1 checkpoint passes the current five-direction session gate.")
    else:
        decisions.append("V1 checkpoint does not pass the current session gate; improve before moving to V2.")

    if affine_mean <= model_mean * 1.1:
        decisions.append("Traditional pupil-center affine baseline is close to CNN; prioritize calibration/bias analysis before enlarging the CNN.")
    else:
        decisions.append("CNN is meaningfully ahead of pupil-center affine baseline on validation.")

    if adapted_mean < model_mean * 0.85:
        decisions.append("Quick per-session calibration reduces validation error; implement this layer before temporal modeling.")
    else:
        decisions.append("Quick per-session calibration did not materially improve current validation error.")

    if missing_required:
        if found_missing_elsewhere:
            decisions.append("Some required gaze stages are missing from the manifest but exist elsewhere on disk; rebuild manifest with those sessions first.")
        else:
            decisions.append("No corner-stage samples were found on disk; record corner gaze sessions before claiming usable continuous eye tracking.")

    next_actions: list[str] = []
    if missing_required and not found_missing_elsewhere:
        next_actions.append("Record a new validation-first session with left_up/right_up/left_down/right_down plus center/left/right/up/down.")
    if model_mean > mean_gate or model_p90 > p90_gate:
        if not missing_required and affine_mean > model_mean * 1.1:
            next_actions.append("Continue image-model work first: train longer, inspect worst validation eye crops, and check session domain shift before changing runtime affine calibration.")
        else:
            next_actions.append("Inspect label/session consistency and compare image model against the affine baseline before changing runtime calibration.")
    elif adapted_mean < model_mean * 0.85:
        next_actions.append("Promote quick calibration layer into the runtime path: corrected = A * raw + b.")
    elif missing_required:
        next_actions.append("Keep current checkpoint as five-direction V1, but expand data coverage before V2.")
    else:
        next_actions.append("Proceed to V3 quick calibration layer; only move to V2 temporal if live output jitters.")

    review = {
        "gate": {
            "mean_l2_max": mean_gate,
            "p90_l2_max": p90_gate,
            "max_l2_max": max_gate,
            "val_mean_l2": model_mean,
            "val_p90_l2": model_p90,
            "val_max_l2": model_max,
            "passed_current_stage_gate": model_mean <= mean_gate and model_p90 <= p90_gate and model_max <= max_gate,
        },
        "affine_baseline": {
            "val_mean_l2": affine_mean,
            "cnn_val_mean_l2": model_mean,
            "cnn_better_ratio": affine_mean / model_mean if model_mean > 0 else math.inf,
        },
        "quick_calibration": {
            **quick_calibration,
            "heldout_val_mean_l2": adapted_mean,
            "heldout_val_p90_l2": adapted_p90,
            "heldout_val_max_l2": adapted_max,
            "heldout_count": len(adapted_eval),
        },
        "coverage": {
            "manifest_stages": sorted(manifest_stages),
            "missing_required_gaze_stages": missing_required,
            "found_missing_stages_elsewhere": found_missing_elsewhere,
            "available_sessions": session_scan,
        },
        "summaries": {
            "split": split_summary,
            "stage": stage_summary,
            "session": session_summary,
        },
        "worst_val_samples": worst,
        "decisions": decisions,
        "next_actions": next_actions,
        "counts": {
            "train": len(train),
            "val": len(val),
        },
    }

    md_lines = [
        "# Gaze Model Auto Review",
        "",
        "## Gate",
        "",
        f"- val mean L2: {model_mean:.6f} (gate <= {mean_gate})",
        f"- val p90 L2: {model_p90:.6f} (gate <= {p90_gate})",
        f"- val max L2: {model_max:.6f} (gate <= {max_gate})",
        f"- passed current stage gate: {review['gate']['passed_current_stage_gate']}",
        "",
        "## Baseline Check",
        "",
        f"- CNN val mean L2: {model_mean:.6f}",
        f"- affine pupil-center val mean L2: {affine_mean:.6f}",
        f"- affine / CNN ratio: {review['affine_baseline']['cnn_better_ratio']:.3f}",
        "",
        "## Quick Calibration Simulation",
        "",
        f"- held-out val mean L2 after correction: {adapted_mean:.6f}",
        f"- held-out val p90 L2 after correction: {adapted_p90:.6f}",
        f"- held-out val max L2 after correction: {adapted_max:.6f}",
        f"- held-out samples: {len(adapted_eval)}",
        "- layer artifact: `quick_calibration_layer.json`",
        "",
        "## Coverage",
        "",
        f"- manifest stages: {', '.join(sorted(manifest_stages))}",
        f"- missing required gaze stages: {', '.join(missing_required) if missing_required else 'none'}",
        "",
        "## Decisions",
        "",
    ]
    md_lines.extend(f"- {item}" for item in decisions)
    md_lines.extend(["", "## Next Actions", ""])
    md_lines.extend(f"- {item}" for item in next_actions)
    md_lines.extend(["", "## Worst Validation Samples", ""])
    for row in worst[:8]:
        md_lines.append(
            f"- {row['sample_id']} stage={row['stage']} model_l2={float(row['model_l2']):.6f} "
            f"left={row['left_file']}"
        )
    md_lines.append("")
    return review, "\n".join(md_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--quick-calibration-samples-per-stage", type=int, default=3)
    parser.add_argument("--mean-gate", type=float, default=0.08)
    parser.add_argument("--p90-gate", type=float, default=0.15)
    parser.add_argument("--max-gate", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, image_size, model_type = load_model(args.checkpoint.resolve(), device)
    model_predictions = predict_model(model, rows, image_size, device)
    affine_weights = fit_affine_baseline(rows, args.ridge_alpha)
    affine_predictions = predict_affine(rows, affine_weights)
    adapted_predictions, quick_calibration = simulate_quick_calibration(
        rows,
        model_predictions,
        args.quick_calibration_samples_per_stage,
        args.ridge_alpha,
    )
    records = build_records(
        rows,
        model_predictions,
        affine_predictions,
        adapted_predictions,
        set(quick_calibration["calibration_sample_ids"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    write_predictions(records, predictions_path)
    draw_scatter(records, args.output_dir / "scatter_val.png", "val")
    draw_scatter(records, args.output_dir / "scatter_train.png", "train")

    review, markdown = make_review(
        records,
        scan_available_sessions(args.workspace_root.resolve()),
        quick_calibration,
        args.mean_gate,
        args.p90_gate,
        args.max_gate,
    )
    review["device"] = str(device)
    review["model_type"] = model_type
    review["checkpoint"] = str(args.checkpoint.resolve())
    review["manifest"] = str(args.manifest.resolve())
    review["predictions_csv"] = str(predictions_path.resolve())
    review["scatter_val_png"] = str((args.output_dir / "scatter_val.png").resolve())
    review["scatter_train_png"] = str((args.output_dir / "scatter_train.png").resolve())
    quick_layer = build_quick_calibration_layer(quick_calibration, args.checkpoint, args.manifest)
    quick_layer_path = args.output_dir / "quick_calibration_layer.json"
    review["quick_calibration_layer_json"] = str(quick_layer_path.resolve())

    with (args.output_dir / "auto_review.json").open("w", encoding="utf-8") as handle:
        json.dump(review, handle, indent=2, ensure_ascii=False)
    with quick_layer_path.open("w", encoding="utf-8") as handle:
        json.dump(quick_layer, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "auto_review.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown)

    print(markdown)
    print(f"Wrote quick calibration layer to {quick_layer_path}")
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote review to {args.output_dir / 'auto_review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
