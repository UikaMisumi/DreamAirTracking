#!/usr/bin/env python3
"""Append weak pupil/openness/confidence labels to a gaze manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WEAK_FIELDS = [
    "weak_left_openness",
    "weak_left_pupil_x",
    "weak_left_pupil_y",
    "weak_left_pupil_radius",
    "weak_left_quality",
    "weak_right_openness",
    "weak_right_pupil_x",
    "weak_right_pupil_y",
    "weak_right_pupil_radius",
    "weak_right_quality",
    "weak_pair_quality",
    "weak_blink",
    "weak_label_source",
]


def load_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            current: list[tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                current.append((cx, cy))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            components.append(current)
    return components


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def detect_eye_features(array: np.ndarray, side: str) -> dict[str, float]:
    height, width = array.shape
    if width < 8 or height < 8:
        return empty_features()

    dark_threshold = min(float(np.quantile(array, 0.10)), float(np.median(array) * 0.82), 0.42)
    mask = array <= dark_threshold
    border = max(2, min(width, height) // 32)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    min_area = max(6, int(width * height * 0.0015))
    max_area = max(min_area + 1, int(width * height * 0.045))
    best_score = -1.0
    best: dict[str, float] | None = None

    # BrokenEye's per-eye camera crops are mirrored: the left stream's visible pupil
    # usually sits on the right side of the crop, and vice versa.
    expected_x = 0.86 if side == "left" else 0.16

    for component in connected_components(mask):
        area = len(component)
        if area < min_area or area > max_area:
            continue
        xs = np.array([item[0] for item in component], dtype=np.float32)
        ys = np.array([item[1] for item in component], dtype=np.float32)
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        bbox_w = max_x - min_x + 1.0
        bbox_h = max_y - min_y + 1.0
        if bbox_w <= 1.0 or bbox_h <= 1.0:
            continue
        aspect = bbox_w / bbox_h
        if aspect < 0.35 or aspect > 2.8:
            continue
        fill = area / max(1.0, bbox_w * bbox_h)
        radius_px = math.sqrt(area / math.pi)
        cx = float(xs.mean())
        cy = float(ys.mean())
        cx_norm = cx / max(width - 1, 1)
        edge_penalty = min(cx / width, (width - cx) / width, cy / height, (height - cy) / height)
        darkness = 1.0 - float(array[ys.astype(int), xs.astype(int)].mean())
        roundness = 1.0 - min(1.0, abs(math.log(max(aspect, 1e-6))))
        side_prior = clamp01(1.0 - abs(cx_norm - expected_x) / 0.35)
        size_prior = clamp01(1.0 - abs((radius_px / max(1.0, min(width, height))) - 0.08) / 0.12)
        score = (
            (0.30 * fill)
            + (0.22 * darkness)
            + (0.16 * roundness)
            + (0.22 * side_prior)
            + (0.07 * size_prior)
            + (0.03 * clamp01(edge_penalty * 8.0))
        )
        if score > best_score:
            best_score = score
            best = {
                "pupil_x": cx / max(width - 1, 1),
                "pupil_y": cy / max(height - 1, 1),
                "pupil_radius": radius_px / max(1.0, min(width, height)),
                "quality": clamp01(score),
                "bbox_min_x": min_x,
                "bbox_max_x": max_x,
                "bbox_min_y": min_y,
                "bbox_max_y": max_y,
                "bbox_h": bbox_h,
                "bbox_w": bbox_w,
            }

    if best is None:
        return empty_features()

    pupil_x = best["pupil_x"] * max(width - 1, 1)
    radius_px = best["pupil_radius"] * min(width, height)
    band_half = max(3, int(radius_px * 2.3))
    x0 = max(0, int(round(pupil_x)) - band_half)
    x1 = min(width, int(round(pupil_x)) + band_half + 1)
    band = mask[:, x0:x1]
    rows = np.where(band.any(axis=1))[0]
    if len(rows) > 0:
        dark_span = float(rows.max() - rows.min() + 1)
    else:
        dark_span = float(best["bbox_h"])
    # This is a weak aperture proxy, not anatomical truth.
    openness = clamp01((dark_span / max(height, 1)) * 2.4)
    if best["quality"] < 0.18:
        openness *= 0.5

    return {
        "openness": openness,
        "pupil_x": clamp01(best["pupil_x"]),
        "pupil_y": clamp01(best["pupil_y"]),
        "pupil_radius": clamp01(best["pupil_radius"]),
        "quality": clamp01(best["quality"]),
    }


def empty_features() -> dict[str, float]:
    return {
        "openness": 0.0,
        "pupil_x": 0.0,
        "pupil_y": 0.0,
        "pupil_radius": 0.0,
        "quality": 0.0,
    }


def resolve_image_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def format_float(value: float) -> str:
    return f"{value:.6f}"


def process_manifest(manifest: Path, output: Path, limit: int | None) -> dict[str, object]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        input_fields = list(reader.fieldnames or [])

    if limit is not None:
        rows = rows[:limit]

    fields = list(input_fields)
    for field in WEAK_FIELDS:
        if field not in fields:
            fields.append(field)

    output_rows: list[dict[str, str]] = []
    left_quality: list[float] = []
    right_quality: list[float] = []
    openness_values: list[float] = []
    failures = 0
    for row in rows:
        row = dict(row)
        try:
            left = detect_eye_features(load_gray(resolve_image_path(manifest, row["left_file"])), "left")
            right = detect_eye_features(load_gray(resolve_image_path(manifest, row["right_file"])), "right")
        except Exception:
            left = empty_features()
            right = empty_features()
            failures += 1

        pair_quality = min(left["quality"], right["quality"])
        pair_openness = (left["openness"] + right["openness"]) * 0.5
        blink = 1.0 if pair_openness < 0.18 and pair_quality > 0.12 else 0.0
        for side, features in (("left", left), ("right", right)):
            row[f"weak_{side}_openness"] = format_float(features["openness"])
            row[f"weak_{side}_pupil_x"] = format_float(features["pupil_x"])
            row[f"weak_{side}_pupil_y"] = format_float(features["pupil_y"])
            row[f"weak_{side}_pupil_radius"] = format_float(features["pupil_radius"])
            row[f"weak_{side}_quality"] = format_float(features["quality"])
        row["weak_pair_quality"] = format_float(pair_quality)
        row["weak_blink"] = format_float(blink)
        row["weak_label_source"] = "classical_dark_component_v1"
        left_quality.append(left["quality"])
        right_quality.append(right["quality"])
        openness_values.extend([left["openness"], right["openness"]])
        output_rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "input_manifest": str(manifest.resolve()),
        "output_manifest": str(output.resolve()),
        "row_count": len(output_rows),
        "failures": failures,
        "left_quality_mean": float(np.mean(left_quality)) if left_quality else 0.0,
        "right_quality_mean": float(np.mean(right_quality)) if right_quality else 0.0,
        "pair_quality_median": float(np.median(np.minimum(left_quality, right_quality))) if left_quality and right_quality else 0.0,
        "openness_median": float(np.median(openness_values)) if openness_values else 0.0,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def draw_overlay(image_path: Path, features: dict[str, float], size: tuple[int, int]) -> Image.Image:
    image = Image.open(image_path).convert("RGB").resize(size)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if features["quality"] > 0.0:
        cx = features["pupil_x"] * (width - 1)
        cy = features["pupil_y"] * (height - 1)
        radius = max(2.0, features["pupil_radius"] * min(width, height))
        color = (0, 255, 80) if features["quality"] >= 0.25 else (255, 200, 0)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=2)
        draw.line((cx - 5, cy, cx + 5, cy), fill=color, width=1)
        draw.line((cx, cy - 5, cx, cy + 5), fill=color, width=1)
    draw.text((4, 4), f"open={features['openness']:.2f} q={features['quality']:.2f}", fill=(255, 255, 255))
    return image


def write_contact_sheet(manifest: Path, labeled_manifest: Path, output: Path, max_rows: int, every: int) -> None:
    with labeled_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = rows[:: max(1, every)][:max_rows]
    if not selected:
        return
    tile_w, tile_h = 180, 120
    label_h = 34
    canvas = Image.new("RGB", (tile_w * 2, (tile_h + label_h) * len(selected)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected):
        y0 = index * (tile_h + label_h)
        draw.text((4, y0 + 4), f"{row.get('stage','')} {row.get('sample_id','')[:34]}", fill=(0, 0, 0))
        for side, x0 in (("left", 0), ("right", tile_w)):
            image_path = resolve_image_path(manifest, row[f"{side}_file"])
            features = {
                "openness": float(row[f"weak_{side}_openness"]),
                "pupil_x": float(row[f"weak_{side}_pupil_x"]),
                "pupil_y": float(row[f"weak_{side}_pupil_y"]),
                "pupil_radius": float(row[f"weak_{side}_pupil_radius"]),
                "quality": float(row[f"weak_{side}_quality"]),
            }
            tile = draw_overlay(image_path, features, (tile_w, tile_h))
            canvas.paste(tile, (x0, y0 + label_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--contact-sheet-rows", type=int, default=24)
    parser.add_argument("--contact-sheet-every", type=int, default=80)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    output = args.output.resolve()
    summary = process_manifest(manifest, output, args.limit)
    if args.contact_sheet:
        write_contact_sheet(manifest, output, args.contact_sheet.resolve(), args.contact_sheet_rows, args.contact_sheet_every)
        summary["contact_sheet"] = str(args.contact_sheet.resolve())
        output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
