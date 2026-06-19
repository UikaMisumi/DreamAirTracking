#!/usr/bin/env python3
"""Render overlays for EyeCropNormalizer debugging."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eye_normalization import EyeNormalizationOptions, EyeNormalizationResult, EyeNormalizationState  # noqa: E402
from live_gaze_dry_run import JpegFrame, choose_pair, jpeg_stream  # noqa: E402


@dataclass(frozen=True)
class DebugPair:
    sequence: int
    left: JpegFrame
    right: JpegFrame
    left_result: EyeNormalizationResult
    right_result: EyeNormalizationResult


def capture_live_pairs(args: argparse.Namespace, options: EyeNormalizationOptions) -> list[DebugPair]:
    q: queue.Queue[JpegFrame] = queue.Queue(maxsize=80)
    stop = threading.Event()
    left_url = f"http://{args.host}:{args.port}/eye/left"
    right_url = f"http://{args.host}:{args.port}/eye/right"
    threads = [
        threading.Thread(target=jpeg_stream, args=(left_url, "left", q, stop), daemon=True),
        threading.Thread(target=jpeg_stream, args=(right_url, "right", q, stop), daemon=True),
    ]
    for thread in threads:
        thread.start()

    left_options = side_options(args, "left")
    right_options = side_options(args, "right")
    left_state = EyeNormalizationState()
    right_state = EyeNormalizationState()
    left_frames: list[JpegFrame] = []
    right_frames: list[JpegFrame] = []
    pairs: list[DebugPair] = []
    started = time.time()
    sequence = 0
    try:
        while time.time() - started < args.duration_seconds and len(pairs) < args.max_pairs:
            try:
                frame = q.get(timeout=2.0)
            except queue.Empty:
                break
            if frame.index < 0:
                raise RuntimeError(frame.jpeg.decode("utf-8", errors="replace"))

            target = left_frames if frame.side == "left" else right_frames
            target.append(frame)
            del target[:-10]
            pair = choose_pair(left_frames, right_frames, args.max_delta_ms)
            if pair is None:
                continue
            left, right = pair
            left_frames.remove(left)
            right_frames.remove(right)
            sequence += 1
            left_result = left_state.update(left.jpeg, args.image_size, left_options, openness=1.0, aperture_height=64)
            right_result = right_state.update(right.jpeg, args.image_size, right_options, openness=1.0, aperture_height=64)
            pairs.append(DebugPair(sequence, left, right, left_result, right_result))
    finally:
        stop.set()
    return pairs


def render_pair(pair: DebugPair, image_size: int, left_options: EyeNormalizationOptions, right_options: EyeNormalizationOptions) -> Image.Image:
    panels = [
        render_original(pair.left.jpeg, pair.left_result, image_size, left_options, "left original"),
        render_normalized(pair.left_result, image_size, left_options, "left normalized"),
        render_original(pair.right.jpeg, pair.right_result, image_size, right_options, "right original"),
        render_normalized(pair.right_result, image_size, right_options, "right normalized"),
    ]
    width = image_size * len(panels)
    height = image_size + 40
    row = Image.new("RGB", (width, height), "white")
    for index, panel in enumerate(panels):
        row.paste(panel, (index * image_size, 0))
    draw = ImageDraw.Draw(row)
    draw.text((6, image_size + 4), f"seq {pair.sequence}", fill=(0, 0, 0))
    draw.text(
        (72, image_size + 4),
        format_metadata("L", pair.left_result),
        fill=(0, 0, 0),
    )
    draw.text(
        (72, image_size + 20),
        format_metadata("R", pair.right_result),
        fill=(0, 0, 0),
    )
    return row


def render_original(
    jpeg: bytes,
    result: EyeNormalizationResult,
    image_size: int,
    options: EyeNormalizationOptions,
    title: str,
) -> Image.Image:
    image = Image.open(BytesIO(jpeg)).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    target = (int(options.target_x * image_size), int(options.target_y * image_size))
    pupil = (int(result.metadata.pupil_x * image_size), int(result.metadata.pupil_y * image_size))
    draw_cross(draw, target, (40, 160, 255), 7)
    draw_cross(draw, pupil, (255, 70, 70), 7)
    draw.line([target, pupil], fill=(255, 210, 0), width=2)
    draw.text((4, 4), title, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image


def render_normalized(result: EyeNormalizationResult, image_size: int, options: EyeNormalizationOptions, title: str) -> Image.Image:
    gray = np.clip(result.image * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(gray, mode="L").convert("RGB").resize((image_size, image_size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    center = (int(options.target_x * image_size), int(options.target_y * image_size))
    draw_cross(draw, center, (40, 160, 255), 7)
    draw.text((4, 4), title, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return image


def draw_cross(draw: ImageDraw.ImageDraw, point: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    draw.line((x - radius - 3, y, x + radius + 3, y), fill=color, width=2)
    draw.line((x, y - radius - 3, x, y + radius + 3), fill=color, width=2)


def format_metadata(label: str, result: EyeNormalizationResult) -> str:
    meta = result.metadata
    reason = "" if not meta.drop_reason else f" reason={meta.drop_reason}"
    return (
        f"{label}: found={int(meta.found)} q={meta.confidence:.3f} "
        f"pupil=({meta.pupil_x:.3f},{meta.pupil_y:.3f}) "
        f"shift=({meta.shift_x:.3f},{meta.shift_y:.3f}) scale={meta.scale:.3f}{reason}"
    )


def save_contact_sheet(
    pairs: list[DebugPair],
    output_dir: Path,
    image_size: int,
    left_options: EyeNormalizationOptions,
    right_options: EyeNormalizationOptions,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [render_pair(pair, image_size, left_options, right_options) for pair in pairs]
    if not rows:
        raise RuntimeError("No synchronized pairs captured.")
    sheet = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * row.height))
    path = output_dir / f"eye_normalization_debug_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    sheet.save(path, quality=92)
    return path


def build_options(args: argparse.Namespace) -> EyeNormalizationOptions:
    return EyeNormalizationOptions(
        mode=args.normalization_mode,
        target_x=args.normalization_target_x,
        target_y=args.normalization_target_y,
        max_shift_x=args.normalization_max_shift_x,
        max_shift_y=args.normalization_max_shift_y,
        scale_min=args.normalization_scale_min,
        scale_max=args.normalization_scale_max,
        hold_frames=args.normalization_hold_frames,
    )


def side_options(args: argparse.Namespace, side: str) -> EyeNormalizationOptions:
    x = args.normalization_target_x
    y = args.normalization_target_y
    if side == "left":
        x = args.normalization_left_target_x if args.normalization_left_target_x is not None else x
        y = args.normalization_left_target_y if args.normalization_left_target_y is not None else y
    else:
        x = args.normalization_right_target_x if args.normalization_right_target_x is not None else x
        y = args.normalization_right_target_y if args.normalization_right_target_y is not None else y
    return EyeNormalizationOptions(
        mode=args.normalization_mode,
        target_x=x,
        target_y=y,
        max_shift_x=args.normalization_max_shift_x,
        max_shift_y=args.normalization_max_shift_y,
        scale_min=args.normalization_scale_min,
        scale_max=args.normalization_scale_max,
        hold_frames=args.normalization_hold_frames,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eye_normalization_debug"))
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--normalization-mode", choices=("off", "pupil_center"), default="pupil_center")
    parser.add_argument("--normalization-target-x", type=float, default=0.50)
    parser.add_argument("--normalization-target-y", type=float, default=0.54)
    parser.add_argument("--normalization-left-target-x", type=float)
    parser.add_argument("--normalization-right-target-x", type=float)
    parser.add_argument("--normalization-left-target-y", type=float)
    parser.add_argument("--normalization-right-target-y", type=float)
    parser.add_argument("--normalization-max-shift-x", type=float, default=0.18)
    parser.add_argument("--normalization-max-shift-y", type=float, default=0.16)
    parser.add_argument("--normalization-scale-min", type=float, default=0.86)
    parser.add_argument("--normalization-scale-max", type=float, default=1.18)
    parser.add_argument("--normalization-hold-frames", type=int, default=5)
    args = parser.parse_args()

    options = build_options(args)
    pairs = capture_live_pairs(args, options)
    path = save_contact_sheet(pairs, args.output_dir, args.image_size, side_options(args, "left"), side_options(args, "right"))
    print(f"Wrote {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
