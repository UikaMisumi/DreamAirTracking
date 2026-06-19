#!/usr/bin/env python3
"""Run ONNX gaze inference on live BrokenEye eye-image streams without driving VRCFT."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


SOI = b"\xff\xd8"
EOI = b"\xff\xd9"

STAGE_PRESETS = {
    "five": ["center", "left", "right", "up", "down"],
    "nine": ["center", "left", "right", "up", "down", "left_up", "right_up", "left_down", "right_down"],
}


@dataclass(frozen=True)
class JpegFrame:
    side: str
    index: int
    received_at: float
    jpeg: bytes


def jpeg_stream(url: str, side: str, output: queue.Queue[JpegFrame], stop: threading.Event) -> None:
    index = 0
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            buffer = bytearray()
            while not stop.is_set():
                chunk = response.read(4096)
                if not chunk:
                    time.sleep(0.01)
                    continue
                buffer.extend(chunk)
                while True:
                    start = buffer.find(SOI)
                    end = buffer.find(EOI, start + 2) if start >= 0 else -1
                    if start < 0:
                        if len(buffer) > 1024 * 1024:
                            del buffer[:-4096]
                        break
                    if end < 0:
                        if start > 0:
                            del buffer[:start]
                        break
                    jpeg = bytes(buffer[start : end + 2])
                    del buffer[: end + 2]
                    index += 1
                    try:
                        output.put_nowait(JpegFrame(side, index, time.time(), jpeg))
                    except queue.Full:
                        try:
                            output.get_nowait()
                        except queue.Empty:
                            pass
                        output.put_nowait(JpegFrame(side, index, time.time(), jpeg))
    except Exception as exc:
        if not stop.is_set():
            output.put(JpegFrame(side, -1, time.time(), f"ERROR:{exc}".encode("utf-8", errors="replace")))


def preprocess(jpeg: bytes, image_size: int) -> np.ndarray:
    image = Image.open(BytesIO(jpeg)).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


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
    return np.array(layer["A"], dtype=np.float32), np.array(layer["b"], dtype=np.float32)


def apply_layer(raw: np.ndarray, layer: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if layer is None:
        return raw
    a, b = layer
    return (a @ raw + b).astype(np.float32)


def soft_clamp(value: np.ndarray, limit: float, knee: float) -> np.ndarray:
    if limit <= 0:
        return value
    knee = max(0.0, min(knee, limit * 0.98))
    magnitude = np.abs(value)
    sign = np.sign(value)
    linear = magnitude <= knee
    compressed = knee + (limit - knee) * np.tanh((magnitude - knee) / max(limit - knee, 1e-6))
    return np.where(linear, value, sign * compressed)


def apply_runtime_map(value: np.ndarray, offset: np.ndarray, gain: np.ndarray, clamp: float, clamp_mode: str, soft_knee: float) -> np.ndarray:
    mapped = (value - offset) * gain
    if clamp > 0 and clamp_mode == "hard":
        mapped = np.clip(mapped, -clamp, clamp)
    elif clamp > 0 and clamp_mode == "soft":
        mapped = soft_clamp(mapped, clamp, soft_knee)
    return mapped.astype(np.float32)


class OutputGazeMapper:
    """Post-map gaze for VRCFT output without changing model/raw calibration values."""

    def __init__(
        self,
        *,
        mode: str = "stable_center",
        deadzone: float = 0.015,
        gamma: float = 1.0,
        limit: float = 1.0,
        center_radius: float = 0.08,
        center_tau: float = 6.0,
        center_max_step: float = 0.025,
        min_openness: float = 0.35,
        min_confidence: float = 0.05,
    ) -> None:
        self.mode = mode
        self.deadzone = float(np.clip(deadzone, 0.0, max(limit * 0.8, 0.0)))
        self.gamma = float(np.clip(gamma, 0.35, 3.0))
        self.limit = float(max(limit, 1e-6))
        self.center_radius = float(max(center_radius, 0.0))
        self.center_tau = float(max(center_tau, 1e-3))
        self.center_max_step = float(max(center_max_step, 0.0))
        self.min_openness = float(np.clip(min_openness, 0.0, 1.0))
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.center = np.zeros(2, dtype=np.float32)
        self._last_value: np.ndarray | None = None
        self._last_time: float | None = None

    def update(
        self,
        value: np.ndarray,
        *,
        timestamp: float | None = None,
        confidence: float = 1.0,
        openness: float = 1.0,
    ) -> np.ndarray:
        value = value.astype(np.float32)
        now = time.time() if timestamp is None else timestamp
        self._update_center(value, now, confidence, openness)
        centered = value - self.center if self.mode != "off" else value
        return self._apply_curve(centered).astype(np.float32)

    def _update_center(self, value: np.ndarray, now: float, confidence: float, openness: float) -> None:
        if self.mode != "stable_center":
            self._last_value = value
            self._last_time = now
            return
        if confidence < self.min_confidence or openness < self.min_openness:
            self._last_value = value
            self._last_time = now
            return
        if float(np.linalg.norm(value)) > self.center_radius:
            self._last_value = value
            self._last_time = now
            return
        if self._last_value is not None and self.center_max_step > 0:
            if float(np.linalg.norm(value - self._last_value)) > self.center_max_step:
                self._last_value = value
                self._last_time = now
                return
        dt = 1.0 / 60.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.25))
        alpha = 1.0 - float(np.exp(-dt / self.center_tau))
        self.center = ((1.0 - alpha) * self.center + alpha * value).astype(np.float32)
        self._last_value = value
        self._last_time = now

    def _apply_curve(self, value: np.ndarray) -> np.ndarray:
        if self.deadzone <= 0.0 and abs(self.gamma - 1.0) < 1e-6:
            return np.clip(value, -self.limit, self.limit)
        magnitude = np.abs(value)
        sign = np.sign(value)
        denom = max(self.limit - self.deadzone, 1e-6)
        normalized = np.clip((magnitude - self.deadzone) / denom, 0.0, 1.0)
        curved = normalized**self.gamma
        return sign * curved * self.limit


def update_smooth(previous: np.ndarray | None, current: np.ndarray, alpha: float, max_step: float) -> np.ndarray:
    if previous is None:
        return current
    target = alpha * current + (1.0 - alpha) * previous
    if max_step > 0:
        delta = target - previous
        norm = float(np.linalg.norm(delta))
        if norm > max_step:
            target = previous + delta * (max_step / norm)
    return target.astype(np.float32)


def save_pair_snapshots(output_dir: Path, seq: int, left: JpegFrame, right: JpegFrame) -> tuple[str, str]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    left_path = frames_dir / f"{seq:06d}_left.jpg"
    right_path = frames_dir / f"{seq:06d}_right.jpg"
    left_path.write_bytes(left.jpeg)
    right_path.write_bytes(right.jpeg)
    return str(left_path.resolve()), str(right_path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("runs/gaze_image_v1_cnn_validsessions_60e/gaze_baseline.onnx"))
    parser.add_argument("--metadata", type=Path, default=Path("runs/gaze_image_v1_cnn_validsessions_60e/gaze_baseline.metadata.json"))
    parser.add_argument("--quick-layer", type=Path, default=Path("runs/gaze_image_v1_cnn_validsessions_60e/eval/quick_calibration_layer.json"))
    parser.add_argument("--quick-layer-session")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/live_gaze_dry_run"))
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--direction-preset", choices=("five", "nine"), help="Show visible gaze prompts and tag CSV rows by stage.")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="Prompt settle time before capturing each guided stage.")
    parser.add_argument("--stage-seconds", type=float, default=3.0, help="Capture time per guided stage.")
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--center-offset-x", type=float, default=0.0, help="Subtract this x offset after optional quick layer.")
    parser.add_argument("--center-offset-y", type=float, default=0.0, help="Subtract this y offset after optional quick layer.")
    parser.add_argument("--x-gain", type=float, default=1.0, help="Runtime x gain after center offset.")
    parser.add_argument("--y-gain", type=float, default=1.0, help="Runtime y gain after center offset.")
    parser.add_argument("--clamp", type=float, default=1.0, help="Clamp mapped gaze to [-clamp, clamp]; <=0 disables clamp.")
    parser.add_argument("--clamp-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--soft-knee", type=float, default=0.72, help="Soft clamp remains linear up to this absolute value.")
    parser.add_argument("--max-step", type=float, default=0.0, help="Optional maximum smoothed 2D step per output frame; <=0 disables.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--snapshot-every", type=int, default=30)
    parser.add_argument("--no-quick-layer", action="store_true")
    args = parser.parse_args()

    metadata = json.load(args.metadata.open("r", encoding="utf-8"))
    image_size = int(metadata.get("image_size", 96))
    session = ort.InferenceSession(str(args.onnx.resolve()), providers=["CPUExecutionProvider"])
    layer = None if args.no_quick_layer else load_quick_layer(args.quick_layer.resolve(), args.quick_layer_session)
    runtime_offset = np.array([args.center_offset_x, args.center_offset_y], dtype=np.float32)
    runtime_gain = np.array([args.x_gain, args.y_gain], dtype=np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"live_gaze_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    status_path = args.output_dir / "latest_status.json"
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

    fields = [
        "sequence",
        "timestamp",
        "stage",
        "stage_elapsed",
        "left_index",
        "right_index",
        "delta_ms",
        "raw_x",
        "raw_y",
        "corrected_x",
        "corrected_y",
        "smooth_x",
        "smooth_y",
        "left_snapshot",
        "right_snapshot",
    ]
    left_frames: list[JpegFrame] = []
    right_frames: list[JpegFrame] = []
    smooth: np.ndarray | None = None
    sequence = 0
    started = time.time()
    guided_started = started
    current_stage = ""
    stage_started = started
    capture_enabled = args.direction_preset is None
    last_error: str | None = None
    print(f"BrokenEye left:  {left_url}")
    print(f"BrokenEye right: {right_url}")
    print(f"ONNX: {args.onnx.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            stages = STAGE_PRESETS.get(args.direction_preset or "", [])
            stage_index = -1
            stage_capture_until = started + args.duration_seconds
            if stages:
                args.duration_seconds = len(stages) * (args.settle_seconds + args.stage_seconds)
                print("")
                print("Guided direction check")
                print("Look at each prompted direction. Keep your head still; move only your eyes if possible.")
                print("")

            while time.time() - started < args.duration_seconds:
                now = time.time()
                if stages:
                    elapsed = now - guided_started
                    block = args.settle_seconds + args.stage_seconds
                    next_stage_index = min(int(elapsed // block), len(stages) - 1)
                    stage_phase = elapsed - next_stage_index * block
                    if next_stage_index != stage_index:
                        stage_index = next_stage_index
                        current_stage = stages[stage_index]
                        stage_started = now
                        stage_capture_until = now + args.settle_seconds + args.stage_seconds
                        label = current_stage.upper().replace("_", " ")
                        print("")
                        print("=" * 56)
                        print(f"LOOK {label}")
                        print("=" * 56)
                    capture_enabled = stage_phase >= args.settle_seconds
                    if not capture_enabled:
                        remaining = max(0.0, args.settle_seconds - stage_phase)
                        if int(remaining * 10) % 10 == 0:
                            print(f"settle {remaining:0.1f}s", flush=True)

                try:
                    frame = q.get(timeout=2.0)
                except queue.Empty:
                    last_error = "Timed out waiting for BrokenEye frames."
                    break
                if frame.index < 0:
                    last_error = frame.jpeg.decode("utf-8", errors="replace")
                    break
                target = left_frames if frame.side == "left" else right_frames
                target.append(frame)
                del target[:-10]

                pair = choose_pair(left_frames, right_frames, args.max_delta_ms)
                if pair is None:
                    continue
                left, right = pair
                left_frames.remove(left)
                right_frames.remove(right)
                if not capture_enabled:
                    continue
                sequence += 1

                left_img = preprocess(left.jpeg, image_size)
                right_img = preprocess(right.jpeg, image_size)
                image = np.stack([left_img, right_img], axis=0)[None, ...].astype(np.float32)
                raw = session.run(["gaze_xy"], {"image": image})[0][0].astype(np.float32)
                corrected = apply_runtime_map(apply_layer(raw, layer), runtime_offset, runtime_gain, args.clamp, args.clamp_mode, args.soft_knee)
                smooth = update_smooth(smooth, corrected, args.ema_alpha, args.max_step)
                delta_ms = abs(left.received_at - right.received_at) * 1000.0
                left_snapshot = ""
                right_snapshot = ""
                if args.snapshot_every > 0 and sequence % args.snapshot_every == 0:
                    left_snapshot, right_snapshot = save_pair_snapshots(args.output_dir, sequence, left, right)

                writer.writerow(
                    {
                        "sequence": sequence,
                        "timestamp": f"{time.time():.6f}",
                        "stage": current_stage,
                        "stage_elapsed": f"{time.time() - stage_started:0.3f}",
                        "left_index": left.index,
                        "right_index": right.index,
                        "delta_ms": f"{delta_ms:.3f}",
                        "raw_x": f"{raw[0]:.6f}",
                        "raw_y": f"{raw[1]:.6f}",
                        "corrected_x": f"{corrected[0]:.6f}",
                        "corrected_y": f"{corrected[1]:.6f}",
                        "smooth_x": f"{smooth[0]:.6f}",
                        "smooth_y": f"{smooth[1]:.6f}",
                        "left_snapshot": left_snapshot,
                        "right_snapshot": right_snapshot,
                    }
                )
                if args.print_every > 0 and sequence % args.print_every == 0:
                    print(
                        f"{sequence:06d} dt={delta_ms:0.1f} "
                        f"raw=({raw[0]:+0.3f},{raw[1]:+0.3f}) "
                        f"corr=({corrected[0]:+0.3f},{corrected[1]:+0.3f}) "
                        f"smooth=({smooth[0]:+0.3f},{smooth[1]:+0.3f})"
                    )
    finally:
        stop.set()

    status = {
        "state": "complete" if last_error is None and sequence > 0 else "failed",
        "error": last_error,
        "csv": str(csv_path.resolve()),
        "sequence_count": sequence,
        "duration_seconds": time.time() - started,
        "onnx": str(args.onnx.resolve()),
        "quick_layer": None if layer is None else str(args.quick_layer.resolve()),
        "runtime_map": {
            "center_offset_x": args.center_offset_x,
            "center_offset_y": args.center_offset_y,
            "x_gain": args.x_gain,
            "y_gain": args.y_gain,
            "clamp": args.clamp,
            "clamp_mode": args.clamp_mode,
            "soft_knee": args.soft_knee,
            "max_step": args.max_step,
        },
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["state"] == "complete" else 2


def choose_pair(left_frames: list[JpegFrame], right_frames: list[JpegFrame], max_delta_ms: float) -> tuple[JpegFrame, JpegFrame] | None:
    best: tuple[JpegFrame, JpegFrame] | None = None
    best_delta = float("inf")
    for left in left_frames:
        for right in right_frames:
            delta = abs(left.received_at - right.received_at) * 1000.0
            if delta < best_delta:
                best = (left, right)
                best_delta = delta
    if best is None or best_delta > max_delta_ms:
        return None
    return best


if __name__ == "__main__":
    raise SystemExit(main())
