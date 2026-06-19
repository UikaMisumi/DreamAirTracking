#!/usr/bin/env python3
"""Live gaze prediction runtime for BrokenEye streams.

Default mode is safe: write CSV/logs only. Pass --udp-port explicitly to emit
DreamAirTracking bridge packets.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_gaze_dry_run import (  # noqa: E402
    JpegFrame,
    OutputGazeMapper,
    apply_runtime_map,
    choose_pair,
    jpeg_stream,
    preprocess,
    save_pair_snapshots,
    update_smooth,
)
from eye_normalization import (  # noqa: E402
    EyeNormalizationMetadata,
    EyeNormalizationOptions,
    EyeNormalizationState,
)
from wear_templates import load_store, match_template, signature_from_csv  # noqa: E402


DEFAULT_ONNX = Path("runs/gaze_mobilenetv3_small_live_round1_round2/gaze_baseline.onnx")
DEFAULT_METADATA = Path("runs/gaze_mobilenetv3_small_live_round1_round2/gaze_baseline.metadata.json")


@dataclass
class OpennessOptions:
    mode: str
    fixed: float
    roi_x0: float
    roi_y0: float
    roi_x1: float
    roi_y1: float
    adaptive_percentile: float
    adaptive_margin: int
    max_threshold: int
    row_smoothing_radius: int
    aperture_gate_fraction: float
    closed_peak_fraction: float
    open_peak_fraction: float
    min_aperture_height_for_partial_open: int


@dataclass
class OpennessDetection:
    found: bool
    openness: float
    aperture_height: int
    peak_dark_fraction: float
    threshold: int
    reason: str


class OpennessFilter:
    def __init__(self) -> None:
        self.left = EyeOpennessFilterState()
        self.right = EyeOpennessFilterState()

    def update(self, left: OpennessDetection, right: OpennessDetection) -> tuple[float, float]:
        raw_left = self.left.normalize(left)
        raw_right = self.right.normalize(right)
        both_closed = raw_left <= 0.20 and raw_right <= 0.20
        return self.left.smooth(raw_left, both_closed), self.right.smooth(raw_right, both_closed)


class CalibratedOpennessFilter:
    def __init__(self) -> None:
        self.left = 1.0
        self.right = 1.0

    def update(self, left: float, right: float) -> tuple[float, float]:
        both_closed = left <= 0.20 and right <= 0.20
        self.left = smooth_openness(self.left, left, both_closed)
        self.right = smooth_openness(self.right, right, both_closed)
        return self.left, self.right


def smooth_openness(previous: float, target: float, both_closed: bool) -> float:
    alpha = 0.75 if target >= previous else 0.90 if both_closed else 0.65
    return float(np.clip(previous + (target - previous) * alpha, 0.0, 1.0))


class EyeOpennessFilterState:
    def __init__(self) -> None:
        self.open_peak = 0.35
        self.open_height: float | None = None
        self.previous = 1.0

    def normalize(self, detection: OpennessDetection) -> float:
        if not detection.found:
            return 0.0
        if detection.aperture_height < 18 and detection.peak_dark_fraction <= 0.30:
            return 0.0
        if detection.aperture_height >= 18 and detection.peak_dark_fraction >= 0.30:
            self.open_peak = max(0.30, self.open_peak + (detection.peak_dark_fraction - self.open_peak) * 0.08)
            if self.open_height is None:
                self.open_height = float(detection.aperture_height)
            else:
                alpha = 0.08 if detection.aperture_height >= self.open_height else 0.01
                self.open_height += (detection.aperture_height - self.open_height) * alpha

        denominator = max(self.open_peak - 0.18, 0.001)
        peak_open = (detection.peak_dark_fraction - 0.18) / denominator
        height_open = detection.aperture_height / self.open_height if self.open_height and self.open_height > 0 else detection.openness
        raw = (peak_open * 0.70) + (height_open * 0.30)
        raw = float(np.clip(raw, 0.0, 1.0) ** 1.35)
        if detection.openness >= 0.90 and detection.aperture_height >= 18:
            raw = max(raw, 0.75)
        return float(np.clip(raw, 0.0, 1.0))

    def smooth(self, target: float, both_closed: bool) -> float:
        alpha = 0.75 if target >= self.previous else 0.90 if both_closed else 0.55
        self.previous = float(np.clip(self.previous + (target - self.previous) * alpha, 0.0, 1.0))
        return self.previous


def decode_gray(jpeg: bytes) -> np.ndarray:
    image = Image.open(BytesIO(jpeg)).convert("L")
    return np.asarray(image, dtype=np.uint8)


def detect_openness(jpeg: bytes, options: OpennessOptions) -> OpennessDetection:
    if options.mode == "fixed":
        value = float(np.clip(options.fixed, 0.0, 1.0))
        return OpennessDetection(True, value, 0, value, 0, "fixed")

    try:
        image = decode_gray(jpeg)
    except Exception as exc:
        return OpennessDetection(False, 0.0, 0, 0.0, 0, f"decode failed: {exc}")

    height, width = image.shape
    x0 = int(np.clip(options.roi_x0, 0.0, 0.98) * width)
    y0 = int(np.clip(options.roi_y0, 0.0, 0.98) * height)
    x1 = int(np.clip(options.roi_x1, 0.02, 1.0) * width)
    y1 = int(np.clip(options.roi_y1, 0.02, 1.0) * height)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return OpennessDetection(False, 0.0, 0, 0.0, 0, "empty roi")

    roi = image[y0:y1, x0:x1]
    threshold = int(
        min(
            np.percentile(roi, np.clip(options.adaptive_percentile, 0.0, 1.0) * 100.0)
            + options.adaptive_margin,
            options.max_threshold,
        )
    )
    dark_rows = np.mean(roi <= threshold, axis=1).astype(np.float32)
    radius = max(0, options.row_smoothing_radius)
    if radius > 0 and dark_rows.size > 0:
        kernel = np.ones((radius * 2) + 1, dtype=np.float32) / ((radius * 2) + 1)
        dark_rows = np.convolve(dark_rows, kernel, mode="same")

    peak = float(np.max(dark_rows)) if dark_rows.size else 0.0
    active = np.flatnonzero(dark_rows >= options.aperture_gate_fraction)
    aperture = int(active[-1] - active[0] + 1) if active.size else 0
    openness = normalize_openness_peak(peak, options.closed_peak_fraction, options.open_peak_fraction)
    if aperture == 0 and peak < 0.30:
        openness = 0.0
    elif aperture > 0 and aperture < options.min_aperture_height_for_partial_open and peak < options.open_peak_fraction:
        openness = 0.0

    return OpennessDetection(True, openness, aperture, peak, threshold, "ok" if aperture > 0 else "no coherent dark aperture")


def normalize_openness_peak(peak: float, closed: float, open_value: float) -> float:
    if open_value <= closed:
        return 1.0 if peak >= open_value else 0.0
    return float(np.clip((peak - closed) / (open_value - closed), 0.0, 1.0))


def load_openness_calibration(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return json.load(path.resolve().open("r", encoding="utf-8"))


def calibrated_openness(detection: OpennessDetection, calibration: dict[str, object] | None, side: str) -> float:
    if calibration is None:
        return detection.openness
    side_payload = calibration.get(side)
    if not isinstance(side_payload, dict):
        return detection.openness
    open_score = float(side_payload.get("open_score_median", 0.0))
    closed_score = float(side_payload.get("closed_score_median", 0.0))
    denominator = open_score - closed_score
    if denominator <= 1e-6:
        return detection.openness
    score = openness_score(detection)
    return float(np.clip((score - closed_score) / denominator, 0.0, 1.0))


def openness_score(detection: OpennessDetection) -> float:
    return float(detection.aperture_height * detection.peak_dark_fraction)


def apply_center_bias(value: np.ndarray, amount: float, start: float, limit: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.75))
    if amount <= 0.0:
        return value.astype(np.float32)
    start = max(0.0, min(start, max(limit, 1e-6) * 0.98))
    end = max(limit, start + 1e-6)
    magnitude = np.abs(value)
    t = np.clip((magnitude - start) / max(end - start, 1e-6), 0.0, 1.0)
    smooth_t = t * t * (3.0 - 2.0 * t)
    scale = 1.0 - amount * smooth_t
    return (value * scale).astype(np.float32)


def boost_open_openness(value: float, gamma: float, full_open_threshold: float, knee: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    knee = float(np.clip(knee, 0.0, 0.95))
    full_open_threshold = float(np.clip(full_open_threshold, knee + 0.01, 1.0))
    if value <= knee:
        return value
    if value >= full_open_threshold:
        return 1.0
    gamma = float(np.clip(gamma, 0.20, 1.50))
    normalized = (value - knee) / max(full_open_threshold - knee, 1e-6)
    boosted = knee + (normalized**gamma) * (1.0 - knee)
    return float(np.clip(boosted, 0.0, 1.0))


def bridge_eye(
    raw_x: float,
    raw_y: float,
    mapped_x: float,
    mapped_y: float,
    smooth_x: float,
    smooth_y: float,
    confidence: float,
    openness: float,
    calibrated_openness_value: float,
    openness_detection: OpennessDetection | None,
    normalization: EyeNormalizationMetadata,
) -> dict[str, object]:
    aperture_openness = openness_detection.openness if openness_detection is not None else openness
    aperture_peak = openness_detection.peak_dark_fraction if openness_detection is not None else 0.0
    aperture_height = openness_detection.aperture_height if openness_detection is not None else 0
    aperture_reason = openness_detection.reason if openness_detection is not None else ""
    return {
        "found": True,
        "confidence": confidence,
        "rawX": raw_x,
        "rawY": raw_y,
        "normalizedX": smooth_x,
        "normalizedY": smooth_y,
        "openness": openness,
        "pupilOpenness": openness,
        "apertureOpenness": aperture_openness,
        "calibratedOpenness": calibrated_openness_value,
        "outputOpenness": openness,
        "aperturePeakDarkFraction": aperture_peak,
        "apertureHeight": aperture_height,
        "apertureReason": aperture_reason,
        "rawNormalizedX": raw_x,
        "rawNormalizedY": raw_y,
        "monocularNormalizedX": mapped_x,
        "monocularNormalizedY": mapped_y,
        "pupilDiameterFound": False,
        "pupilDiameterQuality": False,
        "pupilDiameterNormalized": 0.5,
        "normalizationFound": normalization.found,
        "normalizationConfidence": normalization.confidence,
        "normalizationShiftX": normalization.shift_x,
        "normalizationShiftY": normalization.shift_y,
        "normalizationScale": normalization.scale,
        "normalizationPupilX": normalization.pupil_x,
        "normalizationPupilY": normalization.pupil_y,
        "normalizationDropReason": normalization.drop_reason,
    }


def bridge_packet(
    sequence: int,
    delta_ms: float,
    raw_x: float,
    raw_y: float,
    mapped_x: float,
    mapped_y: float,
    smooth_x: float,
    smooth_y: float,
    confidence: float,
    left_openness: float,
    right_openness: float,
    left_calibrated_openness: float,
    right_calibrated_openness: float,
    left_openness_detection: OpennessDetection | None,
    right_openness_detection: OpennessDetection | None,
    normalization_mode: str,
    left_normalization: EyeNormalizationMetadata,
    right_normalization: EyeNormalizationMetadata,
) -> dict[str, object]:
    left_eye = bridge_eye(
        raw_x,
        raw_y,
        mapped_x,
        mapped_y,
        smooth_x,
        smooth_y,
        confidence,
        left_openness,
        left_calibrated_openness,
        left_openness_detection,
        left_normalization,
    )
    right_eye = bridge_eye(
        raw_x,
        raw_y,
        mapped_x,
        mapped_y,
        smooth_x,
        smooth_y,
        confidence,
        right_openness,
        right_calibrated_openness,
        right_openness_detection,
        right_normalization,
    )
    return {
        "sequence": sequence,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z",
        "deltaMs": delta_ms,
        "eyeTrackingEnabled": True,
        "pupilDiameterEnabled": False,
        "pupilDiameterMode": "off",
        "normalizationMode": normalization_mode,
        "left": left_eye,
        "right": right_eye,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predict_live"))
    parser.add_argument("--duration-seconds", "--duration", dest="duration_seconds", type=float, default=0.0, help="0 means run until Ctrl+C.")
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument("--center-offset-x", type=float, default=0.119408)
    parser.add_argument("--center-offset-y", type=float, default=0.000840)
    parser.add_argument("--x-gain", type=float, default=1.0)
    parser.add_argument("--y-gain", type=float, default=1.0)
    parser.add_argument("--clamp", type=float, default=0.95)
    parser.add_argument("--clamp-mode", choices=("hard", "soft"), default="soft")
    parser.add_argument("--soft-knee", type=float, default=0.72)
    parser.add_argument("--center-bias", type=float, default=0.10, help="Pull non-center gaze slightly toward center after runtime mapping; 0 disables.")
    parser.add_argument("--center-bias-start", type=float, default=0.25, help="Absolute gaze value where center bias starts.")
    parser.add_argument("--output-map-mode", choices=("off", "stable_center"), default="stable_center")
    parser.add_argument("--output-deadzone", type=float, default=0.015)
    parser.add_argument("--output-curve-gamma", type=float, default=1.0)
    parser.add_argument("--output-center-radius", type=float, default=0.08)
    parser.add_argument("--output-center-tau", type=float, default=6.0)
    parser.add_argument("--output-center-max-step", type=float, default=0.025)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-step", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--openness", type=float, default=1.0)
    parser.add_argument("--openness-mode", choices=("image", "fixed"), default="image")
    parser.add_argument("--openness-calibration", type=Path, help="Optional open/closed calibration JSON from live_openness_calibration.py.")
    parser.add_argument("--openness-roi-x0", type=float, default=0.15)
    parser.add_argument("--openness-roi-y0", type=float, default=0.12)
    parser.add_argument("--openness-roi-x1", type=float, default=0.85)
    parser.add_argument("--openness-roi-y1", type=float, default=0.88)
    parser.add_argument("--openness-adaptive-percentile", type=float, default=0.18)
    parser.add_argument("--openness-adaptive-margin", type=int, default=8)
    parser.add_argument("--openness-max-threshold", type=int, default=100)
    parser.add_argument("--openness-row-smoothing-radius", type=int, default=2)
    parser.add_argument("--openness-aperture-gate", type=float, default=0.25)
    parser.add_argument("--openness-closed-peak", type=float, default=0.25)
    parser.add_argument("--openness-open-peak", type=float, default=0.35)
    parser.add_argument("--openness-min-aperture", type=int, default=10)
    parser.add_argument("--openness-boost-gamma", type=float, default=0.75, help="Gamma below 1 makes open eyes reach higher openness sooner.")
    parser.add_argument("--openness-full-open-threshold", type=float, default=0.82, help="Openness at or above this value is reported as fully open.")
    parser.add_argument("--openness-boost-knee", type=float, default=0.25, help="Do not boost openness below this value.")
    parser.add_argument(
        "--normalization-mode",
        choices=("off", "probe_only", "pupil_center"),
        default="off",
        help="off disables pupil probing; probe_only logs pupil metadata while feeding original images; pupil_center applies image transforms.",
    )
    parser.add_argument("--normalization-target-x", type=float, default=0.50)
    parser.add_argument("--normalization-target-y", type=float, default=0.54)
    parser.add_argument("--normalization-left-target-x", type=float, help="Optional left-eye target x. Defaults to --normalization-target-x.")
    parser.add_argument("--normalization-right-target-x", type=float, help="Optional right-eye target x. Defaults to --normalization-target-x.")
    parser.add_argument("--normalization-left-target-y", type=float, help="Optional left-eye target y. Defaults to --normalization-target-y.")
    parser.add_argument("--normalization-right-target-y", type=float, help="Optional right-eye target y. Defaults to --normalization-target-y.")
    parser.add_argument("--normalization-max-shift-x", type=float, default=0.18)
    parser.add_argument("--normalization-max-shift-y", type=float, default=0.16)
    parser.add_argument("--normalization-scale-min", type=float, default=0.86)
    parser.add_argument("--normalization-scale-max", type=float, default=1.18)
    parser.add_argument("--normalization-hold-frames", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--snapshot-every", type=int, default=120)
    parser.add_argument("--udp-port", type=int, help="Optional DreamAirTracking VRCFT bridge UDP port, usually 9400.")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--monitor-udp-port", type=int, default=0, help="Optional DreamAirTracking App monitor UDP port, usually 9401; 0 disables.")
    parser.add_argument("--monitor-udp-host", default="127.0.0.1")
    parser.add_argument("--wear-template-path", type=Path, help="Optional wear_templates.json path. Writes a wear-template match report after capture.")
    args = parser.parse_args()

    metadata = json.load(args.metadata.resolve().open("r", encoding="utf-8"))
    image_size = int(metadata.get("image_size", 96))
    session = ort.InferenceSession(str(args.onnx.resolve()), providers=["CPUExecutionProvider"])
    runtime_offset = np.array([args.center_offset_x, args.center_offset_y], dtype=np.float32)
    runtime_gain = np.array([args.x_gain, args.y_gain], dtype=np.float32)
    output_mapper = OutputGazeMapper(
        mode=args.output_map_mode,
        deadzone=args.output_deadzone,
        gamma=args.output_curve_gamma,
        limit=args.clamp if args.clamp > 0 else 1.0,
        center_radius=args.output_center_radius,
        center_tau=args.output_center_tau,
        center_max_step=args.output_center_max_step,
    )
    openness_options = OpennessOptions(
        mode=args.openness_mode,
        fixed=args.openness,
        roi_x0=args.openness_roi_x0,
        roi_y0=args.openness_roi_y0,
        roi_x1=args.openness_roi_x1,
        roi_y1=args.openness_roi_y1,
        adaptive_percentile=args.openness_adaptive_percentile,
        adaptive_margin=args.openness_adaptive_margin,
        max_threshold=args.openness_max_threshold,
        row_smoothing_radius=args.openness_row_smoothing_radius,
        aperture_gate_fraction=args.openness_aperture_gate,
        closed_peak_fraction=args.openness_closed_peak,
        open_peak_fraction=args.openness_open_peak,
        min_aperture_height_for_partial_open=args.openness_min_aperture,
    )
    openness_filter = OpennessFilter()
    calibrated_openness_filter = CalibratedOpennessFilter()
    openness_calibration = load_openness_calibration(args.openness_calibration)
    left_normalization_options = EyeNormalizationOptions(
        mode=args.normalization_mode,
        target_x=args.normalization_left_target_x if args.normalization_left_target_x is not None else args.normalization_target_x,
        target_y=args.normalization_left_target_y if args.normalization_left_target_y is not None else args.normalization_target_y,
        max_shift_x=args.normalization_max_shift_x,
        max_shift_y=args.normalization_max_shift_y,
        scale_min=args.normalization_scale_min,
        scale_max=args.normalization_scale_max,
        hold_frames=args.normalization_hold_frames,
    )
    right_normalization_options = EyeNormalizationOptions(
        mode=args.normalization_mode,
        target_x=args.normalization_right_target_x if args.normalization_right_target_x is not None else args.normalization_target_x,
        target_y=args.normalization_right_target_y if args.normalization_right_target_y is not None else args.normalization_target_y,
        max_shift_x=args.normalization_max_shift_x,
        max_shift_y=args.normalization_max_shift_y,
        scale_min=args.normalization_scale_min,
        scale_max=args.normalization_scale_max,
        hold_frames=args.normalization_hold_frames,
    )
    left_normalizer = EyeNormalizationState()
    right_normalizer = EyeNormalizationState()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"predict_live_{run_id}.csv"
    status_path = args.output_dir / "latest_status.json"
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp_port else None
    monitor_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.monitor_udp_port > 0 else None

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
        "left_index",
        "right_index",
        "delta_ms",
        "raw_x",
        "raw_y",
        "mapped_x",
        "mapped_y",
        "smooth_x",
        "smooth_y",
        "left_openness_raw",
        "right_openness_raw",
        "left_openness_calibrated",
        "right_openness_calibrated",
        "left_openness",
        "right_openness",
        "left_aperture_height",
        "right_aperture_height",
        "left_peak_dark_fraction",
        "right_peak_dark_fraction",
        "left_openness_reason",
        "right_openness_reason",
        "left_norm_found",
        "right_norm_found",
        "left_norm_confidence",
        "right_norm_confidence",
        "left_norm_shift_x",
        "left_norm_shift_y",
        "right_norm_shift_x",
        "right_norm_shift_y",
        "left_norm_scale",
        "right_norm_scale",
        "left_norm_pupil_x",
        "left_norm_pupil_y",
        "right_norm_pupil_x",
        "right_norm_pupil_y",
        "normalization_mode",
        "normalization_drop_reason",
        "left_snapshot",
        "right_snapshot",
    ]
    left_frames: list[JpegFrame] = []
    right_frames: list[JpegFrame] = []
    smooth: np.ndarray | None = None
    sequence = 0
    started = time.time()
    last_error: str | None = None

    print(f"BrokenEye left:  {left_url}")
    print(f"BrokenEye right: {right_url}")
    print(f"ONNX: {args.onnx.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")
    if args.udp_port:
        print(f"UDP:  {args.udp_host}:{args.udp_port}")
    else:
        print("UDP:  disabled")
    if args.monitor_udp_port > 0:
        print(f"MON:  {args.monitor_udp_host}:{args.monitor_udp_port}")

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while args.duration_seconds <= 0 or time.time() - started < args.duration_seconds:
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

                sequence += 1
                delta_ms = abs(left.received_at - right.received_at) * 1000.0
                left_open_raw = detect_openness(left.jpeg, openness_options)
                right_open_raw = detect_openness(right.jpeg, openness_options)
                if openness_calibration is None:
                    left_openness, right_openness = openness_filter.update(left_open_raw, right_open_raw)
                else:
                    left_openness, right_openness = calibrated_openness_filter.update(
                        calibrated_openness(left_open_raw, openness_calibration, "left"),
                        calibrated_openness(right_open_raw, openness_calibration, "right"),
                    )
                left_calibrated_openness = left_openness
                right_calibrated_openness = right_openness
                left_openness = boost_open_openness(
                    left_openness,
                    args.openness_boost_gamma,
                    args.openness_full_open_threshold,
                    args.openness_boost_knee,
                )
                right_openness = boost_open_openness(
                    right_openness,
                    args.openness_boost_gamma,
                    args.openness_full_open_threshold,
                    args.openness_boost_knee,
                )
                left_norm = left_normalizer.update(
                    left.jpeg,
                    image_size,
                    left_normalization_options,
                    left_openness,
                    left_open_raw.aperture_height,
                )
                right_norm = right_normalizer.update(
                    right.jpeg,
                    image_size,
                    right_normalization_options,
                    right_openness,
                    right_open_raw.aperture_height,
                )
                image = np.stack([left_norm.image, right_norm.image], axis=0)[None, ...].astype(np.float32)
                raw = session.run(["gaze_xy"], {"image": image})[0][0].astype(np.float32)
                mapped = apply_runtime_map(raw, runtime_offset, runtime_gain, args.clamp, args.clamp_mode, args.soft_knee)
                mapped = apply_center_bias(mapped, args.center_bias, args.center_bias_start, args.clamp)
                mapped = output_mapper.update(
                    mapped,
                    timestamp=max(left.received_at, right.received_at),
                    confidence=args.confidence,
                    openness=(left_openness + right_openness) * 0.5,
                )
                smooth = update_smooth(smooth, mapped, args.ema_alpha, args.max_step)

                if (udp is not None and args.udp_port is not None) or (monitor_udp is not None and args.monitor_udp_port > 0):
                    packet = bridge_packet(
                        sequence,
                        delta_ms,
                        float(raw[0]),
                        float(raw[1]),
                        float(mapped[0]),
                        float(mapped[1]),
                        float(smooth[0]),
                        float(smooth[1]),
                        args.confidence,
                        left_openness,
                        right_openness,
                        left_calibrated_openness,
                        right_calibrated_openness,
                        left_open_raw,
                        right_open_raw,
                        args.normalization_mode,
                        left_norm.metadata,
                        right_norm.metadata,
                    )
                    payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
                    if udp is not None and args.udp_port is not None:
                        udp.sendto(payload, (args.udp_host, args.udp_port))
                    if monitor_udp is not None and args.monitor_udp_port > 0:
                        monitor_udp.sendto(payload, (args.monitor_udp_host, args.monitor_udp_port))

                left_snapshot = ""
                right_snapshot = ""
                if args.snapshot_every > 0 and sequence % args.snapshot_every == 0:
                    left_snapshot, right_snapshot = save_pair_snapshots(args.output_dir, sequence, left, right)

                writer.writerow(
                    {
                        "sequence": sequence,
                        "timestamp": f"{time.time():.6f}",
                        "left_index": left.index,
                        "right_index": right.index,
                        "delta_ms": f"{delta_ms:.3f}",
                        "raw_x": f"{raw[0]:.6f}",
                        "raw_y": f"{raw[1]:.6f}",
                        "mapped_x": f"{mapped[0]:.6f}",
                        "mapped_y": f"{mapped[1]:.6f}",
                        "smooth_x": f"{smooth[0]:.6f}",
                        "smooth_y": f"{smooth[1]:.6f}",
                        "left_openness_raw": f"{left_open_raw.openness:.6f}",
                        "right_openness_raw": f"{right_open_raw.openness:.6f}",
                        "left_openness_calibrated": f"{left_calibrated_openness:.6f}",
                        "right_openness_calibrated": f"{right_calibrated_openness:.6f}",
                        "left_openness": f"{left_openness:.6f}",
                        "right_openness": f"{right_openness:.6f}",
                        "left_aperture_height": left_open_raw.aperture_height,
                        "right_aperture_height": right_open_raw.aperture_height,
                        "left_peak_dark_fraction": f"{left_open_raw.peak_dark_fraction:.6f}",
                        "right_peak_dark_fraction": f"{right_open_raw.peak_dark_fraction:.6f}",
                        "left_openness_reason": left_open_raw.reason,
                        "right_openness_reason": right_open_raw.reason,
                        "left_norm_found": int(left_norm.metadata.found),
                        "right_norm_found": int(right_norm.metadata.found),
                        "left_norm_confidence": f"{left_norm.metadata.confidence:.6f}",
                        "right_norm_confidence": f"{right_norm.metadata.confidence:.6f}",
                        "left_norm_shift_x": f"{left_norm.metadata.shift_x:.6f}",
                        "left_norm_shift_y": f"{left_norm.metadata.shift_y:.6f}",
                        "right_norm_shift_x": f"{right_norm.metadata.shift_x:.6f}",
                        "right_norm_shift_y": f"{right_norm.metadata.shift_y:.6f}",
                        "left_norm_scale": f"{left_norm.metadata.scale:.6f}",
                        "right_norm_scale": f"{right_norm.metadata.scale:.6f}",
                        "left_norm_pupil_x": f"{left_norm.metadata.pupil_x:.6f}",
                        "left_norm_pupil_y": f"{left_norm.metadata.pupil_y:.6f}",
                        "right_norm_pupil_x": f"{right_norm.metadata.pupil_x:.6f}",
                        "right_norm_pupil_y": f"{right_norm.metadata.pupil_y:.6f}",
                        "normalization_mode": args.normalization_mode,
                        "normalization_drop_reason": "; ".join(
                            reason
                            for reason in (
                                left_norm.metadata.drop_reason,
                                right_norm.metadata.drop_reason,
                            )
                            if reason
                        ),
                        "left_snapshot": left_snapshot,
                        "right_snapshot": right_snapshot,
                    }
                )
                if args.print_every > 0 and sequence % args.print_every == 0:
                    print(
                        f"{sequence:06d} dt={delta_ms:0.1f} "
                        f"raw=({raw[0]:+0.3f},{raw[1]:+0.3f}) "
                        f"map=({mapped[0]:+0.3f},{mapped[1]:+0.3f}) "
                        f"smooth=({smooth[0]:+0.3f},{smooth[1]:+0.3f}) "
                        f"open=({left_openness:0.2f},{right_openness:0.2f}) "
                        f"norm=({left_norm.metadata.confidence:0.2f},{right_norm.metadata.confidence:0.2f})",
                        flush=True,
                    )
    except KeyboardInterrupt:
        last_error = None
    finally:
        stop.set()
        if udp is not None:
            udp.close()
        if monitor_udp is not None:
            monitor_udp.close()

    wear_template_report = None
    if args.wear_template_path is not None:
        try:
            store = load_store(args.wear_template_path.resolve())
            payload = signature_from_csv(csv_path.resolve())
            wear_template_report = {
                "quality": payload["quality"],
                "match": match_template(payload["signature"], store),
                "signature": payload["signature"],
                "templates": str(args.wear_template_path.resolve()),
            }
            (args.output_dir / "wear_template_probe.json").write_text(
                json.dumps(wear_template_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            wear_template_report = {"error": str(exc), "templates": str(args.wear_template_path.resolve())}

    status = {
        "state": "complete" if last_error is None and sequence > 0 else "failed",
        "error": last_error,
        "csv": str(csv_path.resolve()),
        "sequence_count": sequence,
        "duration_seconds": time.time() - started,
        "onnx": str(args.onnx.resolve()),
        "udp": None if args.udp_port is None else {"host": args.udp_host, "port": args.udp_port},
        "monitor_udp": None if args.monitor_udp_port <= 0 else {"host": args.monitor_udp_host, "port": args.monitor_udp_port},
        "runtime_map": {
            "center_offset_x": args.center_offset_x,
            "center_offset_y": args.center_offset_y,
            "x_gain": args.x_gain,
            "y_gain": args.y_gain,
            "clamp": args.clamp,
            "clamp_mode": args.clamp_mode,
            "soft_knee": args.soft_knee,
            "center_bias": args.center_bias,
            "center_bias_start": args.center_bias_start,
            "ema_alpha": args.ema_alpha,
            "max_step": args.max_step,
        },
        "openness": {
            "mode": args.openness_mode,
            "fixed": args.openness,
            "calibration": None if args.openness_calibration is None else str(args.openness_calibration.resolve()),
            "roi": [args.openness_roi_x0, args.openness_roi_y0, args.openness_roi_x1, args.openness_roi_y1],
            "adaptive_percentile": args.openness_adaptive_percentile,
            "adaptive_margin": args.openness_adaptive_margin,
            "max_threshold": args.openness_max_threshold,
            "aperture_gate_fraction": args.openness_aperture_gate,
            "closed_peak_fraction": args.openness_closed_peak,
            "open_peak_fraction": args.openness_open_peak,
            "min_aperture_height_for_partial_open": args.openness_min_aperture,
            "boost_gamma": args.openness_boost_gamma,
            "full_open_threshold": args.openness_full_open_threshold,
            "boost_knee": args.openness_boost_knee,
        },
        "normalization": {
            "mode": args.normalization_mode,
            "target_x": args.normalization_target_x,
            "target_y": args.normalization_target_y,
            "left_target_x": args.normalization_left_target_x,
            "right_target_x": args.normalization_right_target_x,
            "left_target_y": args.normalization_left_target_y,
            "right_target_y": args.normalization_right_target_y,
            "max_shift_x": args.normalization_max_shift_x,
            "max_shift_y": args.normalization_max_shift_y,
            "scale_min": args.normalization_scale_min,
            "scale_max": args.normalization_scale_max,
            "hold_frames": args.normalization_hold_frames,
        },
        "wear_template": wear_template_report,
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["state"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
