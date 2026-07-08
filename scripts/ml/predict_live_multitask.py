#!/usr/bin/env python3
"""Run live BrokenEye inference with the MobileNetV3 multitask eye model.

Default mode is a dry-run: write CSV/status only, do not drive VRCFT.
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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from live_gaze_dry_run import (  # noqa: E402
    OutputGazeMapper,
    STAGE_PRESETS,
    JpegFrame,
    apply_runtime_map,
    jpeg_stream,
    preprocess,
    save_pair_snapshots,
    update_smooth,
)


def load_image_size(metadata_path: Path | None, fallback: int) -> int:
    if metadata_path is None or not metadata_path.exists():
        return fallback
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(payload.get("image_size", fallback))


def run_onnx(
    session: ort.InferenceSession,
    left: JpegFrame,
    right: JpegFrame,
    image_size: int,
    output_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    image = np.stack([preprocess(left.jpeg, image_size), preprocess(right.jpeg, image_size)], axis=0)
    batch = image[None].astype(np.float32)
    available = {item.name for item in session.get_outputs()}
    if output_names is None:
        names = ["gaze_xy", "openness_lr"]
        if "wide_lr" in available:
            names.append("wide_lr")
        if "squint_lr" in available:
            names.append("squint_lr")
        names.extend(["pupil_lr", "confidence"])
    else:
        names = [name for name in output_names if name in available]
        missing = sorted(set(output_names) - available)
        if missing:
            raise RuntimeError(f"ONNX missing requested outputs: {missing}")
    outputs = session.run(names, {"image": batch})
    result = {name: value[0].astype(np.float32) for name, value in zip(names, outputs)}
    result.setdefault("wide_lr", np.zeros(2, dtype=np.float32))
    result.setdefault("squint_lr", np.zeros(2, dtype=np.float32))
    return result


def stage_state(
    stages: list[str],
    started: float,
    settle_seconds: float,
    stage_seconds: float,
    previous_index: int,
) -> tuple[int, str, bool, float]:
    if not stages:
        return previous_index, "", True, time.time() - started
    now = time.time()
    block = settle_seconds + stage_seconds
    index = min(int((now - started) // block), len(stages) - 1)
    phase = now - started - index * block
    stage = stages[index]
    capture = phase >= settle_seconds
    stage_elapsed = max(0.0, phase - settle_seconds)
    return index, stage, capture, stage_elapsed


def write_status(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


PUPIL_NEUTRAL = 0.50
PUPIL_CONSTRICTED_ON_WIDE = 0.20


def clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def normalize_pupil_output_mode(mode: str) -> str:
    normalized = (mode or "off").strip().lower()
    if normalized in {"assist", "eye_wide_assist", "wide_assist", "expression"}:
        return "expression_constrict_on_wide"
    if normalized in {"diameter", "raw", "model", "model_radius"}:
        return "model_radius"
    if normalized == "expression_constrict_on_wide":
        return normalized
    return "off"


def eye_expression_from_runtime(
    openness: np.ndarray,
    model_openness: np.ndarray,
    confidence: np.ndarray,
    wide_source: str,
    wide_model_threshold: float,
    wide_output_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    wide = np.zeros(2, dtype=np.float32)
    squint = np.zeros(2, dtype=np.float32)
    model_threshold = float(np.clip(wide_model_threshold, 0.50, 0.999))
    output_threshold = float(np.clip(wide_output_threshold, 0.50, 1.0))
    if wide_source == "openness_heuristic":
        for index in range(2):
            if confidence[index] >= 0.05 and openness[index] >= 0.93:
                wide[index] = clamp01((float(openness[index]) - 0.93) / 0.07)
    elif wide_source == "model_openness_heuristic":
        for index in range(2):
            if (
                confidence[index] >= 0.05
                and openness[index] >= output_threshold
                and model_openness[index] >= model_threshold
            ):
                wide[index] = clamp01((float(model_openness[index]) - model_threshold) / max(0.001, 1.0 - model_threshold))
    return wide, squint


def apply_openness_curve(
    openness: np.ndarray,
    mode: str,
    full_open_threshold: float,
    boost_knee: float,
    boost_gamma: float,
) -> np.ndarray:
    values = np.clip(openness.astype(np.float32), 0.0, 1.0)
    if mode == "off":
        return values

    threshold = float(np.clip(full_open_threshold, 0.05, 1.0))
    knee = float(np.clip(boost_knee, 0.0, threshold - 0.001))
    gamma = max(0.05, float(boost_gamma))
    if mode == "soft_open_plateau":
        curved = np.power(values, gamma)
        curved[values >= threshold] = 1.0
        return np.clip(curved, 0.0, 1.0).astype(np.float32)

    if mode == "blink_s_curve":
        t = np.clip((values - knee) / max(0.001, threshold - knee), 0.0, 1.0)
        smoothstep = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        curved = np.power(smoothstep, gamma)
        curved[values >= threshold] = 1.0
        return np.clip(curved, 0.0, 1.0).astype(np.float32)

    curved = values.copy()
    high = values >= threshold
    mid = (values > knee) & ~high
    curved[high] = 1.0
    if np.any(mid):
        t = (values[mid] - knee) / max(0.001, threshold - knee)
        curved[mid] = knee + (1.0 - knee) * np.power(t, gamma)
    return np.clip(curved, 0.0, 1.0).astype(np.float32)


def normalize_per_eye(
    model_openness: np.ndarray,
    open_p95: np.ndarray,
    closed_p05: np.ndarray,
    min_range: float = 0.05,
) -> np.ndarray:
    """Linearly normalize per-eye model openness using calibrated open/closed refs.

    Each eye: clip((v - closed_p05) / (open_p95 - closed_p05), 0, 1). When a
    per-eye range is degenerate (< min_range) that eye passes through unchanged,
    so an empty/bad calibration can never corrupt output. NaNs are coerced.
    Mirrors the dead-legacy predict_live.calibrated_openness, vectorized per eye.
    """
    x = np.clip(np.nan_to_num(model_openness.astype(np.float32), nan=0.0), 0.0, 1.0)
    hi = np.broadcast_to(np.nan_to_num(np.asarray(open_p95, dtype=np.float32), nan=1.0), x.shape).astype(np.float32)
    lo = np.broadcast_to(np.nan_to_num(np.asarray(closed_p05, dtype=np.float32), nan=0.0), x.shape).astype(np.float32)
    rng = hi - lo
    scaled = (x - lo) / np.maximum(rng, 1e-6)
    out = np.where(rng >= float(min_range), scaled, x)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_per_eye_openness_calibration(
    path: "Path | None",
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Load per-eye (open_p95, closed_p05) from a v2 openness_calibration.json.

    Returns (open_p95[2], closed_p05[2]) or None when the path is missing, the
    file is absent/unreadable, or lacks per-eye model p95/p05 fields (e.g. a
    v1-only file). None -> caller keeps raw model openness (identity, harmless).
    """
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        left = payload["left"]
        right = payload["right"]
        open_p95 = np.array([float(left["open_p95"]), float(right["open_p95"])], dtype=np.float32)
        closed_p05 = np.array([float(left["closed_p05"]), float(right["closed_p05"])], dtype=np.float32)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not np.all(np.isfinite(open_p95)) or not np.all(np.isfinite(closed_p05)):
        return None
    return open_p95, closed_p05


def apply_eye_shape_curve(values: np.ndarray, scale: float, gamma: float, deadzone: float) -> np.ndarray:
    shaped = np.clip(values.astype(np.float32), 0.0, 1.0)
    deadzone = float(np.clip(deadzone, 0.0, 0.95))
    if deadzone > 0.0:
        shaped = np.clip((shaped - deadzone) / max(0.001, 1.0 - deadzone), 0.0, 1.0)
    gamma = max(0.05, float(gamma))
    if abs(gamma - 1.0) > 1e-6:
        shaped = np.power(shaped, gamma)
    return np.clip(shaped * float(np.clip(scale, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)


def pupil_output_state(
    mode: str,
    openness: float,
    wide: float,
    pupil_radius: float,
    confidence: float,
) -> dict[str, object]:
    if mode == "off":
        return {
            "found": False,
            "quality": False,
            "confidence": float(confidence),
            "normalized": PUPIL_NEUTRAL,
            "expressionNormalized": PUPIL_NEUTRAL,
            "reason": "pupil-output-off",
        }

    if confidence < 0.05 or openness < 0.45:
        return {
            "found": False,
            "quality": False,
            "confidence": float(confidence),
            "normalized": PUPIL_NEUTRAL,
            "expressionNormalized": PUPIL_NEUTRAL,
            "reason": "pupil output gated by confidence or closed eyelid",
        }

    if mode == "expression_constrict_on_wide":
        expression = PUPIL_NEUTRAL + ((PUPIL_CONSTRICTED_ON_WIDE - PUPIL_NEUTRAL) * clamp01(wide))
        return {
            "found": True,
            "quality": True,
            "confidence": float(confidence),
            "normalized": float(np.clip(expression, 0.0, 1.0)),
            "expressionNormalized": float(np.clip(expression, 0.0, 1.0)),
            "reason": "expression-constrict-on-wide",
        }

    if not np.isfinite(pupil_radius) or pupil_radius <= 0.02 or pupil_radius >= 0.35:
        return {
            "found": False,
            "quality": False,
            "confidence": float(confidence),
            "normalized": PUPIL_NEUTRAL,
            "expressionNormalized": PUPIL_NEUTRAL,
            "reason": "model pupil radius gated",
        }

    return {
        "found": True,
        "quality": True,
        "confidence": float(confidence),
        "normalized": clamp01(pupil_radius),
        "expressionNormalized": PUPIL_NEUTRAL,
        "reason": "model-radius-debug",
    }


def update_pupil_wide_state(
    raw_wide: np.ndarray,
    state: np.ndarray,
    hold_remaining: np.ndarray,
    enter_threshold: float,
    exit_threshold: float,
    hold_frames: int,
    ema_alpha: float,
) -> np.ndarray:
    raw = np.clip(raw_wide.astype(np.float32), 0.0, 1.0)
    enter = float(np.clip(enter_threshold, 0.0, 1.0))
    exit_ = float(np.clip(exit_threshold, 0.0, enter))
    frames = max(0, int(hold_frames))
    alpha = float(np.clip(ema_alpha, 0.01, 1.0))
    for index in range(2):
        value = float(raw[index])
        if value >= enter:
            hold_remaining[index] = frames
            target = clamp01((value - enter) / max(0.001, 1.0 - enter))
        elif value <= exit_:
            if hold_remaining[index] > 0:
                hold_remaining[index] -= 1
                target = float(state[index])
            else:
                target = 0.0
        else:
            if hold_remaining[index] > 0:
                hold_remaining[index] -= 1
                target = float(state[index])
            else:
                target = 0.0
        state[index] = state[index] + alpha * (target - state[index])
    return np.clip(state, 0.0, 1.0).astype(np.float32)


TRACKING_STATES = (
    "TRACKING_OK",
    "FIXATING",
    "SACCADE",
    "BLINKING",
    "CLOSED",
    "REOPENING",
    "REACQUIRE",
    "ONE_EYE_LOST",
)


@dataclass
class TrackingStateParams:
    """Parameters for TrackingStateMachine.

    Numeric defaults are the production constants ported from the legacy C#
    BridgeOpennessFilterOptions / BridgeClosureGazeStabilizerOptions. The two
    master switches default to OFF so the machine is a pure passthrough unless
    explicitly enabled (compat mode == today's behavior, byte-identical).
    """

    # master switches (compat defaults => identity passthrough)
    openness_machine: bool = False
    gaze_hold: bool = False
    # per-eye openness (BridgeOpennessFilterOptions)
    closed_threshold: float = 0.20
    other_eye_open_threshold: float = 0.60
    open_evidence_threshold: float = 0.90
    open_evidence_floor: float = 0.75
    open_hold_threshold: float = 0.80
    single_eye_drop_threshold: float = 0.70
    other_eye_fully_open_hold_threshold: float = 0.78
    asymmetric_drop_confirm_frames: int = 2
    rise_alpha: float = 0.75
    fall_alpha: float = 0.55
    both_closed_fall_alpha: float = 0.90
    # binocular closure gaze hold (BridgeClosureGazeStabilizerOptions)
    closing_openness_threshold: float = 0.50
    closed_openness_threshold: float = 0.20
    reopen_openness_threshold: float = 0.70
    closing_drop_threshold: float = 0.28
    closing_drop_max_openness: float = 0.65
    hold_frames_after_closing: int = 4
    reopen_release_alpha: float = 0.35
    # gaze saccade/fixate (reported only in this version; None => no re-smooth)
    saccade_alpha: "float | None" = None
    fixate_alpha: "float | None" = None
    saccade_delta: float = 0.06
    fixate_delta: float = 0.02
    fixate_frames: int = 8


class TrackingStateMachine:
    """Explicit per-eye openness + binocular gaze tracking state machine (E1).

    Ports the legacy C# BridgeOpennessFilter (per-eye rise/fall smoothing,
    single-eye false-close suppression, open-evidence floor) and
    BridgeClosureGazeStabilizer (hold/lerp gaze while closing/closed) into the
    production Python runtime, and reports one of TRACKING_STATES per frame.

    Compat: with openness_machine=False and gaze_hold=False it returns the exact
    input objects (openness == curved, gaze == input) -> byte-identical output.
    Model openness is already normalized+curved upstream, so the image baseline
    math from BridgeOpennessFilter.NormalizeAdaptive is intentionally not ported
    (per-eye normalization is handled separately by normalize_per_eye).
    """

    def __init__(self, params: TrackingStateParams, ema_alpha: float, min_confidence: float = 0.05) -> None:
        self.p = params
        self.ema_alpha = float(ema_alpha)
        self.min_confidence = float(min_confidence)
        self._open_state = np.array([1.0, 1.0], dtype=np.float32)
        self._open_initialized = False
        self._left_hold = 0
        self._right_hold = 0
        self._prev_gaze = None
        self._prev_avg_open = 1.0
        self._gaze_hold_frames = 0
        self._fixate_count = 0
        self.state = "TRACKING_OK"
        self.left_substate = "OPEN"
        self.right_substate = "OPEN"

    def _hold_asymmetric_drop(self, target, other_raw, previous, hold_frames):
        # Suppress a single-eye false close: keep this eye's previous value for a
        # few frames when it was open, is dropping, and the other eye stays open.
        p = self.p
        if (
            previous < p.open_hold_threshold
            or target >= p.single_eye_drop_threshold
            or other_raw < p.other_eye_open_threshold
        ):
            return target, 0
        hold_frames += 1
        if other_raw >= p.other_eye_fully_open_hold_threshold or hold_frames <= p.asymmetric_drop_confirm_frames:
            return previous, hold_frames
        return target, hold_frames

    def _smooth(self, previous, target, both_closed):
        p = self.p
        if target >= previous:
            alpha = p.rise_alpha
        elif both_closed:
            alpha = p.both_closed_fall_alpha
        else:
            alpha = p.fall_alpha
        alpha = min(1.0, max(0.0, alpha))
        return float(min(1.0, max(0.0, previous + (target - previous) * alpha)))

    def _openness(self, curved):
        p = self.p
        if not p.openness_machine:
            self._open_state = curved.astype(np.float32)
            return curved  # passthrough: same values (already clipped upstream)
        raw = np.clip(curved.astype(np.float32), 0.0, 1.0).copy()
        for i in range(2):
            if float(curved[i]) >= p.open_evidence_threshold:
                raw[i] = max(float(raw[i]), p.open_evidence_floor)
        if not self._open_initialized:
            self._open_initialized = True
            self._open_state = raw.copy()
            return self._open_state
        both_closed = bool(raw[0] <= p.closed_threshold and raw[1] <= p.closed_threshold)
        target = raw.copy()
        if not both_closed:
            target[0], self._left_hold = self._hold_asymmetric_drop(
                float(raw[0]), float(raw[1]), float(self._open_state[0]), self._left_hold
            )
            target[1], self._right_hold = self._hold_asymmetric_drop(
                float(raw[1]), float(raw[0]), float(self._open_state[1]), self._right_hold
            )
        else:
            self._left_hold = 0
            self._right_hold = 0
        self._open_state[0] = self._smooth(float(self._open_state[0]), float(target[0]), both_closed)
        self._open_state[1] = self._smooth(float(self._open_state[1]), float(target[1]), both_closed)
        return self._open_state

    def _gaze(self, gaze, avg_open):
        p = self.p
        if not p.gaze_hold:
            self._prev_gaze = np.asarray(gaze, dtype=np.float32).copy()
            self._prev_avg_open = avg_open
            return gaze  # passthrough: same object
        g = np.clip(np.asarray(gaze, dtype=np.float32), -1.0, 1.0)
        if self._prev_gaze is None:
            self._prev_gaze = g.copy()
            self._prev_avg_open = avg_open
            return self._prev_gaze
        closing_drop = (self._prev_avg_open - avg_open >= p.closing_drop_threshold) and (
            avg_open <= p.closing_drop_max_openness
        )
        closing = closing_drop or avg_open <= p.closing_openness_threshold
        closed = avg_open <= p.closed_openness_threshold
        if closing:
            self._gaze_hold_frames = max(self._gaze_hold_frames, max(0, int(p.hold_frames_after_closing)))
        if closed or (self._gaze_hold_frames > 0 and avg_open < p.reopen_openness_threshold):
            out = self._prev_gaze.copy()
            if self._gaze_hold_frames > 0:
                self._gaze_hold_frames -= 1
        elif self._gaze_hold_frames > 0:
            alpha = min(1.0, max(0.0, p.reopen_release_alpha))
            out = self._prev_gaze + (g - self._prev_gaze) * alpha
            self._prev_gaze = out.copy()
            self._gaze_hold_frames -= 1
        else:
            out = g.copy()
            self._prev_gaze = out.copy()
        self._prev_avg_open = avg_open
        return out.astype(np.float32)

    def _substate(self, value):
        p = self.p
        if value <= p.closed_threshold:
            return "CLOSED"
        if value >= p.open_hold_threshold:
            return "OPEN"
        return "PARTIAL"

    def _classify(self, open_out, avg_open, prev_avg, hold_before, pair_conf, gaze_delta):
        p = self.p
        if pair_conf < self.min_confidence:
            return "REACQUIRE"
        left_closed = float(open_out[0]) <= p.closed_threshold
        right_closed = float(open_out[1]) <= p.closed_threshold
        if left_closed != right_closed:
            return "ONE_EYE_LOST"
        if avg_open <= p.closed_openness_threshold:
            return "CLOSED"
        closing_drop = (prev_avg - avg_open >= p.closing_drop_threshold) and (avg_open <= p.closing_drop_max_openness)
        if closing_drop:
            return "BLINKING"
        if hold_before > 0 and avg_open < p.reopen_openness_threshold:
            return "REOPENING"
        if gaze_delta >= p.saccade_delta:
            self._fixate_count = 0
            return "SACCADE"
        if gaze_delta < p.fixate_delta:
            self._fixate_count += 1
            if self._fixate_count >= p.fixate_frames:
                return "FIXATING"
        else:
            self._fixate_count = 0
        return "TRACKING_OK"

    def update(self, curved_openness, model_openness, confidence, gaze):
        curved = curved_openness
        open_out = self._openness(curved)
        avg_open = float(np.clip((float(open_out[0]) + float(open_out[1])) * 0.5, 0.0, 1.0))
        conf = np.asarray(confidence, dtype=np.float32)
        pair_conf = float(conf[2]) if conf.size > 2 else float(np.min(conf[:2]))
        prev_gaze_snapshot = None if self._prev_gaze is None else self._prev_gaze.copy()
        prev_avg_snapshot = self._prev_avg_open
        hold_before = self._gaze_hold_frames
        gaze_out = self._gaze(gaze, avg_open)
        if prev_gaze_snapshot is not None:
            gi = np.asarray(gaze, dtype=np.float32)
            gaze_delta = float(np.linalg.norm(gi[:2] - prev_gaze_snapshot[:2]))
        else:
            gaze_delta = 0.0
        self.state = self._classify(open_out, avg_open, prev_avg_snapshot, hold_before, pair_conf, gaze_delta)
        self.left_substate = self._substate(float(open_out[0]))
        self.right_substate = self._substate(float(open_out[1]))
        return SimpleNamespace(
            openness=open_out,
            gaze=gaze_out,
            state=self.state,
            left_substate=self.left_substate,
            right_substate=self.right_substate,
        )


def build_tracking_params(args) -> TrackingStateParams:
    return TrackingStateParams(
        openness_machine=(args.tracking_openness_machine == "on"),
        gaze_hold=bool(args.tracking_gaze_hold),
        closed_threshold=args.tracking_openness_closed_threshold,
        other_eye_open_threshold=args.tracking_other_eye_open_threshold,
        open_evidence_threshold=args.tracking_open_evidence_threshold,
        open_evidence_floor=args.tracking_open_evidence_floor,
        open_hold_threshold=args.tracking_open_hold_threshold,
        single_eye_drop_threshold=args.tracking_single_eye_drop_threshold,
        other_eye_fully_open_hold_threshold=args.tracking_other_eye_fully_open_hold_threshold,
        asymmetric_drop_confirm_frames=args.tracking_asymmetric_drop_confirm_frames,
        rise_alpha=args.tracking_openness_rise_alpha,
        fall_alpha=args.tracking_openness_fall_alpha,
        both_closed_fall_alpha=args.tracking_openness_both_closed_fall_alpha,
        closing_openness_threshold=args.tracking_gaze_closing_openness_threshold,
        closed_openness_threshold=args.tracking_gaze_closed_openness_threshold,
        reopen_openness_threshold=args.tracking_gaze_reopen_openness_threshold,
        closing_drop_threshold=args.tracking_gaze_closing_drop_threshold,
        closing_drop_max_openness=args.tracking_gaze_closing_drop_max_openness,
        hold_frames_after_closing=args.tracking_gaze_hold_frames,
        reopen_release_alpha=args.tracking_gaze_reopen_release_alpha,
        saccade_alpha=args.tracking_gaze_saccade_alpha,
        fixate_alpha=args.tracking_gaze_fixate_alpha,
        saccade_delta=args.tracking_gaze_saccade_delta,
        fixate_delta=args.tracking_gaze_fixate_delta,
        fixate_frames=args.tracking_gaze_fixate_frames,
    )


def bridge_eye(
    raw: np.ndarray,
    mapped: np.ndarray,
    smooth: np.ndarray,
    model_openness: float,
    openness: float,
    wide: float,
    squint: float,
    pupil_x: float,
    pupil_y: float,
    pupil_radius: float,
    confidence: float,
    pupil_output: dict[str, object],
) -> dict[str, object]:
    return {
        "found": True,
        "confidence": float(confidence),
        "rawX": float(raw[0]),
        "rawY": float(raw[1]),
        "normalizedX": float(smooth[0]),
        "normalizedY": float(smooth[1]),
        "openness": float(openness),
        "pupilOpenness": float(openness),
        "apertureOpenness": float(model_openness),
        "calibratedOpenness": float(model_openness),
        "outputOpenness": float(openness),
        "wide": clamp01(wide),
        "squint": clamp01(squint),
        "aperturePeakDarkFraction": 0.0,
        "apertureHeight": 0,
        "apertureReason": "multitask-openness-head+runtime-curve",
        "rawNormalizedX": float(raw[0]),
        "rawNormalizedY": float(raw[1]),
        "monocularNormalizedX": float(mapped[0]),
        "monocularNormalizedY": float(mapped[1]),
        "pupilDiameterFound": bool(pupil_output["found"]),
        "pupilDiameterQuality": bool(pupil_output["quality"]),
        "pupilDiameterConfidence": float(pupil_output["confidence"]),
        "pupilDiameterPx": 0.0,
        "pupilDiameterNormalized": float(pupil_output["normalized"]),
        "pupilExpressionNormalized": float(pupil_output["expressionNormalized"]),
        "pupilGeometryRadius": float(pupil_radius),
        "pupilDiameterAxisRatio": 1.0,
        "pupilDiameterReason": str(pupil_output["reason"]),
        "normalizationFound": True,
        "normalizationConfidence": float(confidence),
        "normalizationShiftX": 0.0,
        "normalizationShiftY": 0.0,
        "normalizationScale": float(max(0.05, pupil_radius)),
        "normalizationPupilX": float(pupil_x),
        "normalizationPupilY": float(pupil_y),
        "normalizationDropReason": "multitask-pupil-head",
    }


def bridge_packet(
    sequence: int,
    delta_ms: float,
    raw: np.ndarray,
    mapped: np.ndarray,
    smooth: np.ndarray,
    model_openness: np.ndarray,
    openness: np.ndarray,
    pupil: np.ndarray,
    confidence: np.ndarray,
    wide: np.ndarray,
    squint: np.ndarray,
    pupil_output_mode: str,
    eye_expression_mode: str,
    pupil_wide: np.ndarray | None = None,
) -> bytes:
    pupil_wide_values = wide if pupil_wide is None else pupil_wide
    left_pupil_output = pupil_output_state(pupil_output_mode, float(openness[0]), float(pupil_wide_values[0]), float(pupil[2]), float(confidence[0]))
    right_pupil_output = pupil_output_state(pupil_output_mode, float(openness[1]), float(pupil_wide_values[1]), float(pupil[5]), float(confidence[1]))
    payload = {
        "sequence": sequence,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z",
        "deltaMs": float(delta_ms),
        "eyeTrackingEnabled": True,
        "eyeExpressionEnabled": eye_expression_mode != "off",
        "eyeExpressionMode": eye_expression_mode,
        "pupilDiameterEnabled": pupil_output_mode != "off",
        "pupilDiameterMode": pupil_output_mode,
        "normalizationMode": "multitask_diagnostic",
        "left": bridge_eye(raw, mapped, smooth, model_openness[0], openness[0], wide[0], squint[0], pupil[0], pupil[1], pupil[2], confidence[0], left_pupil_output),
        "right": bridge_eye(raw, mapped, smooth, model_openness[1], openness[1], wide[1], squint[1], pupil[3], pupil[4], pupil[5], confidence[1], right_pupil_output),
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("runs/eye_multitask_mobilenetv3_fullparam_round1_twochannel_retarget070/eye_multitask.onnx"))
    parser.add_argument("--metadata", type=Path, default=Path("runs/eye_multitask_mobilenetv3_fullparam_round1_twochannel_retarget070/eye_multitask.metadata.json"))
    parser.add_argument("--expression-onnx", type=Path, help="Optional secondary ONNX used only for wide_lr/squint_lr.")
    parser.add_argument("--expression-metadata", type=Path, help="Optional metadata for --expression-onnx image size.")
    parser.add_argument("--expression-every-n-frames", type=int, default=1, help="Run secondary expression ONNX every N emitted frames and hold between updates.")
    parser.add_argument("--expression-ema-alpha", type=float, default=1.0, help="EMA alpha for secondary wide/squint outputs; 1 disables smoothing.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/live_multitask_dry_run"))
    parser.add_argument("--duration-seconds", "--duration", dest="duration_seconds", type=float, default=0.0, help="0 means run until Ctrl+C.")
    parser.add_argument("--direction-preset", choices=("five", "nine"), help="Prompt directions and tag captured rows.")
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--stage-seconds", type=float, default=3.0)
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--center-offset-x", type=float, default=0.0)
    parser.add_argument("--center-offset-y", type=float, default=0.0)
    parser.add_argument("--x-gain", type=float, default=1.0)
    parser.add_argument("--y-gain", type=float, default=1.0)
    parser.add_argument("--clamp", type=float, default=1.0)
    parser.add_argument("--clamp-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--soft-knee", type=float, default=0.72)
    parser.add_argument("--max-step", type=float, default=0.0)
    parser.add_argument("--output-map-mode", choices=("off", "stable_center"), default="stable_center")
    parser.add_argument("--output-deadzone", type=float, default=0.015)
    parser.add_argument("--output-curve-gamma", type=float, default=1.0)
    parser.add_argument("--output-center-radius", type=float, default=0.08)
    parser.add_argument("--output-center-tau", type=float, default=6.0)
    parser.add_argument("--output-center-max-step", type=float, default=0.025)
    parser.add_argument("--pupil-output-mode", default="off", help="off, expression_constrict_on_wide, model_radius; legacy aliases: eye_wide_assist, diameter.")
    parser.add_argument("--eye-expression-mode", choices=("off", "steamlink_eye_shapes"), default="off")
    parser.add_argument("--wide-source", choices=("none", "model_head", "openness_heuristic", "model_openness_heuristic"), default="model_head")
    parser.add_argument("--wide-model-threshold", type=float, default=0.93)
    parser.add_argument("--wide-output-threshold", type=float, default=0.98)
    parser.add_argument("--eye-shape-wide-scale", type=float, default=1.0, help="Scale applied only to VRCFT EyeWide output; pupil output still uses raw wide.")
    parser.add_argument("--eye-shape-squint-scale", type=float, default=1.0, help="Scale applied only to VRCFT EyeSquint output.")
    parser.add_argument("--eye-shape-gamma", type=float, default=1.0, help="Gamma applied to VRCFT EyeWide/EyeSquint after deadzone.")
    parser.add_argument("--eye-shape-deadzone", type=float, default=0.0, help="Deadzone applied only to VRCFT EyeWide/EyeSquint.")
    parser.add_argument("--pupil-wide-enter-threshold", type=float, default=0.90, help="Raw/model wide value that turns on pupil constriction assist.")
    parser.add_argument("--pupil-wide-exit-threshold", type=float, default=0.86, help="Raw/model wide value below which pupil constriction assist fades out.")
    parser.add_argument("--pupil-wide-hold-frames", type=int, default=8, help="Frames to hold pupil constriction assist after a wide trigger.")
    parser.add_argument("--pupil-wide-ema-alpha", type=float, default=0.45, help="EMA alpha for pupil constriction assist wide amount.")
    parser.add_argument("--openness-curve-mode", choices=("off", "full_open_plateau", "blink_s_curve", "soft_open_plateau"), default="blink_s_curve")
    parser.add_argument("--openness-full-open-threshold", type=float, default=0.90)
    parser.add_argument("--openness-boost-knee", type=float, default=0.28)
    parser.add_argument("--openness-boost-gamma", type=float, default=1.25)
    parser.add_argument(
        "--openness-per-eye-calibration",
        type=Path,
        default=None,
        help="Optional v2 openness_calibration.json with per-eye open_p95/closed_p05; "
        "absent -> raw model openness (identity).",
    )
    parser.add_argument(
        "--tracking-openness-machine",
        choices=("off", "on"),
        default="off",
        help="Per-eye openness state machine (rise/fall smoothing, single-eye false-close suppression, "
        "open-evidence floor). off -> passthrough (default, byte-identical to legacy).",
    )
    parser.add_argument(
        "--tracking-gaze-hold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hold/lerp gaze while eyes are closing/closed (ports BridgeClosureGazeStabilizer).",
    )
    parser.add_argument("--tracking-openness-closed-threshold", type=float, default=0.20)
    parser.add_argument("--tracking-openness-rise-alpha", type=float, default=0.75)
    parser.add_argument("--tracking-openness-fall-alpha", type=float, default=0.55)
    parser.add_argument("--tracking-openness-both-closed-fall-alpha", type=float, default=0.90)
    parser.add_argument("--tracking-open-evidence-threshold", type=float, default=0.90)
    parser.add_argument("--tracking-open-evidence-floor", type=float, default=0.75)
    parser.add_argument("--tracking-open-hold-threshold", type=float, default=0.80)
    parser.add_argument("--tracking-single-eye-drop-threshold", type=float, default=0.70)
    parser.add_argument("--tracking-other-eye-open-threshold", type=float, default=0.60)
    parser.add_argument("--tracking-other-eye-fully-open-hold-threshold", type=float, default=0.78)
    parser.add_argument("--tracking-asymmetric-drop-confirm-frames", type=int, default=2)
    parser.add_argument("--tracking-gaze-closing-openness-threshold", type=float, default=0.50)
    parser.add_argument("--tracking-gaze-closed-openness-threshold", type=float, default=0.20)
    parser.add_argument("--tracking-gaze-reopen-openness-threshold", type=float, default=0.70)
    parser.add_argument("--tracking-gaze-closing-drop-threshold", type=float, default=0.28)
    parser.add_argument("--tracking-gaze-closing-drop-max-openness", type=float, default=0.65)
    parser.add_argument("--tracking-gaze-hold-frames", type=int, default=4)
    parser.add_argument("--tracking-gaze-reopen-release-alpha", type=float, default=0.35)
    parser.add_argument("--tracking-gaze-saccade-alpha", type=float, default=None)
    parser.add_argument("--tracking-gaze-fixate-alpha", type=float, default=None)
    parser.add_argument("--tracking-gaze-saccade-delta", type=float, default=0.06)
    parser.add_argument("--tracking-gaze-fixate-delta", type=float, default=0.02)
    parser.add_argument("--tracking-gaze-fixate-frames", type=int, default=8)
    parser.add_argument("--udp-port", type=int, help="Optional DreamAirTracking VRCFT bridge UDP port, usually 9400.")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--monitor-udp-port", type=int, default=0, help="Optional App monitor UDP port, usually 9401; 0 disables.")
    parser.add_argument("--monitor-udp-host", default="127.0.0.1")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--snapshot-on-low-openness", action="store_true")
    parser.add_argument("--snapshot-low-openness-threshold", type=float, default=0.80)
    parser.add_argument("--snapshot-low-openness-min-confidence", type=float, default=0.95)
    parser.add_argument("--snapshot-low-openness-min-interval", type=int, default=30)
    args = parser.parse_args()

    image_size = load_image_size(args.metadata.resolve() if args.metadata else None, args.image_size)
    session = ort.InferenceSession(str(args.onnx.resolve()), providers=["CPUExecutionProvider"])
    expression_session = None
    expression_image_size = image_size
    expression_every_n = max(1, args.expression_every_n_frames)
    expression_ema_alpha = min(1.0, max(0.0, args.expression_ema_alpha))
    if args.expression_onnx:
        expression_image_size = load_image_size(
            args.expression_metadata.resolve() if args.expression_metadata else None,
            image_size,
        )
        expression_session = ort.InferenceSession(str(args.expression_onnx.resolve()), providers=["CPUExecutionProvider"])
        expression_outputs = {item.name for item in expression_session.get_outputs()}
        if not {"wide_lr", "squint_lr"}.issubset(expression_outputs):
            raise SystemExit(f"Expression ONNX outputs {sorted(expression_outputs)}; expected wide_lr and squint_lr.")
    output_names = [item.name for item in session.get_outputs()]
    required = {"gaze_xy", "openness_lr", "pupil_lr", "confidence"}
    if not required.issubset(set(output_names)):
        raise SystemExit(f"ONNX outputs {output_names}; expected {sorted(required)}.")

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
    pupil_output_mode = normalize_pupil_output_mode(args.pupil_output_mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"live_multitask_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    status_path = args.output_dir / "latest_status.json"
    q: queue.Queue[JpegFrame] = queue.Queue(maxsize=80)
    stop = threading.Event()
    left_url = f"http://{args.host}:{args.port}/eye/left"
    right_url = f"http://{args.host}:{args.port}/eye/right"
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp_port else None
    monitor_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.monitor_udp_port > 0 else None
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
        "mapped_x",
        "mapped_y",
        "smooth_x",
        "smooth_y",
        "left_model_openness",
        "right_model_openness",
        "left_openness",
        "right_openness",
        "left_norm_openness",
        "right_norm_openness",
        "tracking_state",
        "left_substate",
        "right_substate",
        "left_model_wide",
        "right_model_wide",
        "left_pupil_wide",
        "right_pupil_wide",
        "left_model_squint",
        "right_model_squint",
        "left_wide",
        "right_wide",
        "left_squint",
        "right_squint",
        "left_pupil_expression",
        "right_pupil_expression",
        "left_pupil_x",
        "left_pupil_y",
        "left_pupil_radius",
        "right_pupil_x",
        "right_pupil_y",
        "right_pupil_radius",
        "left_confidence",
        "right_confidence",
        "pair_confidence",
        "left_snapshot",
        "right_snapshot",
    ]

    stages = STAGE_PRESETS.get(args.direction_preset or "", [])
    if stages:
        args.duration_seconds = len(stages) * (args.settle_seconds + args.stage_seconds)

    print(f"BrokenEye left:  {left_url}")
    print(f"BrokenEye right: {right_url}")
    print(f"ONNX: {args.onnx.resolve()}")
    if expression_session is not None and args.expression_onnx is not None:
        print(
            f"Expression ONNX: {args.expression_onnx.resolve()} "
            f"(every {expression_every_n} frame(s), ema_alpha={expression_ema_alpha:0.2f})"
        )
    print(f"CSV:  {csv_path.resolve()}")
    if args.udp_port:
        print(f"UDP:  {args.udp_host}:{args.udp_port}")
    if args.monitor_udp_port > 0:
        print(f"MON:  {args.monitor_udp_host}:{args.monitor_udp_port}")
    print(
        f"Pupil output: {pupil_output_mode}; eye expression: {args.eye_expression_mode}; "
        f"wide source: {args.wide_source} (model>={args.wide_model_threshold:0.2f}, output>={args.wide_output_threshold:0.2f}); "
        f"pupil wide: enter={args.pupil_wide_enter_threshold:0.2f}, exit={args.pupil_wide_exit_threshold:0.2f}, "
        f"hold={args.pupil_wide_hold_frames}, ema={args.pupil_wide_ema_alpha:0.2f}; "
        f"openness curve: {args.openness_curve_mode} "
        f"(threshold={args.openness_full_open_threshold:0.2f}, knee={args.openness_boost_knee:0.2f}, gamma={args.openness_boost_gamma:0.2f})"
    )
    if stages:
        print("")
        print("Guided multitask dry-run")
        print("Follow the prompted direction; rows are recorded after the settle period.")

    left_frames: list[JpegFrame] = []
    right_frames: list[JpegFrame] = []
    smooth: np.ndarray | None = None
    sequence = 0
    started = time.time()
    stage_index = -1
    last_error: str | None = None
    last_low_openness_snapshot_sequence = -1_000_000
    last_expression_result: dict[str, np.ndarray] | None = None
    last_expression_sequence = -1_000_000
    pupil_wide_state = np.zeros(2, dtype=np.float32)
    pupil_wide_hold_remaining = np.zeros(2, dtype=np.int32)
    openness_cal = load_per_eye_openness_calibration(args.openness_per_eye_calibration)
    if openness_cal is not None:
        print(
            f"Per-eye openness calibration: open_p95={openness_cal[0].tolist()} "
            f"closed_p05={openness_cal[1].tolist()}"
        )
    tracking = TrackingStateMachine(build_tracking_params(args), args.ema_alpha)
    print(
        f"Tracking state machine: openness_machine={args.tracking_openness_machine} "
        f"gaze_hold={bool(args.tracking_gaze_hold)}"
    )

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while args.duration_seconds <= 0 or time.time() - started < args.duration_seconds:
                new_stage_index, stage, capture_enabled, stage_elapsed = stage_state(
                    stages,
                    started,
                    args.settle_seconds,
                    args.stage_seconds,
                    stage_index,
                )
                if new_stage_index != stage_index:
                    stage_index = new_stage_index
                    if stages:
                        phase = "settle" if not capture_enabled else "capture"
                        print("")
                        print(f"[{stage_index + 1}/{len(stages)}] {stage.upper().replace('_', ' ')} ({phase})")

                try:
                    frame = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if frame.index < 0:
                    raise RuntimeError(frame.jpeg.decode("utf-8", errors="replace"))
                if frame.side == "left":
                    left_frames.append(frame)
                    left_frames = left_frames[-8:]
                else:
                    right_frames.append(frame)
                    right_frames = right_frames[-8:]
                if not left_frames or not right_frames:
                    continue

                left = min(left_frames, key=lambda item: abs(item.received_at - frame.received_at))
                right = min(right_frames, key=lambda item: abs(item.received_at - left.received_at))
                delta_ms = abs(left.received_at - right.received_at) * 1000.0
                if delta_ms > args.max_delta_ms or not capture_enabled:
                    continue

                result = run_onnx(session, left, right, image_size)
                if expression_session is not None:
                    next_sequence = sequence + 1
                    should_update_expression = (
                        last_expression_result is None
                        or next_sequence - last_expression_sequence >= expression_every_n
                    )
                    if should_update_expression:
                        expression_result = run_onnx(
                            expression_session,
                            left,
                            right,
                            expression_image_size,
                            output_names=["wide_lr", "squint_lr"],
                        )
                        wide_raw = expression_result["wide_lr"]
                        squint_raw = expression_result["squint_lr"]
                        if last_expression_result is not None and expression_ema_alpha < 1.0:
                            wide_raw = (
                                expression_ema_alpha * wide_raw
                                + (1.0 - expression_ema_alpha) * last_expression_result["wide_lr"]
                            ).astype(np.float32)
                            squint_raw = (
                                expression_ema_alpha * squint_raw
                                + (1.0 - expression_ema_alpha) * last_expression_result["squint_lr"]
                            ).astype(np.float32)
                        last_expression_result = {
                            "wide_lr": wide_raw.astype(np.float32),
                            "squint_lr": squint_raw.astype(np.float32),
                        }
                        last_expression_sequence = next_sequence
                    if last_expression_result is not None:
                        result["wide_lr"] = last_expression_result["wide_lr"]
                        result["squint_lr"] = last_expression_result["squint_lr"]
                model_openness = result["openness_lr"]
                if openness_cal is not None:
                    norm_openness = normalize_per_eye(model_openness, openness_cal[0], openness_cal[1])
                else:
                    norm_openness = model_openness
                curved = apply_openness_curve(
                    norm_openness,
                    args.openness_curve_mode,
                    args.openness_full_open_threshold,
                    args.openness_boost_knee,
                    args.openness_boost_gamma,
                )
                confidence = result["confidence"]
                frame_timestamp = max(left.received_at, right.received_at)
                raw = result["gaze_xy"]
                mapped = apply_runtime_map(raw, runtime_offset, runtime_gain, args.clamp, args.clamp_mode, args.soft_knee)
                mapped = output_mapper.update(
                    mapped,
                    timestamp=frame_timestamp,
                    confidence=float(confidence[2]),
                    openness=float(np.mean(curved)),
                )
                smooth = update_smooth(smooth, mapped, args.ema_alpha, args.max_step)
                tracking_result = tracking.update(curved, model_openness, confidence, smooth)
                openness = tracking_result.openness
                smooth = tracking_result.gaze

                sequence += 1
                snapshot_left = ""
                snapshot_right = ""
                if args.snapshot_every > 0 and sequence % args.snapshot_every == 0:
                    snapshot_left, snapshot_right = save_pair_snapshots(args.output_dir, sequence, left, right)
                elif (
                    args.snapshot_on_low_openness
                    and sequence - last_low_openness_snapshot_sequence >= max(1, args.snapshot_low_openness_min_interval)
                    and float(np.min(confidence[:2])) >= args.snapshot_low_openness_min_confidence
                    and float(np.min(openness)) < args.snapshot_low_openness_threshold
                ):
                    snapshot_left, snapshot_right = save_pair_snapshots(args.output_dir, sequence, left, right)
                    last_low_openness_snapshot_sequence = sequence

                pupil = result["pupil_lr"]
                if args.wide_source == "model_head":
                    model_wide = np.clip(result.get("wide_lr", np.zeros(2, dtype=np.float32)), 0.0, 1.0).astype(np.float32)
                    model_squint = np.clip(result.get("squint_lr", np.zeros(2, dtype=np.float32)), 0.0, 1.0).astype(np.float32)
                else:
                    model_wide, model_squint = eye_expression_from_runtime(
                        openness,
                        model_openness,
                        confidence,
                        args.wide_source,
                        args.wide_model_threshold,
                        args.wide_output_threshold,
                    )
                shape_wide = apply_eye_shape_curve(model_wide, args.eye_shape_wide_scale, args.eye_shape_gamma, args.eye_shape_deadzone)
                shape_squint = apply_eye_shape_curve(model_squint, args.eye_shape_squint_scale, args.eye_shape_gamma, args.eye_shape_deadzone)
                pupil_wide = update_pupil_wide_state(
                    model_wide,
                    pupil_wide_state,
                    pupil_wide_hold_remaining,
                    args.pupil_wide_enter_threshold,
                    args.pupil_wide_exit_threshold,
                    args.pupil_wide_hold_frames,
                    args.pupil_wide_ema_alpha,
                )
                left_pupil_output = pupil_output_state(pupil_output_mode, float(openness[0]), float(pupil_wide[0]), float(pupil[2]), float(confidence[0]))
                right_pupil_output = pupil_output_state(pupil_output_mode, float(openness[1]), float(pupil_wide[1]), float(pupil[5]), float(confidence[1]))
                if udp is not None or monitor_udp is not None:
                    packet = bridge_packet(
                        sequence,
                        delta_ms,
                        raw,
                        mapped,
                        smooth,
                        model_openness,
                        openness,
                        pupil,
                        confidence,
                        shape_wide,
                        shape_squint,
                        pupil_output_mode,
                        args.eye_expression_mode,
                        pupil_wide=pupil_wide,
                    )
                    if udp is not None and args.udp_port is not None:
                        udp.sendto(packet, (args.udp_host, args.udp_port))
                    if monitor_udp is not None and args.monitor_udp_port > 0:
                        monitor_udp.sendto(packet, (args.monitor_udp_host, args.monitor_udp_port))
                row = {
                    "sequence": sequence,
                    "timestamp": f"{time.time():.6f}",
                    "stage": stage,
                    "stage_elapsed": f"{stage_elapsed:.3f}",
                    "left_index": left.index,
                    "right_index": right.index,
                    "delta_ms": f"{delta_ms:.3f}",
                    "raw_x": f"{raw[0]:.6f}",
                    "raw_y": f"{raw[1]:.6f}",
                    "mapped_x": f"{mapped[0]:.6f}",
                    "mapped_y": f"{mapped[1]:.6f}",
                    "smooth_x": f"{smooth[0]:.6f}",
                    "smooth_y": f"{smooth[1]:.6f}",
                    "left_model_openness": f"{model_openness[0]:.6f}",
                    "right_model_openness": f"{model_openness[1]:.6f}",
                    "left_openness": f"{openness[0]:.6f}",
                    "right_openness": f"{openness[1]:.6f}",
                    "left_norm_openness": f"{norm_openness[0]:.6f}",
                    "right_norm_openness": f"{norm_openness[1]:.6f}",
                    "tracking_state": tracking_result.state,
                    "left_substate": tracking_result.left_substate,
                    "right_substate": tracking_result.right_substate,
                    "left_model_wide": f"{model_wide[0]:.6f}",
                    "right_model_wide": f"{model_wide[1]:.6f}",
                    "left_pupil_wide": f"{pupil_wide[0]:.6f}",
                    "right_pupil_wide": f"{pupil_wide[1]:.6f}",
                    "left_model_squint": f"{model_squint[0]:.6f}",
                    "right_model_squint": f"{model_squint[1]:.6f}",
                    "left_wide": f"{shape_wide[0]:.6f}",
                    "right_wide": f"{shape_wide[1]:.6f}",
                    "left_squint": f"{shape_squint[0]:.6f}",
                    "right_squint": f"{shape_squint[1]:.6f}",
                    "left_pupil_expression": f"{float(left_pupil_output['normalized']):.6f}",
                    "right_pupil_expression": f"{float(right_pupil_output['normalized']):.6f}",
                    "left_pupil_x": f"{pupil[0]:.6f}",
                    "left_pupil_y": f"{pupil[1]:.6f}",
                    "left_pupil_radius": f"{pupil[2]:.6f}",
                    "right_pupil_x": f"{pupil[3]:.6f}",
                    "right_pupil_y": f"{pupil[4]:.6f}",
                    "right_pupil_radius": f"{pupil[5]:.6f}",
                    "left_confidence": f"{confidence[0]:.6f}",
                    "right_confidence": f"{confidence[1]:.6f}",
                    "pair_confidence": f"{confidence[2]:.6f}",
                    "left_snapshot": snapshot_left,
                    "right_snapshot": snapshot_right,
                }
                writer.writerow(row)
                if sequence % max(1, args.print_every) == 0:
                    print(
                        f"{sequence:06d} {stage or '-':>10} "
                        f"raw=({raw[0]:+0.3f},{raw[1]:+0.3f}) "
                        f"smooth=({smooth[0]:+0.3f},{smooth[1]:+0.3f}) "
                        f"open=({openness[0]:0.2f},{openness[1]:0.2f}) "
                        f"model_open=({model_openness[0]:0.2f},{model_openness[1]:0.2f}) "
                        f"wide=({shape_wide[0]:0.2f},{shape_wide[1]:0.2f}) "
                        f"pupil=({float(left_pupil_output['normalized']):0.2f},{float(right_pupil_output['normalized']):0.2f}) "
                        f"conf=({confidence[0]:0.2f},{confidence[1]:0.2f},{confidence[2]:0.2f})"
                    )
                if sequence % 10 == 0:
                    handle.flush()
                    write_status(
                        status_path,
                        {
                            "state": "running",
                            "sequence": sequence,
                            "csv": str(csv_path.resolve()),
                            "onnx": str(args.onnx.resolve()),
                            "expression_onnx": str(args.expression_onnx.resolve()) if args.expression_onnx else None,
                            "expression_every_n_frames": expression_every_n if expression_session is not None else None,
                            "expression_ema_alpha": expression_ema_alpha if expression_session is not None else None,
                            "stage": stage,
                            "raw": [float(raw[0]), float(raw[1])],
                            "smooth": [float(smooth[0]), float(smooth[1])],
                            "model_openness": [float(model_openness[0]), float(model_openness[1])],
                            "openness": [float(openness[0]), float(openness[1])],
                            "openness_curve": {
                                "mode": args.openness_curve_mode,
                                "full_open_threshold": args.openness_full_open_threshold,
                                "boost_knee": args.openness_boost_knee,
                                "boost_gamma": args.openness_boost_gamma,
                            },
                            "tracking_state_machine": {
                                "openness_machine": args.tracking_openness_machine == "on",
                                "gaze_hold": bool(args.tracking_gaze_hold),
                                "calibration_loaded": openness_cal is not None,
                                "state": tracking_result.state,
                                "left_substate": tracking_result.left_substate,
                                "right_substate": tracking_result.right_substate,
                            },
                            "model_wide": [float(model_wide[0]), float(model_wide[1])],
                            "pupil_wide": [float(pupil_wide[0]), float(pupil_wide[1])],
                            "pupil_wide_curve": {
                                "enter_threshold": args.pupil_wide_enter_threshold,
                                "exit_threshold": args.pupil_wide_exit_threshold,
                                "hold_frames": args.pupil_wide_hold_frames,
                                "ema_alpha": args.pupil_wide_ema_alpha,
                            },
                            "model_squint": [float(model_squint[0]), float(model_squint[1])],
                            "wide": [float(shape_wide[0]), float(shape_wide[1])],
                            "squint": [float(shape_squint[0]), float(shape_squint[1])],
                            "pupil_output_mode": pupil_output_mode,
                            "eye_expression_mode": args.eye_expression_mode,
                            "eye_shape_curve": {
                                "wide_scale": args.eye_shape_wide_scale,
                                "squint_scale": args.eye_shape_squint_scale,
                                "gamma": args.eye_shape_gamma,
                                "deadzone": args.eye_shape_deadzone,
                            },
                            "wide_source": args.wide_source,
                            "wide_model_threshold": args.wide_model_threshold,
                            "wide_output_threshold": args.wide_output_threshold,
                            "pupil": [float(x) for x in pupil],
                            "pupil_expression": [
                                float(left_pupil_output["normalized"]),
                                float(right_pupil_output["normalized"]),
                            ],
                            "confidence": [float(x) for x in confidence],
                        },
                    )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        last_error = str(exc)
        print(f"ERROR: {last_error}", file=sys.stderr)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=1.0)
        if udp is not None:
            udp.close()
        if monitor_udp is not None:
            monitor_udp.close()

    write_status(
        status_path,
        {
            "state": "complete" if last_error is None else "error",
            "error": last_error,
            "csv": str(csv_path.resolve()),
            "onnx": str(args.onnx.resolve()),
            "expression_onnx": str(args.expression_onnx.resolve()) if args.expression_onnx else None,
            "expression_every_n_frames": expression_every_n if expression_session is not None else None,
            "expression_ema_alpha": expression_ema_alpha if expression_session is not None else None,
            "sequence_count": sequence,
            "duration_seconds": time.time() - started,
            "image_size": image_size,
            "pupil_output_mode": pupil_output_mode,
            "eye_expression_mode": args.eye_expression_mode,
            "eye_shape_curve": {
                "wide_scale": args.eye_shape_wide_scale,
                "squint_scale": args.eye_shape_squint_scale,
                "gamma": args.eye_shape_gamma,
                "deadzone": args.eye_shape_deadzone,
            },
            "wide_source": args.wide_source,
            "wide_model_threshold": args.wide_model_threshold,
            "wide_output_threshold": args.wide_output_threshold,
            "openness_curve": {
                "mode": args.openness_curve_mode,
                "full_open_threshold": args.openness_full_open_threshold,
                "boost_knee": args.openness_boost_knee,
                "boost_gamma": args.openness_boost_gamma,
            },
        },
    )
    print("")
    print(json.dumps(json.loads(status_path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    return 0 if last_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
