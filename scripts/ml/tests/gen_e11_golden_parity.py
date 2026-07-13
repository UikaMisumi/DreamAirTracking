"""Generate the E11 end-to-end golden-parity reference.

Feeds a fixed, deterministic sequence of (left,right) testset frame pairs through the
EXACT predict_live_multitask post-processing loop (same functions, same order as
PLM:1015-1136) and dumps the per-frame VRCFT-consumed fields. The C# golden test
(GoldenParityTests) drives NeuralEyeRuntime.ProcessPair on the same pairs and asserts
the consumed fields match within 1e-4 — validating preprocessing (ImageSharp vs PIL) +
ONNX inference + the full post-processing chain end-to-end.

Run:  python scripts/ml/tests/gen_e11_golden_parity.py
Out:  <scratchpad>/e11_golden_parity.json
"""
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

import sys
ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
import predict_live_multitask as P  # noqa: E402
import live_gaze_dry_run as L  # noqa: E402

LOCALAPP = Path(r"C:\Users\Zhong\AppData\Local\DreamAirTracking\models")
MAIN = LOCALAPP / "dreamair-main-current" / "model.onnx"
MAIN_META = LOCALAPP / "dreamair-main-current" / "metadata.json"
EXPR = LOCALAPP / "dreamair-expression-current" / "model.onnx"
EXPR_META = LOCALAPP / "dreamair-expression-current" / "metadata.json"
FRAMES = Path(r"C:\Users\Zhong\AppData\Local\Temp\claude\C--Users-Zhong-Desktop-SumiruiTracking\4f29c9c5-2603-476a-805e-5864c2f38245\scratchpad\testset_frames")
MANIFEST = Path(r"C:\Users\Zhong\Desktop\SumiruiTracking\hf\testset\manifest_test.csv")
OUT = Path(r"C:\Users\Zhong\AppData\Local\Temp\claude\C--Users-Zhong-Desktop-SumiruiTracking\4f29c9c5-2603-476a-805e-5864c2f38245\scratchpad\e11_golden_parity.json")

IMAGE_SIZE = P.load_image_size(MAIN_META, 128)
EXPR_SIZE = P.load_image_size(EXPR_META, IMAGE_SIZE)

# pick a spread across stages
rows = []
with open(MANIFEST, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
by_stage = {}
for r in rows:
    by_stage.setdefault(r.get("stage"), []).append(r)
pairs = []
for st, srows in sorted(by_stage.items()):
    for r in srows[:6]:
        pairs.append(r)
pairs = pairs[:48]

session = ort.InferenceSession(str(MAIN), sess_options=P.make_session_options(2), providers=["CPUExecutionProvider"])
expr_session = ort.InferenceSession(str(EXPR), sess_options=P.make_session_options(2), providers=["CPUExecutionProvider"])

# config = app defaults (mirror NeuralRuntimeConfig defaults)
runtime_offset = np.array([0.0, 0.0], dtype=np.float32)
runtime_gain = np.array([1.0, 1.0], dtype=np.float32)
CLAMP, CLAMP_MODE, SOFT_KNEE = 1.0, "soft", 0.72
EMA_ALPHA, MAX_STEP = 0.35, 0.0
EXPR_EVERY_N, EXPR_EMA = 3, 0.65
PUPIL_MODE = "off"
EYE_EXPR_MODE = "off"

output_mapper = L.OutputGazeMapper(mode="off", deadzone=0.015, gamma=1.0, limit=1.0,
                                   center_radius=0.08, center_tau=6.0, center_max_step=0.025)
params = P.TrackingStateParams(openness_machine=True, gaze_hold=True)
tracking = P.TrackingStateMachine(params, EMA_ALPHA)
pupil_wide_state = np.zeros(2, dtype=np.float32)
pupil_wide_hold = np.zeros(2, dtype=np.int64)

sequence = 0
last_expr = None
last_expr_seq = 0
smooth = None
out_frames = []

for i, r in enumerate(pairs):
    lf = FRAMES / r["left_file"]
    rf = FRAMES / r["right_file"]
    if not lf.exists() or not rf.exists():
        continue
    left = L.JpegFrame("left", i, float(i), lf.read_bytes())
    right = L.JpegFrame("right", i, float(i), rf.read_bytes())

    result = P.run_onnx(session, left, right, IMAGE_SIZE)
    next_sequence = sequence + 1
    should = last_expr is None or next_sequence - last_expr_seq >= EXPR_EVERY_N
    if should:
        er = P.run_onnx(expr_session, left, right, EXPR_SIZE, output_names=["wide_lr", "squint_lr"])
        wide_raw, squint_raw = er["wide_lr"], er["squint_lr"]
        if last_expr is not None and EXPR_EMA < 1.0:
            wide_raw = (EXPR_EMA * wide_raw + (1 - EXPR_EMA) * last_expr["wide_lr"]).astype(np.float32)
            squint_raw = (EXPR_EMA * squint_raw + (1 - EXPR_EMA) * last_expr["squint_lr"]).astype(np.float32)
        last_expr = {"wide_lr": wide_raw.astype(np.float32), "squint_lr": squint_raw.astype(np.float32)}
        last_expr_seq = next_sequence
    if last_expr is not None:
        result["wide_lr"] = last_expr["wide_lr"]
        result["squint_lr"] = last_expr["squint_lr"]

    model_openness = result["openness_lr"]
    norm_openness = model_openness
    curved = P.apply_openness_curve(norm_openness, "blink_s_curve", 0.90, 0.28, 1.25)
    confidence = result["confidence"]
    raw = result["gaze_xy"]
    mapped = L.apply_runtime_map(raw, runtime_offset, runtime_gain, CLAMP, CLAMP_MODE, SOFT_KNEE)
    mapped = output_mapper.update(mapped, timestamp=float(i), confidence=float(confidence[2]), openness=float(np.mean(curved)))
    smooth = L.update_smooth(smooth, mapped, EMA_ALPHA, MAX_STEP)
    tr = tracking.update(curved, model_openness, confidence, smooth)
    openness = tr.openness
    smooth = tr.gaze
    sequence += 1

    pupil = result["pupil_lr"]
    model_wide = np.clip(result.get("wide_lr", np.zeros(2, dtype=np.float32)), 0.0, 1.0).astype(np.float32)
    model_squint = np.clip(result.get("squint_lr", np.zeros(2, dtype=np.float32)), 0.0, 1.0).astype(np.float32)
    shape_wide = P.apply_eye_shape_curve(model_wide, 0.60, 1.15, 0.03)
    shape_squint = P.apply_eye_shape_curve(model_squint, 0.55, 1.15, 0.03)
    pupil_wide = P.update_pupil_wide_state(model_wide, pupil_wide_state, pupil_wide_hold, 0.90, 0.86, 8, 0.45)
    lpo = P.pupil_output_state(PUPIL_MODE, float(openness[0]), float(pupil_wide[0]), float(pupil[2]), float(confidence[0]))
    rpo = P.pupil_output_state(PUPIL_MODE, float(openness[1]), float(pupil_wide[1]), float(pupil[5]), float(confidence[1]))

    out_frames.append({
        "left_file": str(lf), "right_file": str(rf),
        "sequence": sequence,
        "normalizedX": float(smooth[0]), "normalizedY": float(smooth[1]),
        "rawX": float(raw[0]), "rawY": float(raw[1]),
        "monocularX": float(mapped[0]), "monocularY": float(mapped[1]),
        "opennessL": float(openness[0]), "opennessR": float(openness[1]),
        "wideL": float(P.clamp01(shape_wide[0])), "wideR": float(P.clamp01(shape_wide[1])),
        "squintL": float(P.clamp01(shape_squint[0])), "squintR": float(P.clamp01(shape_squint[1])),
        "pupilNormL": float(lpo["normalized"]), "pupilNormR": float(rpo["normalized"]),
        "pupilExprL": float(lpo["expressionNormalized"]), "pupilExprR": float(rpo["expressionNormalized"]),
        "state": tr.state,
    })

fixture = {
    "main_onnx": str(MAIN), "expression_onnx": str(EXPR),
    "image_size": IMAGE_SIZE, "expression_image_size": EXPR_SIZE,
    "frames": out_frames,
}
OUT.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
print(f"wrote {OUT} with {len(out_frames)} frames (image_size={IMAGE_SIZE}, expr_size={EXPR_SIZE})")
