"""Generate parity fixtures for the E11 C# post-processing port.

Runs the REAL Python runtime post-processing functions on known inputs (incl.
sequences for stateful components) and dumps a JSON the C# unit tests replay and
assert against (tolerance 1e-5). Source of truth = predict_live_multitask.py +
live_gaze_dry_run.py.

Run:  python scripts/ml/tests/gen_e11_parity_fixtures.py
Out:  tests/DreamAirTracking.Tests/Runtime/fixtures/postprocess_parity.json
"""
import json
from pathlib import Path

import numpy as np

import sys
ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

import predict_live_multitask as P  # noqa: E402
import live_gaze_dry_run as L  # noqa: E402

OUT = ML.parents[1] / "tests" / "DreamAirTracking.Tests" / "Runtime" / "fixtures" / "postprocess_parity.json"


def arr(a):
    return [float(x) for x in np.asarray(a, dtype=np.float64).ravel()]


fx = {}

# ---- openness_curve (apply_openness_curve) ----
oc = []
vals = [[0.0, 0.0], [0.1, 0.9], [0.28, 0.5], [0.5, 0.85], [0.9, 0.95], [1.0, 0.7], [0.3, 0.31]]
for mode in ["off", "soft_open_plateau", "blink_s_curve", "full_open_plateau"]:
    for v in vals:
        out = P.apply_openness_curve(np.array(v, dtype=np.float32), mode, 0.90, 0.28, 1.25)
        oc.append({"input": v, "mode": mode, "threshold": 0.90, "knee": 0.28, "gamma": 1.25, "output": arr(out)})
# a couple with different params
for v in [[0.4, 0.6], [0.7, 0.2]]:
    out = P.apply_openness_curve(np.array(v, dtype=np.float32), "blink_s_curve", 0.80, 0.20, 1.5)
    oc.append({"input": v, "mode": "blink_s_curve", "threshold": 0.80, "knee": 0.20, "gamma": 1.5, "output": arr(out)})
fx["openness_curve"] = oc

# ---- normalize_per_eye ----
npe = []
cases = [
    ([0.5, 0.8], [0.95, 0.9], [0.1, 0.05]),
    ([0.2, 1.0], [0.9, 0.92], [0.2, 0.1]),
    ([0.5, 0.5], [0.52, 0.9], [0.5, 0.05]),   # left degenerate range -> passthrough
    ([0.0, 0.3], [1.0, 0.7], [0.0, 0.6]),     # right lo>hi-ish
]
for m, hi, lo in cases:
    out = P.normalize_per_eye(np.array(m, dtype=np.float32), np.array(hi, dtype=np.float32), np.array(lo, dtype=np.float32))
    npe.append({"model": m, "open_p95": hi, "closed_p05": lo, "min_range": 0.05, "output": arr(out)})
fx["normalize_per_eye"] = npe

# ---- normalize_per_eye with S2 half anchor (piecewise) ----
npeh = []
half_cases = [
    ([0.7, 0.9], [1.0, 1.0], [0.1, 0.68], [0.7, 0.9]),       # anchors pinned to 0.5 (right = collapsed range)
    ([0.4, 0.95], [1.0, 1.0], [0.1, 0.68], [0.7, 0.9]),      # below/above anchor
    ([0.55, 0.55], [1.0, 1.0], [0.1, 0.1], [0.11, 0.999]),   # degenerate segments -> linear fallback
    ([0.3, 0.8], [0.95, 0.9], [0.05, 0.1], [0.5, 0.55]),
]
for m, hi, lo, half in half_cases:
    out = P.normalize_per_eye(
        np.array(m, dtype=np.float32), np.array(hi, dtype=np.float32), np.array(lo, dtype=np.float32),
        half_p50=np.array(half, dtype=np.float32))
    npeh.append({"model": m, "open_p95": hi, "closed_p05": lo, "half_p50": half, "min_range": 0.05, "output": arr(out)})
fx["normalize_per_eye_half"] = npeh

# ---- openness dual_path (S1, stateful sequence) ----
def dual_path_seq(seq, threshold=0.90, knee=0.28, gamma=1.25, enter=0.06, exit_=0.02, hold=6):
    dp = P.OpennessDualPath()
    steps = []
    for v in seq:
        out = dp.apply(np.array(v, dtype=np.float32), threshold, knee, gamma, enter, exit_, hold)
        steps.append({"input": v, "output": arr(out)})
    return steps

dp_seq = (
    [[1.0, 1.0]] * 3
    + [[round(1.0 - 0.01 * i, 4)] * 2 for i in range(1, 41)]   # slow ramp to 0.6 (hover path)
    + [[0.6, 0.6], [0.2, 0.2], [0.03, 0.03], [0.03, 0.03]]      # blink drop (fast path)
    + [[0.5, 0.5], [0.97, 0.97], [1.0, 1.0], [1.0, 1.0]]        # reopen
)
fx["openness_dual_path"] = [{
    "threshold": 0.90, "knee": 0.28, "gamma": 1.25,
    "enter_velocity": 0.06, "exit_velocity": 0.02, "hold_frames": 6,
    "steps": dual_path_seq(dp_seq),
}]

# ---- wide hysteresis (L3, stateful sequence) ----
def wide_hyst_seq(seq, confirm=4, enter=0.22, exit_=0.12):
    wh = P.WideHysteresis()
    steps = []
    for v in seq:
        out = wh.apply(np.array(v, dtype=np.float32), confirm, enter, exit_)
        steps.append({"input": v, "output": arr(out)})
    return steps

wh_seq = (
    [[0.0, 0.0]] * 2
    + [[0.3, 0.05], [0.0, 0.05]]                 # 1-frame transient: must never fire
    + [[0.3, 0.0]] * 3 + [[0.05, 0.0]]           # 3 frames < confirm=4: still suppressed, then reset
    + [[0.35, 0.0]] * 5                          # sustained: fires on frame 4
    + [[0.30, 0.0], [0.08, 0.0], [0.3, 0.0]]     # dip below exit -> instant release, needs re-confirm
    + [[0.0, 0.4]] * 5 + [[0.0, 0.0]]            # right eye independent
)
fx["wide_hysteresis"] = [{
    "confirm_frames": 4, "enter_threshold": 0.22, "exit_threshold": 0.12,
    "steps": wide_hyst_seq(wh_seq),
}]

# ---- P0-2 pupil recenter (centroid + shift math parity) ----
import pupil_recenter as PR

def synth(h, w, cx, cy, r, bg, dark):
    img = np.full((h, w), bg, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    img[(xx - cx * w) ** 2 + (yy - cy * h) ** 2 <= r * r] = dark
    return img

pr_cases = []
for (h, w, cx, cy, r, bg, dark, canon, is_left) in [
    (60, 80, 0.50, 0.40, 6, 160, 12, PR.CANONICAL_LEFT, True),
    (60, 80, 0.55, 0.70, 5, 140, 8, PR.CANONICAL_RIGHT, False),
    (48, 48, 0.62, 0.71, 4, 200, 20, PR.CANONICAL_LEFT, True),    # near-zero shift
    (60, 80, 0.35, 0.10, 5, 150, 10, PR.CANONICAL_LEFT, True),    # clamped shift (dy huge)
    (60, 80, 0.62, 0.50, 0, 150, 150, PR.CANONICAL_LEFT, True),   # no dark blob -> degenerate
]:
    img = synth(h, w, cx, cy, r, bg, dark)
    ecx, ecy, conf = PR.estimate_dark_centroid(img, is_left)
    out = PR.recenter_gray(img, (ecx, ecy), canon) if conf > 0.5 else img
    pr_cases.append({
        "height": h, "width": w, "is_left": is_left,
        "image": img.reshape(-1).tolist(),
        "canonical": list(canon),
        "centroid": [float(ecx), float(ecy), float(conf)],
        "output": out.reshape(-1).tolist(),
    })
fx["pupil_recenter"] = pr_cases

# ---- eye_shape_curve ----
esc = []
for v in [[0.0, 0.5], [0.3, 0.9], [0.6, 1.0], [0.03, 0.05]]:
    for scale, gamma, dz in [(0.60, 1.15, 0.03), (0.55, 1.0, 0.0), (1.0, 2.0, 0.1)]:
        out = P.apply_eye_shape_curve(np.array(v, dtype=np.float32), scale, gamma, dz)
        esc.append({"values": v, "scale": scale, "gamma": gamma, "deadzone": dz, "output": arr(out)})
fx["eye_shape_curve"] = esc

# ---- runtime_map (apply_runtime_map) ----
rm = []
for v in [[0.0, 0.0], [0.5, -0.3], [1.2, 0.8], [-0.9, 1.5]]:
    for clamp, mode, knee in [(1.0, "hard", 0.72), (1.0, "soft", 0.72), (0.0, "hard", 0.72)]:
        out = L.apply_runtime_map(np.array(v, dtype=np.float32), np.array([0.02, -0.01], dtype=np.float32),
                                  np.array([1.1, 0.9], dtype=np.float32), clamp, mode, knee)
        rm.append({"value": v, "offset": [0.02, -0.01], "gain": [1.1, 0.9], "clamp": clamp,
                   "clamp_mode": mode, "soft_knee": knee, "output": arr(out)})
fx["runtime_map"] = rm

# ---- update_smooth (stateful via previous threading) ----
def smooth_seq(currents, alpha, max_step):
    prev = None
    steps = []
    for c in currents:
        out = L.update_smooth(prev, np.array(c, dtype=np.float32), alpha, max_step)
        steps.append({"current": c, "output": arr(out)})
        prev = np.asarray(out, dtype=np.float32)
    return steps

seq = [[0.0, 0.0], [0.3, 0.1], [0.31, 0.12], [0.9, -0.5], [0.85, -0.48], [0.0, 0.0]]
fx["update_smooth"] = [
    {"alpha": 0.35, "max_step": 0.0, "steps": smooth_seq(seq, 0.35, 0.0)},
    {"alpha": 0.35, "max_step": 0.05, "steps": smooth_seq(seq, 0.35, 0.05)},
]

# ---- OutputGazeMapper (stateful) ----
def ogm_seq(cfg, steps_in):
    m = L.OutputGazeMapper(**cfg)
    out = []
    for s in steps_in:
        r = m.update(np.array(s["value"], dtype=np.float32), timestamp=s["t"],
                     confidence=s["conf"], openness=s["open"])
        out.append({**s, "output": arr(r)})
    return out

steps_in = [
    {"value": [0.0, 0.0], "t": 0.0, "conf": 1.0, "open": 1.0},
    {"value": [0.02, 0.01], "t": 0.016, "conf": 1.0, "open": 1.0},
    {"value": [0.5, -0.3], "t": 0.032, "conf": 0.9, "open": 0.9},
    {"value": [0.03, 0.02], "t": 0.048, "conf": 0.9, "open": 0.9},
    {"value": [-0.7, 0.6], "t": 0.064, "conf": 0.2, "open": 0.3},
]
cfg_off = {"mode": "off", "deadzone": 0.015, "gamma": 1.0, "limit": 1.0}
cfg_sc = {"mode": "stable_center", "deadzone": 0.015, "gamma": 1.0, "limit": 1.0,
          "center_radius": 0.08, "center_tau": 6.0, "center_max_step": 0.025}
fx["output_gaze_mapper"] = [
    {"config": cfg_off, "steps": ogm_seq(cfg_off, steps_in)},
    {"config": cfg_sc, "steps": ogm_seq(cfg_sc, steps_in)},
]

# ---- pupil_wide (stateful) ----
def pw_seq(raws, enter, exit_, holds, ema):
    state = np.zeros(2, dtype=np.float32)
    hold = np.zeros(2, dtype=np.int64)
    steps = []
    for r in raws:
        out = P.update_pupil_wide_state(np.array(r, dtype=np.float32), state, hold, enter, exit_, holds, ema)
        state = np.asarray(out, dtype=np.float32)
        steps.append({"raw": r, "output": arr(out)})
    return steps

raws = [[0.0, 0.0], [0.95, 0.5], [0.88, 0.92], [0.87, 0.1], [0.5, 0.05], [0.0, 0.0], [0.0, 0.0]]
fx["pupil_wide"] = [{"enter": 0.90, "exit": 0.86, "hold_frames": 8, "ema": 0.45, "steps": pw_seq(raws, 0.90, 0.86, 8, 0.45)}]

# ---- pupil_output ----
po = []
for mode_raw in ["off", "assist", "expression_constrict_on_wide", "diameter", "model_radius"]:
    mode = P.normalize_pupil_output_mode(mode_raw)
    for openness, wide, radius, conf in [(1.0, 0.0, 0.1, 1.0), (1.0, 0.8, 0.1, 1.0), (0.3, 0.5, 0.1, 1.0),
                                          (0.9, 0.2, 0.01, 0.9), (0.9, 0.2, 0.2, 0.02)]:
        r = P.pupil_output_state(mode, openness, wide, radius, conf)
        po.append({"mode_raw": mode_raw, "mode": mode, "openness": openness, "wide": wide,
                   "radius": radius, "confidence": conf,
                   "output": {"found": bool(r["found"]), "quality": bool(r["quality"]),
                              "normalized": float(r["normalized"]),
                              "expressionNormalized": float(r["expressionNormalized"]),
                              "reason": r["reason"]}})
fx["pupil_output"] = po

# ---- TrackingStateMachine (stateful sequences) ----
def tracking_seq(params, ema, seq):
    m = P.TrackingStateMachine(params, ema)
    steps = []
    for curved, pair_conf, gaze in seq:
        conf = np.array([pair_conf, pair_conf, pair_conf], dtype=np.float32)
        r = m.update(np.array(curved, dtype=np.float32), np.array(curved, dtype=np.float32), conf, np.array(gaze, dtype=np.float32))
        steps.append({"curved": curved, "pair_conf": pair_conf, "gaze": gaze,
                      "out_openness": arr(r.openness), "out_gaze": arr(r.gaze),
                      "state": r.state, "left_sub": r.left_substate, "right_sub": r.right_substate})
    return steps

# rich sequence: open, tiny motion (fixate), saccade, blink (both drop), one-eye close, closed, reopen
tseq = [
    ([1.0, 1.0], 0.9, [0.0, 0.0]),
    ([1.0, 1.0], 0.9, [0.005, 0.0]),
    ([0.98, 0.99], 0.9, [0.01, 0.005]),
    ([0.97, 0.98], 0.9, [0.4, -0.2]),      # saccade
    ([0.6, 0.1], 0.9, [0.41, -0.21]),      # blink-ish drop
    ([0.15, 0.12], 0.9, [0.42, -0.22]),    # closed
    ([0.1, 0.9], 0.9, [0.4, -0.2]),        # one eye closed
    ([0.5, 0.95], 0.9, [0.3, -0.1]),       # reopening
    ([0.95, 0.98], 0.03, [0.1, 0.0]),      # low confidence -> REACQUIRE
    ([0.99, 0.99], 0.9, [0.1, 0.0]),
]
params_on = P.TrackingStateParams(openness_machine=True, gaze_hold=True)
params_off = P.TrackingStateParams(openness_machine=False, gaze_hold=False)
# P0-3: shake dips (never confirm), a real blink (confirms), half-open jitter (deadband holds)
tseq_p03 = [
    ([1.0, 1.0], 0.9, [0.0, 0.0]),
    ([0.4, 0.5], 0.9, [0.0, 0.02]),        # shake dip starts
    ([0.1, 0.15], 0.9, [0.0, -0.02]),      # below threshold (1 frame)
    ([0.05, 0.1], 0.9, [0.0, 0.03]),       # below threshold (2 frames) — still unconfirmed
    ([0.8, 0.9], 0.9, [0.0, 0.0]),         # recovered: dip must not have read closed
    ([1.0, 1.0], 0.9, [0.0, 0.0]),
    ([0.5, 0.5], 0.9, [0.0, 0.0]),         # settle into half-open band
    ([0.52, 0.48], 0.9, [0.0, 0.0]),       # jitter within deadband
    ([0.49, 0.53], 0.9, [0.0, 0.0]),
    ([0.15, 0.12], 0.9, [0.0, 0.0]),       # real closure starts (1)
    ([0.05, 0.05], 0.9, [0.0, 0.0]),       # (2)
    ([0.02, 0.02], 0.9, [0.0, 0.0]),       # (3) -> confirmed
    ([0.02, 0.02], 0.9, [0.0, 0.0]),       # fast both-closed fall now active
    ([0.9, 0.95], 0.9, [0.0, 0.0]),        # reopen
]
params_p03 = P.TrackingStateParams(
    openness_machine=True, gaze_hold=True,
    close_confirm_frames=3, close_confirm_floor=0.22, partial_deadband=0.04,
)
fx["tracking"] = [
    {"openness_machine": True, "gaze_hold": True, "ema": 0.35, "steps": tracking_seq(params_on, 0.35, tseq)},
    {"openness_machine": False, "gaze_hold": False, "ema": 0.35, "steps": tracking_seq(params_off, 0.35, tseq)},
    {"openness_machine": True, "gaze_hold": True, "ema": 0.35,
     "close_confirm_frames": 3, "close_confirm_floor": 0.22, "partial_deadband": 0.04,
     "steps": tracking_seq(params_p03, 0.35, tseq_p03)},
]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(fx, indent=1), encoding="utf-8")
print(f"wrote {OUT}")
for k, v in fx.items():
    n = len(v) if isinstance(v, list) else 1
    print(f"  {k}: {n} groups")
