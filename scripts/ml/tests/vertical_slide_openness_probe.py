"""Acceptance probe: openness robustness to vertical headset slide (false-close).

All input frames are OPEN eyes. We translate each eye image vertically (edge-replicated
fill) to simulate the headset sliding up/down, and measure predicted openness. A robust
model keeps openness high across shifts; a model that false-closes on slide drops.

Metric per model: mean openness + false-close rate (frac frames openness<0.5) vs shift.
This is the acceptance test for the P0/P1 openness-fit-robustness work (shipped vs C vs C'').

Run:  python scripts/ml/tests/vertical_slide_openness_probe.py
"""
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS_DIR = Path(r"C:\Users\Zhong\AppData\Local\DreamAirTracking\models")
FRAMES = Path(r"C:\Users\Zhong\AppData\Local\Temp\claude\C--Users-Zhong-Desktop-SumiruiTracking\4f29c9c5-2603-476a-805e-5864c2f38245\scratchpad\testset_frames")
MANIFEST = Path(r"C:\Users\Zhong\Desktop\SumiruiTracking\hf\testset\manifest_test.csv")
SIZE = 128
SHIFTS = [-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]  # frac of source H; +down / -up

MODELS = [
    ("shipped", MODELS_DIR / "dreamair-main-current" / "model.onnx"),
    ("C(clean-v6)", MODELS_DIR / "dreamair-cclean-current" / "model.onnx"),
    ("C''(+P0-1 aug)", MODELS_DIR / "dreamair-cclean2-current" / "model.onnx"),
    ("C'''(+P0-1+P1-7)", MODELS_DIR / "dreamair-cclean3-current" / "model.onnx"),
]


def load_open_frames(per_stage=6):
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_stage = {}
    for r in rows:
        by_stage.setdefault(r.get("stage"), []).append(r)
    picked = []
    for st in ["center", "left", "right", "up", "down"]:
        for r in by_stage.get(st, [])[:per_stage]:
            lp, rp = FRAMES / r["left_file"], FRAMES / r["right_file"]
            if lp.exists() and rp.exists():
                picked.append((lp, rp))
    return picked


def shift_vertical(arr: np.ndarray, dy: int) -> np.ndarray:
    """Shift image content vertically by dy px (+down), edge-replicated fill."""
    if dy == 0:
        return arr
    h = arr.shape[0]
    out = np.empty_like(arr)
    if dy > 0:
        out[dy:] = arr[: h - dy]
        out[:dy] = arr[0]
    else:
        d = -dy
        out[: h - d] = arr[d:]
        out[h - d :] = arr[-1]
    return out


def prep(path: Path, shift_frac: float) -> np.ndarray:
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.uint8)
    dy = int(round(shift_frac * arr.shape[0]))
    arr = shift_vertical(arr, dy)
    im2 = Image.fromarray(arr).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(im2, dtype=np.float32) / 255.0


def run(sess, la, ra):
    batch = np.stack([la, ra], axis=0)[None].astype(np.float32)
    o = sess.run(["openness_lr"], {"image": batch})[0][0]
    return float(o[0]), float(o[1])


def main():
    frames = load_open_frames()
    print(f"loaded {len(frames)} OPEN-eye frame pairs; simulating vertical slide (+down / -up)\n")
    for label, path in MODELS:
        if not path.exists():
            print(f"[skip] {label}: {path} not found\n")
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        print(f"=== {label} ===")
        print(f"  {'shift':>7} {'mean_open':>10} {'false_close_rate':>17}")
        for s in SHIFTS:
            ops = []
            for lp, rp in frames:
                l, r = run(sess, prep(lp, s), prep(rp, s))
                ops.append((l + r) / 2.0)
            ops = np.array(ops)
            fc = float(np.mean(ops < 0.5))
            print(f"  {s:>+7.2f} {ops.mean():10.3f} {fc:17.2f}")
        print()


if __name__ == "__main__":
    main()
