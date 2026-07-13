"""P0-2: pupil-recentered crop — geometric input normalization.

Why: the facial gasket / wear position moves the eye 10-17% inside the camera frame
(measured), so the model is forced to fit contradictory "wear x appearance -> output"
mappings; per-wear directional gaze collapse and eyelid mid-band breakdown follow.
This is the appearance-based-gaze "data normalization" step (MPIIGaze / ETH-XGaze
standard) reduced to 2D: shift the frame so the eye sits at a fixed canonical position
before the model ever sees it.

Anchor: dark centroid of the EYE REGION. Crucially the raw frames contain a static
vertical dark band from the headset rim (left-eye frames: right edge; right-eye frames:
left edge) that dominates a naive dark percentile (std_y ~0.27 vertical strip, identical
for open and closed) — so the estimator restricts its scan to a per-side region that
excludes the rim band and top/bottom shadow. Measured on real frames the masked centroid
is a compact blob (std ~0.06-0.11) that tracks wear shifts; the closed-eye centroid sits
within ~4% of the open one, so blink transients are absorbed by the EMA and confidence
only needs to catch degenerate frames.

Used by BOTH the Python runtime (temporal EMA/freeze) and the trainer (per-frame with a
per-session fallback). The C# native runtime mirrors this file 1:1 (PupilRecenter in
DreamAirTracking.Core.Runtime) — keep in sync, parity-tested via fixtures.
"""

from __future__ import annotations

import numpy as np

# Canonical per-eye centers (fraction of FULL frame), = mean masked dark-centroid of open
# frames across all ramp sessions (old + new gasket, 2026-07-13). Training and inference
# MUST use the same values.
CANONICAL_LEFT = (0.69, 0.75)
CANONICAL_RIGHT = (0.33, 0.69)

MAX_SHIFT_FRAC = 0.25
DARK_DELTA = 26           # threshold = region min + DARK_DELTA (the pupil is the darkest blob;
                          # a percentile threshold sweeps in soft skin/nose shadows instead)
TRIM_FRAC = 0.12          # median-refinement window (fraction of frame)
MIN_DARK_PIXELS = 20      # below this the estimate is degenerate -> confidence 0

# per-side scan region (fractions of frame) — excludes the static headset-rim band and
# top/bottom shadow. Rim measured from column-darkness profiles: left-eye frames have the
# rim at x>=0.77, right-eye at x<=0.26.
LEFT_EYE_REGION = (0.00, 0.76, 0.06, 0.94)    # x0, x1, y0, y1
RIGHT_EYE_REGION = (0.27, 1.00, 0.06, 0.94)


def eye_region(is_left: bool, height: int, width: int) -> tuple[int, int, int, int]:
    x0f, x1f, y0f, y1f = LEFT_EYE_REGION if is_left else RIGHT_EYE_REGION
    return int(x0f * width), int(x1f * width), int(y0f * height), int(y1f * height)


def estimate_dark_centroid(gray: np.ndarray, is_left: bool) -> tuple[float, float, float]:
    """Return (cx_frac, cy_frac, confidence) of the pupil blob inside the eye region.

    Pipeline (deterministic integer math, mirrored exactly in C#):
    1. threshold = region minimum + DARK_DELTA — the pupil is the darkest structure;
       percentile thresholds sweep in soft skin/nose shadows and are unstable under shift;
    2. structural rejection: drop columns/rows whose dark run covers >50% of the region
       (headset-rim penumbra = full-height columns, border shadow = full-width rows);
    3. per-axis MEDIAN of the surviving dark pixels (robust to truncated lash tails);
    4. one refinement pass: re-median inside a ±TRIM_FRAC window around the first median
       (drops disconnected shadow patches).
    Validated: per-session median anchor tracks real gasket shifts (y Δ0.11 across 7
    sessions) with per-frame jitter ~4% that the runtime EMA averages out.
    """
    h, w = gray.shape
    x0, x1, y0, y1 = eye_region(is_left, h, w)
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.5, 0.5, 0.0
    thr = int(sub.min()) + DARK_DELTA
    mask = sub <= thr
    sub_h, sub_w = mask.shape
    col_counts = mask.sum(axis=0)
    row_counts = mask.sum(axis=1)
    mask[:, col_counts > sub_h * 0.5] = False
    mask[row_counts > sub_w * 0.5, :] = False

    def _median(m: np.ndarray) -> "tuple[int, int] | None":
        ys, xs = np.where(m)
        if xs.size < MIN_DARK_PIXELS:
            return None
        half = (xs.size + 1) // 2
        mx = int(np.searchsorted(np.cumsum(np.bincount(xs, minlength=sub_w)), half))
        my = int(np.searchsorted(np.cumsum(np.bincount(ys, minlength=sub_h)), half))
        return mx, my

    first = _median(mask)
    if first is None:
        return 0.5, 0.5, 0.0
    tw, th_ = int(TRIM_FRAC * w), int(TRIM_FRAC * h)
    box = np.zeros_like(mask)
    yl, yh2 = max(0, first[1] - th_), min(sub_h, first[1] + th_)
    xl, xh2 = max(0, first[0] - tw), min(sub_w, first[0] + tw)
    box[yl:yh2, xl:xh2] = mask[yl:yh2, xl:xh2]
    refined = _median(box) or first
    return (refined[0] + x0) / w, (refined[1] + y0) / h, 1.0


def _round(x: float) -> int:
    # floor(x+0.5): deterministic and identical to the C# mirror (python round() is
    # banker's rounding and C# AwayFromZero differs on negatives)
    return int(np.floor(x + 0.5))


def recenter_gray(
    gray: np.ndarray,
    center_frac: tuple[float, float],
    canonical_frac: tuple[float, float],
    max_shift_frac: float = MAX_SHIFT_FRAC,
) -> np.ndarray:
    """Shift the image so center_frac lands on canonical_frac (clamped), edge fill = mean.

    Integer-pixel shift (no resampling) so the downstream Pillow-exact resize stays the
    only interpolation in the pipeline — identical math on both runtimes.
    """
    h, w = gray.shape
    dx = _round((canonical_frac[0] - center_frac[0]) * w)
    dy = _round((canonical_frac[1] - center_frac[1]) * h)
    max_dx = _round(max_shift_frac * w)
    max_dy = _round(max_shift_frac * h)
    dx = max(-max_dx, min(max_dx, dx))
    dy = max(-max_dy, min(max_dy, dy))
    if dx == 0 and dy == 0:
        return gray
    fill = min(255, max(0, _round(float(gray.mean()))))
    out = np.full_like(gray, fill)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_x0, dst_y0 = max(0, dx), max(0, dy)
    out[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = \
        gray[src_y0:src_y1, src_x0:src_x1]
    return out


class PupilRecenterState:
    """Runtime temporal state: EMA while the estimate is non-degenerate, freeze while not.

    Blink transients only move the masked centroid by ~4% (measured) and are further
    damped by the EMA; the freeze protects against fully degenerate frames.
    """

    def __init__(self, ema_alpha: float = 0.1) -> None:
        self._alpha = float(ema_alpha)
        self._center: "tuple[float, float] | None" = None

    def update(self, cx: float, cy: float, confidence: float) -> "tuple[float, float] | None":
        if confidence > 0.5:
            if self._center is None:
                self._center = (cx, cy)
            else:
                px, py = self._center
                self._center = (px + (cx - px) * self._alpha, py + (cy - py) * self._alpha)
        return self._center  # None until the first confident estimate -> caller skips recentering


def recenter_frame(
    gray: np.ndarray,
    state: PupilRecenterState,
    canonical_frac: tuple[float, float],
    is_left: bool,
) -> np.ndarray:
    """Runtime path: estimate -> temporal update -> shift. No-op until first lock."""
    cx, cy, conf = estimate_dark_centroid(gray, is_left)
    center = state.update(cx, cy, conf)
    if center is None:
        return gray
    return recenter_gray(gray, center, canonical_frac)
