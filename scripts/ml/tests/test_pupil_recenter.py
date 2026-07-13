"""Unit tests for P0-2 pupil recentering (pure numpy, no hardware).

Run: python -m pytest scripts/ml/tests -q
"""

import numpy as np

from pupil_recenter import (
    CANONICAL_LEFT,
    MAX_SHIFT_FRAC,
    PupilRecenterState,
    estimate_dark_centroid,
    recenter_frame,
    recenter_gray,
)


def synthetic_eye(h=200, w=200, cx=0.5, cy=0.5, r=12, bg=160, pupil=10):
    """Bright background with a dark pupil disk at (cx, cy) fractions."""
    img = np.full((h, w), bg, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - cx * w) ** 2 + (yy - cy * h) ** 2 <= r * r
    img[mask] = pupil
    return img


def test_centroid_finds_dark_disk_in_region():
    img = synthetic_eye(cx=0.50, cy=0.41)   # inside the left-eye scan region
    cx, cy, conf = estimate_dark_centroid(img, is_left=True)
    assert conf == 1.0
    assert abs(cx - 0.50) < 0.02 and abs(cy - 0.41) < 0.02


def test_centroid_ignores_rim_band():
    # a huge dark band at the rim side (right edge for the left eye) must not drag the centroid
    img = synthetic_eye(cx=0.45, cy=0.50)
    img[:, int(0.75 * img.shape[1]):] = 5   # rim band, darker than the pupil
    cx, cy, conf = estimate_dark_centroid(img, is_left=True)
    assert conf == 1.0
    assert abs(cx - 0.45) < 0.03 and abs(cy - 0.50) < 0.03


def test_centroid_degenerate_without_dark_pixels():
    flat = np.full((120, 120), 150, dtype=np.uint8)
    _, _, conf = estimate_dark_centroid(flat, is_left=True)
    # flat image: the percentile mask is the whole region until the plateau guard trims it
    # to nothing -> degenerate
    assert conf == 0.0


def test_recenter_moves_pupil_to_canonical():
    # start within the +-MAX_SHIFT_FRAC clamp of the canonical target
    img = synthetic_eye(cx=0.50, cy=0.55)
    out = recenter_gray(img, (0.50, 0.55), CANONICAL_LEFT)
    cx, cy, conf = estimate_dark_centroid(out, is_left=True)
    assert conf == 1.0
    assert abs(cx - CANONICAL_LEFT[0]) < 0.02 and abs(cy - CANONICAL_LEFT[1]) < 0.02


def test_recenter_shift_is_clamped():
    img = synthetic_eye(cx=0.30, cy=0.10)
    out = recenter_gray(img, (0.30, 0.10), CANONICAL_LEFT)  # dy=0.65 -> clamped to 0.25
    cx, cy, _ = estimate_dark_centroid(out, is_left=True)
    assert abs(cy - (0.10 + MAX_SHIFT_FRAC)) < 0.02


def test_state_freezes_on_degenerate():
    st = PupilRecenterState()
    assert st.update(0.6, 0.4, 1.0) is not None
    held = st.update(0.1, 0.9, 0.0)  # degenerate estimate -> ignored
    assert held is not None
    assert abs(held[0] - 0.6) < 1e-9 and abs(held[1] - 0.4) < 1e-9


def test_translation_invariance_of_recentered_input():
    """The P0-2 acceptance core: shifting the source frame must not change the
    recentered output (beyond the fill border)."""
    base = synthetic_eye(cx=0.50, cy=0.55)
    shifted = recenter_gray(base, (0.5, 0.5), (0.62, 0.62))  # synthetic wear shift +12%

    out_a = recenter_frame(base, PupilRecenterState(), CANONICAL_LEFT, is_left=True)
    out_b = recenter_frame(shifted, PupilRecenterState(), CANONICAL_LEFT, is_left=True)

    h, w = out_a.shape
    ca = out_a[h // 4: 3 * h // 4, w // 4: 3 * w // 4].astype(np.float32)
    cb = out_b[h // 4: 3 * h // 4, w // 4: 3 * w // 4].astype(np.float32)
    mae = float(np.abs(ca - cb).mean()) / 255.0
    assert mae < 0.02, mae
