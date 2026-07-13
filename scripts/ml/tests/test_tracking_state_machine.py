"""Unit tests for TrackingStateMachine (E1, Phase B).

Pure numpy logic — no ONNX, no hardware, no sockets.
Run: python -m pytest scripts/ml/tests -q
"""

import numpy as np

from predict_live_multitask import TRACKING_STATES, TrackingStateMachine, TrackingStateParams

CONF = np.array([0.9, 0.9, 0.9], dtype=np.float32)


def _open(x0, x1):
    return np.array([x0, x1], dtype=np.float32)


def test_compat_is_pure_passthrough():
    m = TrackingStateMachine(TrackingStateParams(), ema_alpha=0.35)  # both switches off
    for i in range(20):
        curved = _open(0.1 * (i % 10), 1.0 - 0.05 * (i % 7))
        gaze = np.array([0.01 * i, -0.01 * i], dtype=np.float32)
        r = m.update(curved, curved, CONF, gaze)
        assert r.openness is curved          # identical object -> byte-identical output
        assert r.gaze is gaze                # identical object
        assert r.state in TRACKING_STATES


def test_compat_gaze_passthrough_even_when_closed():
    m = TrackingStateMachine(TrackingStateParams(), 0.35)  # gaze_hold off
    m.update(_open(1.0, 1.0), None, CONF, np.array([0.1, 0.1], dtype=np.float32))
    g = np.array([0.9, 0.9], dtype=np.float32)
    r = m.update(_open(0.0, 0.0), None, CONF, g)
    assert r.gaze is g  # not frozen when gaze_hold is off


def test_openness_machine_slow_rise():
    m = TrackingStateMachine(TrackingStateParams(openness_machine=True), 0.35)  # rise 0.75
    gaze = np.array([0.0, 0.0], dtype=np.float32)
    m.update(_open(0.0, 0.0), None, CONF, gaze)          # init at 0
    r = m.update(_open(1.0, 1.0), None, CONF, gaze)       # step to 1
    # first step rises by rise_alpha*(1-0) = 0.75, not instantly to 1.0
    assert abs(float(r.openness[0]) - 0.75) < 1e-4
    assert float(r.openness[0]) < 1.0


def test_openness_machine_fall_alpha():
    m = TrackingStateMachine(TrackingStateParams(openness_machine=True), 0.35)  # fall 0.55
    gaze = np.array([0.0, 0.0], dtype=np.float32)
    m.update(_open(1.0, 1.0), None, CONF, gaze)           # init at 1
    r = m.update(_open(0.3, 0.3), None, CONF, gaze)        # both drop but not fully closed
    # 1.0 + 0.55*(0.3-1.0) = 0.615
    assert abs(float(r.openness[0]) - 0.615) < 1e-4


def test_closed_freezes_gaze():
    m = TrackingStateMachine(TrackingStateParams(openness_machine=True, gaze_hold=True), 0.35)
    m.update(_open(1.0, 1.0), None, CONF, np.array([0.5, 0.2], dtype=np.float32))  # open, gaze here
    r = m.update(_open(0.0, 0.0), None, CONF, np.array([0.9, 0.9], dtype=np.float32))  # close, gaze jumps
    assert r.state in ("CLOSED", "BLINKING")
    # gaze held near last-open (0.5, 0.2), NOT the new (0.9, 0.9)
    assert abs(float(r.gaze[0]) - 0.5) < 0.2
    assert abs(float(r.gaze[1]) - 0.2) < 0.2


def test_single_eye_false_close_suppressed_while_other_fully_open():
    m = TrackingStateMachine(TrackingStateParams(openness_machine=True), 0.35)
    gaze = np.array([0.0, 0.0], dtype=np.float32)
    m.update(_open(1.0, 1.0), None, CONF, gaze)                 # both open
    r1 = m.update(_open(0.0, 1.0), None, CONF, gaze)            # left dips, right fully open
    r2 = m.update(_open(0.0, 1.0), None, CONF, gaze)
    # left is held open (suppressed as a likely single-eye tracking glitch)
    assert float(r1.openness[0]) > 0.9
    assert float(r2.openness[0]) > 0.9


def test_wink_registers_when_other_eye_not_fully_open():
    m = TrackingStateMachine(TrackingStateParams(openness_machine=True), 0.35)
    gaze = np.array([0.0, 0.0], dtype=np.float32)
    m.update(_open(1.0, 0.65), None, CONF, gaze)               # right only partly open
    last = None
    for _ in range(6):
        last = m.update(_open(0.0, 0.65), None, CONF, gaze)     # sustained left close
    assert float(last.openness[0]) < 0.3      # left eventually drops through
    assert float(last.openness[1]) > 0.4      # right stays open
    assert last.state == "ONE_EYE_LOST"


def test_reacquire_on_low_confidence():
    m = TrackingStateMachine(TrackingStateParams(openness_machine=True, gaze_hold=True), 0.35)
    m.update(_open(1.0, 1.0), None, np.array([0.9, 0.9, 0.9], dtype=np.float32),
             np.array([0.3, 0.3], dtype=np.float32))
    r = m.update(_open(1.0, 1.0), None, np.array([0.0, 0.0, 0.0], dtype=np.float32),
                 np.array([0.8, 0.8], dtype=np.float32))
    assert r.state == "REACQUIRE"


def test_saccade_and_fixate_reported_without_regressing_gaze():
    # saccade/fixate alphas are None -> reported only, gaze passes through unchanged
    m = TrackingStateMachine(TrackingStateParams(openness_machine=True, gaze_hold=True), 0.35)
    m.update(_open(1.0, 1.0), None, CONF, np.array([0.0, 0.0], dtype=np.float32))
    big = np.array([0.5, 0.5], dtype=np.float32)
    r = m.update(_open(1.0, 1.0), None, CONF, big)   # large gaze delta
    assert r.state == "SACCADE"
    assert np.allclose(r.gaze, big, atol=1e-6)       # gaze not altered by saccade state


# ---- P0-3: closure confirmation + half-open deadband ----

P03 = dict(openness_machine=True, close_confirm_frames=3, close_confirm_floor=0.22, partial_deadband=0.04)


def test_p03_transient_dip_never_reads_closed():
    # headset shake: both eyes dip below closed threshold for 2 frames, then recover.
    m = TrackingStateMachine(TrackingStateParams(**P03), 0.35)
    gaze = np.array([0.0, 0.0], dtype=np.float32)
    m.update(_open(1.0, 1.0), None, CONF, gaze)
    lows = []
    for v in (0.4, 0.1, 0.05, 0.7, 1.0):  # only 2 consecutive frames below 0.20
        r = m.update(_open(v, v), None, CONF, gaze)
        lows.append(float(r.openness[0]))
        assert r.state != "CLOSED"
    # unconfirmed closure is floored: with fall smoothing the output never gets near closed
    assert min(lows) > 0.20 + 1e-6
    assert lows[-1] > 0.8  # fast recovery


def test_p03_real_blink_still_confirms_closed():
    m = TrackingStateMachine(TrackingStateParams(**P03), 0.35)
    gaze = np.array([0.0, 0.0], dtype=np.float32)
    m.update(_open(1.0, 1.0), None, CONF, gaze)
    seq = [0.6, 0.15, 0.05, 0.02, 0.02, 0.02, 0.02]  # sustained closure
    outs = []
    for v in seq:
        r = m.update(_open(v, v), None, CONF, gaze)
        outs.append(float(r.openness[0]))
    # after confirmation (3rd below-threshold frame) the fast both-closed alpha kicks in
    assert outs[-1] < 0.05
    assert m.state == "CLOSED"


def test_p03_confirm_zero_is_legacy_identical():
    # close_confirm_frames=0 + deadband=0 must be byte-identical to the pre-P0-3 machine
    legacy = TrackingStateMachine(TrackingStateParams(openness_machine=True, gaze_hold=True), 0.35)
    p03off = TrackingStateMachine(
        TrackingStateParams(openness_machine=True, gaze_hold=True, close_confirm_frames=0, partial_deadband=0.0), 0.35
    )
    rng = np.random.default_rng(7)
    for _ in range(200):
        v = rng.random(2).astype(np.float32)
        g = (rng.random(2).astype(np.float32) - 0.5) * 2.0
        ra = legacy.update(v.copy(), None, CONF, g.copy())
        rb = p03off.update(v.copy(), None, CONF, g.copy())
        assert np.array_equal(np.asarray(ra.openness), np.asarray(rb.openness))
        assert np.array_equal(np.asarray(ra.gaze), np.asarray(rb.gaze))
        assert ra.state == rb.state


def test_p03_half_open_deadband_holds_steady():
    m = TrackingStateMachine(TrackingStateParams(**P03), 0.35)
    gaze = np.array([0.0, 0.0], dtype=np.float32)
    m.update(_open(0.5, 0.5), None, CONF, gaze)  # init mid-band
    outs = []
    for v in (0.52, 0.48, 0.51, 0.49, 0.53):  # jitter within +-0.04 of held value
        r = m.update(_open(v, v), None, CONF, gaze)
        outs.append(float(r.openness[0]))
    assert max(outs) - min(outs) < 1e-6  # rock steady
    # a real move beyond the deadband still tracks
    r = m.update(_open(0.75, 0.75), None, CONF, gaze)
    assert float(r.openness[0]) > 0.6
