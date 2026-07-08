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
