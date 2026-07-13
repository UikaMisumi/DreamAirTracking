namespace DreamAirTracking.Core.Runtime.PostProcess;

/// <summary>Tracking states reported per frame (mirrors Python TRACKING_STATES).</summary>
public static class TrackingStates
{
    public const string TrackingOk = "TRACKING_OK";
    public const string Fixating = "FIXATING";
    public const string Saccade = "SACCADE";
    public const string Blinking = "BLINKING";
    public const string Closed = "CLOSED";
    public const string Reopening = "REOPENING";
    public const string Reacquire = "REACQUIRE";
    public const string OneEyeLost = "ONE_EYE_LOST";
}

/// <summary>
/// Parameters for <see cref="TrackingStateMachine"/>. Numeric defaults are the production
/// constants ported from the legacy C# BridgeOpennessFilterOptions /
/// BridgeClosureGazeStabilizerOptions. The two master switches default OFF so the machine
/// is a pure passthrough (compat == today's Python behavior) unless enabled.
/// Mirrors Python TrackingStateParams (predict_live_multitask.py:372-410).
/// </summary>
public sealed class TrackingStateParams
{
    // master switches (compat defaults => identity passthrough)
    public bool OpennessMachine { get; init; }
    public bool GazeHold { get; init; }
    // per-eye openness (BridgeOpennessFilterOptions)
    public double ClosedThreshold { get; init; } = 0.20;
    public double OtherEyeOpenThreshold { get; init; } = 0.60;
    public double OpenEvidenceThreshold { get; init; } = 0.90;
    public double OpenEvidenceFloor { get; init; } = 0.75;
    public double OpenHoldThreshold { get; init; } = 0.80;
    public double SingleEyeDropThreshold { get; init; } = 0.70;
    public double OtherEyeFullyOpenHoldThreshold { get; init; } = 0.78;
    public int AsymmetricDropConfirmFrames { get; init; } = 2;
    public double RiseAlpha { get; init; } = 0.75;
    public double FallAlpha { get; init; } = 0.55;
    public double BothClosedFallAlpha { get; init; } = 0.90;
    // P0-3 closure confirmation + half-open stability (defaults OFF = legacy byte-identical).
    // A closure only "confirms" after this many consecutive frames at/below ClosedThreshold;
    // until confirmed the output is floored at CloseConfirmFloor and the fast both-closed fall
    // alpha is not applied. Kills transient dips from headset shake (1-3 frame dips never
    // confirm) while a real blink stays below threshold and confirms ~25ms later at 120fps.
    public int CloseConfirmFrames { get; init; }
    public double CloseConfirmFloor { get; init; } = 0.22;
    // Within the half-open band (ClosedThreshold..OpenHoldThreshold) ignore target changes
    // smaller than this so a half-closed lid reads steady instead of flickering.
    public double PartialDeadband { get; init; }
    // binocular closure gaze hold (BridgeClosureGazeStabilizerOptions)
    public double ClosingOpennessThreshold { get; init; } = 0.50;
    public double ClosedOpennessThreshold { get; init; } = 0.20;
    public double ReopenOpennessThreshold { get; init; } = 0.70;
    public double ClosingDropThreshold { get; init; } = 0.28;
    public double ClosingDropMaxOpenness { get; init; } = 0.65;
    public int HoldFramesAfterClosing { get; init; } = 4;
    public double ReopenReleaseAlpha { get; init; } = 0.35;
    // gaze saccade/fixate (reported only; no re-smooth in this version)
    public double SaccadeDelta { get; init; } = 0.06;
    public double FixateDelta { get; init; } = 0.02;
    public int FixateFrames { get; init; } = 8;
}

/// <summary>Result of one <see cref="TrackingStateMachine"/> update.</summary>
public readonly record struct TrackingResult(EyePair Openness, Vec2 Gaze, string State, string LeftSubstate, string RightSubstate);

/// <summary>
/// Explicit per-eye openness + binocular gaze tracking state machine (E1).
/// Port of Python <c>TrackingStateMachine</c> (predict_live_multitask.py:413-590).
/// Compat: with both master switches off it returns the input openness/gaze unchanged
/// (byte-identical), only computing the reported state. Openness is already
/// normalized+curved upstream.
/// </summary>
public sealed class TrackingStateMachine
{
    private readonly TrackingStateParams _p;
    private readonly double _minConfidence;

    private double _openL = 1.0, _openR = 1.0;
    private bool _openInitialized;
    private int _leftHold, _rightHold;
    private int _closeCountL, _closeCountR; // P0-3 consecutive frames at/below ClosedThreshold per eye
    private Vec2? _prevGaze;
    private double _prevAvgOpen = 1.0;
    private int _gazeHoldFrames;
    private int _fixateCount;

    public string State { get; private set; } = TrackingStates.TrackingOk;
    public string LeftSubstate { get; private set; } = "OPEN";
    public string RightSubstate { get; private set; } = "OPEN";

    public TrackingStateMachine(TrackingStateParams parameters, double minConfidence = 0.05)
    {
        _p = parameters;
        _minConfidence = minConfidence;
    }

    // (target, hold) — suppress a single-eye false close.
    private (double target, int hold) HoldAsymmetricDrop(double target, double otherRaw, double previous, int holdFrames)
    {
        if (previous < _p.OpenHoldThreshold || target >= _p.SingleEyeDropThreshold || otherRaw < _p.OtherEyeOpenThreshold)
            return (target, 0);
        holdFrames += 1;
        if (otherRaw >= _p.OtherEyeFullyOpenHoldThreshold || holdFrames <= _p.AsymmetricDropConfirmFrames)
            return (previous, holdFrames);
        return (target, holdFrames);
    }

    private bool InPartialBand(double value)
        => value > _p.ClosedThreshold && value < _p.OpenHoldThreshold;

    private double Smooth(double previous, double target, bool bothClosed)
    {
        double alpha = target >= previous ? _p.RiseAlpha : (bothClosed ? _p.BothClosedFallAlpha : _p.FallAlpha);
        alpha = Math.Min(1.0, Math.Max(0.0, alpha));
        return Math.Min(1.0, Math.Max(0.0, previous + (target - previous) * alpha));
    }

    private EyePair Openness(EyePair curved)
    {
        if (!_p.OpennessMachine)
        {
            _openL = curved.Left;
            _openR = curved.Right;
            return curved;
        }
        double rawL = EyeMath.Clamp(curved.Left, 0.0, 1.0);
        double rawR = EyeMath.Clamp(curved.Right, 0.0, 1.0);
        if (curved.Left >= _p.OpenEvidenceThreshold) rawL = Math.Max(rawL, _p.OpenEvidenceFloor);
        if (curved.Right >= _p.OpenEvidenceThreshold) rawR = Math.Max(rawR, _p.OpenEvidenceFloor);
        if (!_openInitialized)
        {
            _openInitialized = true;
            _openL = rawL;
            _openR = rawR;
            return new EyePair(_openL, _openR);
        }
        bool belowL = rawL <= _p.ClosedThreshold;
        bool belowR = rawR <= _p.ClosedThreshold;
        // P0-3: a closure only counts once it persists CloseConfirmFrames consecutive frames.
        bool confirmedL, confirmedR;
        if (_p.CloseConfirmFrames > 0)
        {
            _closeCountL = belowL ? _closeCountL + 1 : 0;
            _closeCountR = belowR ? _closeCountR + 1 : 0;
            confirmedL = _closeCountL >= _p.CloseConfirmFrames;
            confirmedR = _closeCountR >= _p.CloseConfirmFrames;
        }
        else
        {
            confirmedL = belowL; // legacy: below == instantly confirmed
            confirmedR = belowR;
        }
        bool bothClosedRaw = belowL && belowR;
        bool bothClosed = confirmedL && confirmedR; // gates the fast both-closed fall alpha
        double targetL = rawL, targetR = rawR;
        if (!bothClosedRaw)
        {
            (targetL, _leftHold) = HoldAsymmetricDrop(rawL, rawR, _openL, _leftHold);
            (targetR, _rightHold) = HoldAsymmetricDrop(rawR, rawL, _openR, _rightHold);
        }
        else
        {
            _leftHold = 0;
            _rightHold = 0;
        }
        if (_p.CloseConfirmFrames > 0)
        {
            // unconfirmed closures cannot read fully closed (transient dip suppression)
            if (belowL && !confirmedL) targetL = Math.Max(targetL, _p.CloseConfirmFloor);
            if (belowR && !confirmedR) targetR = Math.Max(targetR, _p.CloseConfirmFloor);
        }
        if (_p.PartialDeadband > 0.0)
        {
            // half-open band deadband: hold steady against sub-threshold flicker
            if (InPartialBand(_openL) && InPartialBand(targetL) && Math.Abs(targetL - _openL) <= _p.PartialDeadband)
                targetL = _openL;
            if (InPartialBand(_openR) && InPartialBand(targetR) && Math.Abs(targetR - _openR) <= _p.PartialDeadband)
                targetR = _openR;
        }
        _openL = Smooth(_openL, targetL, bothClosed);
        _openR = Smooth(_openR, targetR, bothClosed);
        return new EyePair(_openL, _openR);
    }

    private Vec2 Gaze(Vec2 gaze, double avgOpen)
    {
        if (!_p.GazeHold)
        {
            _prevGaze = gaze;
            _prevAvgOpen = avgOpen;
            return gaze;
        }
        var g = new Vec2(EyeMath.Clamp(gaze.X, -1.0, 1.0), EyeMath.Clamp(gaze.Y, -1.0, 1.0));
        if (_prevGaze is not Vec2 prev)
        {
            _prevGaze = g;
            _prevAvgOpen = avgOpen;
            return g;
        }
        bool closingDrop = (_prevAvgOpen - avgOpen >= _p.ClosingDropThreshold) && (avgOpen <= _p.ClosingDropMaxOpenness);
        bool closing = closingDrop || avgOpen <= _p.ClosingOpennessThreshold;
        bool closed = avgOpen <= _p.ClosedOpennessThreshold;
        if (closing)
            _gazeHoldFrames = Math.Max(_gazeHoldFrames, Math.Max(0, _p.HoldFramesAfterClosing));
        Vec2 outv;
        if (closed || (_gazeHoldFrames > 0 && avgOpen < _p.ReopenOpennessThreshold))
        {
            outv = prev;
            if (_gazeHoldFrames > 0) _gazeHoldFrames -= 1;
        }
        else if (_gazeHoldFrames > 0)
        {
            double alpha = Math.Min(1.0, Math.Max(0.0, _p.ReopenReleaseAlpha));
            outv = new Vec2(prev.X + (g.X - prev.X) * alpha, prev.Y + (g.Y - prev.Y) * alpha);
            _prevGaze = outv;
            _gazeHoldFrames -= 1;
        }
        else
        {
            outv = g;
            _prevGaze = outv;
        }
        _prevAvgOpen = avgOpen;
        return outv;
    }

    private string Substate(double value)
    {
        if (value <= _p.ClosedThreshold) return "CLOSED";
        if (value >= _p.OpenHoldThreshold) return "OPEN";
        return "PARTIAL";
    }

    private string Classify(EyePair openOut, double avgOpen, double prevAvg, int holdBefore, double pairConf, double gazeDelta)
    {
        if (pairConf < _minConfidence) return TrackingStates.Reacquire;
        bool leftClosed = openOut.Left <= _p.ClosedThreshold;
        bool rightClosed = openOut.Right <= _p.ClosedThreshold;
        if (leftClosed != rightClosed) return TrackingStates.OneEyeLost;
        if (avgOpen <= _p.ClosedOpennessThreshold) return TrackingStates.Closed;
        bool closingDrop = (prevAvg - avgOpen >= _p.ClosingDropThreshold) && (avgOpen <= _p.ClosingDropMaxOpenness);
        if (closingDrop) return TrackingStates.Blinking;
        if (holdBefore > 0 && avgOpen < _p.ReopenOpennessThreshold) return TrackingStates.Reopening;
        if (gazeDelta >= _p.SaccadeDelta)
        {
            _fixateCount = 0;
            return TrackingStates.Saccade;
        }
        if (gazeDelta < _p.FixateDelta)
        {
            _fixateCount += 1;
            if (_fixateCount >= _p.FixateFrames) return TrackingStates.Fixating;
        }
        else
        {
            _fixateCount = 0;
        }
        return TrackingStates.TrackingOk;
    }

    /// <summary>
    /// One frame update. <paramref name="curvedOpenness"/> is the post-curve openness,
    /// <paramref name="pairConfidence"/> is confidence[2], <paramref name="gaze"/> the smoothed gaze.
    /// </summary>
    public TrackingResult Update(EyePair curvedOpenness, double pairConfidence, Vec2 gaze)
    {
        EyePair openOut = Openness(curvedOpenness);
        double avgOpen = EyeMath.Clamp((openOut.Left + openOut.Right) * 0.5, 0.0, 1.0);
        Vec2? prevGazeSnapshot = _prevGaze;
        double prevAvgSnapshot = _prevAvgOpen;
        int holdBefore = _gazeHoldFrames;
        Vec2 gazeOut = Gaze(gaze, avgOpen);
        double gazeDelta = prevGazeSnapshot is Vec2 pg
            ? Math.Sqrt((gaze.X - pg.X) * (gaze.X - pg.X) + (gaze.Y - pg.Y) * (gaze.Y - pg.Y))
            : 0.0;
        State = Classify(openOut, avgOpen, prevAvgSnapshot, holdBefore, pairConfidence, gazeDelta);
        LeftSubstate = Substate(openOut.Left);
        RightSubstate = Substate(openOut.Right);
        return new TrackingResult(openOut, gazeOut, State, LeftSubstate, RightSubstate);
    }
}
