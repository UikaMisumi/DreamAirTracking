namespace DreamAirTracking.Core.Runtime.PostProcess;

/// <summary>
/// Maps raw model gaze to VRCFT space: (raw - offset) * gain, then optional hard/soft clamp.
/// Port of Python <c>apply_runtime_map</c> + <c>soft_clamp</c> (live_gaze_dry_run.py:110-127).
/// </summary>
public static class RuntimeGazeMap
{
    public static Vec2 Apply(Vec2 value, Vec2 offset, Vec2 gain, double clamp, string clampMode, double softKnee)
    {
        double mx = (value.X - offset.X) * gain.X;
        double my = (value.Y - offset.Y) * gain.Y;
        var mapped = new Vec2(mx, my);
        if (clamp > 0 && clampMode == "hard")
            return new Vec2(EyeMath.Clamp(mx, -clamp, clamp), EyeMath.Clamp(my, -clamp, clamp));
        if (clamp > 0 && clampMode == "soft")
            return SoftClamp(mapped, clamp, softKnee);
        return mapped;
    }

    private static Vec2 SoftClamp(Vec2 v, double limit, double knee)
    {
        if (limit <= 0) return v;
        knee = Math.Max(0.0, Math.Min(knee, limit * 0.98));
        return new Vec2(SoftClampComp(v.X, limit, knee), SoftClampComp(v.Y, limit, knee));
    }

    private static double SoftClampComp(double v, double limit, double knee)
    {
        double mag = Math.Abs(v);
        if (mag <= knee) return v; // linear region: value unchanged (np.where linear -> value)
        double sign = v < 0 ? -1.0 : (v > 0 ? 1.0 : 0.0);
        double compressed = knee + (limit - knee) * Math.Tanh((mag - knee) / Math.Max(limit - knee, 1e-6));
        return sign * compressed;
    }
}

/// <summary>
/// Gaze EMA smoothing with optional max-step. Port of Python <c>update_smooth</c>
/// (live_gaze_dry_run.py:208-217). Stateless like the Python function: the caller owns
/// the <c>previous</c> value (which the TrackingStateMachine may overwrite each frame).
/// </summary>
public static class GazeSmoother
{
    public static Vec2 Update(Vec2? previous, Vec2 current, double alpha, double maxStep)
    {
        if (previous is not Vec2 prev) return current;
        double tx = alpha * current.X + (1.0 - alpha) * prev.X;
        double ty = alpha * current.Y + (1.0 - alpha) * prev.Y;
        if (maxStep > 0)
        {
            double dx = tx - prev.X, dy = ty - prev.Y;
            double norm = Math.Sqrt(dx * dx + dy * dy);
            if (norm > maxStep)
            {
                tx = prev.X + dx * (maxStep / norm);
                ty = prev.Y + dy * (maxStep / norm);
            }
        }
        return new Vec2(tx, ty);
    }
}

/// <summary>
/// Post-map gaze mapper for VRCFT output. Port of Python <c>OutputGazeMapper</c>
/// (live_gaze_dry_run.py:130-205). Stateful (holds an EMA center for stable_center mode).
/// The app forces mode="off", but the deadzone/gamma output curve still applies.
/// </summary>
public sealed class OutputGazeMapper
{
    private readonly string _mode;
    private readonly double _deadzone;
    private readonly double _gamma;
    private readonly double _limit;
    private readonly double _centerRadius;
    private readonly double _centerTau;
    private readonly double _centerMaxStep;
    private readonly double _minOpenness;
    private readonly double _minConfidence;
    private Vec2 _center;
    private Vec2? _lastValue;
    private double? _lastTime;

    public OutputGazeMapper(
        string mode = "stable_center",
        double deadzone = 0.015,
        double gamma = 1.0,
        double limit = 1.0,
        double centerRadius = 0.08,
        double centerTau = 6.0,
        double centerMaxStep = 0.025,
        double minOpenness = 0.35,
        double minConfidence = 0.05)
    {
        _mode = mode;
        _limit = Math.Max(limit, 1e-6);
        _deadzone = EyeMath.Clamp(deadzone, 0.0, Math.Max(_limit * 0.8, 0.0));
        _gamma = EyeMath.Clamp(gamma, 0.35, 3.0);
        _centerRadius = Math.Max(centerRadius, 0.0);
        _centerTau = Math.Max(centerTau, 1e-3);
        _centerMaxStep = Math.Max(centerMaxStep, 0.0);
        _minOpenness = EyeMath.Clamp(minOpenness, 0.0, 1.0);
        _minConfidence = EyeMath.Clamp(minConfidence, 0.0, 1.0);
        _center = new Vec2(0.0, 0.0);
    }

    public Vec2 Update(Vec2 value, double timestamp, double confidence = 1.0, double openness = 1.0)
    {
        UpdateCenter(value, timestamp, confidence, openness);
        Vec2 centered = _mode != "off" ? new Vec2(value.X - _center.X, value.Y - _center.Y) : value;
        return ApplyCurve(centered);
    }

    private void UpdateCenter(Vec2 value, double now, double confidence, double openness)
    {
        if (_mode != "stable_center") { _lastValue = value; _lastTime = now; return; }
        if (confidence < _minConfidence || openness < _minOpenness) { _lastValue = value; _lastTime = now; return; }
        if (Norm(value) > _centerRadius) { _lastValue = value; _lastTime = now; return; }
        if (_lastValue is Vec2 lv && _centerMaxStep > 0 &&
            Norm(new Vec2(value.X - lv.X, value.Y - lv.Y)) > _centerMaxStep)
        {
            _lastValue = value; _lastTime = now; return;
        }
        double dt = _lastTime is double lt ? Math.Max(0.0, Math.Min(now - lt, 0.25)) : 1.0 / 60.0;
        double alpha = 1.0 - Math.Exp(-dt / _centerTau);
        _center = new Vec2((1.0 - alpha) * _center.X + alpha * value.X, (1.0 - alpha) * _center.Y + alpha * value.Y);
        _lastValue = value;
        _lastTime = now;
    }

    private Vec2 ApplyCurve(Vec2 value)
    {
        if (_deadzone <= 0.0 && Math.Abs(_gamma - 1.0) < 1e-6)
            return new Vec2(EyeMath.Clamp(value.X, -_limit, _limit), EyeMath.Clamp(value.Y, -_limit, _limit));
        return new Vec2(CurveComp(value.X), CurveComp(value.Y));
    }

    private double CurveComp(double v)
    {
        double sign = v < 0 ? -1.0 : (v > 0 ? 1.0 : 0.0);
        double denom = Math.Max(_limit - _deadzone, 1e-6);
        double normalized = EyeMath.Clamp((Math.Abs(v) - _deadzone) / denom, 0.0, 1.0);
        double curved = Math.Pow(normalized, _gamma);
        return sign * curved * _limit;
    }

    private static double Norm(Vec2 v) => Math.Sqrt(v.X * v.X + v.Y * v.Y);
}
