namespace DreamAirTracking.Core.Runtime.PostProcess;

/// <summary>
/// Per-eye openness normalization using calibrated open/closed references.
/// Port of Python <c>normalize_per_eye</c> (predict_live_multitask.py:207-226).
/// </summary>
public static class PerEyeOpennessNormalizer
{
    public static EyePair Normalize(EyePair modelOpenness, EyePair openP95, EyePair closedP05, double minRange = 0.05)
        => Normalize(modelOpenness, openP95, closedP05, minRange, null);

    /// <summary>
    /// With a valid half anchor (S2 / P0-4) the map is piecewise-linear, pinning the user's
    /// calibrated half-open to exactly 0.5 — [closed..half] -> [0..0.5], [half..open] -> [0.5..1].
    /// A degenerate segment falls back to the linear map for that eye. Mirrors Python
    /// <c>normalize_per_eye(..., half_p50=...)</c>.
    /// </summary>
    public static EyePair Normalize(EyePair modelOpenness, EyePair openP95, EyePair closedP05, double minRange, EyePair? halfP50)
        => new(
            NormOne(modelOpenness.Left, openP95.Left, closedP05.Left, minRange, halfP50?.Left),
            NormOne(modelOpenness.Right, openP95.Right, closedP05.Right, minRange, halfP50?.Right));

    private static double NormOne(double v, double hi, double lo, double minRange, double? half = null)
    {
        double x = EyeMath.Clamp(EyeMath.NanToNum(v, 0.0), 0.0, 1.0);
        hi = EyeMath.NanToNum(hi, 1.0);
        lo = EyeMath.NanToNum(lo, 0.0);
        double rng = hi - lo;
        double scaled = (x - lo) / Math.Max(rng, 1e-6);
        if (half is double mid0)
        {
            double mid = EyeMath.NanToNum(mid0, -1.0);
            double segMin = minRange * 0.5;
            if (mid - lo >= segMin && hi - mid >= segMin)
            {
                scaled = x <= mid
                    ? 0.5 * (x - lo) / Math.Max(mid - lo, 1e-6)
                    : 0.5 + 0.5 * (x - mid) / Math.Max(hi - mid, 1e-6);
            }
        }
        double outv = rng >= minRange ? scaled : x;
        return EyeMath.Clamp(outv, 0.0, 1.0);
    }
}

/// <summary>
/// Velocity-gated dual-path openness curve (S1). Port of Python <c>OpennessDualPath</c>.
/// Fast lid motion (a blink) takes the snappy blink_s_curve; slow motion takes a near-linear
/// hover path (v / full_open_threshold, mid slope ~1.1) so half-open states read accurately
/// and hold steady. A short per-eye crossfade avoids pops on mode switches. Stateful.
/// </summary>
public sealed class OpennessDualPath
{
    private const double BlendStep = 0.34; // ~3 frames to fully switch paths

    private EyePair? _prev;
    private double _blendL, _blendR;       // 0 = hover path, 1 = blink path
    private int _fastHoldL, _fastHoldR;

    public EyePair Apply(EyePair values, double fullOpenThreshold, double boostKnee, double boostGamma,
        double enterVelocity = 0.06, double exitVelocity = 0.02, int holdFrames = 6)
    {
        var x = new EyePair(EyeMath.Clamp(values.Left, 0.0, 1.0), EyeMath.Clamp(values.Right, 0.0, 1.0));
        EyePair prev = _prev ?? x;
        double deltaL = Math.Abs(x.Left - prev.Left);
        double deltaR = Math.Abs(x.Right - prev.Right);
        _prev = x;

        EyePair fastOut = OpennessCurve.Apply(x, "blink_s_curve", fullOpenThreshold, boostKnee, boostGamma);
        double threshold = EyeMath.Clamp(fullOpenThreshold, 0.05, 1.0);
        double slowL = EyeMath.Clamp(x.Left / Math.Max(0.001, threshold), 0.0, 1.0);
        double slowR = EyeMath.Clamp(x.Right / Math.Max(0.001, threshold), 0.0, 1.0);

        double outL = PathOne(deltaL, fastOut.Left, slowL, enterVelocity, exitVelocity, holdFrames, ref _fastHoldL, ref _blendL);
        double outR = PathOne(deltaR, fastOut.Right, slowR, enterVelocity, exitVelocity, holdFrames, ref _fastHoldR, ref _blendR);
        return new EyePair(EyeMath.Clamp(outL, 0.0, 1.0), EyeMath.Clamp(outR, 0.0, 1.0));
    }

    private static double PathOne(double delta, double fastOut, double slowOut,
        double enterVelocity, double exitVelocity, int holdFrames, ref int fastHold, ref double blend)
    {
        if (delta >= enterVelocity)
            fastHold = Math.Max(1, holdFrames);
        else if (fastHold > 0 && delta <= exitVelocity)
            fastHold -= 1;
        double target = fastHold > 0 ? 1.0 : 0.0;
        blend = target > blend ? Math.Min(1.0, blend + BlendStep) : Math.Max(0.0, blend - BlendStep);
        return blend * fastOut + (1.0 - blend) * slowOut;
    }
}

/// <summary>
/// Openness response curve. Port of Python <c>apply_openness_curve</c>
/// (predict_live_multitask.py:171-204). Modes: off | soft_open_plateau |
/// blink_s_curve (app default) | full_open_plateau (else). Applied per eye.
/// </summary>
public static class OpennessCurve
{
    public static EyePair Apply(EyePair openness, string mode, double fullOpenThreshold, double boostKnee, double boostGamma)
    {
        if (mode == "off")
            return new EyePair(EyeMath.Clamp(openness.Left, 0.0, 1.0), EyeMath.Clamp(openness.Right, 0.0, 1.0));

        double threshold = EyeMath.Clamp(fullOpenThreshold, 0.05, 1.0);
        double knee = EyeMath.Clamp(boostKnee, 0.0, threshold - 0.001);
        double gamma = Math.Max(0.05, boostGamma);
        return new EyePair(
            CurveOne(openness.Left, mode, threshold, knee, gamma),
            CurveOne(openness.Right, mode, threshold, knee, gamma));
    }

    private static double CurveOne(double v, string mode, double threshold, double knee, double gamma)
    {
        double values = EyeMath.Clamp(v, 0.0, 1.0);
        if (mode == "soft_open_plateau")
        {
            double curved = Math.Pow(values, gamma);
            if (values >= threshold) curved = 1.0;
            return EyeMath.Clamp(curved, 0.0, 1.0);
        }
        if (mode == "blink_s_curve")
        {
            double t = EyeMath.Clamp((values - knee) / Math.Max(0.001, threshold - knee), 0.0, 1.0);
            double s = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            double curved = Math.Pow(s, gamma);
            if (values >= threshold) curved = 1.0;
            return EyeMath.Clamp(curved, 0.0, 1.0);
        }
        // full_open_plateau (Python's else branch)
        double c = values;
        if (values >= threshold)
        {
            c = 1.0;
        }
        else if (values > knee)
        {
            double t = (values - knee) / Math.Max(0.001, threshold - knee);
            c = knee + (1.0 - knee) * Math.Pow(t, gamma);
        }
        return EyeMath.Clamp(c, 0.0, 1.0);
    }
}
