namespace DreamAirTracking.Core.Runtime.PostProcess;

/// <summary>
/// Per-eye openness normalization using calibrated open/closed references.
/// Port of Python <c>normalize_per_eye</c> (predict_live_multitask.py:207-226).
/// </summary>
public static class PerEyeOpennessNormalizer
{
    public static EyePair Normalize(EyePair modelOpenness, EyePair openP95, EyePair closedP05, double minRange = 0.05)
        => new(
            NormOne(modelOpenness.Left, openP95.Left, closedP05.Left, minRange),
            NormOne(modelOpenness.Right, openP95.Right, closedP05.Right, minRange));

    private static double NormOne(double v, double hi, double lo, double minRange)
    {
        double x = EyeMath.Clamp(EyeMath.NanToNum(v, 0.0), 0.0, 1.0);
        hi = EyeMath.NanToNum(hi, 1.0);
        lo = EyeMath.NanToNum(lo, 0.0);
        double rng = hi - lo;
        double scaled = (x - lo) / Math.Max(rng, 1e-6);
        double outv = rng >= minRange ? scaled : x;
        return EyeMath.Clamp(outv, 0.0, 1.0);
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
