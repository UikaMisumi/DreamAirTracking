namespace DreamAirTracking.Core.Runtime.PostProcess;

/// <summary>
/// Temporal confirmation for the VRCFT EyeWide output (L3). Port of Python <c>WideHysteresis</c>.
/// A genuine widen is a sustained expression; extrapolation noise at off-forward gaze is
/// transient — so gate in the TIME domain: 0 until the shaped wide exceeds enterThreshold for
/// confirmFrames consecutive frames; releases immediately below exitThreshold. Never touches gaze.
/// </summary>
public sealed class WideHysteresis
{
    private int _countL, _countR;
    private bool _activeL, _activeR;

    public EyePair Apply(EyePair shapedWide, int confirmFrames, double enterThreshold, double exitThreshold)
    {
        if (confirmFrames <= 0)
            return shapedWide;
        double l = One(shapedWide.Left, confirmFrames, enterThreshold, exitThreshold, ref _countL, ref _activeL);
        double r = One(shapedWide.Right, confirmFrames, enterThreshold, exitThreshold, ref _countR, ref _activeR);
        return new EyePair(EyeMath.Clamp(l, 0.0, 1.0), EyeMath.Clamp(r, 0.0, 1.0));
    }

    private static double One(double v, int confirmFrames, double enter, double exit, ref int count, ref bool active)
    {
        if (active)
        {
            if (v < exit)
            {
                active = false;
                count = 0;
                return 0.0;
            }
            return v;
        }
        count = v >= enter ? count + 1 : 0;
        if (count >= confirmFrames)
        {
            active = true;
            return v;
        }
        return 0.0;
    }
}

/// <summary>
/// VRCFT wide/squint shaping. Port of Python <c>apply_eye_shape_curve</c>
/// (predict_live_multitask.py:256-264). Applied per eye.
/// </summary>
public static class EyeShapeCurve
{
    public static EyePair Apply(EyePair values, double scale, double gamma, double deadzone)
    {
        double dz = EyeMath.Clamp(deadzone, 0.0, 0.95);
        double g = Math.Max(0.05, gamma);
        double s = EyeMath.Clamp(scale, 0.0, 1.0);
        return new EyePair(ShapeOne(values.Left, s, g, dz), ShapeOne(values.Right, s, g, dz));
    }

    private static double ShapeOne(double v, double scale, double gamma, double deadzone)
    {
        double shaped = EyeMath.Clamp(v, 0.0, 1.0);
        if (deadzone > 0.0)
            shaped = EyeMath.Clamp((shaped - deadzone) / Math.Max(0.001, 1.0 - deadzone), 0.0, 1.0);
        if (Math.Abs(gamma - 1.0) > 1e-6)
            shaped = Math.Pow(shaped, gamma);
        return EyeMath.Clamp(shaped * scale, 0.0, 1.0);
    }
}

/// <summary>
/// Pupil-wide hold state (drives pupil constriction). Port of Python
/// <c>update_pupil_wide_state</c> (predict_live_multitask.py:325-357). Stateful:
/// holds per-eye smoothed state and a hold counter. Uses RAW model wide, not shaped.
/// </summary>
public sealed class PupilWideHold
{
    private double _stateL, _stateR;
    private int _holdL, _holdR;

    public EyePair Update(EyePair rawWide, double enterThreshold, double exitThreshold, int holdFrames, double emaAlpha)
    {
        double enter = EyeMath.Clamp(enterThreshold, 0.0, 1.0);
        double exit = EyeMath.Clamp(exitThreshold, 0.0, enter);
        int frames = Math.Max(0, holdFrames);
        double alpha = EyeMath.Clamp(emaAlpha, 0.01, 1.0);
        _stateL = Step(EyeMath.Clamp(rawWide.Left, 0.0, 1.0), ref _stateL, ref _holdL, enter, exit, frames, alpha);
        _stateR = Step(EyeMath.Clamp(rawWide.Right, 0.0, 1.0), ref _stateR, ref _holdR, enter, exit, frames, alpha);
        return new EyePair(EyeMath.Clamp(_stateL, 0.0, 1.0), EyeMath.Clamp(_stateR, 0.0, 1.0));
    }

    private static double Step(double value, ref double state, ref int hold, double enter, double exit, int frames, double alpha)
    {
        double target;
        if (value >= enter)
        {
            hold = frames;
            target = EyeMath.Clamp01((value - enter) / Math.Max(0.001, 1.0 - enter));
        }
        else if (value <= exit)
        {
            if (hold > 0) { hold -= 1; target = state; }
            else target = 0.0;
        }
        else
        {
            if (hold > 0) { hold -= 1; target = state; }
            else target = 0.0;
        }
        return state + alpha * (target - state);
    }
}

/// <summary>Result of <see cref="PupilOutput"/>, mirrors the Python pupil_output_state dict.</summary>
public readonly record struct PupilResult(bool Found, bool Quality, double Confidence, double Normalized, double ExpressionNormalized, string Reason);

/// <summary>
/// Per-eye pupil output. Port of Python <c>pupil_output_state</c> +
/// <c>normalize_pupil_output_mode</c> (predict_live_multitask.py:133-141, 267-322).
/// Modes: off | expression_constrict_on_wide | model_radius.
/// </summary>
public static class PupilOutput
{
    public const double Neutral = 0.50;            // PUPIL_NEUTRAL
    public const double ConstrictedOnWide = 0.20;  // PUPIL_CONSTRICTED_ON_WIDE

    public static string NormalizeMode(string? mode)
    {
        string m = (mode ?? "off").Trim().ToLowerInvariant();
        if (m is "assist" or "eye_wide_assist" or "wide_assist" or "expression") return "expression_constrict_on_wide";
        if (m is "diameter" or "raw" or "model" or "model_radius") return "model_radius";
        if (m == "expression_constrict_on_wide") return m;
        return "off";
    }

    public static PupilResult Compute(string mode, double openness, double wide, double pupilRadius, double confidence)
    {
        if (mode == "off")
            return new PupilResult(false, false, confidence, Neutral, Neutral, "pupil-output-off");

        if (confidence < 0.05 || openness < 0.45)
            return new PupilResult(false, false, confidence, Neutral, Neutral, "pupil output gated by confidence or closed eyelid");

        if (mode == "expression_constrict_on_wide")
        {
            double expression = EyeMath.Clamp(Neutral + (ConstrictedOnWide - Neutral) * EyeMath.Clamp01(wide), 0.0, 1.0);
            return new PupilResult(true, true, confidence, expression, expression, "expression-constrict-on-wide");
        }

        if (!double.IsFinite(pupilRadius) || pupilRadius <= 0.02 || pupilRadius >= 0.35)
            return new PupilResult(false, false, confidence, Neutral, Neutral, "model pupil radius gated");

        return new PupilResult(true, true, confidence, EyeMath.Clamp01(pupilRadius), Neutral, "model-radius-debug");
    }
}
