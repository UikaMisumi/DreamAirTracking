namespace DreamAirTracking.Core.Runtime;

/// <summary>2D gaze vector (or any XY pair). Mirrors a numpy float array of size 2.</summary>
public readonly record struct Vec2(double X, double Y);

/// <summary>Per-eye scalar pair (left, right). Mirrors a numpy float array of size 2.</summary>
public readonly record struct EyePair(double Left, double Right);

/// <summary>
/// Shared scalar helpers ported from the Python runtime (predict_live_multitask.py).
/// All post-processing math runs in double internally; the Python side uses float32
/// arrays with float64 scalar intermediates, so double here matches within ~1e-7.
/// </summary>
internal static class EyeMath
{
    /// <summary>Python <c>clamp01</c> (PLM:127-130): non-finite -> 0, else clamp to [0,1].</summary>
    public static double Clamp01(double v) => double.IsFinite(v) ? Math.Min(1.0, Math.Max(0.0, v)) : 0.0;

    /// <summary>Mirrors <c>np.clip(v, lo, hi)</c>: NaN propagates (unlike Clamp01).</summary>
    public static double Clamp(double v, double lo, double hi) => Math.Min(hi, Math.Max(lo, v));

    /// <summary>Mirrors <c>np.nan_to_num(v, nan=...)</c> for a scalar (inf coerced to +/-nan replacement is out of range here).</summary>
    public static double NanToNum(double v, double nan) =>
        double.IsNaN(v) ? nan : (double.IsPositiveInfinity(v) ? double.MaxValue : (double.IsNegativeInfinity(v) ? double.MinValue : v));
}
