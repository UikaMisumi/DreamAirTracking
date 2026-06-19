namespace DreamAirTracking.Core.Tracking;

public static class PupilAssistEstimator
{
    public const double NeutralNormalized = 0.50;
    public const double ConstrictedNormalized = 0.20;
    public const double WideStartOpenness = 0.93;
    public const double FullWideOpenness = 1.00;
    public const int WideStartApertureHeight = 75;
    public const int FullWideApertureHeight = 86;
    public const double MinTrackedOpenness = 0.45;
    public const double MinConfidence = 0.05;

    public static PupilDiameterTrackingState Estimate(double openness, double confidence)
    {
        if (!double.IsFinite(openness) || !double.IsFinite(confidence))
        {
            return PupilDiameterTrackingState.NotFound("pupil assist has non-finite input");
        }

        if (openness < MinTrackedOpenness || confidence < MinConfidence)
        {
            return PupilDiameterTrackingState.NotFound("pupil assist gated by openness or confidence");
        }

        var wideAmount = openness >= WideStartOpenness
            ? SmoothStep(Math.Clamp(
                (openness - WideStartOpenness) / (FullWideOpenness - WideStartOpenness),
                0.0,
                1.0))
            : 0.0;
        var normalized = NeutralNormalized + ((ConstrictedNormalized - NeutralNormalized) * wideAmount);

        return new PupilDiameterTrackingState(
            true,
            true,
            Math.Clamp(confidence, 0.0, 1.0),
            0,
            Math.Clamp(normalized, 0.0, 1.0),
            1.0,
            "eye-wide pupil assist");
    }

    public static (PupilDiameterTrackingState Left, PupilDiameterTrackingState Right) EstimateLinked(
        double leftOpenness,
        double leftConfidence,
        double rightOpenness,
        double rightConfidence)
    {
        var leftUsable = IsUsable(leftOpenness, leftConfidence);
        var rightUsable = IsUsable(rightOpenness, rightConfidence);
        if (!leftUsable && !rightUsable)
        {
            var missing = PupilDiameterTrackingState.NotFound("pupil assist gated by both eyes");
            return (missing, missing);
        }

        var openness = Math.Max(leftUsable ? leftOpenness : 0.0, rightUsable ? rightOpenness : 0.0);
        var confidence = Math.Max(leftUsable ? leftConfidence : 0.0, rightUsable ? rightConfidence : 0.0);
        var linked = Estimate(openness, confidence);
        return (linked, linked);
    }

    public static (PupilDiameterTrackingState Left, PupilDiameterTrackingState Right) EstimateLinked(
        double leftOpenness,
        double leftConfidence,
        double rightOpenness,
        double rightConfidence,
        int leftApertureHeight,
        int rightApertureHeight)
    {
        var leftUsable = IsUsable(leftOpenness, leftConfidence);
        var rightUsable = IsUsable(rightOpenness, rightConfidence);
        if (!leftUsable && !rightUsable)
        {
            var missing = PupilDiameterTrackingState.NotFound("pupil assist gated by both eyes");
            return (missing, missing);
        }

        var openness = Math.Max(leftUsable ? leftOpenness : 0.0, rightUsable ? rightOpenness : 0.0);
        var confidence = Math.Max(leftUsable ? leftConfidence : 0.0, rightUsable ? rightConfidence : 0.0);
        var apertureHeight = Math.Max(leftUsable ? leftApertureHeight : 0, rightUsable ? rightApertureHeight : 0);
        var linked = EstimateFromAperture(openness, confidence, apertureHeight);
        return (linked, linked);
    }

    public static PupilDiameterTrackingState EstimateFromAperture(double openness, double confidence, int apertureHeight)
    {
        if (!double.IsFinite(openness) || !double.IsFinite(confidence))
        {
            return PupilDiameterTrackingState.NotFound("pupil assist has non-finite input");
        }

        if (openness < MinTrackedOpenness || confidence < MinConfidence)
        {
            return PupilDiameterTrackingState.NotFound("pupil assist gated by openness or confidence");
        }

        var wideAmount = apertureHeight >= WideStartApertureHeight
            ? SmoothStep(Math.Clamp(
                (apertureHeight - WideStartApertureHeight) / (double)Math.Max(1, FullWideApertureHeight - WideStartApertureHeight),
                0.0,
                1.0))
            : 0.0;
        var normalized = NeutralNormalized + ((ConstrictedNormalized - NeutralNormalized) * wideAmount);

        return new PupilDiameterTrackingState(
            true,
            true,
            Math.Clamp(confidence, 0.0, 1.0),
            0,
            Math.Clamp(normalized, 0.0, 1.0),
            1.0,
            "eye-wide pupil assist");
    }

    private static bool IsUsable(double openness, double confidence) =>
        double.IsFinite(openness) &&
        double.IsFinite(confidence) &&
        openness >= MinTrackedOpenness &&
        confidence >= MinConfidence;

    private static double SmoothStep(double value) => value * value * (3.0 - (2.0 * value));
}
