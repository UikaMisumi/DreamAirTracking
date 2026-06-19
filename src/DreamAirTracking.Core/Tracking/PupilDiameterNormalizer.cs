namespace DreamAirTracking.Core.Tracking;

public sealed class PupilDiameterNormalizer
{
    private readonly PupilDiameterCalibrationProfile _profile;

    public PupilDiameterNormalizer(PupilDiameterCalibrationProfile profile)
    {
        _profile = profile;
    }

    public PupilDiameterTrackingState Normalize(
        EyeSide side,
        PupilDiameterEstimate estimate,
        double openness)
    {
        var calibration = _profile.GetEye(side);
        if (!calibration.Valid)
        {
            return PupilDiameterTrackingState.NotFound("pupil diameter calibration is not valid");
        }

        if (!estimate.Found)
        {
            return PupilDiameterTrackingState.NotFound(estimate.Reason ?? "pupil diameter not found");
        }

        if (!double.IsFinite(estimate.EquivalentDiameterPx) ||
            !double.IsFinite(estimate.AxisRatio) ||
            !double.IsFinite(openness))
        {
            return PupilDiameterTrackingState.NotFound("pupil diameter has non-finite input");
        }

        var range = calibration.RangePx;
        if (!double.IsFinite(range) || range <= 0)
        {
            range = calibration.DarkPx - calibration.BrightPx;
        }

        if (!double.IsFinite(range) || range < 3)
        {
            return PupilDiameterTrackingState.NotFound("pupil diameter calibration range is too small");
        }

        var quality = estimate.Confidence >= _profile.MinConfidence &&
            estimate.AxisRatio <= _profile.MaxAxisRatio &&
            openness >= _profile.MinOpenness;
        var normalized = Math.Clamp((estimate.EquivalentDiameterPx - calibration.BrightPx) / range, 0.0, 1.0);

        return new PupilDiameterTrackingState(
            true,
            quality,
            estimate.Confidence,
            estimate.EquivalentDiameterPx,
            normalized,
            estimate.AxisRatio,
            quality ? string.Empty : "pupil diameter failed quality gate");
    }
}
