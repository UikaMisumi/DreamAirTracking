using DreamAirTracking.Core.Calibration;
using DreamAirTracking.Core.Profiles;

namespace DreamAirTracking.Core.Tracking;

public sealed class PupilOffsetMapper
{
    public NormalizedEyePoint Map(EyeSide side, PupilDetection detection, TrackingProfile profile)
    {
        if (!detection.Found)
        {
            return new NormalizedEyePoint(false, 0, 0, 0);
        }

        var eye = profile.GetEye(side);
        var raw = eye.InputTransform.Apply(detection.CenterX, detection.CenterY);
        if (eye.GazeModel is { } gazeModel && IsSupportedGazeModel(gazeModel))
        {
            var mapped = gazeModel.Map(raw.X, raw.Y);
            return new NormalizedEyePoint(true, detection.Confidence, mapped.X, mapped.Y);
        }

        var calibration = eye.Calibration;
        var x = Normalize(raw.X, calibration.CenterX, calibration.HorizontalRange, calibration.HorizontalOutputSign, calibration.HorizontalOutputOffset);
        var y = Normalize(raw.Y, calibration.CenterY, calibration.VerticalRange, calibration.VerticalOutputSign, calibration.VerticalOutputOffset);
        return new NormalizedEyePoint(true, detection.Confidence, x, y);
    }

    private static bool IsSupportedGazeModel(Affine2DGazeModel gazeModel)
    {
        if (gazeModel.OutX.Length < 3 || gazeModel.OutY.Length < 3)
        {
            return false;
        }

        return gazeModel.Type.Equals("affine2d", StringComparison.OrdinalIgnoreCase) ||
            gazeModel.Type.Equals("quadratic2d", StringComparison.OrdinalIgnoreCase) ||
            gazeModel.Type.Equals("polynomial2d", StringComparison.OrdinalIgnoreCase);
    }

    private static double Normalize(double value, double center, double range, double outputSign, double outputOffset)
    {
        if (range <= 0)
        {
            return 0;
        }

        var sign = outputSign < 0 ? -1 : 1;
        return Math.Clamp((((value - center) / range) * sign) + outputOffset, -1, 1);
    }
}
