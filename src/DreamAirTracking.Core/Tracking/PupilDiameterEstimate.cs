using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Tracking;

public sealed record PupilDiameterEstimate(
    bool Found,
    double Confidence,
    double EquivalentDiameterPx,
    double MajorAxisPx,
    double MinorAxisPx,
    int Area,
    double CenterX,
    double CenterY,
    PixelRect Bounds,
    double AxisRatio,
    int Threshold,
    double OrientationRadians,
    string? Reason = null)
{
    public static PupilDiameterEstimate NotFound(string reason) =>
        new(false, 0, double.NaN, double.NaN, double.NaN, 0, double.NaN, double.NaN, new PixelRect(0, 0, 0, 0), double.NaN, 0, double.NaN, reason);
}
