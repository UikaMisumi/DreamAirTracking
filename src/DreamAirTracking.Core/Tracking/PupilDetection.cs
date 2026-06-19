using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Tracking;

public sealed record PupilDetection(
    bool Found,
    double Confidence,
    double CenterX,
    double CenterY,
    int Threshold,
    PixelRect Bounds,
    int Area,
    double FillRatio,
    double MomentXX = double.NaN,
    double MomentYY = double.NaN,
    double MomentXY = double.NaN,
    string? Reason = null)
{
    public static PupilDetection NotFound(string reason) =>
        new(false, 0, double.NaN, double.NaN, 0, new PixelRect(0, 0, 0, 0), 0, 0, double.NaN, double.NaN, double.NaN, reason);
}
