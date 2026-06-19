namespace DreamAirTracking.Core.Tracking;

public sealed record NormalizedEyePoint(
    bool Found,
    double Confidence,
    double X,
    double Y);
