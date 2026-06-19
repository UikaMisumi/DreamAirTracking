namespace DreamAirTracking.Core.Tracking;

public sealed record PupilDiameterTrackingState(
    bool Found,
    bool Quality,
    double Confidence,
    double DiameterPx,
    double Normalized,
    double AxisRatio,
    string Reason = "")
{
    public static PupilDiameterTrackingState NotFound(string reason) =>
        new(false, false, 0, 0, 0, 0, reason);
}
