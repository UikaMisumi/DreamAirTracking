namespace DreamAirTracking.Core.Tracking;

public sealed record EyeOpennessDetection(
    bool Found,
    double Openness,
    int UpperY,
    int LowerY,
    int ApertureHeight,
    double PeakDarkFraction,
    double MeanDarkFraction,
    int DarkThreshold,
    string Reason)
{
    public static EyeOpennessDetection NotFound(string reason) =>
        new(false, 0, 0, 0, 0, 0, 0, 0, reason);
}
