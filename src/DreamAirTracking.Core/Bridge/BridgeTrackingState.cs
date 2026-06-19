namespace DreamAirTracking.Core.Bridge;

public sealed record BridgeTrackingState(
    long Sequence,
    DateTimeOffset Timestamp,
    double DeltaMs,
    BridgeEyeState Left,
    BridgeEyeState Right)
{
    public bool EyeTrackingEnabled { get; init; } = true;
    public bool EyeExpressionEnabled { get; init; }
    public string EyeExpressionMode { get; init; } = "off";
    public bool PupilDiameterEnabled { get; init; }
    public string PupilDiameterMode { get; init; } = "off";
    public string NormalizationMode { get; init; } = "off";
}
