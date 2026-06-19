namespace DreamAirTracking.Core.Tracking;

public sealed class PupilDiameterEstimatorOptions
{
    public double MinPupilDetectionConfidence { get; init; } = 0.05;
    public int ThresholdPadding { get; init; } = 6;
    public int MinArea { get; init; } = 40;
    public double MaxAxisRatio { get; init; } = 3.0;
}
