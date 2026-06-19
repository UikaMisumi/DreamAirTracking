using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Tracking;

public sealed class PupilDetectorOptions
{
    public int MinThreshold { get; init; } = 45;
    public int MaxThreshold { get; init; } = 100;
    public int ThresholdStep { get; init; } = 5;
    public int MinArea { get; init; } = 80;
    public int MaxArea { get; init; } = 2500;
    public int MaxWidth { get; init; } = 55;
    public int MaxHeight { get; init; } = 70;
    public double MinAspectRatio { get; init; } = 0.25;
    public double MaxAspectRatio { get; init; } = 1.8;
    public double MinFillRatio { get; init; } = 0.25;
    public int BorderMargin { get; init; } = 1;
    public PixelRect? Roi { get; init; }
    public double? ExpectedCenterX { get; init; }
    public double? ExpectedCenterY { get; init; }
    public double PreviousCenterWeight { get; init; } = 1.2;
}
