using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Tracking;

public sealed class EyeOpennessDetectorOptions
{
    public PixelRect? Roi { get; set; }
    public int? DarkThreshold { get; set; }
    public double AdaptivePercentile { get; set; } = 0.18;
    public int AdaptiveMargin { get; set; } = 8;
    public int MaxAdaptiveThreshold { get; set; } = 100;
    public int RowSmoothingRadius { get; set; } = 2;
    public double ApertureGateFraction { get; set; } = 0.25;
    public double ClosedPeakFraction { get; set; } = 0.25;
    public double OpenPeakFraction { get; set; } = 0.35;
    public double ClosedPeakWhenNoAperture { get; set; } = 0.30;
    public int MinApertureHeightForPartialOpen { get; set; } = 10;
}
