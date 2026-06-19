using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Diagnostics;

public sealed class BrightSpotDetectorOptions
{
    public PixelRect? Roi { get; set; }
    public byte Threshold { get; set; } = 235;
    public int MinArea { get; set; } = 1;
    public int MaxArea { get; set; } = 120;
    public int MaxWidth { get; set; } = 18;
    public int MaxHeight { get; set; } = 18;
    public int MaxCandidates { get; set; } = 8;
}
