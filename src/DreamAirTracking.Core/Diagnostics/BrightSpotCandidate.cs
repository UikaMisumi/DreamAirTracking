using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Diagnostics;

public sealed record BrightSpotCandidate(
    PixelRect Bounds,
    int Area,
    double CenterX,
    double CenterY,
    double MeanIntensity,
    byte PeakIntensity,
    double Score);
