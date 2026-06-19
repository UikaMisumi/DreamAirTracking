using DreamAirTracking.Core;
using DreamAirTracking.Core.Diagnostics;
using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Tests;

public sealed class BrightSpotDetectorTests
{
    [Fact]
    public void FindsSmallBrightComponentsInsideRoi()
    {
        var pixels = Enumerable.Repeat((byte)40, 40 * 30).ToArray();
        pixels[12 * 40 + 15] = 250;
        pixels[12 * 40 + 16] = 251;
        pixels[13 * 40 + 15] = 252;
        pixels[24 * 40 + 35] = 255;
        var frame = new EyeFrame(EyeSide.Left, 40, 30, 8, pixels, DateTimeOffset.UtcNow);

        var candidates = new BrightSpotDetector().Detect(frame, new BrightSpotDetectorOptions
        {
            Roi = new PixelRect(10, 8, 12, 10),
            Threshold = 240
        });

        var candidate = Assert.Single(candidates);
        Assert.Equal(3, candidate.Area);
        Assert.InRange(candidate.CenterX, 15.0, 15.5);
        Assert.InRange(candidate.CenterY, 12.0, 12.5);
        Assert.Equal(252, candidate.PeakIntensity);
    }

    [Fact]
    public void RejectsOversizedBrightRegions()
    {
        var pixels = Enumerable.Repeat((byte)40, 30 * 30).ToArray();
        for (var y = 5; y < 25; y++)
        {
            for (var x = 5; x < 25; x++)
            {
                pixels[y * 30 + x] = 250;
            }
        }

        var frame = new EyeFrame(EyeSide.Left, 30, 30, 8, pixels, DateTimeOffset.UtcNow);
        var candidates = new BrightSpotDetector().Detect(frame, new BrightSpotDetectorOptions { Threshold = 240 });

        Assert.Empty(candidates);
    }
}
