using DreamAirTracking.Core;
using DreamAirTracking.Core.Geometry;
using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Tests;

public sealed class EyeOpennessDetectorTests
{
    [Fact]
    public void ReportsHighOpennessForWideDarkBand()
    {
        var frame = MakeFrameWithDarkBand(width: 80, height: 60, bandTop: 22, bandBottom: 38, bandLeft: 18, bandRight: 62);

        var detection = new EyeOpennessDetector().Detect(frame, new EyeOpennessDetectorOptions
        {
            Roi = new PixelRect(10, 10, 60, 40),
            DarkThreshold = 80,
            ClosedPeakFraction = 0.15,
            OpenPeakFraction = 0.35
        });

        Assert.True(detection.Found);
        Assert.True(detection.Openness > 0.9);
        Assert.InRange(detection.ApertureHeight, 15, 19);
        Assert.True(detection.PeakDarkFraction > 0.6);
    }

    [Fact]
    public void ReportsLowOpennessForNarrowDarkLine()
    {
        var frame = MakeFrameWithDarkBand(width: 80, height: 60, bandTop: 30, bandBottom: 31, bandLeft: 35, bandRight: 42);

        var detection = new EyeOpennessDetector().Detect(frame, new EyeOpennessDetectorOptions
        {
            Roi = new PixelRect(10, 10, 60, 40),
            DarkThreshold = 80,
            ClosedPeakFraction = 0.15,
            OpenPeakFraction = 0.35
        });

        Assert.True(detection.Found);
        Assert.True(detection.Openness < 0.1);
        Assert.Equal(0, detection.ApertureHeight);
        Assert.True(detection.PeakDarkFraction < 0.15);
    }

    private static EyeFrame MakeFrameWithDarkBand(int width, int height, int bandTop, int bandBottom, int bandLeft, int bandRight)
    {
        var pixels = Enumerable.Repeat((byte)150, width * height).ToArray();
        for (var y = bandTop; y <= bandBottom; y++)
        {
            for (var x = bandLeft; x <= bandRight; x++)
            {
                pixels[y * width + x] = 45;
            }
        }

        return new EyeFrame(EyeSide.Left, width, height, 8, pixels, DateTimeOffset.UtcNow);
    }
}
