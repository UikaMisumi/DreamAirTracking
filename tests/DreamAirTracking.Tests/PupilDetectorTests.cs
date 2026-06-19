using DreamAirTracking.Core;
using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Tests;

public sealed class PupilDetectorTests
{
    [Fact]
    public void DetectsSyntheticDarkOval()
    {
        var frame = MakeSyntheticFrame(200, 200, 112, 96, 12, 20);
        var detection = new PupilDetector().Detect(frame);

        Assert.True(detection.Found);
        Assert.InRange(detection.CenterX, 109, 115);
        Assert.InRange(detection.CenterY, 93, 99);
        Assert.InRange(detection.Area, 500, 900);
        Assert.True(double.IsFinite(detection.MomentXX));
        Assert.True(double.IsFinite(detection.MomentYY));
        Assert.True(double.IsFinite(detection.MomentXY));
    }

    [Fact]
    public void ReturnsNotFoundWhenNoDarkComponentExists()
    {
        var pixels = Enumerable.Repeat((byte)140, 200 * 200).ToArray();
        var frame = new EyeFrame(EyeSide.Left, 200, 200, 8, pixels, DateTimeOffset.UtcNow);

        var detection = new PupilDetector().Detect(frame);

        Assert.False(detection.Found);
        Assert.Equal(0, detection.Confidence);
    }

    private static EyeFrame MakeSyntheticFrame(int width, int height, int centerX, int centerY, int radiusX, int radiusY)
    {
        var pixels = Enumerable.Repeat((byte)135, width * height).ToArray();

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var dx = (x - centerX) / (double)radiusX;
                var dy = (y - centerY) / (double)radiusY;
                if (dx * dx + dy * dy <= 1)
                {
                    pixels[y * width + x] = 45;
                }
            }
        }

        return new EyeFrame(EyeSide.Left, width, height, 8, pixels, DateTimeOffset.UtcNow);
    }
}
