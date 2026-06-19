using DreamAirTracking.Core;
using DreamAirTracking.Core.Geometry;
using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Tests;

public sealed class PupilDiameterEstimatorTests
{
    [Fact]
    public void EstimatesEquivalentDiameterFromDarkPupilArea()
    {
        var frame = MakeEyeFrame(radiusX: 10, radiusY: 8);
        var detection = new PupilDetector().Detect(frame, DetectorOptions());

        var estimate = new PupilDiameterEstimator().Estimate(frame, detection);

        Assert.True(estimate.Found);
        Assert.InRange(estimate.EquivalentDiameterPx, 17.0, 20.5);
        Assert.InRange(estimate.MajorAxisPx, 18.0, 23.0);
        Assert.InRange(estimate.MinorAxisPx, 14.0, 19.0);
    }

    [Fact]
    public void TracksRelativePupilDiameterChange()
    {
        var smallFrame = MakeEyeFrame(radiusX: 7, radiusY: 6);
        var largeFrame = MakeEyeFrame(radiusX: 13, radiusY: 11);
        var detector = new PupilDetector();
        var estimator = new PupilDiameterEstimator();

        var small = estimator.Estimate(smallFrame, detector.Detect(smallFrame, DetectorOptions()));
        var large = estimator.Estimate(largeFrame, detector.Detect(largeFrame, DetectorOptions()));

        Assert.True(small.Found);
        Assert.True(large.Found);
        Assert.True(large.EquivalentDiameterPx > small.EquivalentDiameterPx * 1.55);
    }

    [Fact]
    public void UsesSelectedComponentMomentsInsteadOfAllDarkPixelsInsideBounds()
    {
        var frame = MakeEyeFrame(radiusX: 7, radiusY: 6);
        var detection = new PupilDetection(
            true,
            0.9,
            50,
            40,
            70,
            new PixelRect(30, 20, 40, 40),
            81,
            0.6,
            6.667,
            6.667,
            0);

        var estimate = new PupilDiameterEstimator().Estimate(frame, detection, new PupilDiameterEstimatorOptions
        {
            MinArea = 20,
            ThresholdPadding = 200
        });

        Assert.True(estimate.Found);
        Assert.InRange(estimate.EquivalentDiameterPx, 9.5, 11.0);
        Assert.InRange(estimate.MajorAxisPx, 9.5, 11.0);
        Assert.InRange(estimate.MinorAxisPx, 9.5, 11.0);
    }

    private static PupilDetectorOptions DetectorOptions() => new()
    {
        Roi = new PixelRect(10, 10, 80, 60),
        MinThreshold = 60,
        MaxThreshold = 90,
        MinArea = 20,
        MaxArea = 1500,
        MaxWidth = 40,
        MaxHeight = 40,
        MinFillRatio = 0.35,
        ExpectedCenterX = 50,
        ExpectedCenterY = 40
    };

    private static EyeFrame MakeEyeFrame(int radiusX, int radiusY)
    {
        const int width = 100;
        const int height = 80;
        const int centerX = 50;
        const int centerY = 40;
        var pixels = Enumerable.Repeat((byte)150, width * height).ToArray();
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var nx = (x - centerX) / (double)radiusX;
                var ny = (y - centerY) / (double)radiusY;
                if ((nx * nx) + (ny * ny) <= 1)
                {
                    pixels[y * width + x] = 35;
                }
            }
        }

        return new EyeFrame(EyeSide.Left, width, height, 8, pixels, DateTimeOffset.UtcNow);
    }
}
