using DreamAirTracking.Core;
using DreamAirTracking.Core.Geometry;
using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Tests;

public sealed class PupilDiameterNormalizerTests
{
    [Fact]
    public void MapsDiameterIntoNormalizedCalibrationRange()
    {
        var normalizer = new PupilDiameterNormalizer(MakeProfile());
        var estimate = new PupilDiameterEstimate(
            true,
            0.5,
            35,
            12,
            10,
            500,
            80,
            60,
            new PixelRect(1, 1, 10, 10),
            1.2,
            70,
            0);

        var state = normalizer.Normalize(EyeSide.Left, estimate, openness: 1.0);

        Assert.True(state.Found);
        Assert.True(state.Quality);
        Assert.Equal(0.5, state.Normalized, precision: 3);
    }

    [Fact]
    public void KeepsFoundButMarksLowConfidenceAsLowQuality()
    {
        var normalizer = new PupilDiameterNormalizer(MakeProfile());
        var estimate = new PupilDiameterEstimate(
            true,
            0.1,
            35,
            12,
            10,
            500,
            80,
            60,
            new PixelRect(1, 1, 10, 10),
            1.2,
            70,
            0);

        var state = normalizer.Normalize(EyeSide.Left, estimate, openness: 1.0);

        Assert.True(state.Found);
        Assert.False(state.Quality);
        Assert.Equal(0.5, state.Normalized, precision: 3);
    }

    private static PupilDiameterCalibrationProfile MakeProfile() => new()
    {
        MinConfidence = 0.28,
        MaxAxisRatio = 1.8,
        MinOpenness = 0.75,
        Left = new PupilDiameterEyeCalibrationProfile
        {
            Valid = true,
            BrightPx = 30,
            DarkPx = 40,
            RangePx = 10
        },
        Right = new PupilDiameterEyeCalibrationProfile
        {
            Valid = true,
            BrightPx = 32,
            DarkPx = 42,
            RangePx = 10
        }
    };
}
