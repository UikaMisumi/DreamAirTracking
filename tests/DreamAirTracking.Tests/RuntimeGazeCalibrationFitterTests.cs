using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class RuntimeGazeCalibrationFitterTests
{
    [Fact]
    public void UsesCenterStageOffsetAndKeepsConservativeGainByDefault()
    {
        var samples = Samples(
            ("center", 0.10, -0.02),
            ("left", -0.78, -0.01),
            ("right", 1.02, 0.01),
            ("up", 0.12, 0.86),
            ("down", 0.08, -0.94));

        var fit = new RuntimeGazeCalibrationFitter().Fit(samples, new RuntimeGazeCalibrationOptions
        {
            MinSamplesPerStage = 2
        });

        Assert.True(fit.Succeeded);
        Assert.Equal(0.10, fit.CenterOffsetX, precision: 6);
        Assert.Equal(-0.02, fit.CenterOffsetY, precision: 6);
        Assert.Equal(1.0, fit.XGain, precision: 6);
        Assert.Equal(1.0, fit.YGain, precision: 6);
        Assert.Equal(2.0 / 1.8, fit.FittedXGain, precision: 6);
        Assert.Equal(2.0 / 1.8, fit.FittedYGain, precision: 6);
    }

    [Fact]
    public void CanUseAxisMidpointOffsetWhenExplicitlyRequested()
    {
        var samples = Samples(
            ("center", 0.10, -0.02),
            ("left", -0.78, -0.01),
            ("right", 1.02, 0.01),
            ("up", 0.12, 0.86),
            ("down", 0.08, -0.94));

        var fit = new RuntimeGazeCalibrationFitter().Fit(samples, new RuntimeGazeCalibrationOptions
        {
            MinSamplesPerStage = 2,
            OffsetMode = RuntimeGazeOffsetMode.AxisMidpoint
        });

        Assert.True(fit.Succeeded);
        Assert.Equal(0.12, fit.CenterOffsetX, precision: 6);
        Assert.Equal(-0.04, fit.CenterOffsetY, precision: 6);
    }

    [Fact]
    public void CanApplyFittedGainWhenExplicitlyEnabled()
    {
        var samples = Samples(
            ("center", 0.10, -0.02),
            ("left", -0.40, -0.01),
            ("right", 0.60, 0.01),
            ("up", 0.12, 0.52),
            ("down", 0.08, -0.48));

        var fit = new RuntimeGazeCalibrationFitter().Fit(samples, new RuntimeGazeCalibrationOptions
        {
            MinSamplesPerStage = 2,
            ApplyFittedGain = true
        });

        Assert.True(fit.Succeeded);
        Assert.Equal(2.0, fit.XGain, precision: 6);
        Assert.Equal(2.0, fit.YGain, precision: 6);
    }

    [Fact]
    public void FailsWhenRequiredStageIsMissing()
    {
        var samples = Samples(
            ("center", 0.10, -0.02),
            ("left", -0.78, -0.01),
            ("right", 1.02, 0.01),
            ("up", 0.12, 0.86));

        var fit = new RuntimeGazeCalibrationFitter().Fit(samples, new RuntimeGazeCalibrationOptions
        {
            MinSamplesPerStage = 2
        });

        Assert.False(fit.Succeeded);
        Assert.Contains("down", fit.MissingStages);
    }

    [Fact]
    public void FailsWhenVerticalSeparationIsTooSmall()
    {
        var samples = Samples(
            ("center", 0.13, 0.66),
            ("left", -0.68, 0.92),
            ("right", 0.81, 0.97),
            ("up", 0.22, 0.95),
            ("down", -0.89, 0.41));

        var fit = new RuntimeGazeCalibrationFitter().Fit(samples, new RuntimeGazeCalibrationOptions
        {
            MinSamplesPerStage = 2
        });

        Assert.False(fit.Succeeded);
        Assert.Contains("vertical separation", fit.Reason);
    }

    [Fact]
    public void KeepsDiagnosticsWhenGeometryValidationFails()
    {
        var samples = Samples(
            ("center", 0.00, 0.00),
            ("left", -0.20, 0.00),
            ("right", 0.10, 0.00),
            ("up", 0.00, 0.90),
            ("down", 0.00, -0.90));

        var fit = new RuntimeGazeCalibrationFitter().Fit(samples, new RuntimeGazeCalibrationOptions
        {
            MinSamplesPerStage = 2
        });

        Assert.False(fit.Succeeded);
        Assert.Contains("horizontal separation", fit.Reason);
        Assert.Equal(10, fit.SampleCount);
        Assert.Equal(0.30, fit.XSpan, precision: 6);
        Assert.Equal(1.80, fit.YSpan, precision: 6);
        Assert.Equal(5, fit.StageMedians.Count);
        Assert.Equal(2, fit.StageMedians["right"].Count);
    }

    private static IReadOnlyList<RuntimeGazeCalibrationSample> Samples(params (string Stage, double X, double Y)[] centers)
    {
        var samples = new List<RuntimeGazeCalibrationSample>();
        var sequence = 1L;
        foreach (var (stage, x, y) in centers)
        {
            samples.Add(new RuntimeGazeCalibrationSample(stage, x - 0.01, y - 0.01, sequence++, DateTimeOffset.UtcNow));
            samples.Add(new RuntimeGazeCalibrationSample(stage, x + 0.01, y + 0.01, sequence++, DateTimeOffset.UtcNow));
        }

        return samples;
    }
}
