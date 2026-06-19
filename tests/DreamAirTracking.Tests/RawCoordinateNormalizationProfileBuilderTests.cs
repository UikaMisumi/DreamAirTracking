using DreamAirTracking.Core.Calibration;
using DreamAirTracking.Core.Profiles;

namespace DreamAirTracking.Tests;

public sealed class RawCoordinateNormalizationProfileBuilderTests
{
    [Fact]
    public void BuildsMaskNormalizationFromAcceptedStageMedians()
    {
        var source = new TrackingProfile();
        source.Left.Calibration.CenterX = 100;
        source.Left.Calibration.CenterY = 80;
        source.Left.Calibration.HorizontalRange = 20;
        source.Left.Calibration.VerticalRange = 10;
        var labels = new List<CalibrationLabel>
        {
            Label("center", 0, 0, 1),
            Label("right", 1, 0, 2),
            Label("up", 0, 1, 3)
        };
        var pairs = new List<CalibrationFramePair>
        {
            Pair(1, "center", leftX: 130, leftY: 70),
            Pair(2, "right", leftX: 170, leftY: 70),
            Pair(3, "up", leftX: 130, leftY: 90)
        };

        var result = new RawCoordinateNormalizationProfileBuilder().BuildProfile(source, pairs, labels);

        Assert.True(result.Succeeded);
        Assert.True(result.Left.Succeeded);
        Assert.Equal("0.3-mask-normalized", result.Profile.Version);
        Assert.True(result.Profile.Left.InputTransform.Enabled);
        Assert.Equal(130, result.Profile.Left.InputTransform.CurrentCenterX, 3);
        Assert.Equal(70, result.Profile.Left.InputTransform.CurrentCenterY, 3);
        Assert.Equal(100, result.Profile.Left.InputTransform.TargetCenterX, 3);
        Assert.Equal(80, result.Profile.Left.InputTransform.TargetCenterY, 3);
        Assert.Equal(0.5, result.Profile.Left.InputTransform.ScaleX, 3);
        Assert.Equal(0.5, result.Profile.Left.InputTransform.ScaleY, 3);
    }

    [Fact]
    public void DoesNotSucceedWithoutCenterStage()
    {
        var source = new TrackingProfile();
        var labels = new List<CalibrationLabel> { Label("right", 1, 0, 1) };
        var pairs = new List<CalibrationFramePair> { Pair(1, "right", leftX: 170, leftY: 70) };

        var result = new RawCoordinateNormalizationProfileBuilder().BuildProfile(source, pairs, labels);

        Assert.False(result.Succeeded);
        Assert.False(result.Left.Succeeded);
    }

    private static CalibrationLabel Label(string stageId, double x, double y, long sequence) =>
        new()
        {
            StageId = stageId,
            Target = new CalibrationTarget(x, y),
            FrameStart = sequence,
            FrameEnd = sequence,
            Accepted = true,
            OperatorVerdict = "good"
        };

    private static CalibrationFramePair Pair(long sequence, string stageId, double leftX, double leftY) =>
        new(
            sequence,
            stageId,
            string.Empty,
            string.Empty,
            0,
            true,
            leftX,
            leftY,
            0.8,
            1.0,
            false,
            0,
            0,
            0,
            0);
}
