using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class GazeCalibrationFitterTests
{
    [Fact]
    public void FitsAffineModelFromAcceptedStageLabels()
    {
        var labels = CreateNinePointLabels();
        var pairs = CreateSyntheticPairs(labels);

        var result = new GazeCalibrationFitter().Fit(pairs, labels);

        Assert.True(result.Succeeded);
        Assert.NotNull(result.LeftModel);
        Assert.NotNull(result.RightModel);
        Assert.Equal("good", result.Metrics.Quality);
        Assert.Equal("affine2d", result.Metrics.Model);
        Assert.True(result.Metrics.ValidPairs > 0);
        AssertTarget(result.LeftModel!, new CalibrationTarget(0, 0), 100, 90);
        AssertTarget(result.LeftModel!, new CalibrationTarget(1, 0), 130, 85);
        AssertTarget(result.LeftModel!, new CalibrationTarget(0, 1), 110, 115);
        Assert.True(result.LeftModel!.CenterDeadzoneX > 0);
    }

    [Fact]
    public void AutoSelectsQuadraticMapWhenNinePointDataIsCurved()
    {
        var labels = CreateNinePointLabels();
        var pairs = CreateCurvedPairs(labels);

        var result = new GazeCalibrationFitter().Fit(pairs, labels);

        Assert.True(result.Succeeded);
        Assert.Equal("quadratic2d", result.LeftModel!.Type);
        Assert.Equal("quadratic2d", result.RightModel!.Type);
        Assert.Equal("quadratic2d", result.Metrics.Model);
        AssertTarget(result.LeftModel!, new CalibrationTarget(-1, 1), 49.5, 120);
        AssertTarget(result.LeftModel!, new CalibrationTarget(1, -1), 109.5, 60);
        Assert.True(result.Metrics.Eyes["left"].AverageResidual < 0.03);
    }

    [Fact]
    public void UsesOperatorCorrectionsAsGroundTruth()
    {
        var labels = CreateNinePointLabels();
        var corrected = labels.Single(label => label.StageId == "left");
        corrected.OperatorOverrideTarget = new CalibrationTarget(-0.5, 0);
        corrected.Edited = true;
        var pairs = CreateSyntheticPairs(labels);

        var result = new GazeCalibrationFitter().Fit(pairs, labels);
        var predicted = result.LeftModel!.Map(85, 92.5);

        Assert.Equal(-0.5, predicted.X, 1);
        Assert.Equal(0, predicted.Y, 1);
    }

    [Fact]
    public void RejectsBadFramesAndOutliers()
    {
        var labels = CreateNinePointLabels();
        var center = labels.Single(label => label.StageId == "center");
        center.BadFrames.Add(center.FrameStart);
        var pairs = CreateSyntheticPairs(labels).ToList();
        pairs.Add(new CalibrationFramePair(center.FrameStart, "center", "bad-left.jpg", "bad-right.jpg", 1, true, 400, 400, 0.9, 1, true, 400, 400, 0.9, 1));

        var result = new GazeCalibrationFitter().Fit(pairs, labels);
        var predicted = result.LeftModel!.Map(100, 90);

        Assert.Equal(0, predicted.X, 1);
        Assert.Equal(0, predicted.Y, 1);
    }

    [Fact]
    public void IgnoresOpennessStagesForGazeFit()
    {
        var labels = CreateNinePointLabels();
        labels.Add(new CalibrationLabel
        {
            StageId = "closed",
            Target = new CalibrationTarget(0, 0),
            FrameStart = 100,
            FrameEnd = 104,
            Accepted = true
        });
        var pairs = CreateSyntheticPairs(labels).ToList();

        var result = new GazeCalibrationFitter().Fit(pairs, labels);

        Assert.DoesNotContain("closed", result.Metrics.StageResiduals.Keys);
        AssertTarget(result.LeftModel!, new CalibrationTarget(0, 0), 100, 90);
    }

    [Fact]
    public void AnchorsCenterStageEvenWhenOtherStagesAreBiased()
    {
        var labels = CreateNinePointLabels();
        var pairs = CreateSyntheticPairs(labels)
            .Select(pair => pair.StageId == "center"
                ? pair
                : pair with
                {
                    LeftRawX = pair.LeftRawX + 3,
                    RightRawX = pair.RightRawX + 3
                })
            .ToList();

        var result = new GazeCalibrationFitter().Fit(pairs, labels);

        Assert.True(result.Succeeded);
        AssertTarget(result.LeftModel!, new CalibrationTarget(0, 0), 100, 90);
    }

    private static List<CalibrationLabel> CreateNinePointLabels()
    {
        var stages = new (string Id, CalibrationTarget Target)[]
        {
            ("center", new CalibrationTarget(0, 0)),
            ("left", new CalibrationTarget(-1, 0)),
            ("right", new CalibrationTarget(1, 0)),
            ("up", new CalibrationTarget(0, 1)),
            ("down", new CalibrationTarget(0, -1)),
            ("left_up", new CalibrationTarget(-1, 1)),
            ("right_up", new CalibrationTarget(1, 1)),
            ("left_down", new CalibrationTarget(-1, -1)),
            ("right_down", new CalibrationTarget(1, -1)),
            ("center_confirm", new CalibrationTarget(0, 0))
        };

        var labels = new List<CalibrationLabel>();
        var sequence = 1;
        foreach (var stage in stages)
        {
            labels.Add(new CalibrationLabel
            {
                StageId = stage.Id,
                Target = stage.Target,
                FrameStart = sequence,
                FrameEnd = sequence + 4,
                Accepted = true
            });
            sequence += 5;
        }

        return labels;
    }

    private static IReadOnlyList<CalibrationFramePair> CreateSyntheticPairs(IReadOnlyList<CalibrationLabel> labels)
    {
        var pairs = new List<CalibrationFramePair>();
        foreach (var label in labels)
        {
            var raw = RawFromTarget(label.EffectiveTarget);
            for (var sequence = label.FrameStart; sequence <= label.FrameEnd; sequence++)
            {
                var jitter = (sequence - label.FrameStart - 2) * 0.15;
                pairs.Add(new CalibrationFramePair(
                    sequence,
                    label.StageId,
                    $"left_{sequence:0000}.jpg",
                    $"right_{sequence:0000}.jpg",
                    1.2,
                    true,
                    raw.X + jitter,
                    raw.Y - jitter,
                    0.8,
                    1,
                    true,
                    raw.X + 1 + jitter,
                    raw.Y - 1 - jitter,
                    0.75,
                    1));
            }
        }

        return pairs;
    }

    private static IReadOnlyList<CalibrationFramePair> CreateCurvedPairs(IReadOnlyList<CalibrationLabel> labels)
    {
        var pairs = new List<CalibrationFramePair>();
        foreach (var label in labels)
        {
            var raw = CurvedRawFromTarget(label.EffectiveTarget);
            for (var sequence = label.FrameStart; sequence <= label.FrameEnd; sequence++)
            {
                var jitter = (sequence - label.FrameStart - 2) * 0.03;
                pairs.Add(new CalibrationFramePair(
                    sequence,
                    label.StageId,
                    $"left_{sequence:0000}.jpg",
                    $"right_{sequence:0000}.jpg",
                    1.2,
                    true,
                    raw.X + jitter,
                    raw.Y - jitter,
                    0.8,
                    1,
                    true,
                    raw.X + 1 + jitter,
                    raw.Y - 1 - jitter,
                    0.75,
                    1));
            }
        }

        return pairs;
    }

    private static (double X, double Y) RawFromTarget(CalibrationTarget target) =>
        (100 + (30 * target.X) + (10 * target.Y), 90 - (5 * target.X) + (25 * target.Y));

    private static (double X, double Y) CurvedRawFromTarget(CalibrationTarget target)
    {
        var rawX = target.X - (0.35 * target.Y * target.Y);
        return (90 + (30 * rawX), 90 + (30 * target.Y));
    }

    private static void AssertTarget(Affine2DGazeModel model, CalibrationTarget expected, double rawX, double rawY)
    {
        var mapped = model.Map(rawX, rawY);
        Assert.Equal(expected.X, mapped.X, 1);
        Assert.Equal(expected.Y, mapped.Y, 1);
    }
}
