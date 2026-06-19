using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class CalibrationProfileApplicationGateTests
{
    [Fact]
    public void AllowsCompleteFairCalibration()
    {
        var decision = CalibrationProfileApplicationGate.Decide(
            Metrics("fair"),
            Labels("center", "up", "down"),
            Stages("center", "up", "down"),
            normalizedProfileSucceeded: true);

        Assert.True(decision.ShouldApply);
    }

    [Fact]
    public void RejectsPoorCalibration()
    {
        var decision = CalibrationProfileApplicationGate.Decide(
            Metrics("poor"),
            Labels("center", "up", "down"),
            Stages("center", "up", "down"),
            normalizedProfileSucceeded: true);

        Assert.False(decision.ShouldApply);
        Assert.Contains("poor", decision.Reason);
    }

    [Fact]
    public void RejectsMissingAcceptedStages()
    {
        var decision = CalibrationProfileApplicationGate.Decide(
            Metrics("good"),
            Labels("center", "up"),
            Stages("center", "up", "down"),
            normalizedProfileSucceeded: true);

        Assert.False(decision.ShouldApply);
        Assert.Equal(new[] { "down" }, decision.MissingAcceptedStageIds);
    }

    [Fact]
    public void RejectsHighResidualStages()
    {
        var metrics = Metrics("fair");
        metrics.StageResiduals["left_down"] = new CalibrationResidual { Distance = 0.72 };

        var decision = CalibrationProfileApplicationGate.Decide(
            metrics,
            Labels("center", "left_down"),
            Stages("center", "left_down"),
            normalizedProfileSucceeded: true);

        Assert.False(decision.ShouldApply);
        Assert.Equal("left_down", Assert.Single(decision.HighResiduals).StageId);
    }

    private static CalibrationMetrics Metrics(string quality) =>
        new()
        {
            Quality = quality,
            StageResiduals =
            {
                ["center"] = new CalibrationResidual { Distance = 0.02 },
                ["up"] = new CalibrationResidual { Distance = 0.06 },
                ["down"] = new CalibrationResidual { Distance = 0.07 }
            }
        };

    private static IReadOnlyList<CalibrationLabel> Labels(params string[] stageIds) =>
        stageIds.Select(stageId => new CalibrationLabel
        {
            StageId = stageId,
            Target = CalibrationTarget.Center,
            Accepted = true,
            OperatorVerdict = "good"
        }).ToArray();

    private static IReadOnlyList<CalibrationStage> Stages(params string[] stageIds) =>
        stageIds.Select((stageId, index) => new CalibrationStage
        {
            Order = index + 1,
            StageId = stageId,
            Target = CalibrationTarget.Center,
            DisplayName = stageId
        }).ToArray();
}
