using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class CalibrationStageCatalogTests
{
    [Fact]
    public void CreatesDefaultGazeAndOpennessStages()
    {
        var stages = CalibrationStageCatalog.CreateDefaultGazeStages();

        Assert.Equal(12, stages.Count);
        Assert.Equal(
            new[] { "center", "up", "down", "left", "right", "left_up", "right_up", "left_down", "right_down" },
            stages.Take(9).Select(stage => stage.StageId).ToArray());
        Assert.Equal(new CalibrationTarget(-1, -1), stages.Single(stage => stage.StageId == "left_down").Target);
        Assert.Equal("center_confirm", stages[9].StageId);
        Assert.Contains(stages, stage => stage.StageId == "closed");
        Assert.Contains(stages, stage => stage.StageId == "open");
    }

    [Fact]
    public void CreatesFiveAndNinePointGazeStageSets()
    {
        Assert.Equal(
            new[] { "center", "up", "down", "left", "right" },
            CalibrationStageCatalog.CreateFivePointGazeStages().Select(stage => stage.StageId).ToArray());
        Assert.Equal(9, CalibrationStageCatalog.CreateNinePointGazeStages().Count);
    }
}
