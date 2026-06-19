using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class CalibrationStageGuidanceAdvisorTests
{
    [Fact]
    public void BuildsLowerCornerGuidanceForWeakStage()
    {
        var guidance = CalibrationStageGuidanceAdvisor.BuildStageGuidance(
            "right_down",
            new[]
            {
                new CalibrationWeakStageSummary("right_down", MissingCount: 8, PoorLabelCount: 4, HighResidualCount: 2)
            });

        Assert.Contains("right down", guidance);
        Assert.Contains("lower target", guidance);
        Assert.Contains("lower eyelid", guidance);
        Assert.Contains("horizontal headset alignment", guidance);
    }

    [Fact]
    public void KeepsDefaultGuidanceForNonWeakStage()
    {
        var guidance = CalibrationStageGuidanceAdvisor.BuildStageGuidance(
            "center",
            new[]
            {
                new CalibrationWeakStageSummary("right_down", MissingCount: 8, PoorLabelCount: 4, HighResidualCount: 2)
            });

        Assert.Equal("Auto map: automatic affine/quadratic fitting will run after accepted points.", guidance);
    }
}
