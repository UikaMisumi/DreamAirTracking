using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class RecordingRequestSummaryParserTests
{
    [Fact]
    public void SummarizesWeakStageFixPlanRows()
    {
        var markdown = new[]
        {
            "# Recording Request",
            "",
            "### Weak Stage Fix Plan",
            "",
            "| Stage | Evidence | Likely cause | Next improvement |",
            "|---|---|---|---|",
            "| right_down | score=27 | lower-target capture is fragile | keep both eyes open; check horizontal alignment |",
            "| left_down | score=26 | geometry/profile mismatch | keep this stage out of automatic profile decisions; record again |",
            "",
            "Required stages:",
        };

        var summary = RecordingRequestSummaryParser.SummarizeWeakStageFixPlan(markdown, maxRows: 2);

        Assert.NotNull(summary);
        Assert.Contains("right_down: lower-target capture is fragile; keep both eyes open", summary);
        Assert.Contains("left_down: geometry/profile mismatch; keep this stage out of automatic profile decisions", summary);
    }

    [Fact]
    public void ReturnsNullWhenFixPlanIsMissing()
    {
        var summary = RecordingRequestSummaryParser.SummarizeWeakStageFixPlan(new[] { "# Recording Request" });

        Assert.Null(summary);
    }
}
