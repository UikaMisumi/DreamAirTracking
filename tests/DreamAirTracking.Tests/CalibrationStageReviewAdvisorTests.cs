using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class CalibrationStageReviewAdvisorTests
{
    [Fact]
    public void AcceptsFairOrGoodStage()
    {
        var recommendation = CalibrationStageReviewAdvisor.Recommend(new CalibrationStageReviewInput(
            "center",
            PairCount: 30,
            ValidPairCount: 22,
            LowOpennessPairs: 0,
            LowConfidencePairs: 0,
            NotFoundPairs: 0,
            SyncLatePairs: 0,
            Quality: "fair"));

        Assert.Equal("accept", recommendation.SuggestedAction);
        Assert.Contains("Enough usable pairs", recommendation.Reason);
    }

    [Fact]
    public void RecommendsHeadsetAndEyelidFixForLowOpennessLowerCorner()
    {
        var recommendation = CalibrationStageReviewAdvisor.Recommend(new CalibrationStageReviewInput(
            "right_down",
            PairCount: 30,
            ValidPairCount: 4,
            LowOpennessPairs: 18,
            LowConfidencePairs: 5,
            NotFoundPairs: 3,
            SyncLatePairs: 0,
            Quality: "poor"));

        Assert.Equal("retry", recommendation.SuggestedAction);
        Assert.Contains("Low eye openness", recommendation.Reason);
        Assert.Contains("lower-corner", recommendation.ImprovementPlan);
    }

    [Fact]
    public void RecommendsStreamRestartForSyncLateDominantFailure()
    {
        var recommendation = CalibrationStageReviewAdvisor.Recommend(new CalibrationStageReviewInput(
            "left",
            PairCount: 30,
            ValidPairCount: 6,
            LowOpennessPairs: 0,
            LowConfidencePairs: 2,
            NotFoundPairs: 1,
            SyncLatePairs: 12,
            Quality: "poor"));

        Assert.Contains("sync", recommendation.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Restart", recommendation.ImprovementPlan);
    }
}
