namespace DreamAirTracking.Core.Calibration;

public static class CalibrationStageReviewAdvisor
{
    public static CalibrationStageReviewRecommendation Recommend(CalibrationStageReviewInput input)
    {
        if (input.PairCount == 0)
        {
            return new CalibrationStageReviewRecommendation(
                "No synchronized frames were captured.",
                "Start or restart BrokenEye streaming, then retry this point.",
                "retry");
        }

        if (input.Quality.Equals("good", StringComparison.OrdinalIgnoreCase) ||
            input.Quality.Equals("fair", StringComparison.OrdinalIgnoreCase))
        {
            return new CalibrationStageReviewRecommendation(
                "Enough usable pairs were captured.",
                "Accept this point unless the saved eye images visibly point somewhere else.",
                "accept");
        }

        var dominant = DominantFailure(input);
        var stageHint = StageSpecificHint(input.StageId);
        var plan = dominant switch
        {
            StageFailureKind.LowOpenness => "Keep both eyes open, reduce eyelid occlusion, and slightly adjust headset height before retrying.",
            StageFailureKind.NotFound => "Check that both eye cameras see the pupil, clean or uncover the lens area, and retry after the pupil is centered in frame.",
            StageFailureKind.LowConfidence => "Reduce glare/reflection, hold gaze steady on the target, and adjust the headset until the pupil edge is clearer.",
            StageFailureKind.SyncLate => "Restart the BrokenEye/Bridge stream or reduce capture load, then retry once left/right frames are synchronized.",
            _ => "Retry while holding the target steadily; if the same point fails again, mark it bad instead of accepting noisy labels."
        };

        if (!string.IsNullOrWhiteSpace(stageHint))
        {
            plan += " " + stageHint;
        }

        return new CalibrationStageReviewRecommendation(
            ReasonFor(dominant),
            plan,
            "retry");
    }

    private static StageFailureKind DominantFailure(CalibrationStageReviewInput input)
    {
        var failures = new[]
        {
            (Kind: StageFailureKind.LowOpenness, Count: input.LowOpennessPairs),
            (Kind: StageFailureKind.NotFound, Count: input.NotFoundPairs),
            (Kind: StageFailureKind.LowConfidence, Count: input.LowConfidencePairs),
            (Kind: StageFailureKind.SyncLate, Count: input.SyncLatePairs),
        };
        var dominant = failures
            .OrderByDescending(item => item.Count)
            .ThenBy(item => item.Kind)
            .First();
        return dominant.Count > 0 ? dominant.Kind : StageFailureKind.NotEnoughUsablePairs;
    }

    private static string ReasonFor(StageFailureKind kind) =>
        kind switch
        {
            StageFailureKind.LowOpenness => "Low eye openness is the main failure.",
            StageFailureKind.NotFound => "Pupil detection is missing in too many pairs.",
            StageFailureKind.LowConfidence => "Pupil confidence is too low.",
            StageFailureKind.SyncLate => "Left/right frame sync is too late.",
            _ => "Too few usable pairs were captured."
        };

    private static string StageSpecificHint(string stageId)
    {
        var lower = stageId.Contains("down", StringComparison.OrdinalIgnoreCase);
        var corner = stageId.Contains('_', StringComparison.Ordinal);
        if (lower && corner)
        {
            return "For lower-corner targets, move the eyes rather than the head and make sure the lower eyelid is not covering the pupil.";
        }

        if (lower)
        {
            return "For lower targets, keep the headset still and avoid eyelid cover over the pupil.";
        }

        if (corner)
        {
            return "For corner targets, keep the head still and hold the gaze until capture finishes.";
        }

        return string.Empty;
    }

    private enum StageFailureKind
    {
        LowOpenness,
        NotFound,
        LowConfidence,
        SyncLate,
        NotEnoughUsablePairs
    }
}

public sealed record CalibrationStageReviewInput(
    string StageId,
    int PairCount,
    int ValidPairCount,
    int LowOpennessPairs,
    int LowConfidencePairs,
    int NotFoundPairs,
    int SyncLatePairs,
    string Quality);

public sealed record CalibrationStageReviewRecommendation(
    string Reason,
    string ImprovementPlan,
    string SuggestedAction);
