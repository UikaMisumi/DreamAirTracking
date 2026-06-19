namespace DreamAirTracking.Core.Calibration;

public static class CalibrationProfileApplicationGate
{
    public static CalibrationProfileApplicationDecision Decide(
        CalibrationMetrics metrics,
        IReadOnlyList<CalibrationLabel> labels,
        IReadOnlyList<CalibrationStage> requiredStages,
        bool normalizedProfileSucceeded,
        double maxStageResidual = 0.5)
    {
        var missingStages = FindMissingAcceptedStages(labels, requiredStages);
        var highResiduals = FindHighResiduals(metrics, maxStageResidual);
        var isPoor = metrics.Quality.Equals("poor", StringComparison.OrdinalIgnoreCase);

        if (!normalizedProfileSucceeded)
        {
            return new CalibrationProfileApplicationDecision(
                false,
                "No normalized profile was generated.",
                missingStages,
                highResiduals);
        }

        if (missingStages.Count > 0)
        {
            return new CalibrationProfileApplicationDecision(
                false,
                $"Missing accepted stages: {string.Join(", ", missingStages)}.",
                missingStages,
                highResiduals);
        }

        if (isPoor)
        {
            return new CalibrationProfileApplicationDecision(
                false,
                "Calibration metrics are poor.",
                missingStages,
                highResiduals);
        }

        if (highResiduals.Count > 0)
        {
            return new CalibrationProfileApplicationDecision(
                false,
                $"High stage residuals: {string.Join(", ", highResiduals.Select(item => $"{item.StageId}={item.Distance:0.00}"))}.",
                missingStages,
                highResiduals);
        }

        return new CalibrationProfileApplicationDecision(
            true,
            "Calibration passed profile application gates.",
            missingStages,
            highResiduals);
    }

    private static IReadOnlyList<string> FindMissingAcceptedStages(
        IReadOnlyList<CalibrationLabel> labels,
        IReadOnlyList<CalibrationStage> requiredStages)
    {
        var accepted = labels
            .Where(label => label.Accepted && !label.OperatorVerdict.Equals("bad", StringComparison.OrdinalIgnoreCase))
            .Select(label => label.StageId)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        return requiredStages
            .Where(stage => !accepted.Contains(stage.StageId))
            .Select(stage => stage.StageId)
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static IReadOnlyList<CalibrationStageResidualSummary> FindHighResiduals(
        CalibrationMetrics metrics,
        double maxStageResidual) =>
        metrics.StageResiduals
            .Where(item => item.Value.Distance > maxStageResidual)
            .OrderByDescending(item => item.Value.Distance)
            .Select(item => new CalibrationStageResidualSummary(item.Key, item.Value.Distance))
            .ToArray();
}

public sealed record CalibrationProfileApplicationDecision(
    bool ShouldApply,
    string Reason,
    IReadOnlyList<string> MissingAcceptedStageIds,
    IReadOnlyList<CalibrationStageResidualSummary> HighResiduals);

public sealed record CalibrationStageResidualSummary(string StageId, double Distance);
