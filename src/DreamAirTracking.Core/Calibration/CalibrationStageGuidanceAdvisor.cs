namespace DreamAirTracking.Core.Calibration;

public static class CalibrationStageGuidanceAdvisor
{
    public static string BuildStageGuidance(
        string stageId,
        IReadOnlyList<CalibrationWeakStageSummary> weakStages,
        int minimumScore = 3)
    {
        var weak = weakStages.FirstOrDefault(stage =>
            stage.StageId.Equals(stageId, StringComparison.OrdinalIgnoreCase) &&
            stage.Score >= minimumScore);
        if (weak is null)
        {
            return "Auto map: automatic affine/quadratic fitting will run after accepted points.";
        }

        var reason = LikelyReason(stageId, weak);
        var plan = ImprovementPlan(stageId, weak);
        return $"Weak-stage guidance for {stageId.Replace('_', ' ')}: {reason}. {plan}";
    }

    private static string LikelyReason(string stageId, CalibrationWeakStageSummary weak)
    {
        if (weak.HighResidualCount >= weak.MissingCount && weak.HighResidualCount >= weak.PoorLabelCount)
        {
            return "historical sessions show high residuals, so this direction may expose profile or headset-placement mismatch";
        }

        if (stageId.Contains("down", StringComparison.OrdinalIgnoreCase))
        {
            return "historical sessions often miss or reject this lower target, usually from eyelid cover or headset height";
        }

        if (stageId.Contains("left", StringComparison.OrdinalIgnoreCase) ||
            stageId.Contains("right", StringComparison.OrdinalIgnoreCase))
        {
            return "historical sessions often miss or reject this edge target, usually from horizontal alignment or eye-crop drift";
        }

        return "historical sessions show repeated missing or rejected samples";
    }

    private static string ImprovementPlan(string stageId, CalibrationWeakStageSummary weak)
    {
        var parts = new List<string>();
        if (stageId.Contains("down", StringComparison.OrdinalIgnoreCase))
        {
            parts.Add("keep both eyes open and adjust headset height so the lower eyelid does not cover the pupil");
        }

        if (stageId.Contains("left", StringComparison.OrdinalIgnoreCase) ||
            stageId.Contains("right", StringComparison.OrdinalIgnoreCase))
        {
            parts.Add("check horizontal headset alignment and keep the pupil inside both eye crops");
        }

        if (weak.HighResidualCount > 0)
        {
            parts.Add("accept only if the eye image direction visibly matches the target");
        }

        if (parts.Count == 0)
        {
            parts.Add("hold gaze steady until capture finishes");
        }

        return string.Join("; ", parts) + ".";
    }
}
