namespace DreamAirTracking.Core.Calibration;

public static class CalibrationSessionAuditor
{
    public static readonly IReadOnlyList<string> RequiredStageIds = new[]
    {
        "center",
        "up",
        "down",
        "left",
        "right",
        "left_up",
        "right_up",
        "left_down",
        "right_down"
    };

    public static string DefaultCalibrationRoot =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "DreamAirTracking",
            "calibration_data");

    public static async Task<CalibrationSessionAuditSummary> AuditRootAsync(
        string? rootDirectory = null,
        int minStageSamples = 20,
        double maxStageResidual = 0.5,
        CancellationToken cancellationToken = default)
    {
        var root = rootDirectory ?? DefaultCalibrationRoot;
        var sessions = DiscoverSessionDirectories(root)
            .Select(path => AuditSessionAsync(path, minStageSamples, maxStageResidual, cancellationToken))
            .ToArray();
        var reports = await Task.WhenAll(sessions);
        var ordered = reports
            .OrderByDescending(report => report.LastWriteTimeUtc)
            .ThenByDescending(report => report.SessionId, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        return CalibrationSessionAuditSummary.FromReports(root, ordered);
    }

    public static async Task<CalibrationSessionAuditReport> AuditSessionAsync(
        string sessionDirectory,
        int minStageSamples = 20,
        double maxStageResidual = 0.5,
        CancellationToken cancellationToken = default)
    {
        var labels = await CalibrationSessionStore.LoadLabelsAsync(sessionDirectory, cancellationToken);
        var pairs = await CalibrationSessionStore.LoadPairsAsync(sessionDirectory, cancellationToken);
        var metrics = await LoadMetricsOrDefaultAsync(sessionDirectory, cancellationToken);
        var exclusion = await LoadSessionExclusionAsync(sessionDirectory, cancellationToken);
        var acceptedCounts = CountAcceptedPairs(pairs, labels);
        var missing = RequiredStageIds
            .Where(stage => acceptedCounts.GetValueOrDefault(stage) < minStageSamples)
            .ToArray();
        var poorLabels = labels
            .Where(label => !label.Accepted || IsPoorVerdict(label.OperatorVerdict))
            .Select(label => label.StageId)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var highResiduals = metrics.StageResiduals
            .Where(item => item.Value.Distance > maxStageResidual)
            .OrderByDescending(item => item.Value.Distance)
            .Select(item => new CalibrationSessionStageResidual(item.Key, item.Value.Distance))
            .ToArray();
        double? averageResidual = metrics.StageResiduals.Count == 0
            ? null
            : metrics.StageResiduals.Values.Average(item => item.Distance);
        var category = Classify(
            exclusion.IsExcluded,
            metrics.Quality,
            acceptedCounts.Values.Sum(),
            missing,
            poorLabels,
            highResiduals);
        return new CalibrationSessionAuditReport(
            Path.GetFileName(sessionDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)),
            Path.GetFullPath(sessionDirectory),
            Directory.GetLastWriteTimeUtc(sessionDirectory),
            metrics.Quality,
            averageResidual,
            acceptedCounts,
            missing,
            poorLabels,
            highResiduals,
            exclusion.IsExcluded,
            exclusion.Reason,
            category,
            DecisionFor(category));
    }

    private static IReadOnlyList<string> DiscoverSessionDirectories(string rootDirectory)
    {
        if (!Directory.Exists(rootDirectory))
        {
            return Array.Empty<string>();
        }

        var immediate = Directory.EnumerateDirectories(rootDirectory).ToArray();
        if (immediate.Length > 0)
        {
            return immediate;
        }

        return Directory
            .EnumerateFiles(rootDirectory, "pairs.csv", SearchOption.AllDirectories)
            .Select(Path.GetDirectoryName)
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Select(path => path!)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static async Task<CalibrationMetrics> LoadMetricsOrDefaultAsync(
        string sessionDirectory,
        CancellationToken cancellationToken)
    {
        var path = Path.Combine(sessionDirectory, "metrics.json");
        if (!File.Exists(path))
        {
            return new CalibrationMetrics { Quality = "missing" };
        }

        return await CalibrationSessionStore.LoadMetricsAsync(sessionDirectory, cancellationToken);
    }

    private static async Task<SessionExclusion> LoadSessionExclusionAsync(
        string sessionDirectory,
        CancellationToken cancellationToken)
    {
        var path = Path.Combine(sessionDirectory, "session_exclude.json");
        if (!File.Exists(path))
        {
            return new SessionExclusion(false, string.Empty);
        }

        try
        {
            await using var stream = File.OpenRead(path);
            var document = await System.Text.Json.JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
            var root = document.RootElement;
            var excluded = !root.TryGetProperty("exclude", out var excludeValue) ||
                excludeValue.ValueKind != System.Text.Json.JsonValueKind.False;
            var reason = root.TryGetProperty("reason", out var reasonValue) &&
                reasonValue.ValueKind == System.Text.Json.JsonValueKind.String
                    ? reasonValue.GetString() ?? "session excluded"
                    : "session excluded";
            return new SessionExclusion(excluded, reason);
        }
        catch
        {
            return new SessionExclusion(true, "invalid session_exclude.json");
        }
    }

    private static Dictionary<string, int> CountAcceptedPairs(
        IReadOnlyList<CalibrationFramePair> pairs,
        IReadOnlyList<CalibrationLabel> labels)
    {
        var acceptedRanges = labels
            .Where(label => label.Accepted)
            .ToArray();
        var counts = RequiredStageIds.ToDictionary(stage => stage, _ => 0, StringComparer.OrdinalIgnoreCase);
        foreach (var pair in pairs)
        {
            if (!counts.ContainsKey(pair.StageId))
            {
                continue;
            }

            if (acceptedRanges.Any(label =>
                    label.StageId.Equals(pair.StageId, StringComparison.OrdinalIgnoreCase) &&
                    label.FrameStart <= pair.Sequence &&
                    pair.Sequence <= label.FrameEnd))
            {
                counts[pair.StageId]++;
            }
        }

        return counts;
    }

    private static bool IsPoorVerdict(string verdict) =>
        verdict.Equals("poor", StringComparison.OrdinalIgnoreCase) ||
        verdict.Equals("bad", StringComparison.OrdinalIgnoreCase);

    private static CalibrationSessionAuditCategory Classify(
        bool excluded,
        string quality,
        int acceptedTotal,
        IReadOnlyList<string> missing,
        IReadOnlyList<string> poorLabels,
        IReadOnlyList<CalibrationSessionStageResidual> highResiduals)
    {
        if (excluded)
        {
            return CalibrationSessionAuditCategory.Excluded;
        }

        if (acceptedTotal == 0)
        {
            return CalibrationSessionAuditCategory.EmptyOrUnlabeled;
        }

        var fairOrGood = quality.Equals("fair", StringComparison.OrdinalIgnoreCase) ||
            quality.Equals("good", StringComparison.OrdinalIgnoreCase);
        var poor = quality.Equals("poor", StringComparison.OrdinalIgnoreCase);

        if (missing.Count == 0 && fairOrGood && poorLabels.Count == 0 && highResiduals.Count == 0)
        {
            return CalibrationSessionAuditCategory.StrictUsable;
        }

        if (missing.Count == 0 && (poor || highResiduals.Count > 0))
        {
            return CalibrationSessionAuditCategory.CompleteButUnstable;
        }

        if (missing.Count > 0 && fairOrGood)
        {
            return CalibrationSessionAuditCategory.PartialCandidate;
        }

        if (missing.Count > 0)
        {
            return CalibrationSessionAuditCategory.Incomplete;
        }

        return CalibrationSessionAuditCategory.ReviewOnly;
    }

    private static string DecisionFor(CalibrationSessionAuditCategory category) =>
        category switch
        {
            CalibrationSessionAuditCategory.Excluded => "Excluded from automatic use by session_exclude.json.",
            CalibrationSessionAuditCategory.StrictUsable => "Can be used as a fair held-out session or Bridge-layer source.",
            CalibrationSessionAuditCategory.PartialCandidate => "Some stages look usable, but missing accepted stages make it incomplete.",
            CalibrationSessionAuditCategory.CompleteButUnstable => "Complete coverage; usable for image-model review/training, but residuals/quality are too weak for automatic Bridge/profile use.",
            CalibrationSessionAuditCategory.Incomplete => "Missing accepted stages; use only for debugging weak directions.",
            CalibrationSessionAuditCategory.EmptyOrUnlabeled => "No accepted calibration ranges; cannot use for calibration or training.",
            _ => "Keep as review evidence; do not use for automatic model/profile decisions."
        };
}

public sealed record CalibrationSessionAuditSummary(
    string RootDirectory,
    IReadOnlyList<CalibrationSessionAuditReport> Sessions,
    int StrictUsableCount,
    int ImageTrainingUsableCount,
    int ExcludedCount,
    int PartialCandidateCount,
    int CompleteButUnstableCount,
    int IncompleteCount,
    int EmptyOrUnlabeledCount,
    int ReviewOnlyCount,
    IReadOnlyList<CalibrationWeakStageSummary> WeakStages,
    CalibrationFairnessAudit FairnessAudit)
{
    public int TotalCount => Sessions.Count;
    public bool NeedsMoreDataForSessionValidation => StrictUsableCount < 2;
    public int AdditionalStrictSessionsNeeded => Math.Max(0, 2 - StrictUsableCount);
    public bool NeedsMoreImageTrainingData => ImageTrainingUsableCount < 2;
    public int AdditionalImageTrainingSessionsNeeded => Math.Max(0, 2 - ImageTrainingUsableCount);
    public string WeakStageText => WeakStages.Count == 0
        ? "none"
        : string.Join(", ", WeakStages.Take(4).Select(stage => $"{stage.StageId}({stage.Score})"));

    public string BottomLine => NeedsMoreDataForSessionValidation
        ? NeedsMoreImageTrainingData
            ? $"Need {AdditionalImageTrainingSessionsNeeded} more complete accepted non-excluded 9-point image session(s) for image-model validation."
            : $"Enough complete accepted image sessions exist for image-model training. Need {AdditionalStrictSessionsNeeded} more strict profile/affine session(s) for Bridge/profile validation."
        : "Enough strict usable sessions exist for session-based validation.";

    public string NextRecordingPlan
    {
        get
        {
            if (!NeedsMoreDataForSessionValidation && WeakStages.Count == 0)
            {
                return "No extra recording is currently requested by the audit.";
            }

            var basePlan = NeedsMoreImageTrainingData
                ? $"Record {AdditionalImageTrainingSessionsNeeded} new full 9-point image session(s) after reseating the headset."
                : NeedsMoreDataForSessionValidation
                    ? "No immediate image-data recording is requested; record another strict profile/affine session only if Bridge/profile validation is the target."
                : "Run another full 9-point session only if live tracking still looks unstable.";
            return WeakStages.Count == 0
                ? basePlan
                : $"{basePlan} Pay extra attention to weak stages: {WeakStageText}.";
        }
    }

    public static CalibrationSessionAuditSummary FromReports(
        string rootDirectory,
        IReadOnlyList<CalibrationSessionAuditReport> reports)
    {
        var weakStages = BuildWeakStageSummary(reports);
        var fairness = BuildFairnessAudit(reports, weakStages);
        return new(
            rootDirectory,
            reports,
            reports.Count(report => report.Category == CalibrationSessionAuditCategory.StrictUsable),
            reports.Count(report => report.IsImageTrainingUsable),
            reports.Count(report => report.Category == CalibrationSessionAuditCategory.Excluded),
            reports.Count(report => report.Category == CalibrationSessionAuditCategory.PartialCandidate),
            reports.Count(report => report.Category == CalibrationSessionAuditCategory.CompleteButUnstable),
            reports.Count(report => report.Category == CalibrationSessionAuditCategory.Incomplete),
            reports.Count(report => report.Category == CalibrationSessionAuditCategory.EmptyOrUnlabeled),
            reports.Count(report => report.Category == CalibrationSessionAuditCategory.ReviewOnly),
            weakStages,
            fairness);
    }

    private static CalibrationFairnessAudit BuildFairnessAudit(
        IReadOnlyList<CalibrationSessionAuditReport> reports,
        IReadOnlyList<CalibrationWeakStageSummary> weakStages)
    {
        var reviewedCount = reports.Count(report =>
            report.Category != CalibrationSessionAuditCategory.EmptyOrUnlabeled &&
            report.Category != CalibrationSessionAuditCategory.Excluded);
        if (reviewedCount == 0)
        {
            return new CalibrationFairnessAudit(
                false,
                Array.Empty<CalibrationWeakStageSummary>(),
                "Not enough reviewed calibration data to check for stage-specific bias.",
                "Record a full 9-point session before trusting fairness conclusions.");
        }

        if (weakStages.Count == 0)
        {
            return new CalibrationFairnessAudit(
                false,
                Array.Empty<CalibrationWeakStageSummary>(),
                "No repeated stage-specific failure pattern detected by the current audit.",
                "Keep session-based validation; do not loosen automatic gates unless live tracking still looks unstable.");
        }

        var orderedScores = weakStages.Select(stage => stage.Score).Order().ToArray();
        var medianLike = orderedScores[orderedScores.Length / 2];
        var flagged = weakStages
            .Where(stage => stage.Score >= 3 && stage.Score >= Math.Max(2, medianLike))
            .Take(4)
            .ToArray();
        if (flagged.Length == 0)
        {
            return new CalibrationFairnessAudit(
                false,
                weakStages.Take(4).ToArray(),
                $"Weak stages exist, but the pattern is not strong enough to call systematic bias yet: {string.Join(", ", weakStages.Take(4).Select(stage => $"{stage.StageId}={stage.Score}"))}.",
                "Keep collecting session-based evidence before changing calibration thresholds.");
        }

        return new CalibrationFairnessAudit(
            true,
            flagged,
            $"Stage-specific failure pattern detected across {reviewedCount} reviewed sessions: {string.Join(", ", flagged.Select(stage => $"{stage.StageId}={stage.Score}"))}.",
            "Treat this as a calibration-gate bias risk, not proof of user error. Keep flagged stages out of automatic training/profile decisions, but use them to guide the next recording and manual review.");
    }

    private static IReadOnlyList<CalibrationWeakStageSummary> BuildWeakStageSummary(
        IReadOnlyList<CalibrationSessionAuditReport> reports)
    {
        var mutable = CalibrationSessionAuditor.RequiredStageIds.ToDictionary(
            stage => stage,
            _ => new MutableWeakStageSummary(),
            StringComparer.OrdinalIgnoreCase);
        foreach (var report in reports.Where(report =>
            report.Category != CalibrationSessionAuditCategory.EmptyOrUnlabeled &&
            report.Category != CalibrationSessionAuditCategory.Excluded))
        {
            foreach (var stage in report.MissingStageIds)
            {
                mutable[stage].MissingCount++;
            }

            foreach (var stage in report.PoorLabelStageIds)
            {
                mutable[stage].PoorLabelCount++;
            }

            foreach (var residual in report.HighResiduals)
            {
                if (mutable.TryGetValue(residual.StageId, out var item))
                {
                    item.HighResidualCount++;
                }
            }
        }

        return mutable
            .Select(item => new CalibrationWeakStageSummary(
                item.Key,
                item.Value.MissingCount,
                item.Value.PoorLabelCount,
                item.Value.HighResidualCount))
            .Where(item => item.Score > 0)
            .OrderByDescending(item => item.Score)
            .ThenBy(item => StageOrder(item.StageId))
            .ToArray();
    }

    private static int StageOrder(string stageId)
    {
        for (var index = 0; index < CalibrationSessionAuditor.RequiredStageIds.Count; index++)
        {
            if (CalibrationSessionAuditor.RequiredStageIds[index].Equals(stageId, StringComparison.OrdinalIgnoreCase))
            {
                return index;
            }
        }

        return int.MaxValue;
    }

    private sealed class MutableWeakStageSummary
    {
        public int MissingCount { get; set; }
        public int PoorLabelCount { get; set; }
        public int HighResidualCount { get; set; }
    }
}

public sealed record CalibrationSessionAuditReport(
    string SessionId,
    string Directory,
    DateTime LastWriteTimeUtc,
    string Quality,
    double? AverageResidual,
    IReadOnlyDictionary<string, int> AcceptedCounts,
    IReadOnlyList<string> MissingStageIds,
    IReadOnlyList<string> PoorLabelStageIds,
    IReadOnlyList<CalibrationSessionStageResidual> HighResiduals,
    bool IsExcluded,
    string ExcludeReason,
    CalibrationSessionAuditCategory Category,
    string Decision)
{
    public bool IsImageTrainingUsable =>
        !IsExcluded &&
        MissingStageIds.Count == 0 &&
        PoorLabelStageIds.Count == 0 &&
        AcceptedCounts.Values.Sum() > 0;

    public string CategoryLabel => Category switch
    {
        CalibrationSessionAuditCategory.Excluded => "Excluded",
        CalibrationSessionAuditCategory.StrictUsable => "Strict usable",
        CalibrationSessionAuditCategory.PartialCandidate => "Partial candidate",
        CalibrationSessionAuditCategory.CompleteButUnstable => "Complete but unstable",
        CalibrationSessionAuditCategory.Incomplete => "Incomplete",
        CalibrationSessionAuditCategory.EmptyOrUnlabeled => "Empty/unlabeled",
        _ => "Review only"
    };

    public string MissingStageText => MissingStageIds.Count == 0 ? "none" : string.Join(", ", MissingStageIds);
    public string PoorLabelText => PoorLabelStageIds.Count == 0 ? "none" : string.Join(", ", PoorLabelStageIds);
    public string HighResidualText => HighResiduals.Count == 0
        ? "none"
        : string.Join(", ", HighResiduals.Select(item => $"{item.StageId}={item.Distance:0.00}"));
    public string AverageResidualText => AverageResidual.HasValue ? AverageResidual.Value.ToString("0.000") : "n/a";
    public string MissingStageDisplay => $"Missing: {MissingStageText}";
    public string PoorLabelDisplay => $"Poor labels: {PoorLabelText}";
    public string HighResidualDisplay => $"High residual: {HighResidualText}";
    public string ImageTrainingDisplay => IsImageTrainingUsable
        ? "Image training: usable"
        : IsExcluded
            ? $"Image training: excluded ({ExcludeReason})"
            : "Image training: review only";
}

internal sealed record SessionExclusion(bool IsExcluded, string Reason);

public sealed record CalibrationSessionStageResidual(string StageId, double Distance);

public sealed record CalibrationWeakStageSummary(
    string StageId,
    int MissingCount,
    int PoorLabelCount,
    int HighResidualCount)
{
    public int Score => MissingCount + PoorLabelCount + HighResidualCount;
    public string DetailText => $"missing={MissingCount}, poor={PoorLabelCount}, residual={HighResidualCount}";
}

public sealed record CalibrationFairnessAudit(
    bool HasStageSpecificPattern,
    IReadOnlyList<CalibrationWeakStageSummary> FlaggedStages,
    string StatusText,
    string RecommendationText);

public enum CalibrationSessionAuditCategory
{
    Excluded,
    StrictUsable,
    PartialCandidate,
    CompleteButUnstable,
    Incomplete,
    EmptyOrUnlabeled,
    ReviewOnly
}
