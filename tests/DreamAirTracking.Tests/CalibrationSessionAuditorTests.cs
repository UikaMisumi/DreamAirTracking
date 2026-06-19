using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class CalibrationSessionAuditorTests
{
    [Fact]
    public async Task ClassifiesCompleteFairSessionAsStrictUsable()
    {
        var root = CreateTempDirectory();
        try
        {
            var session = Path.Combine(root, "session_strict");
            await WriteSessionAsync(session, "fair", missingStages: Array.Empty<string>(), highResidualStages: Array.Empty<string>());

            var report = await CalibrationSessionAuditor.AuditSessionAsync(session);

            Assert.Equal(CalibrationSessionAuditCategory.StrictUsable, report.Category);
            Assert.Equal("none", report.MissingStageText);
            Assert.Equal(20, report.AcceptedCounts["center"]);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task ClassifiesGoodSessionWithMissingStageAsPartialCandidate()
    {
        var root = CreateTempDirectory();
        try
        {
            var session = Path.Combine(root, "session_partial");
            await WriteSessionAsync(session, "good", missingStages: new[] { "right_down" }, highResidualStages: Array.Empty<string>());

            var report = await CalibrationSessionAuditor.AuditSessionAsync(session);

            Assert.Equal(CalibrationSessionAuditCategory.PartialCandidate, report.Category);
            Assert.Contains("right_down", report.MissingStageIds);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task ClassifiesCompletePoorSessionAsCompleteButUnstable()
    {
        var root = CreateTempDirectory();
        try
        {
            var session = Path.Combine(root, "session_unstable");
            await WriteSessionAsync(session, "poor", missingStages: Array.Empty<string>(), highResidualStages: new[] { "left" });

            var report = await CalibrationSessionAuditor.AuditSessionAsync(session);

            Assert.Equal(CalibrationSessionAuditCategory.CompleteButUnstable, report.Category);
            Assert.True(report.IsImageTrainingUsable);
            Assert.Contains(report.HighResiduals, item => item.StageId == "left");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task ExcludedSessionDoesNotCountAsImageTrainingUsable()
    {
        var root = CreateTempDirectory();
        try
        {
            var session = Path.Combine(root, "session_no_eye");
            await WriteSessionAsync(session, "poor", missingStages: Array.Empty<string>(), highResidualStages: Array.Empty<string>());
            await File.WriteAllTextAsync(
                Path.Combine(session, "session_exclude.json"),
                """
                {
                  "exclude": true,
                  "reason": "no_eye_image"
                }
                """);

            var report = await CalibrationSessionAuditor.AuditSessionAsync(session);
            var summary = await CalibrationSessionAuditor.AuditRootAsync(root);

            Assert.Equal(CalibrationSessionAuditCategory.Excluded, report.Category);
            Assert.False(report.IsImageTrainingUsable);
            Assert.Equal("no_eye_image", report.ExcludeReason);
            Assert.Equal(1, summary.ExcludedCount);
            Assert.Equal(0, summary.ImageTrainingUsableCount);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task RootSummaryReportsMoreDataNeededForSessionValidation()
    {
        var root = CreateTempDirectory();
        try
        {
            await WriteSessionAsync(Path.Combine(root, "session_strict"), "fair", Array.Empty<string>(), Array.Empty<string>());
            await WriteSessionAsync(Path.Combine(root, "session_partial"), "good", new[] { "right_down" }, Array.Empty<string>());

            var summary = await CalibrationSessionAuditor.AuditRootAsync(root);

            Assert.Equal(2, summary.TotalCount);
            Assert.Equal(1, summary.StrictUsableCount);
            Assert.Equal(1, summary.ImageTrainingUsableCount);
            Assert.True(summary.NeedsMoreDataForSessionValidation);
            Assert.Equal(1, summary.AdditionalStrictSessionsNeeded);
            Assert.Contains("Record 1 new full 9-point", summary.NextRecordingPlan);
            Assert.Contains("right_down", summary.WeakStageText);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task RootSummaryFlagsRepeatedStageSpecificPattern()
    {
        var root = CreateTempDirectory();
        try
        {
            await WriteSessionAsync(Path.Combine(root, "session_a"), "good", new[] { "right_down" }, Array.Empty<string>());
            await WriteSessionAsync(Path.Combine(root, "session_b"), "fair", new[] { "right_down" }, Array.Empty<string>());
            await WriteSessionAsync(Path.Combine(root, "session_c"), "poor", Array.Empty<string>(), new[] { "right_down" });

            var summary = await CalibrationSessionAuditor.AuditRootAsync(root);

            Assert.True(summary.FairnessAudit.HasStageSpecificPattern);
            Assert.Contains(summary.FairnessAudit.FlaggedStages, stage => stage.StageId == "right_down");
            Assert.Contains("bias risk", summary.FairnessAudit.RecommendationText);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static async Task WriteSessionAsync(
        string sessionDirectory,
        string quality,
        IReadOnlyCollection<string> missingStages,
        IReadOnlyCollection<string> highResidualStages)
    {
        Directory.CreateDirectory(sessionDirectory);
        var labels = new List<CalibrationLabel>();
        var pairs = new List<CalibrationFramePair>();
        var sequence = 1L;
        foreach (var stage in CalibrationSessionAuditor.RequiredStageIds)
        {
            var sampleCount = missingStages.Contains(stage) ? 10 : 20;
            var start = sequence;
            for (var i = 0; i < sampleCount; i++)
            {
                pairs.Add(new CalibrationFramePair(
                    sequence,
                    stage,
                    "left.jpg",
                    "right.jpg",
                    1,
                    true,
                    0,
                    0,
                    0.9,
                    1,
                    true,
                    0,
                    0,
                    0.9,
                    1));
                sequence++;
            }

            labels.Add(new CalibrationLabel
            {
                StageId = stage,
                FrameStart = start,
                FrameEnd = sequence - 1,
                Accepted = true,
                OperatorVerdict = "good"
            });
        }

        var metrics = new CalibrationMetrics { Quality = quality };
        foreach (var stage in CalibrationSessionAuditor.RequiredStageIds)
        {
            metrics.StageResiduals[stage] = new CalibrationResidual
            {
                Distance = highResidualStages.Contains(stage) ? 0.8 : 0.1
            };
        }

        await CalibrationSessionStore.SaveLabelsAsync(labels, sessionDirectory);
        await CalibrationSessionStore.SavePairsAsync(pairs, sessionDirectory);
        await CalibrationSessionStore.SaveMetricsAsync(metrics, sessionDirectory);
    }

    private static string CreateTempDirectory() =>
        Path.Combine(Path.GetTempPath(), $"dream-air-audit-{Guid.NewGuid():N}");
}
