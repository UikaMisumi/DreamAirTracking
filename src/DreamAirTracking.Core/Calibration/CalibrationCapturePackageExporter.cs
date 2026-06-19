using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace DreamAirTracking.Core.Calibration;

public sealed class CalibrationCapturePackageExporter
{
    private static readonly JsonSerializerOptions JsonOptions = CoreJsonOptions.CamelCase();

    public async Task<CalibrationCapturePackageExportResult> ExportAsync(
        CalibrationCapturePackageExportOptions options,
        CancellationToken cancellationToken = default)
    {
        var sessionDirectory = Path.GetFullPath(options.SessionDirectory);
        if (!Directory.Exists(sessionDirectory))
        {
            throw new DirectoryNotFoundException($"Calibration session directory was not found: {sessionDirectory}");
        }

        var session = await CalibrationSessionStore.LoadSessionAsync(sessionDirectory, cancellationToken);
        var labels = await CalibrationSessionStore.LoadLabelsAsync(sessionDirectory, cancellationToken);
        var pairs = await CalibrationSessionStore.LoadPairsAsync(sessionDirectory, cancellationToken);
        var stageIds = BuildStageIds(session, labels, pairs);
        var protocol = BuildProtocol(options, stageIds);
        var outputPath = BuildOutputPath(options, sessionDirectory, session.SessionId);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        if (File.Exists(outputPath))
        {
            File.Delete(outputPath);
        }

        await using var stream = File.Create(outputPath);
        using var archive = new ZipArchive(stream, ZipArchiveMode.Create);

        await WriteJsonEntryAsync(archive, "manifest.json", BuildManifest(options, session, stageIds), cancellationToken);
        await WriteJsonEntryAsync(archive, "device.json", BuildDevice(options, session), cancellationToken);
        await WriteJsonEntryAsync(archive, "capture_protocol.json", protocol, cancellationToken);
        await WriteJsonEntryAsync(archive, "session.json", SanitizeSession(session), cancellationToken);
        await CopyTextFileIfExistsAsync(archive, sessionDirectory, "labels.jsonl", cancellationToken);
        await CopyTextFileIfExistsAsync(archive, sessionDirectory, "pairs.csv", cancellationToken);
        await CopyTextFileIfExistsAsync(archive, sessionDirectory, "metrics.json", cancellationToken);
        await WriteReportAsync(archive, session, stageIds, pairs.Count, cancellationToken);
        CopyFrames(archive, sessionDirectory, cancellationToken);
        await WriteRuntimeSummaryAsync(archive, sessionDirectory, cancellationToken);

        return new CalibrationCapturePackageExportResult(outputPath, stageIds, pairs.Count);
    }

    private static string BuildOutputPath(
        CalibrationCapturePackageExportOptions options,
        string sessionDirectory,
        string sessionId)
    {
        if (!string.IsNullOrWhiteSpace(options.OutputZipPath))
        {
            return Path.GetFullPath(options.OutputZipPath);
        }

        var outputDirectory = string.IsNullOrWhiteSpace(options.OutputDirectory)
            ? Path.GetDirectoryName(sessionDirectory) ?? sessionDirectory
            : options.OutputDirectory;
        var safeSession = SafeFileName(string.IsNullOrWhiteSpace(sessionId) ? Path.GetFileName(sessionDirectory) : sessionId);
        return Path.Combine(Path.GetFullPath(outputDirectory), $"DreamAirTrackingCapture_{safeSession}.zip");
    }

    private static IReadOnlyList<string> BuildStageIds(
        CalibrationSession session,
        IReadOnlyList<CalibrationLabel> labels,
        IReadOnlyList<CalibrationFramePair> pairs)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var ordered = new List<string>();
        foreach (var stage in session.Stages.Select(stage => stage.StageId)
                     .Concat(labels.Select(label => label.StageId))
                     .Concat(pairs.Select(pair => pair.StageId)))
        {
            if (!string.IsNullOrWhiteSpace(stage) && seen.Add(stage))
            {
                ordered.Add(stage);
            }
        }

        return ordered;
    }

    private static object BuildManifest(
        CalibrationCapturePackageExportOptions options,
        CalibrationSession session,
        IReadOnlyList<string> stageIds) =>
        new
        {
            schema = "dream_air_tracking.capture_package.v1",
            createdAt = DateTimeOffset.Now,
            deviceFamily = options.DeviceFamily,
            subjectId = string.IsNullOrWhiteSpace(options.SubjectId) ? "anonymous-local-id" : options.SubjectId,
            wearId = string.IsNullOrWhiteSpace(options.WearId) ? "wear_unknown" : options.WearId,
            captureProtocol = BuildProtocolId(options.CaptureProtocol, stageIds),
            appVersion = options.AppVersion,
            brokenEyeVersion = options.BrokenEyeVersion,
            vrcftVersion = options.VrcftVersion,
            runtimeModelId = options.RuntimeModelId,
            sessionId = session.SessionId,
            stageIds,
            privacyNote = "Local export only. The app does not upload this package."
        };

    private static object BuildDevice(CalibrationCapturePackageExportOptions options, CalibrationSession session) =>
        new
        {
            schema = "dream_air_tracking.device.v1",
            deviceFamily = options.DeviceFamily,
            subjectId = string.IsNullOrWhiteSpace(options.SubjectId) ? "anonymous-local-id" : options.SubjectId,
            wearId = string.IsNullOrWhiteSpace(options.WearId) ? "wear_unknown" : options.WearId,
            source = session.Device.Source,
            host = session.Device.Host,
            port = session.Device.Port,
            brokenEyeVersion = options.BrokenEyeVersion,
            vrcftVersion = options.VrcftVersion,
            runtimeModelId = options.RuntimeModelId,
            notes = options.Notes
        };

    private static object BuildProtocol(
        CalibrationCapturePackageExportOptions options,
        IReadOnlyList<string> stageIds)
    {
        var heads = new List<string>();
        if (stageIds.Any(IsGazeStage))
        {
            heads.Add("gaze_xy");
        }

        if (stageIds.Any(IsEyelidStage))
        {
            heads.Add("openness_lr");
        }

        return new
        {
            schema = "dream_air_tracking.capture_protocol.v1",
            protocolId = BuildProtocolId(options.CaptureProtocol, stageIds),
            stages = stageIds,
            intendedTrainingHeads = heads,
            headMasksRequired = true
        };
    }

    private static string BuildProtocolId(string? requested, IReadOnlyList<string> stageIds)
    {
        if (!string.IsNullOrWhiteSpace(requested) && !requested.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return requested;
        }

        var stageSet = stageIds.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var hasNinePoint = new[]
        {
            "center", "up", "down", "left", "right", "left_up", "right_up", "left_down", "right_down"
        }.All(stageSet.Contains);
        if (hasNinePoint)
        {
            return "nine_point_gaze";
        }

        var hasFivePoint = new[] { "center", "up", "down", "left", "right" }.All(stageSet.Contains);
        if (hasFivePoint)
        {
            return "five_point_gaze";
        }

        return stageIds.Any(IsEyelidStage) ? "eyelid" : "custom";
    }

    private static CalibrationSession SanitizeSession(CalibrationSession session) =>
        new()
        {
            SchemaVersion = session.SchemaVersion,
            SessionId = session.SessionId,
            CreatedAt = session.CreatedAt,
            Device = session.Device,
            Operator = new CalibrationOperatorInfo
            {
                DisplayName = string.IsNullOrWhiteSpace(session.Operator.DisplayName)
                    ? string.Empty
                    : "local-operator"
            },
            ProfileInput = SanitizePath(session.ProfileInput),
            Notes = session.Notes,
            Stages = session.Stages
        };

    private static async Task CopyTextFileIfExistsAsync(
        ZipArchive archive,
        string sessionDirectory,
        string fileName,
        CancellationToken cancellationToken)
    {
        var path = Path.Combine(sessionDirectory, fileName);
        if (!File.Exists(path))
        {
            return;
        }

        var text = await File.ReadAllTextAsync(path, cancellationToken);
        await WriteTextEntryAsync(archive, fileName, text, cancellationToken);
    }

    private static void CopyFrames(
        ZipArchive archive,
        string sessionDirectory,
        CancellationToken cancellationToken)
    {
        var framesDirectory = Path.Combine(sessionDirectory, "frames");
        if (!Directory.Exists(framesDirectory))
        {
            return;
        }

        foreach (var file in Directory.EnumerateFiles(framesDirectory, "*", SearchOption.AllDirectories))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var relative = Path.GetRelativePath(framesDirectory, file).Replace('\\', '/');
            archive.CreateEntryFromFile(file, $"frames/{relative}", CompressionLevel.Fastest);
        }
    }

    private static async Task WriteRuntimeSummaryAsync(
        ZipArchive archive,
        string sessionDirectory,
        CancellationToken cancellationToken)
    {
        var knownRuntimeFiles = new[]
        {
            "runtime_gaze_calibration.json",
            "generated_profile.json",
            "mask_normalized_profile.json"
        };
        var present = knownRuntimeFiles
            .Where(name => File.Exists(Path.Combine(sessionDirectory, name)))
            .ToArray();
        var payload = new
        {
            schema = "dream_air_tracking.capture_runtime_summary.v1",
            exportedRawRuntimeFiles = false,
            reason = "Raw runtime files can contain local absolute paths. Keep them local unless sanitized by a dedicated exporter.",
            localFilesDetected = present
        };
        await WriteJsonEntryAsync(archive, "runtime/runtime_summary.json", payload, cancellationToken);
    }

    private static async Task WriteReportAsync(
        ZipArchive archive,
        CalibrationSession session,
        IReadOnlyList<string> stageIds,
        int pairCount,
        CancellationToken cancellationToken)
    {
        var builder = new StringBuilder();
        builder.AppendLine("# DreamAirTracking Capture Package");
        builder.AppendLine();
        builder.AppendLine($"- Session: {session.SessionId}");
        builder.AppendLine($"- Created: {session.CreatedAt:O}");
        builder.AppendLine($"- Pair count: {pairCount}");
        builder.AppendLine($"- Stages: {string.Join(", ", stageIds)}");
        await WriteTextEntryAsync(archive, "calibration_report.md", builder.ToString(), cancellationToken);
    }

    private static async Task WriteJsonEntryAsync(
        ZipArchive archive,
        string entryName,
        object payload,
        CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(payload, JsonOptions);
        await WriteTextEntryAsync(archive, entryName, json + "\n", cancellationToken);
    }

    private static async Task WriteTextEntryAsync(
        ZipArchive archive,
        string entryName,
        string text,
        CancellationToken cancellationToken)
    {
        var entry = archive.CreateEntry(entryName, CompressionLevel.Fastest);
        await using var entryStream = entry.Open();
        await using var writer = new StreamWriter(entryStream, new UTF8Encoding(false));
        await writer.WriteAsync(text.AsMemory(), cancellationToken);
    }

    private static string SafeFileName(string value)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var chars = value.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray();
        var safe = new string(chars).Trim();
        return string.IsNullOrWhiteSpace(safe) ? DateTimeOffset.Now.ToString("yyyyMMdd_HHmmss") : safe;
    }

    private static string SanitizePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        return Path.IsPathRooted(path) ? Path.GetFileName(path) : path.Replace('\\', '/');
    }

    private static bool IsGazeStage(string stageId) =>
        stageId.Equals("center", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("up", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("down", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("left", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("right", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("left_up", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("right_up", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("left_down", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("right_down", StringComparison.OrdinalIgnoreCase);

    private static bool IsEyelidStage(string stageId) =>
        stageId.Equals("open", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("closed", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("blink", StringComparison.OrdinalIgnoreCase) ||
        stageId.Equals("half", StringComparison.OrdinalIgnoreCase);
}

public sealed class CalibrationCapturePackageExportOptions
{
    public string SessionDirectory { get; set; } = string.Empty;
    public string? OutputDirectory { get; set; }
    public string? OutputZipPath { get; set; }
    public string DeviceFamily { get; set; } = "Dream Air";
    public string SubjectId { get; set; } = "anonymous-local-id";
    public string WearId { get; set; } = "wear_unknown";
    public string CaptureProtocol { get; set; } = "auto";
    public string? AppVersion { get; set; }
    public string? BrokenEyeVersion { get; set; }
    public string? VrcftVersion { get; set; }
    public string? RuntimeModelId { get; set; }
    public string Notes { get; set; } = string.Empty;
}

public sealed record CalibrationCapturePackageExportResult(
    string OutputZipPath,
    IReadOnlyList<string> StageIds,
    int PairCount);
