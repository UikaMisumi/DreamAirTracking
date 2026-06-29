using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace DreamAirTracking.Core.Calibration;

public sealed class CalibrationCapturePackageExporter
{
    private static readonly JsonSerializerOptions JsonOptions = CoreJsonOptions.CamelCase();

    private static readonly string[] TrainingManifestFields =
    {
        "sample_id",
        "subject_id",
        "wear_id",
        "session_id",
        "session",
        "sequence_id",
        "frame_index",
        "timestamp",
        "split",
        "left_file",
        "right_file",
        "target_x",
        "target_y",
        "target_source",
        "gaze_stage",
        "stage",
        "openness_target_left",
        "openness_target_right",
        "openness_source",
        "weak_left_openness",
        "weak_left_pupil_x",
        "weak_left_pupil_y",
        "weak_left_pupil_radius",
        "weak_left_quality",
        "weak_right_openness",
        "weak_right_pupil_x",
        "weak_right_pupil_y",
        "weak_right_pupil_radius",
        "weak_right_quality",
        "weak_pair_quality",
        "gaze_weight",
        "openness_valid_left",
        "openness_valid_right",
        "wide_valid_left",
        "wide_valid_right",
        "squint_valid_left",
        "squint_valid_right",
        "pupil_valid_left",
        "pupil_valid_right",
        "confidence_valid_left",
        "confidence_valid_right",
        "confidence_valid_pair",
        "sample_weight",
        "left_conf",
        "right_conf",
        "left_center_x",
        "left_center_y",
        "right_center_x",
        "right_center_y",
        "left_found",
        "right_found",
        "source",
        "notes"
    };

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
        if (string.IsNullOrWhiteSpace(session.SessionId))
        {
            session.SessionId = Path.GetFileName(sessionDirectory);
        }

        var labels = await CalibrationSessionStore.LoadLabelsAsync(sessionDirectory, cancellationToken);
        var pairs = await CalibrationSessionStore.LoadPairsAsync(sessionDirectory, cancellationToken);
        var stageIds = BuildStageIds(session, labels, pairs);
        var protocol = BuildProtocol(options, stageIds);
        var identity = CapturePackageIdentityProvider.Create(session.SessionId);
        var subjectId = string.IsNullOrWhiteSpace(options.SubjectId) ? identity.SubjectId : options.SubjectId.Trim();
        var wearId = string.IsNullOrWhiteSpace(options.WearId) ? identity.WearId : options.WearId.Trim();
        var outputPath = BuildOutputPath(options, sessionDirectory, session.SessionId);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        if (File.Exists(outputPath))
        {
            File.Delete(outputPath);
        }

        await using var stream = File.Create(outputPath);
        using var archive = new ZipArchive(stream, ZipArchiveMode.Create);

        await WriteJsonEntryAsync(archive, "manifest.json", BuildManifest(options, session, stageIds, subjectId, wearId, pairs.Count), cancellationToken);
        await WriteJsonEntryAsync(archive, "device.json", BuildDevice(options, identity, subjectId, wearId), cancellationToken);
        await WriteJsonEntryAsync(archive, "capture_protocol.json", protocol, cancellationToken);
        await WriteJsonEntryAsync(archive, "session.json", SanitizeSession(session), cancellationToken);
        await CopyTextFileIfExistsAsync(archive, sessionDirectory, "labels.jsonl", cancellationToken);
        await WriteTextEntryAsync(archive, "pairs.csv", BuildPairsCsv(pairs), cancellationToken);
        await CopyTextFileIfExistsAsync(archive, sessionDirectory, "metrics.json", cancellationToken);
        await WriteTextEntryAsync(
            archive,
            "training_manifest_v3.csv",
            BuildTrainingManifestCsv(session, labels, pairs, subjectId, wearId),
            cancellationToken);
        await WriteReportAsync(archive, session, stageIds, pairs.Count, subjectId, wearId, cancellationToken);
        CopyFrames(archive, sessionDirectory, pairs, cancellationToken);
        await WriteRuntimeSummaryAsync(archive, sessionDirectory, cancellationToken);

        return new CalibrationCapturePackageExportResult(outputPath, stageIds, pairs.Count, subjectId, wearId);
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
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "DreamAirTracking", "capture_packages")
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
        IReadOnlyList<string> stageIds,
        string subjectId,
        string wearId,
        int pairCount) =>
        new
        {
            schema = "dream_air_tracking.capture_package.v1",
            createdAt = DateTimeOffset.Now,
            deviceFamily = options.DeviceFamily,
            subjectId,
            wearId,
            captureProtocol = BuildProtocolId(options.CaptureProtocol, stageIds),
            appVersion = options.AppVersion,
            brokenEyeVersion = options.BrokenEyeVersion,
            vrcftVersion = options.VrcftVersion,
            runtimeModelId = options.RuntimeModelId,
            sessionId = session.SessionId,
            stageIds,
            pairCount,
            privacyNote = "Local export only. Device identity is hashed; raw computer name, user name, and IP are not exported."
        };

    private static object BuildDevice(
        CalibrationCapturePackageExportOptions options,
        CapturePackageIdentity identity,
        string subjectId,
        string wearId) =>
        new
        {
            schema = "dream_air_tracking.device.v1",
            deviceFamily = options.DeviceFamily,
            subjectId,
            wearId,
            brokenEyeVersion = options.BrokenEyeVersion,
            vrcftVersion = options.VrcftVersion,
            runtimeModelId = options.RuntimeModelId,
            notes = options.Notes,
            sourceFields = identity.Fingerprint.SourceFields,
            fingerprint = identity.Fingerprint
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
            return stageIds.Any(IsEyelidStage) ? "nine_point_gaze_plus_eyelid" : "nine_point_gaze";
        }

        var hasFivePoint = new[] { "center", "up", "down", "left", "right" }.All(stageSet.Contains);
        if (hasFivePoint)
        {
            return stageIds.Any(IsEyelidStage) ? "five_point_gaze_plus_eyelid" : "five_point_gaze";
        }

        return stageIds.Any(IsEyelidStage) ? "eyelid" : "custom";
    }

    private static CalibrationSession SanitizeSession(CalibrationSession session) =>
        new()
        {
            SchemaVersion = session.SchemaVersion,
            SessionId = session.SessionId,
            CreatedAt = session.CreatedAt,
            Device = new CalibrationDeviceInfo
            {
                Source = SanitizeDeviceSource(session.Device.Source),
                Host = string.Empty,
                Port = 0
            },
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

    private static string BuildPairsCsv(IReadOnlyList<CalibrationFramePair> pairs)
    {
        var builder = new StringBuilder();
        builder.AppendLine("sequence,stage,left_file,right_file,delta_ms,left_found,left_raw_x,left_raw_y,left_conf,left_open,right_found,right_raw_x,right_raw_y,right_conf,right_open");
        foreach (var pair in pairs)
        {
            builder.AppendLine(string.Join(",", new[]
            {
                pair.Sequence.ToString(CultureInfo.InvariantCulture),
                CsvEscape(pair.StageId),
                CsvEscape(NormalizePackagePath(pair.LeftFile)),
                CsvEscape(NormalizePackagePath(pair.RightFile)),
                F(pair.DeltaMs),
                pair.LeftFound.ToString(),
                F(pair.LeftRawX),
                F(pair.LeftRawY),
                F(pair.LeftConfidence),
                F(pair.LeftOpenness),
                pair.RightFound.ToString(),
                F(pair.RightRawX),
                F(pair.RightRawY),
                F(pair.RightConfidence),
                F(pair.RightOpenness)
            }));
        }

        return builder.ToString();
    }

    private static string BuildTrainingManifestCsv(
        CalibrationSession session,
        IReadOnlyList<CalibrationLabel> labels,
        IReadOnlyList<CalibrationFramePair> pairs,
        string subjectId,
        string wearId)
    {
        var stageTargets = session.Stages
            .Where(stage => !string.IsNullOrWhiteSpace(stage.StageId))
            .GroupBy(stage => stage.StageId, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.Last().Target, StringComparer.OrdinalIgnoreCase);
        var labelsByStage = labels
            .Where(label => !string.IsNullOrWhiteSpace(label.StageId))
            .GroupBy(label => label.StageId, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.ToArray(), StringComparer.OrdinalIgnoreCase);

        var builder = new StringBuilder();
        builder.AppendLine(string.Join(",", TrainingManifestFields));
        foreach (var pair in pairs)
        {
            var label = FindLabel(pair, labelsByStage);
            var accepted = IsPairAccepted(pair, label);
            var isGaze = IsGazeStage(pair.StageId);
            var isEyelid = IsEyelidStage(pair.StageId);
            var target = label?.EffectiveTarget
                         ?? (stageTargets.TryGetValue(pair.StageId, out var stageTarget)
                             ? stageTarget
                             : CalibrationTarget.Center);
            var opennessTargets = OpennessTargetsForStage(pair.StageId);
            var gazeWeight = accepted && isGaze ? 1 : 0;
            var opennessValid = accepted && isEyelid ? 1 : 0;
            var pairQuality = (pair.LeftConfidence + pair.RightConfidence) / 2.0;
            var targetSource = isGaze ? "calibration_label" : "neutral_center";
            var source = !accepted
                ? "audit_only"
                : isGaze
                    ? "app_gaze_calibration"
                    : isEyelid
                        ? "app_eyelid_calibration"
                        : "app_capture";

            var row = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["sample_id"] = $"{session.SessionId}_{SafeToken(pair.StageId)}_{pair.Sequence}",
                ["subject_id"] = subjectId,
                ["wear_id"] = wearId,
                ["session_id"] = session.SessionId,
                ["session"] = session.SessionId,
                ["sequence_id"] = pair.Sequence.ToString(CultureInfo.InvariantCulture),
                ["frame_index"] = pair.Sequence.ToString(CultureInfo.InvariantCulture),
                ["timestamp"] = string.Empty,
                ["split"] = "train",
                ["left_file"] = NormalizePackagePath(pair.LeftFile),
                ["right_file"] = NormalizePackagePath(pair.RightFile),
                ["target_x"] = isGaze ? F(target.X) : "0",
                ["target_y"] = isGaze ? F(target.Y) : "0",
                ["target_source"] = targetSource,
                ["gaze_stage"] = isGaze ? pair.StageId : string.Empty,
                ["stage"] = pair.StageId,
                ["openness_target_left"] = isEyelid ? F(opennessTargets.Left) : string.Empty,
                ["openness_target_right"] = isEyelid ? F(opennessTargets.Right) : string.Empty,
                ["openness_source"] = isEyelid ? "protocol_label" : "not_supervised",
                ["weak_left_openness"] = F(pair.LeftOpenness),
                ["weak_left_pupil_x"] = F(pair.LeftRawX),
                ["weak_left_pupil_y"] = F(pair.LeftRawY),
                ["weak_left_pupil_radius"] = string.Empty,
                ["weak_left_quality"] = F(pair.LeftConfidence),
                ["weak_right_openness"] = F(pair.RightOpenness),
                ["weak_right_pupil_x"] = F(pair.RightRawX),
                ["weak_right_pupil_y"] = F(pair.RightRawY),
                ["weak_right_pupil_radius"] = string.Empty,
                ["weak_right_quality"] = F(pair.RightConfidence),
                ["weak_pair_quality"] = F(pairQuality),
                ["gaze_weight"] = gazeWeight.ToString(CultureInfo.InvariantCulture),
                ["openness_valid_left"] = opennessValid.ToString(CultureInfo.InvariantCulture),
                ["openness_valid_right"] = opennessValid.ToString(CultureInfo.InvariantCulture),
                ["wide_valid_left"] = "0",
                ["wide_valid_right"] = "0",
                ["squint_valid_left"] = "0",
                ["squint_valid_right"] = "0",
                ["pupil_valid_left"] = "0",
                ["pupil_valid_right"] = "0",
                ["confidence_valid_left"] = accepted && pair.LeftFound ? "1" : "0",
                ["confidence_valid_right"] = accepted && pair.RightFound ? "1" : "0",
                ["confidence_valid_pair"] = accepted && pair.LeftFound && pair.RightFound ? "1" : "0",
                ["sample_weight"] = gazeWeight.ToString(CultureInfo.InvariantCulture),
                ["left_conf"] = F(pair.LeftConfidence),
                ["right_conf"] = F(pair.RightConfidence),
                ["left_center_x"] = F(pair.LeftRawX),
                ["left_center_y"] = F(pair.LeftRawY),
                ["right_center_x"] = F(pair.RightRawX),
                ["right_center_y"] = F(pair.RightRawY),
                ["left_found"] = pair.LeftFound ? "1" : "0",
                ["right_found"] = pair.RightFound ? "1" : "0",
                ["source"] = source,
                ["notes"] = accepted ? string.Empty : "rejected_or_bad_frame"
            };

            builder.AppendLine(string.Join(",", TrainingManifestFields.Select(field => CsvEscape(row[field]))));
        }

        return builder.ToString();
    }

    private static CalibrationLabel? FindLabel(
        CalibrationFramePair pair,
        IReadOnlyDictionary<string, CalibrationLabel[]> labelsByStage)
    {
        if (!labelsByStage.TryGetValue(pair.StageId, out var stageLabels) || stageLabels.Length == 0)
        {
            return null;
        }

        return stageLabels.LastOrDefault(label => pair.Sequence >= label.FrameStart && pair.Sequence <= label.FrameEnd)
               ?? stageLabels.Last();
    }

    private static bool IsPairAccepted(CalibrationFramePair pair, CalibrationLabel? label)
    {
        if (label is null)
        {
            return true;
        }

        if (!label.Accepted || label.BadFrames.Contains(pair.Sequence))
        {
            return false;
        }

        return !label.OperatorVerdict.Equals("bad", StringComparison.OrdinalIgnoreCase) &&
               !label.OperatorVerdict.Equals("rejected", StringComparison.OrdinalIgnoreCase);
    }

    private static (double Left, double Right) OpennessTargetsForStage(string stageId)
    {
        var stage = stageId.Trim().ToLowerInvariant();
        var left = 1.0;
        var right = 1.0;

        if (stage.Contains("closed", StringComparison.Ordinal) || stage.Contains("blink", StringComparison.Ordinal))
        {
            left = 0.0;
            right = 0.0;
        }
        else if (stage.Contains("half", StringComparison.Ordinal))
        {
            left = 0.5;
            right = 0.5;
        }
        else if (stage.Contains("squint", StringComparison.Ordinal))
        {
            left = 0.35;
            right = 0.35;
        }

        if (stage.Contains("left_open", StringComparison.Ordinal))
        {
            left = 1.0;
        }

        if (stage.Contains("right_open", StringComparison.Ordinal))
        {
            right = 1.0;
        }

        if (stage.Contains("left_half", StringComparison.Ordinal))
        {
            left = 0.5;
        }

        if (stage.Contains("right_half", StringComparison.Ordinal))
        {
            right = 0.5;
        }

        if (stage.Contains("left_squint", StringComparison.Ordinal) || stage.Contains("left_wide", StringComparison.Ordinal))
        {
            left = stage.Contains("left_wide", StringComparison.Ordinal) ? 1.0 : 0.35;
        }

        if (stage.Contains("right_squint", StringComparison.Ordinal) || stage.Contains("right_wide", StringComparison.Ordinal))
        {
            right = stage.Contains("right_wide", StringComparison.Ordinal) ? 1.0 : 0.35;
        }

        if (stage.Contains("left_closed", StringComparison.Ordinal))
        {
            left = 0.0;
        }

        if (stage.Contains("right_closed", StringComparison.Ordinal))
        {
            right = 0.0;
        }

        return (left, right);
    }

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
        IReadOnlyList<CalibrationFramePair> pairs,
        CancellationToken cancellationToken)
    {
        var added = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var pair in pairs)
        {
            CopyFrameIfExists(archive, sessionDirectory, pair.LeftFile, added, cancellationToken);
            CopyFrameIfExists(archive, sessionDirectory, pair.RightFile, added, cancellationToken);
        }

        var framesDirectory = Path.Combine(sessionDirectory, "frames");
        if (!Directory.Exists(framesDirectory))
        {
            return;
        }

        foreach (var file in Directory.EnumerateFiles(framesDirectory, "*", SearchOption.AllDirectories))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var relative = Path.GetRelativePath(framesDirectory, file).Replace('\\', '/');
            var entryName = $"frames/{relative}";
            if (added.Add(entryName))
            {
                archive.CreateEntryFromFile(file, entryName, CompressionLevel.Fastest);
            }
        }
    }

    private static void CopyFrameIfExists(
        ZipArchive archive,
        string sessionDirectory,
        string framePath,
        HashSet<string> added,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var sourcePath = ResolveFramePath(sessionDirectory, framePath);
        if (sourcePath is null || !File.Exists(sourcePath))
        {
            return;
        }

        var entryName = NormalizePackagePath(framePath);
        if (added.Add(entryName))
        {
            archive.CreateEntryFromFile(sourcePath, entryName, CompressionLevel.Fastest);
        }
    }

    private static string? ResolveFramePath(string sessionDirectory, string framePath)
    {
        if (string.IsNullOrWhiteSpace(framePath))
        {
            return null;
        }

        if (Path.IsPathRooted(framePath))
        {
            return framePath;
        }

        var direct = Path.GetFullPath(Path.Combine(sessionDirectory, framePath));
        if (File.Exists(direct))
        {
            return direct;
        }

        var framesPath = Path.GetFullPath(Path.Combine(sessionDirectory, "frames", Path.GetFileName(framePath)));
        return File.Exists(framesPath) ? framesPath : direct;
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
        string subjectId,
        string wearId,
        CancellationToken cancellationToken)
    {
        var builder = new StringBuilder();
        builder.AppendLine("# DreamAirTracking Capture Package");
        builder.AppendLine();
        builder.AppendLine($"- Session: {session.SessionId}");
        builder.AppendLine($"- Subject: {subjectId}");
        builder.AppendLine($"- Wear: {wearId}");
        builder.AppendLine($"- Created: {session.CreatedAt:O}");
        builder.AppendLine($"- Pair count: {pairCount}");
        builder.AppendLine($"- Stages: {string.Join(", ", stageIds)}");
        builder.AppendLine("- Training entry: training_manifest_v3.csv");
        builder.AppendLine("- Paths inside the manifest are package-relative.");
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

    private static string NormalizePackagePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        if (Path.IsPathRooted(path))
        {
            return $"frames/{Path.GetFileName(path)}";
        }

        var packagePath = path.Replace('\\', '/').TrimStart('/');
        if (string.IsNullOrWhiteSpace(packagePath))
        {
            return string.Empty;
        }

        if (packagePath.StartsWith("frames/", StringComparison.OrdinalIgnoreCase))
        {
            return packagePath;
        }

        return $"frames/{Path.GetFileName(packagePath)}";
    }

    private static string SafeFileName(string value)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var chars = value.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray();
        var safe = new string(chars).Trim();
        return string.IsNullOrWhiteSpace(safe) ? DateTimeOffset.Now.ToString("yyyyMMdd_HHmmss") : safe;
    }

    private static string SafeToken(string value)
    {
        var builder = new StringBuilder(value.Length);
        foreach (var ch in value)
        {
            builder.Append(char.IsLetterOrDigit(ch) || ch is '_' or '-' ? ch : '_');
        }

        var token = builder.ToString().Trim('_');
        return string.IsNullOrWhiteSpace(token) ? "sample" : token;
    }

    private static string SanitizePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        return Path.IsPathRooted(path) ? Path.GetFileName(path) : path.Replace('\\', '/');
    }

    private static string SanitizeDeviceSource(string source)
    {
        if (string.IsNullOrWhiteSpace(source))
        {
            return string.Empty;
        }

        return source.Contains("BrokenEye", StringComparison.OrdinalIgnoreCase)
            ? "BrokenEye HTTP"
            : "local_eye_stream";
    }

    private static string CsvEscape(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
    }

    private static string F(double value) =>
        value.ToString("0.######", CultureInfo.InvariantCulture);

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

    private static bool IsEyelidStage(string stageId)
    {
        var stage = stageId.Trim().ToLowerInvariant();
        return stage is "open" or "closed" or "blink" or "half" ||
               stage.Contains("open", StringComparison.Ordinal) ||
               stage.Contains("closed", StringComparison.Ordinal) ||
               stage.Contains("blink", StringComparison.Ordinal) ||
               stage.Contains("half", StringComparison.Ordinal) ||
               stage.Contains("squint", StringComparison.Ordinal) ||
               stage.Contains("wide", StringComparison.Ordinal);
    }
}

public sealed class CalibrationCapturePackageExportOptions
{
    public string SessionDirectory { get; set; } = string.Empty;
    public string? OutputDirectory { get; set; }
    public string? OutputZipPath { get; set; }
    public string DeviceFamily { get; set; } = "Dream Air";
    public string? SubjectId { get; set; }
    public string? WearId { get; set; }
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
    int PairCount,
    string SubjectId,
    string WearId);
