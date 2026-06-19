using DreamAirTracking.Core;
using DreamAirTracking.Core.Calibration;
using DreamAirTracking.Core.Diagnostics;
using DreamAirTracking.Core.Geometry;
using DreamAirTracking.Core.Input;
using DreamAirTracking.Core.Profiles;
using DreamAirTracking.Core.Tracking;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;

var exitCode = await ProgramMain.RunAsync(args);
return exitCode;

internal static class ProgramMain
{
    private const double PupilDiameterQualityMinConfidence = 0.28;
    private const double PupilDiameterQualityMaxAxisRatio = 1.8;
    private const double PupilDiameterQualityMinOpenness = 0.75;

    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintHelp();
            return 0;
        }

        try
        {
            return args[0] switch
            {
                "analyze-samples" => await AnalyzeSamplesAsync(args.Skip(1).ToArray()),
                "pupil-diameter-samples" => await PupilDiameterSamplesAsync(args.Skip(1).ToArray()),
                "diagnose-session" => await DiagnoseSessionAsync(args.Skip(1).ToArray()),
                "init-profile" => await InitProfileAsync(args.Skip(1).ToArray()),
                "profile-from-samples" => await ProfileFromSamplesAsync(args.Skip(1).ToArray()),
                "calibration-from-capture" => await CalibrationFromCaptureAsync(args.Skip(1).ToArray()),
                "calibration-from-session" => await CalibrationFromSessionAsync(args.Skip(1).ToArray()),
                "fit-gaze-session" => await FitGazeSessionAsync(args.Skip(1).ToArray()),
                "set-roi" => await SetRoiAsync(args.Skip(1).ToArray()),
                "set-calibration" => await SetCalibrationAsync(args.Skip(1).ToArray()),
                "capture-http" => await CaptureHttpAsync(args.Skip(1).ToArray()),
                "live-http" => await LiveHttpAsync(args.Skip(1).ToArray()),
                "pupil-diameter-session-http" => await PupilDiameterSessionHttpAsync(args.Skip(1).ToArray()),
                "fit-pupil-diameter-session" => await FitPupilDiameterSessionAsync(args.Skip(1).ToArray()),
                "calibration-session-http" => await CalibrationSessionHttpAsync(args.Skip(1).ToArray()),
                "capture-raw" => await CaptureRawAsync(args.Skip(1).ToArray()),
                _ => UnknownCommand(args[0])
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }

    private static int UnknownCommand(string command)
    {
        Console.Error.WriteLine($"Unknown command: {command}");
        PrintHelp();
        return 1;
    }

    private static async Task<int> AnalyzeSamplesAsync(string[] args)
    {
        var sampleDir = GetOption(args, "--path") ?? (args.Length > 0 && !args[0].StartsWith("--") ? args[0] : null);
        if (sampleDir is null)
        {
            Console.Error.WriteLine("Missing sample directory.");
            return 1;
        }

        var debugDir = GetOption(args, "--debug");
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(GetOption(args, "--profile"));
        if (debugDir is not null)
        {
            Directory.CreateDirectory(debugDir);
        }

        var detector = new PupilDetector();
        var files = Directory.EnumerateFiles(sampleDir)
            .Where(IsImageFile)
            .OrderBy(static x => x, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        Console.WriteLine("file,side,found,confidence,center_x,center_y,threshold,bounds_x,bounds_y,bounds_w,bounds_h,area,fill");

        foreach (var file in files)
        {
            var side = GuessSide(file);
            var frame = await LoadImageFrameAsync(file, side);
            var detection = detector.Detect(frame, profile.CreateDetectorOptions(side));

            Console.WriteLine(string.Join(',',
                Path.GetFileName(file),
                side.ToString().ToLowerInvariant(),
                detection.Found,
                detection.Confidence.ToString("0.000"),
                detection.CenterX.ToString("0.00"),
                detection.CenterY.ToString("0.00"),
                detection.Threshold,
                detection.Bounds.X,
                detection.Bounds.Y,
                detection.Bounds.Width,
                detection.Bounds.Height,
                detection.Area,
                detection.FillRatio.ToString("0.000")));

            if (debugDir is not null)
            {
                var outPath = Path.Combine(debugDir, Path.GetFileNameWithoutExtension(file) + "_debug.png");
                await SaveDebugImageAsync(file, detection, outPath);
            }
        }

        return 0;
    }

    private static async Task<int> PupilDiameterSamplesAsync(string[] args)
    {
        var sampleDir = GetOption(args, "--path") ?? (args.Length > 0 && !args[0].StartsWith("--") ? args[0] : null);
        if (sampleDir is null)
        {
            Console.Error.WriteLine("Missing sample directory.");
            return 1;
        }

        var profile = await TrackingProfileStore.LoadOrDefaultAsync(GetOption(args, "--profile"));
        var detector = new PupilDetector();
        var diameterEstimator = new PupilDiameterEstimator();
        var opennessDetector = new EyeOpennessDetector();
        var outPath = GetOption(args, "--out");
        var debugDir = GetOption(args, "--debug");
        if (debugDir is not null)
        {
            Directory.CreateDirectory(debugDir);
        }

        var files = Directory.EnumerateFiles(sampleDir)
            .Where(IsImageFile)
            .OrderBy(static x => x, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var csv = new StringBuilder();
        csv.AppendLine("file,side,pupil_found,pupil_confidence,pupil_center_x,pupil_center_y,diameter_found,diameter_confidence,equivalent_diameter_px,major_axis_px,minor_axis_px,diameter_area_px,axis_ratio,diameter_threshold,openness,aperture_height,reason");

        foreach (var file in files)
        {
            var side = GuessSide(file);
            var frame = await LoadImageFrameAsync(file, side);
            var eye = profile.GetEye(side);
            var detection = detector.Detect(frame, profile.CreateDetectorOptions(side));
            var diameter = diameterEstimator.Estimate(frame, detection);
            var openness = opennessDetector.Detect(frame, new EyeOpennessDetectorOptions
            {
                Roi = eye.Roi
            });

            if (debugDir is not null)
            {
                var outDebugPath = Path.Combine(debugDir, Path.GetFileNameWithoutExtension(file) + "_pupil_diameter.png");
                await SavePupilDiameterOverlayAsync(file, eye.Roi, detection, diameter, openness, outDebugPath);
            }

            csv.AppendLine(JoinCsv(new[]
            {
                Path.GetFileName(file),
                side.ToString().ToLowerInvariant(),
                detection.Found.ToString(),
                F(detection.Confidence),
                F(detection.CenterX),
                F(detection.CenterY),
                diameter.Found.ToString(),
                F(diameter.Confidence),
                F(diameter.EquivalentDiameterPx),
                F(diameter.MajorAxisPx),
                F(diameter.MinorAxisPx),
                diameter.Area.ToString(CultureInfo.InvariantCulture),
                F(diameter.AxisRatio),
                diameter.Threshold.ToString(CultureInfo.InvariantCulture),
                F(openness.Openness),
                openness.ApertureHeight.ToString(CultureInfo.InvariantCulture),
                diameter.Reason ?? string.Empty
            }));
        }

        if (outPath is null)
        {
            Console.Write(csv.ToString());
        }
        else
        {
            await File.WriteAllTextAsync(outPath, csv.ToString(), Encoding.UTF8);
            Console.WriteLine($"Wrote pupil diameter CSV: {Path.GetFullPath(outPath)}");
        }

        return 0;
    }

    private static async Task<int> DiagnoseSessionAsync(string[] args)
    {
        var sessionDir = GetOption(args, "--session") ?? (args.Length > 0 && !args[0].StartsWith("--") ? args[0] : null);
        if (sessionDir is null)
        {
            Console.Error.WriteLine("Missing session directory.");
            return 1;
        }

        var profilePath = GetOption(args, "--profile") ?? "tracking_profile_final.json";
        var outDir = GetOption(args, "--out") ?? Path.Combine(sessionDir, "diagnostics");
        var overlayDir = Path.Combine(outDir, "overlays");
        var limitPerStage = int.Parse(GetOption(args, "--limit-per-stage") ?? "8", CultureInfo.InvariantCulture);
        var brightThreshold = (byte)Math.Clamp(int.Parse(GetOption(args, "--bright-threshold") ?? "235", CultureInfo.InvariantCulture), 0, 255);
        var maxBrightCandidates = int.Parse(GetOption(args, "--max-bright-candidates") ?? "8", CultureInfo.InvariantCulture);

        var profile = await TrackingProfileStore.LoadOrDefaultAsync(profilePath);
        var pupilDetector = new PupilDetector();
        var brightDetector = new BrightSpotDetector();
        var opennessDetector = new EyeOpennessDetector();
        Directory.CreateDirectory(overlayDir);

        var files = Directory.EnumerateFiles(sessionDir)
            .Where(IsImageFile)
            .OrderBy(static x => x, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (limitPerStage > 0)
        {
            files = files
                .GroupBy(static file => (Stage: ExtractStageName(file), Side: GuessSide(file)))
                .SelectMany(group => group.Take(limitPerStage))
                .OrderBy(static x => x, StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

        var csv = new StringBuilder();
        csv.AppendLine("file,stage,side,found,confidence,center_x,center_y,bounds_x,bounds_y,bounds_w,bounds_h,area,fill,openness,aperture_upper_y,aperture_lower_y,aperture_height,peak_dark_fraction,mean_dark_fraction,dark_threshold,openness_reason,bright_rank,bright_area,bright_center_x,bright_center_y,bright_peak,bright_mean,bright_bounds_x,bright_bounds_y,bright_bounds_w,bright_bounds_h");

        foreach (var file in files)
        {
            var stage = ExtractStageName(file);
            var side = GuessSide(file);
            var frame = await LoadImageFrameAsync(file, side);
            var eye = profile.GetEye(side);
            var detection = pupilDetector.Detect(frame, profile.CreateDetectorOptions(side));
            var openness = opennessDetector.Detect(frame, new EyeOpennessDetectorOptions
            {
                Roi = eye.Roi
            });
            var brightCandidates = brightDetector.Detect(frame, new BrightSpotDetectorOptions
            {
                Roi = eye.Roi,
                Threshold = brightThreshold,
                MaxCandidates = maxBrightCandidates
            });

            var overlayPath = Path.Combine(overlayDir, Path.GetFileNameWithoutExtension(file) + "_diag.png");
            await SaveDiagnosticOverlayAsync(file, eye.Roi, detection, openness, brightCandidates, overlayPath);
            AppendDiagnosticRows(csv, file, stage, side, detection, openness, brightCandidates);
        }

        var csvPath = Path.Combine(outDir, "diagnostics.csv");
        await File.WriteAllTextAsync(csvPath, csv.ToString(), Encoding.UTF8);

        Console.WriteLine($"Wrote overlays: {Path.GetFullPath(overlayDir)}");
        Console.WriteLine($"Wrote CSV: {Path.GetFullPath(csvPath)}");
        Console.WriteLine($"Frames analyzed: {files.Length}");
        return 0;
    }

    private static async Task<int> CalibrationFromSessionAsync(string[] args)
    {
        var sessionDir = GetOption(args, "--session") ?? (args.Length > 0 && !args[0].StartsWith("--") ? args[0] : null);
        var profilePath = RequireOption(args, "--profile");
        var minConfidence = double.Parse(GetOption(args, "--min-conf") ?? "0.05", CultureInfo.InvariantCulture);
        var minRange = double.Parse(GetOption(args, "--min-range") ?? "8", CultureInfo.InvariantCulture);
        if (sessionDir is null)
        {
            Console.Error.WriteLine("Missing calibration session directory.");
            return 1;
        }

        var csvPath = Path.Combine(sessionDir, "calibration_pairs.csv");
        var rows = LoadStageRows(csvPath);
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(profilePath);
        var stageStats = BuildStageStats(rows, minConfidence);

        UpdateEyeFromStageStats(profile.Left.Calibration, stageStats, "left", minRange);
        UpdateEyeFromStageStats(profile.Right.Calibration, stageStats, "right", minRange);

        await TrackingProfileStore.SaveAsync(profile, profilePath);
        var reportPath = GetOption(args, "--report") ?? Path.Combine(sessionDir, "stage_calibration_report.csv");
        await WriteStageCalibrationReportAsync(reportPath, stageStats);

        Console.WriteLine($"Updated profile: {Path.GetFullPath(profilePath)}");
        Console.WriteLine($"Wrote report: {Path.GetFullPath(reportPath)}");
        return 0;
    }

    private static void UpdateEyeFromStageStats(EyeCalibration calibration, IReadOnlyDictionary<(string Stage, string Side), StageStat> stageStats, string side, double minRange)
    {
        if (!stageStats.TryGetValue(("center", side), out var center) || !center.HasPoint)
        {
            Console.WriteLine($"{side}: missing center stage; calibration unchanged.");
            return;
        }

        calibration.CenterX = center.MedianX!.Value;
        calibration.CenterY = center.MedianY!.Value;

        var horizontalDistances = new List<double>();
        var verticalDistances = new List<double>();
        if (stageStats.TryGetValue(("left", side), out var left) && left.HasPoint)
        {
            horizontalDistances.Add(Math.Abs(left.MedianX!.Value - calibration.CenterX));
        }

        if (stageStats.TryGetValue(("right", side), out var right) && right.HasPoint)
        {
            horizontalDistances.Add(Math.Abs(right.MedianX!.Value - calibration.CenterX));
        }

        if (stageStats.TryGetValue(("up", side), out var up) && up.HasPoint)
        {
            verticalDistances.Add(Math.Abs(up.MedianY!.Value - calibration.CenterY));
        }

        if (stageStats.TryGetValue(("down", side), out var down) && down.HasPoint)
        {
            verticalDistances.Add(Math.Abs(down.MedianY!.Value - calibration.CenterY));
        }

        if (horizontalDistances.Count > 0)
        {
            calibration.HorizontalRange = Math.Max(minRange, horizontalDistances.Max());
        }

        if (verticalDistances.Count > 0)
        {
            calibration.VerticalRange = Math.Max(minRange, verticalDistances.Max());
        }

        calibration.HorizontalOutputSign = InferOutputSign(
            center.MedianX!.Value,
            negativeStage: stageStats.TryGetValue(("left", side), out left) ? left : null,
            positiveStage: stageStats.TryGetValue(("right", side), out right) ? right : null,
            useX: true,
            fallback: calibration.HorizontalOutputSign);

        calibration.VerticalOutputSign = InferOutputSign(
            center.MedianY!.Value,
            negativeStage: stageStats.TryGetValue(("down", side), out var downSign) ? downSign : null,
            positiveStage: stageStats.TryGetValue(("up", side), out var upSign) ? upSign : null,
            useX: false,
            fallback: calibration.VerticalOutputSign);

        Console.WriteLine($"{side}: center=({calibration.CenterX:0.00},{calibration.CenterY:0.00}) range=({calibration.HorizontalRange:0.00},{calibration.VerticalRange:0.00}) sign=({calibration.HorizontalOutputSign:0},{calibration.VerticalOutputSign:0})");
    }

    private static double InferOutputSign(double center, StageStat? negativeStage, StageStat? positiveStage, bool useX, double fallback)
    {
        var candidates = new List<(double Sign, double Weight)>();
        AddSignCandidate(candidates, center, negativeStage, desiredOutputSign: -1, useX);
        AddSignCandidate(candidates, center, positiveStage, desiredOutputSign: 1, useX);
        if (candidates.Count == 0)
        {
            return fallback < 0 ? -1 : 1;
        }

        var score = candidates.Sum(static candidate => candidate.Sign * candidate.Weight);
        if (Math.Abs(score) < 0.001)
        {
            return fallback < 0 ? -1 : 1;
        }

        return score < 0 ? -1 : 1;
    }

    private static void AddSignCandidate(List<(double Sign, double Weight)> candidates, double center, StageStat? stage, int desiredOutputSign, bool useX)
    {
        if (stage is not { HasPoint: true })
        {
            return;
        }

        var median = useX ? stage.MedianX!.Value : stage.MedianY!.Value;
        var observedDelta = median - center;
        if (Math.Abs(observedDelta) < 1)
        {
            return;
        }

        var observedSign = observedDelta < 0 ? -1 : 1;
        var outputSign = desiredOutputSign * observedSign;
        var weight = Math.Abs(observedDelta) * Math.Max(stage.AverageConfidence, 0.001);
        candidates.Add((outputSign, weight));
    }

    private static IReadOnlyDictionary<(string Stage, string Side), StageStat> BuildStageStats(IReadOnlyList<StagePairRow> rows, double minConfidence)
    {
        var result = new Dictionary<(string Stage, string Side), StageStat>();
        foreach (var stage in rows.Select(static row => row.Stage).Distinct(StringComparer.OrdinalIgnoreCase))
        {
            foreach (var side in new[] { "left", "right" })
            {
                var stageRows = rows.Where(row => row.Stage.Equals(stage, StringComparison.OrdinalIgnoreCase)).ToArray();
                var valid = stageRows
                    .Where(row => row.GetFound(side) && row.GetConfidence(side) >= minConfidence && row.GetX(side) is not null && row.GetY(side) is not null)
                    .ToArray();

                var xs = valid.Select(row => row.GetX(side)!.Value).OrderBy(static value => value).ToArray();
                var ys = valid.Select(row => row.GetY(side)!.Value).OrderBy(static value => value).ToArray();
                result[(stage, side)] = new StageStat(
                    stage,
                    side,
                    stageRows.Length,
                    valid.Length,
                    xs.Length > 0 ? Percentile(xs, 0.5) : null,
                    ys.Length > 0 ? Percentile(ys, 0.5) : null,
                    valid.Length > 0 ? valid.Select(row => row.GetConfidence(side)).Average() : 0);
            }
        }

        return result;
    }

    private static async Task WriteStageCalibrationReportAsync(string reportPath, IReadOnlyDictionary<(string Stage, string Side), StageStat> stageStats)
    {
        await using var stream = File.Create(reportPath);
        await using var writer = new StreamWriter(stream);
        await writer.WriteLineAsync("stage,side,total_points,valid_points,median_x,median_y,avg_confidence");
        foreach (var stat in stageStats.Values.OrderBy(static stat => stat.Stage).ThenBy(static stat => stat.Side))
        {
            await writer.WriteLineAsync(string.Join(',',
                stat.Stage,
                stat.Side,
                stat.TotalPoints,
                stat.ValidPoints,
                stat.MedianX?.ToString("0.00", CultureInfo.InvariantCulture) ?? "",
                stat.MedianY?.ToString("0.00", CultureInfo.InvariantCulture) ?? "",
                stat.AverageConfidence.ToString("0.000", CultureInfo.InvariantCulture)));
        }
    }

    private static async Task<int> InitProfileAsync(string[] args)
    {
        var profilePath = RequireOption(args, "--profile");
        await TrackingProfileStore.SaveAsync(new TrackingProfile(), profilePath);
        Console.WriteLine($"Wrote profile: {Path.GetFullPath(profilePath)}");
        return 0;
    }

    private static async Task<int> ProfileFromSamplesAsync(string[] args)
    {
        var sampleDir = GetOption(args, "--path") ?? (args.Length > 0 && !args[0].StartsWith("--") ? args[0] : null);
        var profilePath = RequireOption(args, "--profile");
        if (sampleDir is null)
        {
            Console.Error.WriteLine("Missing sample directory.");
            return 1;
        }

        var margin = int.Parse(GetOption(args, "--margin") ?? "35", CultureInfo.InvariantCulture);
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(profilePath);
        var detector = new PupilDetector();
        var detections = new Dictionary<EyeSide, List<(EyeFrame Frame, PupilDetection Detection)>>
        {
            [EyeSide.Left] = new(),
            [EyeSide.Right] = new()
        };

        foreach (var file in Directory.EnumerateFiles(sampleDir).Where(IsImageFile).OrderBy(static x => x, StringComparer.OrdinalIgnoreCase))
        {
            var side = GuessSide(file);
            var frame = await LoadImageFrameAsync(file, side);
            var detection = detector.Detect(frame);
            if (detection.Found)
            {
                detections[side].Add((frame, detection));
            }
        }

        foreach (var side in new[] { EyeSide.Left, EyeSide.Right })
        {
            var sideDetections = detections[side];
            if (sideDetections.Count == 0)
            {
                Console.WriteLine($"{side}: no detections, profile unchanged");
                continue;
            }

            var eye = profile.GetEye(side);
            eye.Calibration.CenterX = Median(sideDetections.Select(static x => x.Detection.CenterX));
            eye.Calibration.CenterY = Median(sideDetections.Select(static x => x.Detection.CenterY));

            var frameWidth = sideDetections[0].Frame.Width;
            var frameHeight = sideDetections[0].Frame.Height;
            var minX = Math.Max(0, sideDetections.Min(static x => x.Detection.Bounds.X) - margin);
            var minY = Math.Max(0, sideDetections.Min(static x => x.Detection.Bounds.Y) - margin);
            var maxX = Math.Min(frameWidth - 1, sideDetections.Max(static x => x.Detection.Bounds.Right) + margin);
            var maxY = Math.Min(frameHeight - 1, sideDetections.Max(static x => x.Detection.Bounds.Bottom) + margin);
            eye.Roi = new PixelRect(minX, minY, maxX - minX + 1, maxY - minY + 1);

            Console.WriteLine($"{side}: center=({eye.Calibration.CenterX:0.00},{eye.Calibration.CenterY:0.00}) roi=({eye.Roi.Value.X},{eye.Roi.Value.Y},{eye.Roi.Value.Width},{eye.Roi.Value.Height})");
        }

        await TrackingProfileStore.SaveAsync(profile, profilePath);
        Console.WriteLine($"Wrote profile: {Path.GetFullPath(profilePath)}");
        return 0;
    }

    private static async Task<int> CalibrationFromCaptureAsync(string[] args)
    {
        var captureDir = GetOption(args, "--capture") ?? (args.Length > 0 && !args[0].StartsWith("--") ? args[0] : null);
        var profilePath = RequireOption(args, "--profile");
        var minConfidence = double.Parse(GetOption(args, "--min-conf") ?? "0.06", CultureInfo.InvariantCulture);
        var lowerPercentile = double.Parse(GetOption(args, "--low-p") ?? "0.10", CultureInfo.InvariantCulture);
        var upperPercentile = double.Parse(GetOption(args, "--high-p") ?? "0.90", CultureInfo.InvariantCulture);
        if (captureDir is null)
        {
            Console.Error.WriteLine("Missing capture directory.");
            return 1;
        }

        var rows = LoadPairRows(Path.Combine(captureDir, "pairs.csv"));
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(profilePath);
        UpdateEyeCalibrationFromRows(profile.Left.Calibration, rows, "left", minConfidence, lowerPercentile, upperPercentile);
        UpdateEyeCalibrationFromRows(profile.Right.Calibration, rows, "right", minConfidence, lowerPercentile, upperPercentile);
        await TrackingProfileStore.SaveAsync(profile, profilePath);

        var reportPath = GetOption(args, "--report") ?? Path.Combine(captureDir, "calibration_report.csv");
        await WriteCalibrationReportAsync(reportPath, rows, minConfidence);
        Console.WriteLine($"Updated profile: {Path.GetFullPath(profilePath)}");
        Console.WriteLine($"Wrote report: {Path.GetFullPath(reportPath)}");
        return 0;
    }

    private static void UpdateEyeCalibrationFromRows(EyeCalibration calibration, IReadOnlyList<PairRow> rows, string prefix, double minConfidence, double lowP, double highP)
    {
        var xs = rows
            .Where(row => row.GetFound(prefix) && row.GetConfidence(prefix) >= minConfidence)
            .Select(row => row.GetX(prefix))
            .Where(static value => value is not null)
            .Select(static value => value!.Value)
            .OrderBy(static value => value)
            .ToArray();
        var ys = rows
            .Where(row => row.GetFound(prefix) && row.GetConfidence(prefix) >= minConfidence)
            .Select(row => row.GetY(prefix))
            .Where(static value => value is not null)
            .Select(static value => value!.Value)
            .OrderBy(static value => value)
            .ToArray();

        if (xs.Length < 5 || ys.Length < 5)
        {
            Console.WriteLine($"{prefix}: not enough confident points; calibration unchanged.");
            return;
        }

        var lowX = Percentile(xs, lowP);
        var highX = Percentile(xs, highP);
        var lowY = Percentile(ys, lowP);
        var highY = Percentile(ys, highP);
        calibration.CenterX = Percentile(xs, 0.5);
        calibration.CenterY = Percentile(ys, 0.5);
        calibration.HorizontalRange = Math.Max(8, (highX - lowX) / 2);
        calibration.VerticalRange = Math.Max(8, (highY - lowY) / 2);

        Console.WriteLine($"{prefix}: center=({calibration.CenterX:0.00},{calibration.CenterY:0.00}) range=({calibration.HorizontalRange:0.00},{calibration.VerticalRange:0.00}) points={xs.Length}");
    }

    private static async Task WriteCalibrationReportAsync(string reportPath, IReadOnlyList<PairRow> rows, double minConfidence)
    {
        await using var stream = File.Create(reportPath);
        await using var writer = new StreamWriter(stream);
        await writer.WriteLineAsync("side,valid_points,min_conf,min_x,p10_x,p50_x,p90_x,max_x,min_y,p10_y,p50_y,p90_y,max_y");
        foreach (var side in new[] { "left", "right" })
        {
            var valid = rows
                .Where(row => row.GetFound(side) && row.GetConfidence(side) >= minConfidence)
                .ToArray();
            var xs = valid.Select(row => row.GetX(side)).Where(static value => value is not null).Select(static value => value!.Value).OrderBy(static value => value).ToArray();
            var ys = valid.Select(row => row.GetY(side)).Where(static value => value is not null).Select(static value => value!.Value).OrderBy(static value => value).ToArray();
            if (xs.Length == 0 || ys.Length == 0)
            {
                await writer.WriteLineAsync($"{side},0,{minConfidence},,,,,,,,,");
                continue;
            }

            await writer.WriteLineAsync(string.Join(',',
                side,
                xs.Length,
                minConfidence.ToString("0.000", CultureInfo.InvariantCulture),
                xs.First().ToString("0.00", CultureInfo.InvariantCulture),
                Percentile(xs, 0.10).ToString("0.00", CultureInfo.InvariantCulture),
                Percentile(xs, 0.50).ToString("0.00", CultureInfo.InvariantCulture),
                Percentile(xs, 0.90).ToString("0.00", CultureInfo.InvariantCulture),
                xs.Last().ToString("0.00", CultureInfo.InvariantCulture),
                ys.First().ToString("0.00", CultureInfo.InvariantCulture),
                Percentile(ys, 0.10).ToString("0.00", CultureInfo.InvariantCulture),
                Percentile(ys, 0.50).ToString("0.00", CultureInfo.InvariantCulture),
                Percentile(ys, 0.90).ToString("0.00", CultureInfo.InvariantCulture),
                ys.Last().ToString("0.00", CultureInfo.InvariantCulture)));
        }
    }

    private static async Task<int> FitGazeSessionAsync(string[] args)
    {
        var sessionDir = GetOption(args, "--session") ?? (args.Length > 0 && !args[0].StartsWith("--") ? args[0] : null);
        if (sessionDir is null)
        {
            Console.Error.WriteLine("Missing session directory.");
            return 1;
        }

        var profilePath = GetOption(args, "--profile") ?? "tracking_profile_final.json";
        var outProfilePath = GetOption(args, "--out-profile") ?? Path.Combine(sessionDir, "generated_profile.json");
        var options = new GazeCalibrationFitterOptions
        {
            MinConfidence = double.Parse(GetOption(args, "--min-conf") ?? "0.05", CultureInfo.InvariantCulture),
            MinOpenness = double.Parse(GetOption(args, "--min-open") ?? "0.20", CultureInfo.InvariantCulture),
            MaxDeltaMs = double.Parse(GetOption(args, "--max-delta-ms") ?? "20", CultureInfo.InvariantCulture),
            ModelMode = GetOption(args, "--model") ?? "auto"
        };

        var pairs = await CalibrationSessionStore.LoadPairsAsync(sessionDir);
        var labels = await CalibrationSessionStore.LoadLabelsAsync(sessionDir);
        if (pairs.Count == 0)
        {
            Console.Error.WriteLine("No pairs.csv data found.");
            return 1;
        }

        if (labels.Count == 0)
        {
            Console.Error.WriteLine("No labels.jsonl ground truth found. Run human review before fitting.");
            return 1;
        }

        var fit = new GazeCalibrationFitter(options).Fit(pairs, labels);
        await CalibrationSessionStore.SaveMetricsAsync(fit.Metrics, sessionDir);

        if (!fit.Succeeded)
        {
            Console.Error.WriteLine("Gaze fit failed.");
            Console.Error.WriteLine($"left: {fit.Metrics.Eyes.GetValueOrDefault("left")?.FailureReason}");
            Console.Error.WriteLine($"right: {fit.Metrics.Eyes.GetValueOrDefault("right")?.FailureReason}");
            return 1;
        }

        var sourceProfile = await TrackingProfileStore.LoadOrDefaultAsync(profilePath);
        var generatedProfile = new GazeCalibrationProfileBuilder().BuildProfile(sourceProfile, fit);
        await TrackingProfileStore.SaveAsync(generatedProfile, outProfilePath);

        Console.WriteLine($"Quality: {fit.Metrics.Quality}");
        Console.WriteLine($"Model: {fit.Metrics.Model}");
        Console.WriteLine($"Valid pairs: {fit.Metrics.ValidPairs}");
        Console.WriteLine($"Rejected pairs: {fit.Metrics.RejectedPairs}");
        Console.WriteLine($"Wrote metrics: {Path.GetFullPath(Path.Combine(sessionDir, "metrics.json"))}");
        Console.WriteLine($"Wrote profile: {Path.GetFullPath(outProfilePath)}");
        foreach (var residual in fit.Metrics.StageResiduals.OrderBy(pair => pair.Key, StringComparer.OrdinalIgnoreCase))
        {
            Console.WriteLine($"{residual.Key}: residual=({residual.Value.X:0.000},{residual.Value.Y:0.000}) d={residual.Value.Distance:0.000}");
        }

        return 0;
    }

    private static async Task<int> SetRoiAsync(string[] args)
    {
        var profilePath = RequireOption(args, "--profile");
        var side = ParseSide(RequireOption(args, "--side"));
        var x = int.Parse(RequireOption(args, "--x"), CultureInfo.InvariantCulture);
        var y = int.Parse(RequireOption(args, "--y"), CultureInfo.InvariantCulture);
        var width = int.Parse(RequireOption(args, "--w"), CultureInfo.InvariantCulture);
        var height = int.Parse(RequireOption(args, "--h"), CultureInfo.InvariantCulture);

        var profile = await TrackingProfileStore.LoadOrDefaultAsync(profilePath);
        profile.GetEye(side).Roi = new PixelRect(x, y, width, height);
        await TrackingProfileStore.SaveAsync(profile, profilePath);
        Console.WriteLine($"{side} ROI saved to {Path.GetFullPath(profilePath)}");
        return 0;
    }

    private static async Task<int> SetCalibrationAsync(string[] args)
    {
        var profilePath = RequireOption(args, "--profile");
        var side = ParseSide(RequireOption(args, "--side"));
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(profilePath);
        var calibration = profile.GetEye(side).Calibration;

        SetDoubleIfPresent(args, "--center-x", value => calibration.CenterX = value);
        SetDoubleIfPresent(args, "--center-y", value => calibration.CenterY = value);
        SetDoubleIfPresent(args, "--range-x", value => calibration.HorizontalRange = value);
        SetDoubleIfPresent(args, "--range-y", value => calibration.VerticalRange = value);
        SetDoubleIfPresent(args, "--open", value => calibration.OpenThreshold = value);
        SetDoubleIfPresent(args, "--closed", value => calibration.ClosedThreshold = value);

        await TrackingProfileStore.SaveAsync(profile, profilePath);
        Console.WriteLine($"{side} calibration saved to {Path.GetFullPath(profilePath)}");
        return 0;
    }

    private static async Task<int> CaptureHttpAsync(string[] args)
    {
        var host = GetOption(args, "--host") ?? "127.0.0.1";
        var port = int.Parse(GetOption(args, "--port") ?? "5555");
        var frames = int.Parse(GetOption(args, "--frames") ?? "30");
        var outDir = GetOption(args, "--out") ?? "capture_http";
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(GetOption(args, "--profile"));
        Directory.CreateDirectory(outDir);

        var detector = new PupilDetector();
        await CaptureHttpSideAsync(detector, profile, host, port, EyeSide.Left, frames, outDir);
        await CaptureHttpSideAsync(detector, profile, host, port, EyeSide.Right, frames, outDir);
        return 0;
    }

    private static async Task CaptureHttpSideAsync(PupilDetector detector, TrackingProfile profile, string host, int port, EyeSide side, int frames, string outDir)
    {
        var sideName = side == EyeSide.Left ? "left" : "right";
        var uri = new Uri($"http://{host}:{port}/eye/{sideName}");
        var client = new MjpegStreamClient();
        var index = 0;

        await foreach (var jpeg in client.ReadJpegFramesAsync(uri))
        {
            index++;
            var imagePath = Path.Combine(outDir, $"{sideName}_{index:0000}.jpg");
            await File.WriteAllBytesAsync(imagePath, jpeg);

            var frame = await DecodeJpegFrameAsync(jpeg, side);
            var detection = detector.Detect(frame, profile.CreateDetectorOptions(side));
            Console.WriteLine($"{sideName},{index},{detection.Found},{detection.Confidence:0.000},{detection.CenterX:0.00},{detection.CenterY:0.00}");

            if (index >= frames)
            {
                break;
            }
        }
    }

    private static async Task<int> LiveHttpAsync(string[] args)
    {
        var host = GetOption(args, "--host") ?? "127.0.0.1";
        var port = int.Parse(GetOption(args, "--port") ?? "5555", CultureInfo.InvariantCulture);
        var pairs = int.Parse(GetOption(args, "--pairs") ?? GetOption(args, "--frames") ?? "120", CultureInfo.InvariantCulture);
        var outDir = GetOption(args, "--out") ?? $"live_http_{DateTime.Now:yyyyMMdd_HHmmss}";
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(GetOption(args, "--profile"));

        Directory.CreateDirectory(outDir);
        var frameChannel = Channel.CreateBounded<CapturedJpeg>(new BoundedChannelOptions(32)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.DropOldest
        });
        var maxDeltaMs = double.Parse(GetOption(args, "--max-delta-ms") ?? "20", CultureInfo.InvariantCulture);

        using var cts = new CancellationTokenSource();
        var leftTask = Task.Run(() => PumpHttpSideAsync(host, port, EyeSide.Left, frameChannel.Writer, cts.Token));
        var rightTask = Task.Run(() => PumpHttpSideAsync(host, port, EyeSide.Right, frameChannel.Writer, cts.Token));
        var tracker = new EyeTracker(profile);
        var csvPath = Path.Combine(outDir, "pairs.csv");
        var leftQueue = new List<CapturedJpeg>();
        var rightQueue = new List<CapturedJpeg>();

        await using var csvStream = File.Create(csvPath);
        await using var csv = new StreamWriter(csvStream);
        await csv.WriteLineAsync("pair,left_index,right_index,delta_ms,left_found,left_conf,left_x,left_y,right_found,right_conf,right_x,right_y,left_file,right_file");

        for (var pairIndex = 1; pairIndex <= pairs;)
        {
            var packet = await frameChannel.Reader.ReadAsync(cts.Token);
            var queue = packet.Side == EyeSide.Left ? leftQueue : rightQueue;
            queue.Add(packet);
            TrimQueue(queue, 10);

            var pair = TryTakeBestPair(leftQueue, rightQueue, maxDeltaMs);
            if (pair is null)
            {
                continue;
            }

            var (left, right) = pair.Value;
            var leftDetection = tracker.Process(left.Frame);
            var rightDetection = tracker.Process(right.Frame);
            var deltaMs = Math.Abs((left.ReceivedAt - right.ReceivedAt).TotalMilliseconds);

            var leftFile = $"pair_{pairIndex:0000}_left.jpg";
            var rightFile = $"pair_{pairIndex:0000}_right.jpg";
            await File.WriteAllBytesAsync(Path.Combine(outDir, leftFile), left.Jpeg, cts.Token);
            await File.WriteAllBytesAsync(Path.Combine(outDir, rightFile), right.Jpeg, cts.Token);

            await csv.WriteLineAsync(string.Join(',',
                pairIndex,
                left.Index,
                right.Index,
                deltaMs.ToString("0.0", CultureInfo.InvariantCulture),
                leftDetection.Found,
                leftDetection.Confidence.ToString("0.000", CultureInfo.InvariantCulture),
                leftDetection.CenterX.ToString("0.00", CultureInfo.InvariantCulture),
                leftDetection.CenterY.ToString("0.00", CultureInfo.InvariantCulture),
                rightDetection.Found,
                rightDetection.Confidence.ToString("0.000", CultureInfo.InvariantCulture),
                rightDetection.CenterX.ToString("0.00", CultureInfo.InvariantCulture),
                rightDetection.CenterY.ToString("0.00", CultureInfo.InvariantCulture),
                leftFile,
                rightFile));

            Console.WriteLine($"{pairIndex:0000} dt={deltaMs:0.0}ms L=({leftDetection.Found},{leftDetection.CenterX:0.0},{leftDetection.CenterY:0.0}) R=({rightDetection.Found},{rightDetection.CenterX:0.0},{rightDetection.CenterY:0.0})");
            pairIndex++;
        }

        cts.Cancel();
        await Task.WhenAny(Task.WhenAll(leftTask, rightTask), Task.Delay(1000));
        Console.WriteLine($"Wrote live capture: {Path.GetFullPath(outDir)}");
        return 0;
    }

    private static async Task<int> PupilDiameterSessionHttpAsync(string[] args)
    {
        var host = GetOption(args, "--host") ?? "127.0.0.1";
        var port = int.Parse(GetOption(args, "--port") ?? "5555", CultureInfo.InvariantCulture);
        var stagePairs = int.Parse(GetOption(args, "--stage-pairs") ?? "90", CultureInfo.InvariantCulture);
        var maxDeltaMs = double.Parse(GetOption(args, "--max-delta-ms") ?? "20", CultureInfo.InvariantCulture);
        var countdownSeconds = int.Parse(GetOption(args, "--countdown-seconds") ?? "3", CultureInfo.InvariantCulture);
        var settleSeconds = double.Parse(GetOption(args, "--settle-seconds") ?? "0", CultureInfo.InvariantCulture);
        var summaryTailPairs = int.Parse(GetOption(args, "--summary-tail-pairs") ?? "30", CultureInfo.InvariantCulture);
        var outDir = GetOption(args, "--out") ?? $"pupil_diameter_session_{DateTime.Now:yyyyMMdd_HHmmss}";
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(GetOption(args, "--profile"));
        var stages = ParseStages(GetOption(args, "--stages") ?? "center,bright,dark,blink");
        var manualStage = HasFlag(args, "--manual-stage") || HasFlag(args, "--manual");
        var overlayEvery = GetOption(args, "--overlay-every") is { } overlayEveryRaw
            ? int.Parse(overlayEveryRaw, CultureInfo.InvariantCulture)
            : (HasFlag(args, "--debug-overlays") || HasFlag(args, "--overlays") ? 15 : 0);
        var framesDir = Path.Combine(outDir, "frames");
        var overlaysDir = Path.Combine(outDir, "overlays");

        Directory.CreateDirectory(framesDir);
        if (overlayEvery > 0)
        {
            Directory.CreateDirectory(overlaysDir);
        }

        var frameChannel = Channel.CreateBounded<CapturedJpeg>(new BoundedChannelOptions(32)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.DropOldest
        });

        using var cts = new CancellationTokenSource();
        var leftTask = Task.Run(() => PumpHttpSideAsync(host, port, EyeSide.Left, frameChannel.Writer, cts.Token));
        var rightTask = Task.Run(() => PumpHttpSideAsync(host, port, EyeSide.Right, frameChannel.Writer, cts.Token));
        var tracker = new EyeTracker(profile);
        var diameterEstimator = new PupilDiameterEstimator();
        var opennessDetector = new EyeOpennessDetector();
        var leftQueue = new List<CapturedJpeg>();
        var rightQueue = new List<CapturedJpeg>();
        var measurements = new List<PupilDiameterStageMeasurement>();
        var csvPath = Path.Combine(outDir, "pupil_diameter_pairs.csv");

        await using var csvStream = File.Create(csvPath);
        await using var csv = new StreamWriter(csvStream);
        await csv.WriteLineAsync("stage,pair,left_index,right_index,delta_ms,left_file,right_file,left_found,left_conf,left_center_x,left_center_y,left_diameter_found,left_diameter_quality,left_diameter_conf,left_diameter_px,left_major_px,left_minor_px,left_axis_ratio,left_orientation_rad,left_openness,left_aperture_height,left_diameter_reason,right_found,right_conf,right_center_x,right_center_y,right_diameter_found,right_diameter_quality,right_diameter_conf,right_diameter_px,right_major_px,right_minor_px,right_axis_ratio,right_orientation_rad,right_openness,right_aperture_height,right_diameter_reason");

        foreach (var stage in stages)
        {
            Console.WriteLine();
            Console.WriteLine($"Pupil diameter stage: {stage}. Keep gaze centered unless this stage says otherwise.");
            if (manualStage)
            {
                Console.WriteLine("Switch the screen/background for this stage, then press Enter to start recording.");
                Console.ReadLine();
            }

            if (settleSeconds > 0)
            {
                await CountdownAsync(countdownSeconds, cts.Token, "settling");
                Console.WriteLine($"settling {settleSeconds:0.###}s before recording");
                await Task.Delay(TimeSpan.FromSeconds(settleSeconds), cts.Token);
                Console.WriteLine("recording");
            }
            else
            {
                await CountdownAsync(countdownSeconds, cts.Token);
            }

            leftQueue.Clear();
            rightQueue.Clear();
            while (frameChannel.Reader.TryRead(out _))
            {
            }

            for (var pairIndex = 1; pairIndex <= stagePairs;)
            {
                var packet = await frameChannel.Reader.ReadAsync(cts.Token);
                var queue = packet.Side == EyeSide.Left ? leftQueue : rightQueue;
                queue.Add(packet);
                TrimQueue(queue, 10);

                var pair = TryTakeBestPair(leftQueue, rightQueue, maxDeltaMs);
                if (pair is null)
                {
                    continue;
                }

                var (left, right) = pair.Value;
                var leftDetection = tracker.Process(left.Frame);
                var rightDetection = tracker.Process(right.Frame);
                var leftDiameter = diameterEstimator.Estimate(left.Frame, leftDetection);
                var rightDiameter = diameterEstimator.Estimate(right.Frame, rightDetection);
                var leftOpenness = opennessDetector.Detect(left.Frame, new EyeOpennessDetectorOptions { Roi = profile.Left.Roi });
                var rightOpenness = opennessDetector.Detect(right.Frame, new EyeOpennessDetectorOptions { Roi = profile.Right.Roi });
                var leftDiameterQuality = IsQualityPupilDiameter(leftDiameter, leftOpenness);
                var rightDiameterQuality = IsQualityPupilDiameter(rightDiameter, rightOpenness);
                var deltaMs = Math.Abs((left.ReceivedAt - right.ReceivedAt).TotalMilliseconds);

                var leftFile = $"{stage}_{pairIndex:0000}_left.jpg";
                var rightFile = $"{stage}_{pairIndex:0000}_right.jpg";
                await File.WriteAllBytesAsync(Path.Combine(framesDir, leftFile), left.Jpeg, cts.Token);
                await File.WriteAllBytesAsync(Path.Combine(framesDir, rightFile), right.Jpeg, cts.Token);

                if (overlayEvery > 0 && (pairIndex == 1 || pairIndex % overlayEvery == 0))
                {
                    await SavePupilDiameterOverlayAsync(
                        Path.Combine(framesDir, leftFile),
                        profile.Left.Roi,
                        leftDetection,
                        leftDiameter,
                        leftOpenness,
                        Path.Combine(overlaysDir, $"{stage}_{pairIndex:0000}_left.png"));
                    await SavePupilDiameterOverlayAsync(
                        Path.Combine(framesDir, rightFile),
                        profile.Right.Roi,
                        rightDetection,
                        rightDiameter,
                        rightOpenness,
                        Path.Combine(overlaysDir, $"{stage}_{pairIndex:0000}_right.png"));
                }

                measurements.Add(new PupilDiameterStageMeasurement(stage, pairIndex, "left", leftDiameter.Found, leftDiameterQuality, leftDiameter.Confidence, leftDiameter.EquivalentDiameterPx, leftDiameter.AxisRatio, leftOpenness.Openness));
                measurements.Add(new PupilDiameterStageMeasurement(stage, pairIndex, "right", rightDiameter.Found, rightDiameterQuality, rightDiameter.Confidence, rightDiameter.EquivalentDiameterPx, rightDiameter.AxisRatio, rightOpenness.Openness));

                await csv.WriteLineAsync(JoinCsv(new[]
                {
                    stage,
                    pairIndex.ToString(CultureInfo.InvariantCulture),
                    left.Index.ToString(CultureInfo.InvariantCulture),
                    right.Index.ToString(CultureInfo.InvariantCulture),
                    F(deltaMs),
                    Path.Combine("frames", leftFile),
                    Path.Combine("frames", rightFile),
                    leftDetection.Found.ToString(),
                    F(leftDetection.Confidence),
                    F(leftDetection.CenterX),
                    F(leftDetection.CenterY),
                    leftDiameter.Found.ToString(),
                    leftDiameterQuality.ToString(),
                    F(leftDiameter.Confidence),
                    F(leftDiameter.EquivalentDiameterPx),
                    F(leftDiameter.MajorAxisPx),
                    F(leftDiameter.MinorAxisPx),
                    F(leftDiameter.AxisRatio),
                    F(leftDiameter.OrientationRadians),
                    F(leftOpenness.Openness),
                    leftOpenness.ApertureHeight.ToString(CultureInfo.InvariantCulture),
                    leftDiameter.Reason ?? string.Empty,
                    rightDetection.Found.ToString(),
                    F(rightDetection.Confidence),
                    F(rightDetection.CenterX),
                    F(rightDetection.CenterY),
                    rightDiameter.Found.ToString(),
                    rightDiameterQuality.ToString(),
                    F(rightDiameter.Confidence),
                    F(rightDiameter.EquivalentDiameterPx),
                    F(rightDiameter.MajorAxisPx),
                    F(rightDiameter.MinorAxisPx),
                    F(rightDiameter.AxisRatio),
                    F(rightDiameter.OrientationRadians),
                    F(rightOpenness.Openness),
                    rightOpenness.ApertureHeight.ToString(CultureInfo.InvariantCulture),
                    rightDiameter.Reason ?? string.Empty
                }));

                Console.WriteLine($"{stage} {pairIndex:000}/{stagePairs:000} dt={deltaMs:0.0}ms Ld=({leftDiameter.Found},q={leftDiameterQuality},{leftDiameter.EquivalentDiameterPx:0.0}px,o={leftOpenness.Openness:0.00}) Rd=({rightDiameter.Found},q={rightDiameterQuality},{rightDiameter.EquivalentDiameterPx:0.0}px,o={rightOpenness.Openness:0.00})");
                pairIndex++;
            }
        }

        cts.Cancel();
        await Task.WhenAny(Task.WhenAll(leftTask, rightTask), Task.Delay(1000));
        var summaryPath = Path.Combine(outDir, "pupil_diameter_summary.csv");
        await WritePupilDiameterSummaryAsync(summaryPath, measurements, summaryTailPairs);
        Console.WriteLine($"Wrote pupil diameter session: {Path.GetFullPath(outDir)}");
        Console.WriteLine($"Wrote CSV: {Path.GetFullPath(csvPath)}");
        Console.WriteLine($"Wrote summary: {Path.GetFullPath(summaryPath)}");
        return 0;
    }

    private static async Task<int> CalibrationSessionHttpAsync(string[] args)
    {
        var host = GetOption(args, "--host") ?? "127.0.0.1";
        var port = int.Parse(GetOption(args, "--port") ?? "5555", CultureInfo.InvariantCulture);
        var stagePairs = int.Parse(GetOption(args, "--stage-pairs") ?? "30", CultureInfo.InvariantCulture);
        var maxDeltaMs = double.Parse(GetOption(args, "--max-delta-ms") ?? "20", CultureInfo.InvariantCulture);
        var outDir = GetOption(args, "--out") ?? $"calibration_session_{DateTime.Now:yyyyMMdd_HHmmss}";
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(GetOption(args, "--profile"));
        var stages = ParseStages(GetOption(args, "--stages") ?? "center,left,right,up,down,blink,open");

        Directory.CreateDirectory(outDir);
        var frameChannel = Channel.CreateBounded<CapturedJpeg>(new BoundedChannelOptions(32)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.DropOldest
        });

        using var cts = new CancellationTokenSource();
        var leftTask = Task.Run(() => PumpHttpSideAsync(host, port, EyeSide.Left, frameChannel.Writer, cts.Token));
        var rightTask = Task.Run(() => PumpHttpSideAsync(host, port, EyeSide.Right, frameChannel.Writer, cts.Token));
        var tracker = new EyeTracker(profile);
        var leftQueue = new List<CapturedJpeg>();
        var rightQueue = new List<CapturedJpeg>();
        var csvPath = Path.Combine(outDir, "calibration_pairs.csv");

        await using var csvStream = File.Create(csvPath);
        await using var csv = new StreamWriter(csvStream);
        await csv.WriteLineAsync("stage,pair,left_index,right_index,delta_ms,left_found,left_conf,left_x,left_y,right_found,right_conf,right_x,right_y,left_file,right_file");

        foreach (var stage in stages)
        {
            Console.WriteLine();
            Console.WriteLine($"Stage: {stage}. Get ready...");
            await CountdownAsync(3, cts.Token);

            for (var pairIndex = 1; pairIndex <= stagePairs;)
            {
                var packet = await frameChannel.Reader.ReadAsync(cts.Token);
                var queue = packet.Side == EyeSide.Left ? leftQueue : rightQueue;
                queue.Add(packet);
                TrimQueue(queue, 10);

                var pair = TryTakeBestPair(leftQueue, rightQueue, maxDeltaMs);
                if (pair is null)
                {
                    continue;
                }

                var (left, right) = pair.Value;
                var leftDetection = tracker.Process(left.Frame);
                var rightDetection = tracker.Process(right.Frame);
                var deltaMs = Math.Abs((left.ReceivedAt - right.ReceivedAt).TotalMilliseconds);
                var leftFile = $"{stage}_{pairIndex:0000}_left.jpg";
                var rightFile = $"{stage}_{pairIndex:0000}_right.jpg";
                await File.WriteAllBytesAsync(Path.Combine(outDir, leftFile), left.Jpeg, cts.Token);
                await File.WriteAllBytesAsync(Path.Combine(outDir, rightFile), right.Jpeg, cts.Token);

                await csv.WriteLineAsync(string.Join(',',
                    stage,
                    pairIndex,
                    left.Index,
                    right.Index,
                    deltaMs.ToString("0.0", CultureInfo.InvariantCulture),
                    leftDetection.Found,
                    leftDetection.Confidence.ToString("0.000", CultureInfo.InvariantCulture),
                    leftDetection.CenterX.ToString("0.00", CultureInfo.InvariantCulture),
                    leftDetection.CenterY.ToString("0.00", CultureInfo.InvariantCulture),
                    rightDetection.Found,
                    rightDetection.Confidence.ToString("0.000", CultureInfo.InvariantCulture),
                    rightDetection.CenterX.ToString("0.00", CultureInfo.InvariantCulture),
                    rightDetection.CenterY.ToString("0.00", CultureInfo.InvariantCulture),
                    leftFile,
                    rightFile));

                Console.WriteLine($"{stage} {pairIndex:000}/{stagePairs:000} dt={deltaMs:0.0}ms L=({leftDetection.Found},{leftDetection.CenterX:0.0},{leftDetection.CenterY:0.0}) R=({rightDetection.Found},{rightDetection.CenterX:0.0},{rightDetection.CenterY:0.0})");
                pairIndex++;
            }
        }

        cts.Cancel();
        await Task.WhenAny(Task.WhenAll(leftTask, rightTask), Task.Delay(1000));
        Console.WriteLine($"Wrote calibration session: {Path.GetFullPath(outDir)}");
        return 0;
    }

    private static string[] ParseStages(string raw)
    {
        return raw.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
    }

    private static async Task CountdownAsync(int seconds, CancellationToken cancellationToken, string finalMessage = "recording")
    {
        for (var i = seconds; i >= 1; i--)
        {
            Console.WriteLine($"{i}...");
            await Task.Delay(TimeSpan.FromSeconds(1), cancellationToken);
        }

        Console.WriteLine(finalMessage);
    }

    private static async Task WritePupilDiameterSummaryAsync(string path, IReadOnlyList<PupilDiameterStageMeasurement> measurements, int tailPairs)
    {
        var csv = new StringBuilder();
        csv.AppendLine("stage,side,total,found,found_rate,quality,quality_rate,diameter_p10_px,diameter_median_px,diameter_p90_px,diameter_mean_px,diameter_std_px,quality_diameter_median_px,axis_ratio_median,quality_axis_ratio_median,confidence_median,quality_confidence_median,openness_median,tail_pairs,tail_quality,tail_quality_rate,tail_quality_diameter_median_px");

        foreach (var group in measurements.GroupBy(static row => (row.Stage, row.Side)).OrderBy(static group => group.Key.Stage).ThenBy(static group => group.Key.Side))
        {
            var foundRows = group.Where(static row => row.Found && double.IsFinite(row.DiameterPx)).ToArray();
            var qualityRows = foundRows.Where(static row => row.Quality).ToArray();
            var values = foundRows.Select(static row => row.DiameterPx).OrderBy(static value => value).ToArray();
            var qualityValues = qualityRows.Select(static row => row.DiameterPx).OrderBy(static value => value).ToArray();
            var axisRatios = foundRows.Select(static row => row.AxisRatio).Where(double.IsFinite).OrderBy(static value => value).ToArray();
            var qualityAxisRatios = qualityRows.Select(static row => row.AxisRatio).Where(double.IsFinite).OrderBy(static value => value).ToArray();
            var confidences = foundRows.Select(static row => row.Confidence).OrderBy(static value => value).ToArray();
            var qualityConfidences = qualityRows.Select(static row => row.Confidence).OrderBy(static value => value).ToArray();
            var openness = group.Select(static row => row.Openness).Where(double.IsFinite).ToArray();
            var total = group.Count();
            var found = values.Length;
            var quality = qualityValues.Length;
            var maxPair = group.Select(static row => row.Pair).DefaultIfEmpty(0).Max();
            var tailStartPair = Math.Max(1, maxPair - Math.Max(tailPairs, 1) + 1);
            var tailRows = group.Where(row => row.Pair >= tailStartPair).ToArray();
            var tailQualityRows = tailRows.Where(static row => row.Found && row.Quality && double.IsFinite(row.DiameterPx)).ToArray();
            var tailQualityValues = tailQualityRows.Select(static row => row.DiameterPx).OrderBy(static value => value).ToArray();
            var mean = found > 0 ? values.Average() : double.NaN;
            var std = found > 1
                ? Math.Sqrt(values.Select(value => (value - mean) * (value - mean)).Sum() / (found - 1))
                : double.NaN;

            csv.AppendLine(JoinCsv(new[]
            {
                group.Key.Stage,
                group.Key.Side,
                total.ToString(CultureInfo.InvariantCulture),
                found.ToString(CultureInfo.InvariantCulture),
                F(found / (double)Math.Max(total, 1)),
                quality.ToString(CultureInfo.InvariantCulture),
                F(quality / (double)Math.Max(total, 1)),
                F(Percentile(values, 0.10)),
                F(Median(values)),
                F(Percentile(values, 0.90)),
                F(mean),
                F(std),
                F(Median(qualityValues)),
                F(Median(axisRatios)),
                F(Median(qualityAxisRatios)),
                F(Median(confidences)),
                F(Median(qualityConfidences)),
                F(Median(openness)),
                tailRows.Length.ToString(CultureInfo.InvariantCulture),
                tailQualityRows.Length.ToString(CultureInfo.InvariantCulture),
                F(tailQualityRows.Length / (double)Math.Max(tailRows.Length, 1)),
                F(Median(tailQualityValues))
            }));
        }

        await File.WriteAllTextAsync(path, csv.ToString(), Encoding.UTF8);
    }

    private static bool IsQualityPupilDiameter(PupilDiameterEstimate diameter, EyeOpennessDetection openness)
    {
        return diameter.Found &&
            double.IsFinite(diameter.EquivalentDiameterPx) &&
            double.IsFinite(diameter.AxisRatio) &&
            diameter.Confidence >= PupilDiameterQualityMinConfidence &&
            diameter.AxisRatio <= PupilDiameterQualityMaxAxisRatio &&
            openness.Openness >= PupilDiameterQualityMinOpenness;
    }

    private static async Task<int> FitPupilDiameterSessionAsync(string[] args)
    {
        var sessionDir = GetOption(args, "--session") ?? (args.Length > 0 && !args[0].StartsWith("--") ? args[0] : null);
        if (sessionDir is null)
        {
            Console.Error.WriteLine("Missing pupil diameter session directory.");
            return 1;
        }

        var csvPath = Path.Combine(sessionDir, "pupil_diameter_pairs.csv");
        if (!File.Exists(csvPath))
        {
            Console.Error.WriteLine($"Missing pupil diameter pairs CSV: {csvPath}");
            return 1;
        }

        var tailPairs = int.Parse(GetOption(args, "--tail-pairs") ?? "30", CultureInfo.InvariantCulture);
        var minConfidence = double.Parse(GetOption(args, "--min-conf") ?? PupilDiameterQualityMinConfidence.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
        var maxAxisRatio = double.Parse(GetOption(args, "--max-axis-ratio") ?? PupilDiameterQualityMaxAxisRatio.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
        var minOpenness = double.Parse(GetOption(args, "--min-open") ?? PupilDiameterQualityMinOpenness.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
        var brightStages = ParseStages(GetOption(args, "--bright-stages") ?? "bright,bright2");
        var darkStages = ParseStages(GetOption(args, "--dark-stages") ?? "dark,dark2");
        var outPath = GetOption(args, "--out") ?? Path.Combine(sessionDir, "pupil_diameter_calibration.json");
        var reportPath = GetOption(args, "--report") ?? Path.Combine(sessionDir, "pupil_diameter_calibration_report.csv");

        var rows = LoadPupilDiameterPairRows(csvPath);
        if (rows.Count == 0)
        {
            Console.Error.WriteLine("No pupil diameter rows found.");
            return 1;
        }

        var stageOrder = rows.Select(static row => row.Stage)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(static stage => stage, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var leftStageFits = stageOrder.Select(stage => FitPupilDiameterStage(rows, stage, "left", tailPairs, minConfidence, maxAxisRatio, minOpenness)).ToArray();
        var rightStageFits = stageOrder.Select(stage => FitPupilDiameterStage(rows, stage, "right", tailPairs, minConfidence, maxAxisRatio, minOpenness)).ToArray();
        var leftFit = FitPupilDiameterEye("left", leftStageFits, brightStages, darkStages);
        var rightFit = FitPupilDiameterEye("right", rightStageFits, brightStages, darkStages);

        var calibration = new PupilDiameterCalibrationFit(
            DateTimeOffset.UtcNow,
            Path.GetFullPath(sessionDir),
            tailPairs,
            minConfidence,
            maxAxisRatio,
            minOpenness,
            brightStages,
            darkStages,
            leftFit,
            rightFit);

        var outDir = Path.GetDirectoryName(Path.GetFullPath(outPath));
        if (!string.IsNullOrEmpty(outDir))
        {
            Directory.CreateDirectory(outDir);
        }

        await using (var stream = File.Create(outPath))
        {
            await JsonSerializer.SerializeAsync(stream, calibration, new JsonSerializerOptions
            {
                WriteIndented = true
            });
        }

        await WritePupilDiameterCalibrationReportAsync(reportPath, leftStageFits, rightStageFits);

        Console.WriteLine($"Wrote pupil diameter calibration: {Path.GetFullPath(outPath)}");
        Console.WriteLine($"Wrote pupil diameter report: {Path.GetFullPath(reportPath)}");
        Console.WriteLine($"left valid={leftFit.Valid} bright={leftFit.BrightPx:0.00}px dark={leftFit.DarkPx:0.00}px range={leftFit.RangePx:0.00}px repeatability={leftFit.RepeatabilityWarning}");
        Console.WriteLine($"right valid={rightFit.Valid} bright={rightFit.BrightPx:0.00}px dark={rightFit.DarkPx:0.00}px range={rightFit.RangePx:0.00}px repeatability={rightFit.RepeatabilityWarning}");
        return 0;
    }

    private static PupilDiameterStageFit FitPupilDiameterStage(
        IReadOnlyList<PupilDiameterPairRow> rows,
        string stage,
        string side,
        int tailPairs,
        double minConfidence,
        double maxAxisRatio,
        double minOpenness)
    {
        var stageRows = rows.Where(row => row.Stage.Equals(stage, StringComparison.OrdinalIgnoreCase)).ToArray();
        var maxPair = stageRows.Select(static row => row.Pair).DefaultIfEmpty(0).Max();
        var tailStart = Math.Max(1, maxPair - Math.Max(tailPairs, 1) + 1);
        var tailRows = stageRows.Where(row => row.Pair >= tailStart).ToArray();
        var qualityRows = tailRows
            .Select(row => row.GetEye(side))
            .Where(eye => eye.IsQuality(minConfidence, maxAxisRatio, minOpenness))
            .ToArray();
        var diameters = qualityRows.Select(static eye => eye.DiameterPx).OrderBy(static value => value).ToArray();
        var axisRatios = qualityRows.Select(static eye => eye.AxisRatio).Where(double.IsFinite).OrderBy(static value => value).ToArray();
        var confidences = qualityRows.Select(static eye => eye.Confidence).Where(double.IsFinite).OrderBy(static value => value).ToArray();
        var centerXs = qualityRows.Select(static eye => eye.CenterX).Where(double.IsFinite).OrderBy(static value => value).ToArray();
        var centerYs = qualityRows.Select(static eye => eye.CenterY).Where(double.IsFinite).OrderBy(static value => value).ToArray();

        return new PupilDiameterStageFit(
            stage,
            side,
            stageRows.Length,
            tailRows.Length,
            qualityRows.Length,
            qualityRows.Length / (double)Math.Max(tailRows.Length, 1),
            Percentile(diameters, 0.10),
            Median(diameters),
            Percentile(diameters, 0.90),
            Median(axisRatios),
            Median(confidences),
            Median(centerXs),
            Median(centerYs));
    }

    private static PupilDiameterEyeCalibrationFit FitPupilDiameterEye(
        string side,
        IReadOnlyList<PupilDiameterStageFit> stageFits,
        IReadOnlyCollection<string> brightStages,
        IReadOnlyCollection<string> darkStages)
    {
        var brightFits = stageFits
            .Where(stage => brightStages.Contains(stage.Stage, StringComparer.OrdinalIgnoreCase) && double.IsFinite(stage.MedianPx))
            .ToArray();
        var darkFits = stageFits
            .Where(stage => darkStages.Contains(stage.Stage, StringComparer.OrdinalIgnoreCase) && double.IsFinite(stage.MedianPx))
            .ToArray();
        var brightValues = brightFits.Select(static stage => stage.MedianPx).OrderBy(static value => value).ToArray();
        var darkValues = darkFits.Select(static stage => stage.MedianPx).OrderBy(static value => value).ToArray();
        var brightPx = Median(brightValues);
        var darkPx = Median(darkValues);
        var rangePx = darkPx - brightPx;
        var relativeRange = double.IsFinite(rangePx) && brightPx > 0 ? rangePx / brightPx : double.NaN;
        var brightSpread = Spread(brightValues);
        var darkSpread = Spread(darkValues);
        var repeatabilityWarning = (brightSpread > Math.Max(2.5, Math.Abs(rangePx) * 0.25)) ||
            (darkSpread > Math.Max(2.5, Math.Abs(rangePx) * 0.25));
        var valid = brightFits.Length > 0 &&
            darkFits.Length > 0 &&
            double.IsFinite(rangePx) &&
            rangePx >= 3 &&
            brightFits.All(static stage => stage.QualityRate >= 0.5) &&
            darkFits.All(static stage => stage.QualityRate >= 0.5);

        return new PupilDiameterEyeCalibrationFit(
            side,
            valid,
            brightPx,
            darkPx,
            rangePx,
            relativeRange,
            brightSpread,
            darkSpread,
            repeatabilityWarning,
            stageFits);
    }

    private static double Spread(IReadOnlyList<double> values)
    {
        return values.Count == 0 ? double.NaN : values.Max() - values.Min();
    }

    private static async Task WritePupilDiameterCalibrationReportAsync(
        string path,
        IReadOnlyList<PupilDiameterStageFit> left,
        IReadOnlyList<PupilDiameterStageFit> right)
    {
        var outDir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(outDir))
        {
            Directory.CreateDirectory(outDir);
        }

        var csv = new StringBuilder();
        csv.AppendLine("stage,side,total_pairs,tail_pairs,tail_quality,tail_quality_rate,tail_diameter_p10_px,tail_diameter_median_px,tail_diameter_p90_px,tail_axis_ratio_median,tail_confidence_median,tail_center_x_median,tail_center_y_median");
        foreach (var row in left.Concat(right).OrderBy(static row => row.Stage, StringComparer.OrdinalIgnoreCase).ThenBy(static row => row.Side, StringComparer.OrdinalIgnoreCase))
        {
            csv.AppendLine(JoinCsv(new[]
            {
                row.Stage,
                row.Side,
                row.TotalPairs.ToString(CultureInfo.InvariantCulture),
                row.TailPairs.ToString(CultureInfo.InvariantCulture),
                row.QualityPairs.ToString(CultureInfo.InvariantCulture),
                F(row.QualityRate),
                F(row.P10Px),
                F(row.MedianPx),
                F(row.P90Px),
                F(row.AxisRatioMedian),
                F(row.ConfidenceMedian),
                F(row.CenterXMedian),
                F(row.CenterYMedian)
            }));
        }

        await File.WriteAllTextAsync(path, csv.ToString(), Encoding.UTF8);
    }

    private static (CapturedJpeg Left, CapturedJpeg Right)? TryTakeBestPair(List<CapturedJpeg> leftQueue, List<CapturedJpeg> rightQueue, double maxDeltaMs)
    {
        if (leftQueue.Count == 0 || rightQueue.Count == 0)
        {
            return null;
        }

        var bestLeft = -1;
        var bestRight = -1;
        var bestDelta = double.MaxValue;

        for (var li = 0; li < leftQueue.Count; li++)
        {
            for (var ri = 0; ri < rightQueue.Count; ri++)
            {
                var delta = Math.Abs((leftQueue[li].ReceivedAt - rightQueue[ri].ReceivedAt).TotalMilliseconds);
                if (delta < bestDelta)
                {
                    bestDelta = delta;
                    bestLeft = li;
                    bestRight = ri;
                }
            }
        }

        if (bestDelta > maxDeltaMs)
        {
            DropOldestFrame(leftQueue, rightQueue);
            return null;
        }

        var left = leftQueue[bestLeft];
        var right = rightQueue[bestRight];
        leftQueue.RemoveAt(bestLeft);
        rightQueue.RemoveAt(bestRight);
        DropOlderThan(leftQueue, left.ReceivedAt.AddMilliseconds(-maxDeltaMs * 2));
        DropOlderThan(rightQueue, right.ReceivedAt.AddMilliseconds(-maxDeltaMs * 2));
        return (left, right);
    }

    private static void DropOldestFrame(List<CapturedJpeg> leftQueue, List<CapturedJpeg> rightQueue)
    {
        if (leftQueue.Count == 0)
        {
            rightQueue.RemoveAt(0);
            return;
        }

        if (rightQueue.Count == 0)
        {
            leftQueue.RemoveAt(0);
            return;
        }

        if (leftQueue[0].ReceivedAt <= rightQueue[0].ReceivedAt)
        {
            leftQueue.RemoveAt(0);
        }
        else
        {
            rightQueue.RemoveAt(0);
        }
    }

    private static void TrimQueue(List<CapturedJpeg> queue, int maxCount)
    {
        while (queue.Count > maxCount)
        {
            queue.RemoveAt(0);
        }
    }

    private static void DropOlderThan(List<CapturedJpeg> queue, DateTimeOffset cutoff)
    {
        queue.RemoveAll(frame => frame.ReceivedAt < cutoff);
    }

    private static async Task PumpHttpSideAsync(string host, int port, EyeSide side, ChannelWriter<CapturedJpeg> writer, CancellationToken cancellationToken)
    {
        try
        {
            var sideName = side == EyeSide.Left ? "left" : "right";
            var uri = new Uri($"http://{host}:{port}/eye/{sideName}");
            var client = new MjpegStreamClient();
            var index = 0;

            await foreach (var jpeg in client.ReadJpegFramesAsync(uri, cancellationToken))
            {
                index++;
                var frame = await DecodeJpegFrameAsync(jpeg, side);
                var packet = new CapturedJpeg(side, index, jpeg, frame, DateTimeOffset.UtcNow);
                await writer.WriteAsync(packet, cancellationToken);
            }

            writer.TryComplete();
        }
        catch (OperationCanceledException)
        {
            writer.TryComplete();
        }
        catch (Exception ex)
        {
            writer.TryComplete(ex);
        }
    }

    private static async Task<int> CaptureRawAsync(string[] args)
    {
        var host = GetOption(args, "--host") ?? "127.0.0.1";
        var port = int.Parse(GetOption(args, "--port") ?? "5555");
        var frames = int.Parse(GetOption(args, "--frames") ?? "30");
        var outDir = GetOption(args, "--out") ?? "capture_raw";
        var profile = await TrackingProfileStore.LoadOrDefaultAsync(GetOption(args, "--profile"));
        var side = (GetOption(args, "--side") ?? "left").Equals("right", StringComparison.OrdinalIgnoreCase)
            ? EyeSide.Right
            : EyeSide.Left;

        Directory.CreateDirectory(outDir);
        var detector = new PupilDetector();
        var client = new BrokenEyeRawClient();
        var index = 0;

        await foreach (var frame in client.ReadFramesAsync(host, port, side))
        {
            index++;
            var detection = detector.Detect(frame, profile.CreateDetectorOptions(side));
            var sideName = side == EyeSide.Left ? "left" : "right";
            var imagePath = Path.Combine(outDir, $"{sideName}_{index:0000}.pgm");
            await SavePgmAsync(frame, imagePath);
            Console.WriteLine($"{sideName},{index},{detection.Found},{detection.Confidence:0.000},{detection.CenterX:0.00},{detection.CenterY:0.00}");

            if (index >= frames)
            {
                break;
            }
        }

        return 0;
    }

    private static async Task<EyeFrame> LoadImageFrameAsync(string file, EyeSide side)
    {
        var bytes = await File.ReadAllBytesAsync(file);
        return await DecodeJpegFrameAsync(bytes, side);
    }

    private static async Task<EyeFrame> DecodeJpegFrameAsync(byte[] bytes, EyeSide side)
    {
        using var image = Image.Load<L8>(bytes);
        var pixels = new byte[image.Width * image.Height];
        image.CopyPixelDataTo(pixels);
        await Task.CompletedTask;
        return new EyeFrame(side, image.Width, image.Height, 8, pixels, DateTimeOffset.UtcNow);
    }

    private static async Task SaveDebugImageAsync(string inputPath, PupilDetection detection, string outPath)
    {
        using var image = await Image.LoadAsync<Rgba32>(inputPath);
        if (detection.Found)
        {
            DrawRectangle(image, detection.Bounds.X, detection.Bounds.Y, detection.Bounds.Width, detection.Bounds.Height, new Rgba32(255, 40, 40));
            DrawCross(image, (int)Math.Round(detection.CenterX), (int)Math.Round(detection.CenterY), 5, new Rgba32(255, 40, 40));
        }

        await image.SaveAsPngAsync(outPath);
    }

    private static async Task SaveDiagnosticOverlayAsync(
        string inputPath,
        PixelRect? roi,
        PupilDetection detection,
        EyeOpennessDetection openness,
        IReadOnlyList<BrightSpotCandidate> brightCandidates,
        string outPath)
    {
        using var image = await Image.LoadAsync<Rgba32>(inputPath);
        if (roi is { IsEmpty: false })
        {
            DrawRectangle(image, roi.Value.X, roi.Value.Y, roi.Value.Width, roi.Value.Height, new Rgba32(40, 220, 255));
        }

        if (detection.Found)
        {
            DrawRectangle(image, detection.Bounds.X, detection.Bounds.Y, detection.Bounds.Width, detection.Bounds.Height, new Rgba32(255, 40, 40));
            DrawCross(image, (int)Math.Round(detection.CenterX), (int)Math.Round(detection.CenterY), 7, new Rgba32(255, 40, 40));
        }

        if (roi is { IsEmpty: false } && openness.Found && openness.ApertureHeight > 0)
        {
            DrawHorizontalLine(image, roi.Value.X, roi.Value.Right, openness.UpperY, new Rgba32(60, 255, 90));
            DrawHorizontalLine(image, roi.Value.X, roi.Value.Right, openness.LowerY, new Rgba32(60, 255, 90));
        }

        var colors = new[]
        {
            new Rgba32(60, 255, 90),
            new Rgba32(255, 230, 40),
            new Rgba32(80, 150, 255),
            new Rgba32(255, 90, 255),
            new Rgba32(255, 150, 40)
        };

        for (var i = 0; i < brightCandidates.Count; i++)
        {
            var candidate = brightCandidates[i];
            var color = colors[i % colors.Length];
            DrawRectangle(image, candidate.Bounds.X, candidate.Bounds.Y, candidate.Bounds.Width, candidate.Bounds.Height, color);
            DrawCross(image, (int)Math.Round(candidate.CenterX), (int)Math.Round(candidate.CenterY), 4, color);
        }

        await image.SaveAsPngAsync(outPath);
    }

    private static async Task SavePupilDiameterOverlayAsync(
        string inputPath,
        PixelRect? roi,
        PupilDetection detection,
        PupilDiameterEstimate diameter,
        EyeOpennessDetection openness,
        string outPath)
    {
        using var image = await Image.LoadAsync<Rgba32>(inputPath);
        if (roi is { IsEmpty: false })
        {
            DrawRectangle(image, roi.Value.X, roi.Value.Y, roi.Value.Width, roi.Value.Height, new Rgba32(40, 220, 255));
        }

        var quality = IsQualityPupilDiameter(diameter, openness);
        var pupilColor = quality
            ? new Rgba32(60, 255, 90)
            : diameter.Found ? new Rgba32(255, 220, 40) : new Rgba32(255, 60, 60);

        if (detection.Found)
        {
            DrawRectangle(image, detection.Bounds.X, detection.Bounds.Y, detection.Bounds.Width, detection.Bounds.Height, pupilColor);
            DrawCross(image, (int)Math.Round(detection.CenterX), (int)Math.Round(detection.CenterY), 6, pupilColor);
        }

        if (diameter.Found)
        {
            DrawOrientedEllipse(
                image,
                diameter.CenterX,
                diameter.CenterY,
                diameter.MajorAxisPx,
                diameter.MinorAxisPx,
                diameter.OrientationRadians,
                pupilColor);
        }

        if (roi is { IsEmpty: false } && openness.Found && openness.ApertureHeight > 0)
        {
            DrawHorizontalLine(image, roi.Value.X, roi.Value.Right, openness.UpperY, new Rgba32(80, 180, 255));
            DrawHorizontalLine(image, roi.Value.X, roi.Value.Right, openness.LowerY, new Rgba32(80, 180, 255));
        }

        await image.SaveAsPngAsync(outPath);
    }

    private static void AppendDiagnosticRows(
        StringBuilder csv,
        string file,
        string stage,
        EyeSide side,
        PupilDetection detection,
        EyeOpennessDetection openness,
        IReadOnlyList<BrightSpotCandidate> brightCandidates)
    {
        if (brightCandidates.Count == 0)
        {
            csv.AppendLine(BuildDiagnosticRow(file, stage, side, detection, openness, null, 0));
            return;
        }

        for (var i = 0; i < brightCandidates.Count; i++)
        {
            csv.AppendLine(BuildDiagnosticRow(file, stage, side, detection, openness, brightCandidates[i], i + 1));
        }
    }

    private static string BuildDiagnosticRow(
        string file,
        string stage,
        EyeSide side,
        PupilDetection detection,
        EyeOpennessDetection openness,
        BrightSpotCandidate? bright,
        int brightRank)
    {
        var bounds = detection.Found ? detection.Bounds : new PixelRect(0, 0, 0, 0);
        var baseColumns = JoinCsv(new[]
        {
            Path.GetFileName(file),
            stage,
            side.ToString().ToLowerInvariant(),
            detection.Found.ToString(),
            F(detection.Confidence),
            F(detection.CenterX),
            F(detection.CenterY),
            bounds.X.ToString(CultureInfo.InvariantCulture),
            bounds.Y.ToString(CultureInfo.InvariantCulture),
            bounds.Width.ToString(CultureInfo.InvariantCulture),
            bounds.Height.ToString(CultureInfo.InvariantCulture),
            detection.Area.ToString(CultureInfo.InvariantCulture),
            F(detection.FillRatio),
            F(openness.Openness),
            openness.UpperY.ToString(CultureInfo.InvariantCulture),
            openness.LowerY.ToString(CultureInfo.InvariantCulture),
            openness.ApertureHeight.ToString(CultureInfo.InvariantCulture),
            F(openness.PeakDarkFraction),
            F(openness.MeanDarkFraction),
            openness.DarkThreshold.ToString(CultureInfo.InvariantCulture),
            openness.Reason
        });

        if (bright is null)
        {
            return baseColumns + new string(',', 10);
        }

        return baseColumns + "," + JoinCsv(new[]
        {
            brightRank.ToString(CultureInfo.InvariantCulture),
            bright.Area.ToString(CultureInfo.InvariantCulture),
            F(bright.CenterX),
            F(bright.CenterY),
            bright.PeakIntensity.ToString(CultureInfo.InvariantCulture),
            F(bright.MeanIntensity),
            bright.Bounds.X.ToString(CultureInfo.InvariantCulture),
            bright.Bounds.Y.ToString(CultureInfo.InvariantCulture),
            bright.Bounds.Width.ToString(CultureInfo.InvariantCulture),
            bright.Bounds.Height.ToString(CultureInfo.InvariantCulture)
        });
    }

    private static string JoinCsv(IEnumerable<string> columns) => string.Join(",", columns.Select(EscapeCsv));

    private static string EscapeCsv(string value)
    {
        if (value.Contains(',') || value.Contains('"') || value.Contains('\n') || value.Contains('\r'))
        {
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        return value;
    }

    private static void DrawRectangle(Image<Rgba32> image, int x, int y, int width, int height, Rgba32 color)
    {
        for (var px = x; px < x + width; px++)
        {
            SetPixelSafe(image, px, y, color);
            SetPixelSafe(image, px, y + height - 1, color);
        }

        for (var py = y; py < y + height; py++)
        {
            SetPixelSafe(image, x, py, color);
            SetPixelSafe(image, x + width - 1, py, color);
        }
    }

    private static void DrawCross(Image<Rgba32> image, int x, int y, int radius, Rgba32 color)
    {
        for (var d = -radius; d <= radius; d++)
        {
            SetPixelSafe(image, x + d, y, color);
            SetPixelSafe(image, x, y + d, color);
        }
    }

    private static void DrawHorizontalLine(Image<Rgba32> image, int x0, int x1, int y, Rgba32 color)
    {
        for (var x = x0; x <= x1; x++)
        {
            SetPixelSafe(image, x, y, color);
        }
    }

    private static void DrawOrientedEllipse(
        Image<Rgba32> image,
        double centerX,
        double centerY,
        double majorAxis,
        double minorAxis,
        double orientation,
        Rgba32 color)
    {
        if (!double.IsFinite(centerX) ||
            !double.IsFinite(centerY) ||
            !double.IsFinite(majorAxis) ||
            !double.IsFinite(minorAxis) ||
            majorAxis <= 0 ||
            minorAxis <= 0)
        {
            return;
        }

        if (!double.IsFinite(orientation))
        {
            orientation = 0;
        }

        var cosAngle = Math.Cos(orientation);
        var sinAngle = Math.Sin(orientation);
        var majorRadius = majorAxis * 0.5;
        var minorRadius = minorAxis * 0.5;
        double? previousX = null;
        double? previousY = null;

        for (var i = 0; i <= 96; i++)
        {
            var theta = i * 2 * Math.PI / 96;
            var localX = Math.Cos(theta) * majorRadius;
            var localY = Math.Sin(theta) * minorRadius;
            var x = centerX + (localX * cosAngle) - (localY * sinAngle);
            var y = centerY + (localX * sinAngle) + (localY * cosAngle);
            if (previousX is not null && previousY is not null)
            {
                DrawLine(image, previousX.Value, previousY.Value, x, y, color);
            }

            previousX = x;
            previousY = y;
        }

        DrawLine(
            image,
            centerX - (majorRadius * cosAngle),
            centerY - (majorRadius * sinAngle),
            centerX + (majorRadius * cosAngle),
            centerY + (majorRadius * sinAngle),
            color);
        DrawLine(
            image,
            centerX + (minorRadius * sinAngle),
            centerY - (minorRadius * cosAngle),
            centerX - (minorRadius * sinAngle),
            centerY + (minorRadius * cosAngle),
            color);
    }

    private static void DrawLine(Image<Rgba32> image, double x0, double y0, double x1, double y1, Rgba32 color)
    {
        var steps = (int)Math.Ceiling(Math.Max(Math.Abs(x1 - x0), Math.Abs(y1 - y0)));
        if (steps <= 0)
        {
            SetPixelSafe(image, (int)Math.Round(x0), (int)Math.Round(y0), color);
            return;
        }

        for (var i = 0; i <= steps; i++)
        {
            var t = i / (double)steps;
            var x = x0 + ((x1 - x0) * t);
            var y = y0 + ((y1 - y0) * t);
            SetPixelSafe(image, (int)Math.Round(x), (int)Math.Round(y), color);
        }
    }

    private static void SetPixelSafe(Image<Rgba32> image, int x, int y, Rgba32 color)
    {
        if ((uint)x < image.Width && (uint)y < image.Height)
        {
            image[x, y] = color;
        }
    }

    private static async Task SavePgmAsync(EyeFrame frame, string path)
    {
        await using var stream = File.Create(path);
        await using var writer = new StreamWriter(stream, leaveOpen: true);
        await writer.WriteAsync($"P5\n{frame.Width} {frame.Height}\n255\n");
        await writer.FlushAsync();
        await stream.WriteAsync(frame.Pixels.AsMemory(0, frame.Width * frame.Height));
    }

    private static EyeSide GuessSide(string file)
    {
        var name = Path.GetFileNameWithoutExtension(file).ToLowerInvariant();
        var normalized = "_" + name.Replace('-', '_').Replace(' ', '_') + "_";
        if (normalized.Contains("_right_") || normalized.Contains("_r_"))
        {
            return EyeSide.Right;
        }

        if (normalized.Contains("_left_") || normalized.Contains("_l_"))
        {
            return EyeSide.Left;
        }

        return name.Contains("right", StringComparison.OrdinalIgnoreCase) ? EyeSide.Right : EyeSide.Left;
    }

    private static string ExtractStageName(string file)
    {
        var name = Path.GetFileNameWithoutExtension(file);
        var underscore = name.IndexOf('_', StringComparison.Ordinal);
        return underscore > 0 ? name[..underscore] : "unknown";
    }

    private static EyeSide ParseSide(string value)
    {
        return value.Equals("right", StringComparison.OrdinalIgnoreCase) ? EyeSide.Right : EyeSide.Left;
    }

    private static bool IsImageFile(string file)
    {
        var ext = Path.GetExtension(file).ToLowerInvariant();
        return ext is ".jpg" or ".jpeg" or ".png" or ".bmp";
    }

    private static string? GetOption(string[] args, string name)
    {
        for (var i = 0; i < args.Length; i++)
        {
            if (args[i] == name && i + 1 < args.Length)
            {
                return args[i + 1];
            }
        }

        return null;
    }

    private static bool HasFlag(string[] args, string name) =>
        args.Any(arg => arg.Equals(name, StringComparison.OrdinalIgnoreCase));

    private static string RequireOption(string[] args, string name)
    {
        return GetOption(args, name) ?? throw new ArgumentException($"Missing required option {name}.");
    }

    private static List<PairRow> LoadPairRows(string csvPath)
    {
        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException("pairs.csv not found.", csvPath);
        }

        var rows = new List<PairRow>();
        using var reader = new StreamReader(csvPath);
        var header = reader.ReadLine();
        if (header is null)
        {
            return rows;
        }

        var columns = header.Split(',');
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var values = line.Split(',');
            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (var i = 0; i < Math.Min(columns.Length, values.Length); i++)
            {
                map[columns[i]] = values[i];
            }

            rows.Add(PairRow.FromMap(map));
        }

        return rows;
    }

    private static List<StagePairRow> LoadStageRows(string csvPath)
    {
        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException("calibration_pairs.csv not found.", csvPath);
        }

        var rows = new List<StagePairRow>();
        using var reader = new StreamReader(csvPath);
        var header = reader.ReadLine();
        if (header is null)
        {
            return rows;
        }

        var columns = header.Split(',');
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var values = line.Split(',');
            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (var i = 0; i < Math.Min(columns.Length, values.Length); i++)
            {
                map[columns[i]] = values[i];
            }

            rows.Add(StagePairRow.FromMap(map));
        }

        return rows;
    }

    private static List<PupilDiameterPairRow> LoadPupilDiameterPairRows(string csvPath)
    {
        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException("pupil_diameter_pairs.csv not found.", csvPath);
        }

        var rows = new List<PupilDiameterPairRow>();
        foreach (var map in LoadCsvMaps(csvPath))
        {
            rows.Add(PupilDiameterPairRow.FromMap(map));
        }

        return rows;
    }

    private static IEnumerable<Dictionary<string, string>> LoadCsvMaps(string csvPath)
    {
        using var reader = new StreamReader(csvPath);
        var header = reader.ReadLine();
        if (header is null)
        {
            yield break;
        }

        var columns = header.Split(',');
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var values = line.Split(',');
            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (var i = 0; i < Math.Min(columns.Length, values.Length); i++)
            {
                map[columns[i]] = values[i];
            }

            yield return map;
        }
    }

    private static double Percentile(IReadOnlyList<double> sortedValues, double percentile)
    {
        if (sortedValues.Count == 0)
        {
            return double.NaN;
        }

        var index = Math.Clamp((int)Math.Round((sortedValues.Count - 1) * percentile), 0, sortedValues.Count - 1);
        return sortedValues[index];
    }

    private static void SetDoubleIfPresent(string[] args, string name, Action<double> setter)
    {
        var raw = GetOption(args, name);
        if (raw is not null)
        {
            setter(double.Parse(raw, CultureInfo.InvariantCulture));
        }
    }

    private static double Median(IEnumerable<double> values)
    {
        var sorted = values.OrderBy(static x => x).ToArray();
        if (sorted.Length == 0)
        {
            return double.NaN;
        }

        var mid = sorted.Length / 2;
        return sorted.Length % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
    }

    private static string F(double value) => value.ToString("0.###", CultureInfo.InvariantCulture);

    private static bool ParseBool(IReadOnlyDictionary<string, string> map, string key) =>
        map.TryGetValue(key, out var raw) && bool.TryParse(raw, out var value) && value;

    private static int ParseInt(IReadOnlyDictionary<string, string> map, string key)
    {
        return map.TryGetValue(key, out var raw) && int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
            ? value
            : 0;
    }

    private static double ParseDouble(IReadOnlyDictionary<string, string> map, string key)
    {
        if (!map.TryGetValue(key, out var raw) || raw.Equals("NaN", StringComparison.OrdinalIgnoreCase))
        {
            return double.NaN;
        }

        return double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) && double.IsFinite(value)
            ? value
            : double.NaN;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("""
        DreamAirTracking CLI

        Commands:
          analyze-samples <dir> [--debug <dir>]
          pupil-diameter-samples <dir> [--profile tracking_profile_final.json] [--out pupil_diameter.csv] [--debug overlays]
          diagnose-session <dir> [--profile tracking_profile_final.json] [--out diagnostics] [--limit-per-stage 8] [--bright-threshold 235]
          init-profile --profile tracking_profile.json
          profile-from-samples <dir> --profile tracking_profile.json [--margin 35]
          calibration-from-capture <dir> --profile tracking_profile.json [--min-conf 0.06] [--report calibration_report.csv]
          calibration-from-session <dir> --profile tracking_profile_calibrated.json [--min-conf 0.05] [--report stage_calibration_report.csv]
          fit-gaze-session <dir> [--profile tracking_profile_final.json] [--out-profile generated_profile.json] [--min-conf 0.05] [--min-open 0.20] [--max-delta-ms 20]
          set-roi --profile tracking_profile.json --side left|right --x 0 --y 0 --w 200 --h 200
          set-calibration --profile tracking_profile.json --side left|right [--center-x 100] [--center-y 100] [--range-x 40] [--range-y 30]
          capture-http [--profile tracking_profile.json] [--host 127.0.0.1] [--port 5555] [--frames 30] [--out capture_http]
          live-http [--profile tracking_profile.json] [--host 127.0.0.1] [--port 5555] [--pairs 120] [--max-delta-ms 20] [--out live_http]
          pupil-diameter-session-http [--profile tracking_profile_final.json] [--host 127.0.0.1] [--port 5555] [--stage-pairs 90] [--stages center,bright,dark,blink] [--manual-stage] [--settle-seconds 5] [--summary-tail-pairs 30] [--debug-overlays] [--overlay-every 15] [--out pupil_diameter_session]
          fit-pupil-diameter-session <dir> [--tail-pairs 30] [--min-conf 0.28] [--max-axis-ratio 1.8] [--min-open 0.75] [--bright-stages bright,bright2] [--dark-stages dark,dark2] [--out calibration.json] [--report report.csv]
          calibration-session-http [--profile tracking_profile.json] [--host 127.0.0.1] [--port 5555] [--stage-pairs 30] [--stages center,left,right,up,down,blink,open] [--out calibration_session]
          capture-raw  [--profile tracking_profile.json] [--host 127.0.0.1] [--port 5555] [--side left|right] [--frames 30] [--out capture_raw]
        """);
    }

    private sealed record CapturedJpeg(EyeSide Side, int Index, byte[] Jpeg, EyeFrame Frame, DateTimeOffset ReceivedAt);

    private sealed record PupilDiameterCalibrationFit(
        DateTimeOffset GeneratedAtUtc,
        string SourceSession,
        int TailPairs,
        double MinConfidence,
        double MaxAxisRatio,
        double MinOpenness,
        IReadOnlyList<string> BrightStages,
        IReadOnlyList<string> DarkStages,
        PupilDiameterEyeCalibrationFit Left,
        PupilDiameterEyeCalibrationFit Right);

    private sealed record PupilDiameterEyeCalibrationFit(
        string Side,
        bool Valid,
        double BrightPx,
        double DarkPx,
        double RangePx,
        double RelativeRange,
        double BrightSpreadPx,
        double DarkSpreadPx,
        bool RepeatabilityWarning,
        IReadOnlyList<PupilDiameterStageFit> Stages);

    private sealed record PupilDiameterStageFit(
        string Stage,
        string Side,
        int TotalPairs,
        int TailPairs,
        int QualityPairs,
        double QualityRate,
        double P10Px,
        double MedianPx,
        double P90Px,
        double AxisRatioMedian,
        double ConfidenceMedian,
        double CenterXMedian,
        double CenterYMedian);

    private sealed record PupilDiameterPairRow(
        string Stage,
        int Pair,
        PupilDiameterEyePair Left,
        PupilDiameterEyePair Right)
    {
        public static PupilDiameterPairRow FromMap(IReadOnlyDictionary<string, string> map)
        {
            return new PupilDiameterPairRow(
                map.TryGetValue("stage", out var stage) ? stage : "",
                ParseInt(map, "pair"),
                PupilDiameterEyePair.FromMap(map, "left"),
                PupilDiameterEyePair.FromMap(map, "right"));
        }

        public PupilDiameterEyePair GetEye(string side) => side == "left" ? Left : Right;
    }

    private sealed record PupilDiameterEyePair(
        bool Found,
        double Confidence,
        double DiameterPx,
        double AxisRatio,
        double Openness,
        double CenterX,
        double CenterY)
    {
        public static PupilDiameterEyePair FromMap(IReadOnlyDictionary<string, string> map, string prefix)
        {
            return new PupilDiameterEyePair(
                ParseBool(map, $"{prefix}_diameter_found"),
                ParseDouble(map, $"{prefix}_diameter_conf"),
                ParseDouble(map, $"{prefix}_diameter_px"),
                ParseDouble(map, $"{prefix}_axis_ratio"),
                ParseDouble(map, $"{prefix}_openness"),
                ParseDouble(map, $"{prefix}_center_x"),
                ParseDouble(map, $"{prefix}_center_y"));
        }

        public bool IsQuality(double minConfidence, double maxAxisRatio, double minOpenness)
        {
            return Found &&
                double.IsFinite(DiameterPx) &&
                double.IsFinite(AxisRatio) &&
                Confidence >= minConfidence &&
                AxisRatio <= maxAxisRatio &&
                Openness >= minOpenness;
        }
    }

    private sealed record PupilDiameterStageMeasurement(
        string Stage,
        int Pair,
        string Side,
        bool Found,
        bool Quality,
        double Confidence,
        double DiameterPx,
        double AxisRatio,
        double Openness);

    private sealed record StageStat(
        string Stage,
        string Side,
        int TotalPoints,
        int ValidPoints,
        double? MedianX,
        double? MedianY,
        double AverageConfidence)
    {
        public bool HasPoint => MedianX is not null && MedianY is not null;
    }

    private sealed record StagePairRow(
        string Stage,
        bool LeftFound,
        double LeftConfidence,
        double? LeftX,
        double? LeftY,
        bool RightFound,
        double RightConfidence,
        double? RightX,
        double? RightY)
    {
        public static StagePairRow FromMap(IReadOnlyDictionary<string, string> map)
        {
            return new StagePairRow(
                map.TryGetValue("stage", out var stage) ? stage : "",
                ParseBool(map, "left_found"),
                ParseDouble(map, "left_conf") ?? 0,
                ParseDouble(map, "left_x"),
                ParseDouble(map, "left_y"),
                ParseBool(map, "right_found"),
                ParseDouble(map, "right_conf") ?? 0,
                ParseDouble(map, "right_x"),
                ParseDouble(map, "right_y"));
        }

        public bool GetFound(string side) => side == "left" ? LeftFound : RightFound;
        public double GetConfidence(string side) => side == "left" ? LeftConfidence : RightConfidence;
        public double? GetX(string side) => side == "left" ? LeftX : RightX;
        public double? GetY(string side) => side == "left" ? LeftY : RightY;

        private static bool ParseBool(IReadOnlyDictionary<string, string> map, string key) =>
            map.TryGetValue(key, out var raw) && bool.TryParse(raw, out var value) && value;

        private static double? ParseDouble(IReadOnlyDictionary<string, string> map, string key)
        {
            if (!map.TryGetValue(key, out var raw) || raw.Equals("NaN", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            return double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) && double.IsFinite(value)
                ? value
                : null;
        }
    }

    private sealed record PairRow(
        bool LeftFound,
        double LeftConfidence,
        double? LeftX,
        double? LeftY,
        bool RightFound,
        double RightConfidence,
        double? RightX,
        double? RightY)
    {
        public static PairRow FromMap(IReadOnlyDictionary<string, string> map)
        {
            return new PairRow(
                ParseBool(map, "left_found"),
                ParseDouble(map, "left_conf") ?? 0,
                ParseDouble(map, "left_x"),
                ParseDouble(map, "left_y"),
                ParseBool(map, "right_found"),
                ParseDouble(map, "right_conf") ?? 0,
                ParseDouble(map, "right_x"),
                ParseDouble(map, "right_y"));
        }

        public bool GetFound(string side) => side == "left" ? LeftFound : RightFound;
        public double GetConfidence(string side) => side == "left" ? LeftConfidence : RightConfidence;
        public double? GetX(string side) => side == "left" ? LeftX : RightX;
        public double? GetY(string side) => side == "left" ? LeftY : RightY;

        private static bool ParseBool(IReadOnlyDictionary<string, string> map, string key) =>
            map.TryGetValue(key, out var raw) && bool.TryParse(raw, out var value) && value;

        private static double? ParseDouble(IReadOnlyDictionary<string, string> map, string key)
        {
            if (!map.TryGetValue(key, out var raw) || raw.Equals("NaN", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            return double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) && double.IsFinite(value)
                ? value
                : null;
        }
    }
}
