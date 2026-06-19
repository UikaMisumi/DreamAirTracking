using System.Text.Json;
using System.Text.Json.Nodes;
using DreamAirTracking.Core.Bridge;

namespace DreamAirTracking.App.Services;

public sealed class WearTemplateService
{
    private const int RequiredPairs = 60;
    private const double RequiredAverageOpenness = 0.55;
    private const double RequiredMedianConfidence = 0.12;
    private const double RequiredPupilCenterStd = 0.035;
    private const double GoodMatchThreshold = 0.18;
    private const double WeakMatchThreshold = 0.32;
    private const double DefaultLeftPupilX = 0.81;
    private const double DefaultRightPupilX = 0.20;
    private const double DefaultPupilY = 0.54;

    private static readonly TimeSpan ProbeDuration = TimeSpan.FromSeconds(2.5);
    public async Task<WearTemplateProbeResult> ProbeAsync(string repoRoot, CancellationToken cancellationToken)
    {
        var samples = await CollectSamplesAsync(cancellationToken);
        if (samples.Count == 0)
        {
            return WearTemplateProbeResult.NotRun("no monitor packets");
        }

        var signature = WearSignature.FromSamples(samples);
        var requirementsMet = signature.SampleCount >= RequiredPairs
            && signature.AverageOpenness >= RequiredAverageOpenness
            && signature.MedianConfidence >= RequiredMedianConfidence
            && signature.PupilCenterStd <= RequiredPupilCenterStd;

        var requirementsReason =
            $"usable={signature.SampleCount}/{samples.Count}, open={signature.AverageOpenness:0.00}, confidence={signature.MedianConfidence:0.00}, pupilStd={signature.PupilCenterStd:0.000}, rawStd={signature.RawGazeStd:0.00}";

        if (!requirementsMet)
        {
            return new WearTemplateProbeResult(
                "insufficient",
                "suggest_5_point_calibration",
                null,
                null,
                signature.SampleCount,
                false,
                requirementsReason);
        }

        var templates = LoadTemplates(repoRoot);
        if (templates.Count == 0)
        {
            return new WearTemplateProbeResult(
                "new",
                "run_5_point_calibration",
                null,
                null,
                signature.SampleCount,
                true,
                "no templates");
        }

        var best = templates
            .Select(template => new
            {
                Template = template,
                Distance = Distance(signature, template)
            })
            .OrderBy(item => item.Distance)
            .First();

        var status = best.Distance <= GoodMatchThreshold
            ? "good"
            : best.Distance <= WeakMatchThreshold
                ? "weak"
                : "new";
        var action = status switch
        {
            "good" => "load_template",
            "weak" => "suggest_5_point_calibration",
            _ => "run_5_point_calibration"
        };

        return new WearTemplateProbeResult(
            status,
            action,
            best.Template.Id,
            best.Distance,
            signature.SampleCount,
            true,
            requirementsReason,
            best.Template.RuntimeCalibration?.CenterOffsetX,
            best.Template.RuntimeCalibration?.CenterOffsetY,
            best.Template.RuntimeCalibration?.XGain,
            best.Template.RuntimeCalibration?.YGain,
            best.Template.RuntimeCalibration?.Path);
    }

    public bool BindRuntimeCalibration(
        string repoRoot,
        string templateId,
        string runtimeCalibrationPath,
        double centerOffsetX,
        double centerOffsetY,
        double xGain,
        double yGain,
        int sampleCount)
    {
        if (string.IsNullOrWhiteSpace(templateId))
        {
            return false;
        }

        var path = Path.Combine(repoRoot, "runs", "wear_templates", "wear_templates.json");
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            var root = JsonNode.Parse(File.ReadAllText(path))?.AsObject();
            if (root is null || root["templates"] is not JsonArray templates)
            {
                return false;
            }

            foreach (var item in templates)
            {
                if (item is not JsonObject template)
                {
                    continue;
                }

                var id = template["id"]?.GetValue<string>();
                if (!string.Equals(id, templateId, StringComparison.Ordinal))
                {
                    continue;
                }

                var centerOffset = new JsonArray(centerOffsetX, centerOffsetY);
                var gain = new JsonArray(xGain, yGain);
                var updatedAt = DateTimeOffset.UtcNow.ToString("O");
                template["center_offset"] = new JsonArray(centerOffsetX, centerOffsetY);
                template["gain"] = new JsonArray(xGain, yGain);
                template["runtime_calibration"] = new JsonObject
                {
                    ["schema"] = "dream_air_tracking.runtime_template_calibration.v1",
                    ["updated_at"] = updatedAt,
                    ["path"] = runtimeCalibrationPath,
                    ["center_offset"] = centerOffset,
                    ["gain"] = gain,
                    ["sample_count"] = sampleCount,
                    ["reason"] = "ok"
                };
                template["updated_at"] = updatedAt;
                root["updated_at"] = updatedAt;
                File.WriteAllText(path, root.ToJsonString(AppJsonOptions.Web(writeIndented: true)));
                return true;
            }
        }
        catch
        {
            return false;
        }

        return false;
    }

    private static async Task<List<BridgeTrackingState>> CollectSamplesAsync(CancellationToken cancellationToken)
    {
        var samples = new List<BridgeTrackingState>();
        var seen = new HashSet<long>();

        void OnStateReceived(object? _, BridgeTrackingState state)
        {
            if (seen.Add(state.Sequence))
            {
                samples.Add(state);
            }
        }

        BridgeMonitorService.Instance.StateReceived += OnStateReceived;
        try
        {
            if (BridgeMonitorService.Instance.LatestState is not null
                && BridgeMonitorService.Instance.LatestStateAt is not null
                && DateTimeOffset.Now - BridgeMonitorService.Instance.LatestStateAt.Value <= TimeSpan.FromSeconds(1))
            {
                OnStateReceived(null, BridgeMonitorService.Instance.LatestState);
            }

            await Task.Delay(ProbeDuration, cancellationToken);
            return samples;
        }
        catch (OperationCanceledException)
        {
            return samples;
        }
        finally
        {
            BridgeMonitorService.Instance.StateReceived -= OnStateReceived;
        }
    }

    private static IReadOnlyList<WearTemplate> LoadTemplates(string repoRoot)
    {
        var path = Path.Combine(repoRoot, "runs", "wear_templates", "wear_templates.json");
        if (!File.Exists(path))
        {
            return Array.Empty<WearTemplate>();
        }

        try
        {
            var json = File.ReadAllText(path);
            using var document = JsonDocument.Parse(json);
            if (!document.RootElement.TryGetProperty("templates", out var templatesElement)
                || templatesElement.ValueKind != JsonValueKind.Array)
            {
                return Array.Empty<WearTemplate>();
            }

            return templatesElement
                .EnumerateArray()
                .Select(ReadTemplate)
                .Where(template => !string.IsNullOrWhiteSpace(template.Id))
                .ToArray();
        }
        catch
        {
            return Array.Empty<WearTemplate>();
        }
    }

    private static WearTemplate ReadTemplate(JsonElement element) => new()
    {
        Id = ReadString(element, "id"),
        LeftPupilCenterMedian = ReadVector(element, "left_pupil_center_median", DefaultLeftPupilX, DefaultPupilY),
        RightPupilCenterMedian = ReadVector(element, "right_pupil_center_median", DefaultRightPupilX, DefaultPupilY),
        LeftCropShiftMedian = ReadVector(element, "left_crop_shift_median", 0, 0),
        RightCropShiftMedian = ReadVector(element, "right_crop_shift_median", 0, 0),
        OpenOpennessMedian = ReadVector(element, "open_openness_median", 1, 1),
        ConfidenceMedian = ReadVector(element, "confidence_median", 1, 1),
        RawGazeCenterMedian = ReadVector(element, "raw_gaze_center_median", 0, 0),
        RuntimeCalibration = ReadRuntimeCalibration(element)
    };

    private static string ReadString(JsonElement element, string property) =>
        element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : string.Empty;

    private static IReadOnlyList<double> ReadVector(JsonElement element, string property, double fallbackX, double fallbackY)
    {
        if (!element.TryGetProperty(property, out var value)
            || value.ValueKind != JsonValueKind.Array
            || value.GetArrayLength() < 2)
        {
            return new[] { fallbackX, fallbackY };
        }

        var first = value[0].ValueKind == JsonValueKind.Number && value[0].TryGetDouble(out var x) ? x : fallbackX;
        var second = value[1].ValueKind == JsonValueKind.Number && value[1].TryGetDouble(out var y) ? y : fallbackY;
        return new[] { first, second };
    }

    private static double Distance(WearSignature signature, WearTemplate template)
    {
        var pupil = (Distance(signature.LeftPupilCenterMedian, template.LeftPupilCenterMedian)
                     + Distance(signature.RightPupilCenterMedian, template.RightPupilCenterMedian)) * 0.5;
        var crop = (Distance(signature.LeftCropShiftMedian, template.LeftCropShiftMedian)
                    + Distance(signature.RightCropShiftMedian, template.RightCropShiftMedian)) * 0.5;
        var raw = Distance(signature.RawGazeCenterMedian, template.RawGazeCenterMedian);
        var openness = Distance(signature.OpenOpennessMedian, template.OpenOpennessMedian);
        var confidence = Math.Abs(Asymmetry(signature.ConfidenceMedian) - Asymmetry(template.ConfidenceMedian));
        return pupil * 0.45 + crop * 0.30 + raw * 0.05 + openness * 0.10 + confidence * 0.10;
    }

    private static RuntimeTemplateCalibration? ReadRuntimeCalibration(JsonElement element)
    {
        if (!element.TryGetProperty("runtime_calibration", out var calibration)
            || calibration.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        var centerOffset = ReadVector(calibration, "center_offset", 0, 0);
        var gain = ReadVector(calibration, "gain", 1, 1);
        if (centerOffset.Count < 2 || gain.Count < 2
            || !IsFinite(centerOffset[0]) || !IsFinite(centerOffset[1])
            || !IsFinite(gain[0]) || !IsFinite(gain[1]))
        {
            return null;
        }

        return new RuntimeTemplateCalibration(
            centerOffset[0],
            centerOffset[1],
            gain[0],
            gain[1],
            ReadString(calibration, "path"));
    }

    private static double Distance(IReadOnlyList<double> left, IReadOnlyList<double> right)
    {
        if (left.Count != right.Count || left.Count == 0)
        {
            return 1;
        }

        var sum = 0.0;
        for (var index = 0; index < left.Count; index++)
        {
            var delta = left[index] - right[index];
            sum += delta * delta;
        }

        return Math.Sqrt(sum);
    }

    private static double Asymmetry(IReadOnlyList<double> values) =>
        values.Count < 2 ? 0 : Math.Abs(values[0] - values[1]);

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    private sealed record WearTemplate
    {
        public string Id { get; init; } = string.Empty;
        public IReadOnlyList<double> LeftPupilCenterMedian { get; init; } = new[] { DefaultLeftPupilX, DefaultPupilY };
        public IReadOnlyList<double> RightPupilCenterMedian { get; init; } = new[] { DefaultRightPupilX, DefaultPupilY };
        public IReadOnlyList<double> LeftCropShiftMedian { get; init; } = new[] { 0.0, 0.0 };
        public IReadOnlyList<double> RightCropShiftMedian { get; init; } = new[] { 0.0, 0.0 };
        public IReadOnlyList<double> OpenOpennessMedian { get; init; } = new[] { 1.0, 1.0 };
        public IReadOnlyList<double> ConfidenceMedian { get; init; } = new[] { 1.0, 1.0 };
        public IReadOnlyList<double> RawGazeCenterMedian { get; init; } = new[] { 0.0, 0.0 };
        public RuntimeTemplateCalibration? RuntimeCalibration { get; init; }
    }

    private sealed record RuntimeTemplateCalibration(
        double CenterOffsetX,
        double CenterOffsetY,
        double XGain,
        double YGain,
        string Path);

    private sealed record WearSignature(
        int SampleCount,
        double AverageOpenness,
        double MedianConfidence,
        double RawGazeStd,
        double PupilCenterStd,
        IReadOnlyList<double> LeftPupilCenterMedian,
        IReadOnlyList<double> RightPupilCenterMedian,
        IReadOnlyList<double> LeftCropShiftMedian,
        IReadOnlyList<double> RightCropShiftMedian,
        IReadOnlyList<double> OpenOpennessMedian,
        IReadOnlyList<double> ConfidenceMedian,
        IReadOnlyList<double> RawGazeCenterMedian)
    {
        public static WearSignature FromSamples(IReadOnlyList<BridgeTrackingState> samples)
        {
            var usable = samples
                .Where(state => state.Left.NormalizationFound && state.Right.NormalizationFound)
                .ToArray();
            var rawX = usable.Select(RawX).ToArray();
            var rawY = usable.Select(RawY).ToArray();
            var leftConfidence = usable.Select(state => state.Left.NormalizationConfidence).ToArray();
            var rightConfidence = usable.Select(state => state.Right.NormalizationConfidence).ToArray();
            var pairConfidence = leftConfidence.Zip(rightConfidence, (left, right) => (left + right) * 0.5).ToArray();
            var averageOpenness = usable.Select(state => (state.Left.Openness + state.Right.Openness) * 0.5).DefaultIfEmpty(0).Average();
            var rawStd = Math.Sqrt((Variance(rawX) + Variance(rawY)) * 0.5);
            var pupilStd = Math.Sqrt(
                (
                    Variance(usable.Select(state => state.Left.NormalizationPupilX).ToArray())
                    + Variance(usable.Select(state => state.Left.NormalizationPupilY).ToArray())
                    + Variance(usable.Select(state => state.Right.NormalizationPupilX).ToArray())
                    + Variance(usable.Select(state => state.Right.NormalizationPupilY).ToArray())
                ) * 0.25);
            return new WearSignature(
                usable.Length,
                averageOpenness,
                Median(pairConfidence, 0),
                rawStd,
                pupilStd,
                new[] { Median(usable.Select(state => state.Left.NormalizationPupilX), DefaultLeftPupilX), Median(usable.Select(state => state.Left.NormalizationPupilY), DefaultPupilY) },
                new[] { Median(usable.Select(state => state.Right.NormalizationPupilX), DefaultRightPupilX), Median(usable.Select(state => state.Right.NormalizationPupilY), DefaultPupilY) },
                new[] { Median(usable.Select(state => state.Left.NormalizationShiftX), 0), Median(usable.Select(state => state.Left.NormalizationShiftY), 0) },
                new[] { Median(usable.Select(state => state.Right.NormalizationShiftX), 0), Median(usable.Select(state => state.Right.NormalizationShiftY), 0) },
                new[] { Median(usable.Select(state => state.Left.Openness), 1), Median(usable.Select(state => state.Right.Openness), 1) },
                new[] { Median(leftConfidence, 1), Median(rightConfidence, 1) },
                new[] { Median(rawX, 0), Median(rawY, 0) });
        }

        private static double RawX(BridgeTrackingState state) =>
            (RuntimeRawX(state.Left) + RuntimeRawX(state.Right)) * 0.5;

        private static double RawY(BridgeTrackingState state) =>
            (RuntimeRawY(state.Left) + RuntimeRawY(state.Right)) * 0.5;

        private static double RuntimeRawX(BridgeEyeState eye) =>
            Math.Abs(eye.RawNormalizedX) > 0.0000001 ? eye.RawNormalizedX : eye.RawX;

        private static double RuntimeRawY(BridgeEyeState eye) =>
            Math.Abs(eye.RawNormalizedY) > 0.0000001 ? eye.RawNormalizedY : eye.RawY;

        private static double Median(IEnumerable<double> values, double fallback)
        {
            var clean = values.Where(IsFinite).OrderBy(value => value).ToArray();
            if (clean.Length == 0)
            {
                return fallback;
            }

            var middle = clean.Length / 2;
            return clean.Length % 2 == 1
                ? clean[middle]
                : (clean[middle - 1] + clean[middle]) * 0.5;
        }

        private static double Variance(IReadOnlyList<double> values)
        {
            var clean = values.Where(IsFinite).ToArray();
            if (clean.Length < 2)
            {
                return 0;
            }

            var mean = clean.Average();
            return clean.Select(value => (value - mean) * (value - mean)).Average();
        }

        private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
