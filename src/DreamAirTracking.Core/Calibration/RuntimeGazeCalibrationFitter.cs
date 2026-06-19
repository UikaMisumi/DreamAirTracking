namespace DreamAirTracking.Core.Calibration;

public sealed class RuntimeGazeCalibrationFitter
{
    private static readonly string[] RequiredStages = new[] { "center", "left", "right", "up", "down" };

    public RuntimeGazeCalibrationFit Fit(
        IEnumerable<RuntimeGazeCalibrationSample> samples,
        RuntimeGazeCalibrationOptions? options = null)
    {
        options ??= new RuntimeGazeCalibrationOptions();
        var byStage = samples
            .Where(static sample => sample.Accepted && IsFinite(sample.RawX) && IsFinite(sample.RawY))
            .GroupBy(static sample => sample.StageId, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                static group => group.Key,
                static group => group.ToList(),
                StringComparer.OrdinalIgnoreCase);

        var missing = RequiredStages
            .Where(stage => !byStage.TryGetValue(stage, out var stageSamples) || stageSamples.Count < options.MinSamplesPerStage)
            .ToArray();
        if (missing.Length > 0)
        {
            return RuntimeGazeCalibrationFit.Failed(
                $"Missing runtime samples: {string.Join(", ", missing)}.",
                missing);
        }

        var medians = byStage.ToDictionary(
            static item => item.Key,
            static item => MedianPoint(item.Value),
            StringComparer.OrdinalIgnoreCase);

        var xSpan = medians["right"].X - medians["left"].X;
        var ySpan = medians["up"].Y - medians["down"].Y;
        var geometryFailure = ValidateGeometry(medians, xSpan, ySpan, options);
        if (geometryFailure is not null)
        {
            return RuntimeGazeCalibrationFit.FailedWithDiagnostics(
                geometryFailure,
                Array.Empty<string>(),
                xSpan,
                ySpan,
                medians,
                byStage.Sum(static item => item.Value.Count),
                options.OffsetMode,
                options.ApplyFittedGain);
        }

        var offsetX = options.OffsetMode == RuntimeGazeOffsetMode.CenterStage
            ? medians["center"].X
            : (medians["left"].X + medians["right"].X) * 0.5;
        var offsetY = options.OffsetMode == RuntimeGazeOffsetMode.CenterStage
            ? medians["center"].Y
            : (medians["up"].Y + medians["down"].Y) * 0.5;
        var fittedXGain = Math.Abs(xSpan) < options.MinGainDenominator ? 1.0 : 2.0 / xSpan;
        var fittedYGain = Math.Abs(ySpan) < options.MinGainDenominator ? 1.0 : 2.0 / ySpan;
        var xGain = options.ApplyFittedGain ? fittedXGain : 1.0;
        var yGain = options.ApplyFittedGain ? fittedYGain : 1.0;

        return RuntimeGazeCalibrationFit.CreateSucceeded(
            offsetX,
            offsetY,
            xGain,
            yGain,
            fittedXGain,
            fittedYGain,
            xSpan,
            ySpan,
            medians,
            byStage.Sum(static item => item.Value.Count),
            options.OffsetMode,
            options.ApplyFittedGain);
    }

    private static RuntimeGazeStageMedian MedianPoint(IReadOnlyList<RuntimeGazeCalibrationSample> samples)
        => new(
            Median(samples.Select(static sample => sample.RawX).ToArray()),
            Median(samples.Select(static sample => sample.RawY).ToArray()),
            samples.Count);

    private static double Median(double[] values)
    {
        Array.Sort(values);
        var middle = values.Length / 2;
        return values.Length % 2 == 1
            ? values[middle]
            : (values[middle - 1] + values[middle]) * 0.5;
    }

    private static string? ValidateGeometry(
        IReadOnlyDictionary<string, RuntimeGazeStageMedian> medians,
        double xSpan,
        double ySpan,
        RuntimeGazeCalibrationOptions options)
    {
        if (xSpan < options.MinHorizontalSpan)
        {
            return $"Runtime calibration rejected: horizontal separation is too small ({xSpan:0.000} < {options.MinHorizontalSpan:0.000}).";
        }

        if (ySpan < options.MinVerticalSpan)
        {
            return $"Runtime calibration rejected: vertical separation is too small ({ySpan:0.000} < {options.MinVerticalSpan:0.000}).";
        }

        var center = medians["center"];
        if (medians["left"].X > center.X - options.MinDirectionalMargin)
        {
            return "Runtime calibration rejected: left point is not clearly left of center.";
        }

        if (medians["right"].X < center.X + options.MinDirectionalMargin)
        {
            return "Runtime calibration rejected: right point is not clearly right of center.";
        }

        if (medians["up"].Y < center.Y + options.MinDirectionalMargin)
        {
            return "Runtime calibration rejected: up point is not clearly above center.";
        }

        if (medians["down"].Y > center.Y - options.MinDirectionalMargin)
        {
            return "Runtime calibration rejected: down point is not clearly below center.";
        }

        return null;
    }

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}

public sealed record RuntimeGazeCalibrationSample(
    string StageId,
    double RawX,
    double RawY,
    long Sequence,
    DateTimeOffset Timestamp,
    bool Accepted = true);

public sealed record RuntimeGazeCalibrationOptions
{
    public int MinSamplesPerStage { get; init; } = 8;
    public double MinGainDenominator { get; init; } = 0.15;
    public double MinHorizontalSpan { get; init; } = 0.8;
    public double MinVerticalSpan { get; init; } = 0.8;
    public double MinDirectionalMargin { get; init; } = 0.12;
    public RuntimeGazeOffsetMode OffsetMode { get; init; } = RuntimeGazeOffsetMode.CenterStage;
    public bool ApplyFittedGain { get; init; }
}

public enum RuntimeGazeOffsetMode
{
    AxisMidpoint,
    CenterStage
}

public sealed record RuntimeGazeStageMedian(double X, double Y, int Count);

public sealed record RuntimeGazeCalibrationFit
{
    private RuntimeGazeCalibrationFit()
    {
    }

    public bool Succeeded { get; private init; }
    public string Reason { get; private init; } = string.Empty;
    public IReadOnlyList<string> MissingStages { get; private init; } = Array.Empty<string>();
    public double CenterOffsetX { get; private init; }
    public double CenterOffsetY { get; private init; }
    public double XGain { get; private init; } = 1.0;
    public double YGain { get; private init; } = 1.0;
    public double FittedXGain { get; private init; } = 1.0;
    public double FittedYGain { get; private init; } = 1.0;
    public double XSpan { get; private init; }
    public double YSpan { get; private init; }
    public int SampleCount { get; private init; }
    public RuntimeGazeOffsetMode OffsetMode { get; private init; }
    public bool ApplyFittedGain { get; private init; }
    public IReadOnlyDictionary<string, RuntimeGazeStageMedian> StageMedians { get; private init; }
        = new Dictionary<string, RuntimeGazeStageMedian>(StringComparer.OrdinalIgnoreCase);

    public static RuntimeGazeCalibrationFit Failed(string reason, IReadOnlyList<string> missingStages)
        => new()
        {
            Succeeded = false,
            Reason = reason,
            MissingStages = missingStages
        };

    public static RuntimeGazeCalibrationFit FailedWithDiagnostics(
        string reason,
        IReadOnlyList<string> missingStages,
        double xSpan,
        double ySpan,
        IReadOnlyDictionary<string, RuntimeGazeStageMedian> stageMedians,
        int sampleCount,
        RuntimeGazeOffsetMode offsetMode,
        bool applyFittedGain)
        => new()
        {
            Succeeded = false,
            Reason = reason,
            MissingStages = missingStages,
            XSpan = xSpan,
            YSpan = ySpan,
            StageMedians = stageMedians,
            SampleCount = sampleCount,
            OffsetMode = offsetMode,
            ApplyFittedGain = applyFittedGain
        };

    public static RuntimeGazeCalibrationFit CreateSucceeded(
        double centerOffsetX,
        double centerOffsetY,
        double xGain,
        double yGain,
        double fittedXGain,
        double fittedYGain,
        double xSpan,
        double ySpan,
        IReadOnlyDictionary<string, RuntimeGazeStageMedian> stageMedians,
        int sampleCount,
        RuntimeGazeOffsetMode offsetMode,
        bool applyFittedGain)
        => new()
        {
            Succeeded = true,
            Reason = "ok",
            CenterOffsetX = centerOffsetX,
            CenterOffsetY = centerOffsetY,
            XGain = xGain,
            YGain = yGain,
            FittedXGain = fittedXGain,
            FittedYGain = fittedYGain,
            XSpan = xSpan,
            YSpan = ySpan,
            StageMedians = stageMedians,
            SampleCount = sampleCount,
            OffsetMode = offsetMode,
            ApplyFittedGain = applyFittedGain
        };
}
