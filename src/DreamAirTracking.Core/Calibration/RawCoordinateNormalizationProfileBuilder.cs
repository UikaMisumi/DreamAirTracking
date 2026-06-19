using DreamAirTracking.Core;
using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Profiles;

namespace DreamAirTracking.Core.Calibration;

public sealed class RawCoordinateNormalizationProfileBuilder
{
    private readonly GazeCalibrationFitterOptions _options;

    public RawCoordinateNormalizationProfileBuilder(GazeCalibrationFitterOptions? options = null)
    {
        _options = options ?? new GazeCalibrationFitterOptions();
    }

    public RawCoordinateNormalizationProfileResult BuildProfile(
        TrackingProfile source,
        IReadOnlyList<CalibrationFramePair> pairs,
        IReadOnlyList<CalibrationLabel> labels)
    {
        var profile = Clone(source);
        profile.Version = "0.3-mask-normalized";
        profile.Output.OutputOffsetX = 0;
        profile.Output.OutputOffsetY = 0;

        var left = FitEye(source.Left, pairs, labels, EyeSide.Left);
        var right = FitEye(source.Right, pairs, labels, EyeSide.Right);
        if (left.Succeeded)
        {
            profile.Left.InputTransform = left.Transform;
        }

        if (right.Succeeded)
        {
            profile.Right.InputTransform = right.Transform;
        }

        var succeeded = left.Succeeded || right.Succeeded;
        return new RawCoordinateNormalizationProfileResult(profile, succeeded, left, right);
    }

    private RawCoordinateNormalizationEyeResult FitEye(
        EyeTrackingProfile sourceEye,
        IReadOnlyList<CalibrationFramePair> pairs,
        IReadOnlyList<CalibrationLabel> labels,
        EyeSide side)
    {
        var medians = BuildStageMedians(pairs, labels, side);
        if (!medians.TryGetValue("center", out var currentCenter))
        {
            return RawCoordinateNormalizationEyeResult.Failed("center stage is required.");
        }

        var sourceCenter = ExpectedRawFromTarget(sourceEye, CalibrationTarget.Center);
        var xRatios = new List<double>();
        var yRatios = new List<double>();
        foreach (var median in medians.Values)
        {
            if (Math.Abs(median.Target.X) > 0.35)
            {
                AddScaleRatio(
                    xRatios,
                    ExpectedRawFromTarget(sourceEye, median.Target).X - sourceCenter.X,
                    median.RawX - currentCenter.RawX);
            }

            if (Math.Abs(median.Target.Y) > 0.35)
            {
                AddScaleRatio(
                    yRatios,
                    ExpectedRawFromTarget(sourceEye, median.Target).Y - sourceCenter.Y,
                    median.RawY - currentCenter.RawY);
            }
        }

        if (xRatios.Count == 0 && yRatios.Count == 0)
        {
            return RawCoordinateNormalizationEyeResult.Failed("need horizontal or vertical non-center stages.");
        }

        var transform = new RawInputTransform
        {
            Enabled = true,
            CurrentCenterX = currentCenter.RawX,
            CurrentCenterY = currentCenter.RawY,
            TargetCenterX = sourceCenter.X,
            TargetCenterY = sourceCenter.Y,
            ScaleX = xRatios.Count == 0 ? 1 : Math.Clamp(Median(xRatios), 0.45, 1.80),
            ScaleY = yRatios.Count == 0 ? 1 : Math.Clamp(Median(yRatios), 0.45, 1.80)
        };

        return RawCoordinateNormalizationEyeResult.SucceededResult(
            transform,
            medians.Count,
            xRatios.Count,
            yRatios.Count);
    }

    private Dictionary<string, StageMedian> BuildStageMedians(
        IReadOnlyList<CalibrationFramePair> pairs,
        IReadOnlyList<CalibrationLabel> labels,
        EyeSide side)
    {
        var acceptedLabels = labels
            .Where(label =>
                label.Accepted &&
                !_options.IgnoredStageIds.Contains(label.StageId) &&
                !label.OperatorVerdict.Equals("bad", StringComparison.OrdinalIgnoreCase))
            .ToList();
        var medians = new Dictionary<string, StageMedian>(StringComparer.OrdinalIgnoreCase);

        foreach (var label in acceptedLabels)
        {
            var badFrames = label.BadFrames.ToHashSet();
            var samples = pairs
                .Where(pair => IsPairInLabel(pair, label) && !badFrames.Contains(pair.Sequence) && pair.DeltaMs <= _options.MaxDeltaMs)
                .Select(pair => ToSample(pair, side))
                .Where(sample =>
                    sample.Found &&
                    sample.Confidence >= _options.MinConfidence &&
                    sample.Openness >= _options.MinOpenness &&
                    double.IsFinite(sample.RawX) &&
                    double.IsFinite(sample.RawY))
                .ToList();
            if (samples.Count == 0)
            {
                continue;
            }

            var medianX = Median(samples.Select(sample => sample.RawX));
            var medianY = Median(samples.Select(sample => sample.RawY));
            var filtered = samples
                .Where(sample => Distance(sample.RawX, sample.RawY, medianX, medianY) <= _options.MaxStageOutlierDistancePixels)
                .ToList();
            if (filtered.Count == 0)
            {
                filtered = samples;
            }

            medians[label.StageId] = new StageMedian(
                label.StageId,
                label.EffectiveTarget,
                Median(filtered.Select(sample => sample.RawX)),
                Median(filtered.Select(sample => sample.RawY)),
                filtered.Count);
        }

        return medians;
    }

    private static EyeSample ToSample(CalibrationFramePair pair, EyeSide side) =>
        side == EyeSide.Left
            ? new EyeSample(pair.LeftFound, pair.LeftRawX, pair.LeftRawY, pair.LeftConfidence, pair.LeftOpenness)
            : new EyeSample(pair.RightFound, pair.RightRawX, pair.RightRawY, pair.RightConfidence, pair.RightOpenness);

    private static bool IsPairInLabel(CalibrationFramePair pair, CalibrationLabel label)
    {
        if (!pair.StageId.Equals(label.StageId, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (label.FrameStart <= 0 && label.FrameEnd <= 0)
        {
            return true;
        }

        return pair.Sequence >= label.FrameStart && pair.Sequence <= label.FrameEnd;
    }

    private static void AddScaleRatio(List<double> ratios, double sourceDelta, double currentDelta)
    {
        if (!double.IsFinite(sourceDelta) ||
            !double.IsFinite(currentDelta) ||
            Math.Abs(sourceDelta) < 1e-6 ||
            Math.Abs(currentDelta) < 1e-6)
        {
            return;
        }

        var ratio = sourceDelta / currentDelta;
        if (double.IsFinite(ratio) && ratio > 0)
        {
            ratios.Add(ratio);
        }
    }

    private static (double X, double Y) ExpectedRawFromTarget(EyeTrackingProfile eye, CalibrationTarget target)
    {
        if (TryInvertAffineGazeModel(eye.GazeModel, target, out var raw))
        {
            return raw;
        }

        var calibration = eye.Calibration;
        return (
            RawFromNormalized(
                target.X,
                calibration.CenterX,
                calibration.HorizontalRange,
                calibration.HorizontalOutputSign,
                calibration.HorizontalOutputOffset),
            RawFromNormalized(
                target.Y,
                calibration.CenterY,
                calibration.VerticalRange,
                calibration.VerticalOutputSign,
                calibration.VerticalOutputOffset));
    }

    private static bool TryInvertAffineGazeModel(
        Affine2DGazeModel? model,
        CalibrationTarget target,
        out (double X, double Y) raw)
    {
        raw = default;
        if (model is null ||
            !model.Type.Equals("affine2d", StringComparison.OrdinalIgnoreCase) ||
            model.OutX.Length < 3 ||
            model.OutY.Length < 3)
        {
            return false;
        }

        var a = model.OutX;
        var b = model.OutY;
        var det = (a[1] * b[2]) - (a[2] * b[1]);
        if (Math.Abs(det) < 1e-9)
        {
            return false;
        }

        var tx = target.X - a[0];
        var ty = target.Y - b[0];
        var normalizedX = ((tx * b[2]) - (a[2] * ty)) / det;
        var normalizedY = ((a[1] * ty) - (tx * b[1])) / det;
        raw = (
            model.InputCenterX + (normalizedX * model.InputScaleX),
            model.InputCenterY + (normalizedY * model.InputScaleY));
        return double.IsFinite(raw.X) && double.IsFinite(raw.Y);
    }

    private static double RawFromNormalized(double target, double center, double range, double outputSign, double outputOffset)
    {
        if (range <= 0)
        {
            return center;
        }

        var sign = outputSign < 0 ? -1 : 1;
        return center + (((target - outputOffset) / sign) * range);
    }

    private static TrackingProfile Clone(TrackingProfile source) =>
        new()
        {
            Version = source.Version,
            Output = Clone(source.Output),
            Left = Clone(source.Left),
            Right = Clone(source.Right)
        };

    private static BridgeOutputMapperOptions Clone(BridgeOutputMapperOptions source) =>
        new()
        {
            CenterDeadzoneX = source.CenterDeadzoneX,
            CenterDeadzoneY = source.CenterDeadzoneY,
            CenterLockEnterX = source.CenterLockEnterX,
            CenterLockEnterY = source.CenterLockEnterY,
            CenterLockReleaseX = source.CenterLockReleaseX,
            CenterLockReleaseY = source.CenterLockReleaseY,
            OutputGainX = source.OutputGainX,
            OutputGainY = source.OutputGainY,
            OutputOffsetX = source.OutputOffsetX,
            OutputOffsetY = source.OutputOffsetY
        };

    private static EyeTrackingProfile Clone(EyeTrackingProfile source) =>
        new()
        {
            Roi = source.Roi,
            Calibration = new EyeCalibration
            {
                CenterX = source.Calibration.CenterX,
                CenterY = source.Calibration.CenterY,
                HorizontalRange = source.Calibration.HorizontalRange,
                VerticalRange = source.Calibration.VerticalRange,
                HorizontalOutputSign = source.Calibration.HorizontalOutputSign,
                VerticalOutputSign = source.Calibration.VerticalOutputSign,
                HorizontalOutputOffset = source.Calibration.HorizontalOutputOffset,
                VerticalOutputOffset = source.Calibration.VerticalOutputOffset,
                OpenThreshold = source.Calibration.OpenThreshold,
                ClosedThreshold = source.Calibration.ClosedThreshold
            },
            GazeModel = Clone(source.GazeModel),
            InputTransform = Clone(source.InputTransform)
        };

    private static Affine2DGazeModel? Clone(Affine2DGazeModel? source) =>
        source is null
            ? null
            : new Affine2DGazeModel
            {
                Type = source.Type,
                InputCenterX = source.InputCenterX,
                InputCenterY = source.InputCenterY,
                InputScaleX = source.InputScaleX,
                InputScaleY = source.InputScaleY,
                OutX = source.OutX.ToArray(),
                OutY = source.OutY.ToArray(),
                CenterDeadzoneX = source.CenterDeadzoneX,
                CenterDeadzoneY = source.CenterDeadzoneY,
                CenterDeadzoneSoftness = source.CenterDeadzoneSoftness
            };

    private static RawInputTransform Clone(RawInputTransform source) =>
        new()
        {
            Enabled = source.Enabled,
            CurrentCenterX = source.CurrentCenterX,
            CurrentCenterY = source.CurrentCenterY,
            TargetCenterX = source.TargetCenterX,
            TargetCenterY = source.TargetCenterY,
            ScaleX = source.ScaleX,
            ScaleY = source.ScaleY
        };

    private static double Median(IEnumerable<double> values)
    {
        var sorted = values.Order().ToArray();
        if (sorted.Length == 0)
        {
            return 0;
        }

        var middle = sorted.Length / 2;
        return sorted.Length % 2 == 1
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) * 0.5;
    }

    private static double Distance(double x1, double y1, double x2, double y2)
    {
        var dx = x1 - x2;
        var dy = y1 - y2;
        return Math.Sqrt((dx * dx) + (dy * dy));
    }

    private sealed record EyeSample(bool Found, double RawX, double RawY, double Confidence, double Openness);

    private sealed record StageMedian(string StageId, CalibrationTarget Target, double RawX, double RawY, int SampleCount);
}

public sealed record RawCoordinateNormalizationProfileResult(
    TrackingProfile Profile,
    bool Succeeded,
    RawCoordinateNormalizationEyeResult Left,
    RawCoordinateNormalizationEyeResult Right);

public sealed record RawCoordinateNormalizationEyeResult(
    bool Succeeded,
    RawInputTransform Transform,
    int StageCount,
    int HorizontalStageCount,
    int VerticalStageCount,
    string FailureReason)
{
    public static RawCoordinateNormalizationEyeResult SucceededResult(
        RawInputTransform transform,
        int stageCount,
        int horizontalStageCount,
        int verticalStageCount) =>
        new(true, transform, stageCount, horizontalStageCount, verticalStageCount, string.Empty);

    public static RawCoordinateNormalizationEyeResult Failed(string reason) =>
        new(false, new RawInputTransform(), 0, 0, 0, reason);
}
