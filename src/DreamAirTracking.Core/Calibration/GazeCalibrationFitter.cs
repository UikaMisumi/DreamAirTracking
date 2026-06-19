using DreamAirTracking.Core;

namespace DreamAirTracking.Core.Calibration;

public sealed class GazeCalibrationFitter
{
    private readonly GazeCalibrationFitterOptions _options;

    public GazeCalibrationFitter(GazeCalibrationFitterOptions? options = null)
    {
        _options = options ?? new GazeCalibrationFitterOptions();
    }

    public GazeCalibrationFitResult Fit(
        IReadOnlyList<CalibrationFramePair> pairs,
        IReadOnlyList<CalibrationLabel> labels)
    {
        var acceptedLabels = labels
            .Where(label =>
                label.Accepted &&
                !_options.IgnoredStageIds.Contains(label.StageId) &&
                !label.OperatorVerdict.Equals("bad", StringComparison.OrdinalIgnoreCase))
            .ToList();

        var leftSamples = BuildSamples(pairs, acceptedLabels, EyeSide.Left);
        var rightSamples = BuildSamples(pairs, acceptedLabels, EyeSide.Right);

        var leftFit = FitEye(leftSamples);
        var rightFit = FitEye(rightSamples);
        var metrics = BuildMetrics(leftFit, rightFit, leftSamples.Count + rightSamples.Count, pairs.Count * 2);

        return new GazeCalibrationFitResult
        {
            LeftModel = leftFit.Model,
            RightModel = rightFit.Model,
            Metrics = metrics
        };
    }

    private List<CalibrationSample> BuildSamples(
        IReadOnlyList<CalibrationFramePair> pairs,
        IReadOnlyList<CalibrationLabel> labels,
        EyeSide side)
    {
        var samples = new List<CalibrationSample>();

        foreach (var label in labels)
        {
            var badFrames = label.BadFrames.ToHashSet();
            foreach (var pair in pairs)
            {
                if (!IsPairInLabel(pair, label) || badFrames.Contains(pair.Sequence) || pair.DeltaMs > _options.MaxDeltaMs)
                {
                    continue;
                }

                var found = side == EyeSide.Left ? pair.LeftFound : pair.RightFound;
                var rawX = side == EyeSide.Left ? pair.LeftRawX : pair.RightRawX;
                var rawY = side == EyeSide.Left ? pair.LeftRawY : pair.RightRawY;
                var confidence = side == EyeSide.Left ? pair.LeftConfidence : pair.RightConfidence;
                var openness = side == EyeSide.Left ? pair.LeftOpenness : pair.RightOpenness;

                if (!found ||
                    confidence < _options.MinConfidence ||
                    openness < _options.MinOpenness ||
                    !double.IsFinite(rawX) ||
                    !double.IsFinite(rawY))
                {
                    continue;
                }

                samples.Add(new CalibrationSample(
                    label.StageId,
                    label.EffectiveTarget,
                    pair.Sequence,
                    rawX,
                    rawY,
                    confidence,
                    openness));
            }
        }

        return samples;
    }

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

    private EyeFit FitEye(IReadOnlyList<CalibrationSample> samples)
    {
        var medians = BuildStageMedians(samples);
        if (medians.Count < 3)
        {
            return EyeFit.Failed(samples.Count, medians.Count, "Need at least 3 accepted calibration stages.");
        }

        var mode = NormalizeModelMode(_options.ModelMode);
        if (!TryBuildModelCandidate(medians, "affine2d", 3, out var affine))
        {
            return EyeFit.Failed(samples.Count, medians.Count, "Calibration stages are singular or nearly collinear.");
        }

        var selected = affine;
        if (mode.Equals("quadratic2d", StringComparison.OrdinalIgnoreCase))
        {
            if (medians.Count < _options.MinQuadraticStages ||
                !TryBuildModelCandidate(medians, "quadratic2d", 6, out selected))
            {
                return EyeFit.Failed(samples.Count, medians.Count, "Need a non-singular 6-term quadratic calibration map.");
            }
        }
        else if (mode.Equals("auto", StringComparison.OrdinalIgnoreCase) &&
            medians.Count >= _options.MinQuadraticStages &&
            TryBuildModelCandidate(medians, "quadratic2d", 6, out var quadratic) &&
            ShouldPreferQuadratic(affine, quadratic))
        {
            selected = quadratic;
        }

        var model = selected.Model;
        EstimateCenterDeadzone(model, samples);
        var residuals = BuildResiduals(model, medians);
        var averageResidual = residuals.Count == 0 ? double.NaN : residuals.Values.Average(value => value.Distance);

        return new EyeFit(model, samples.Count, medians.Count, averageResidual, residuals, string.Empty);
    }

    private List<StageMedian> BuildStageMedians(IReadOnlyList<CalibrationSample> samples)
    {
        var medians = new List<StageMedian>();
        foreach (var group in samples.GroupBy(sample => sample.StageId))
        {
            var target = group.First().Target;
            var points = group.ToList();
            if (points.Count == 0)
            {
                continue;
            }

            var medianX = Median(points.Select(point => point.RawX));
            var medianY = Median(points.Select(point => point.RawY));
            var filtered = points
                .Where(point => Distance(point.RawX, point.RawY, medianX, medianY) <= _options.MaxStageOutlierDistancePixels)
                .ToList();
            if (filtered.Count == 0)
            {
                filtered = points;
            }

            medians.Add(new StageMedian(
                group.Key,
                target,
                Median(filtered.Select(point => point.RawX)),
                Median(filtered.Select(point => point.RawY)),
                filtered.Count));
        }

        return medians;
    }

    private bool ShouldPreferQuadratic(ModelCandidate affine, ModelCandidate quadratic)
    {
        var improvement = affine.AverageResidual - quadratic.AverageResidual;
        if (!double.IsFinite(improvement) || improvement <= 0)
        {
            return false;
        }

        var ratio = affine.AverageResidual > 1e-9
            ? improvement / affine.AverageResidual
            : 0;

        return (improvement >= _options.MinQuadraticResidualImprovement ||
                ratio >= _options.MinQuadraticResidualImprovementRatio) &&
            quadratic.AverageResidual <= _options.MaxAutoQuadraticAverageResidual &&
            MaxResidualDistance(quadratic) <= _options.MaxAutoQuadraticStageResidual;
    }

    private static double MaxResidualDistance(ModelCandidate candidate) =>
        candidate.Residuals.Count == 0
            ? double.PositiveInfinity
            : candidate.Residuals.Values.Max(value => value.Distance);

    private bool TryBuildModelCandidate(
        IReadOnlyList<StageMedian> medians,
        string modelType,
        int coefficientCount,
        out ModelCandidate candidate)
    {
        candidate = new ModelCandidate(new Affine2DGazeModel(), double.NaN);
        var normalization = BuildNormalization(medians);
        var normal = new double[coefficientCount, coefficientCount];
        var rhsX = new double[coefficientCount];
        var rhsY = new double[coefficientCount];

        foreach (var median in medians)
        {
            var feature = BuildFeatureVector(median, normalization, coefficientCount);
            var weight = Math.Max(1, Math.Sqrt(median.Count));
            if (IsCenterStage(median))
            {
                weight *= Math.Max(1, _options.CenterStageWeightMultiplier);
            }

            for (var row = 0; row < coefficientCount; row++)
            {
                rhsX[row] += weight * feature[row] * median.Target.X;
                rhsY[row] += weight * feature[row] * median.Target.Y;
                for (var column = 0; column < coefficientCount; column++)
                {
                    normal[row, column] += weight * feature[row] * feature[column];
                }
            }
        }

        if (!TrySolve(normal, rhsX, out var outX) ||
            !TrySolve(normal, rhsY, out var outY))
        {
            return false;
        }

        var model = new Affine2DGazeModel
        {
            Type = modelType,
            InputCenterX = normalization.CenterX,
            InputCenterY = normalization.CenterY,
            InputScaleX = normalization.ScaleX,
            InputScaleY = normalization.ScaleY,
            OutX = outX,
            OutY = outY
        };
        AnchorModelAtCenter(model, medians);
        var residuals = BuildResiduals(model, medians);
        var averageResidual = residuals.Count == 0 ? double.NaN : residuals.Values.Average(value => value.Distance);
        candidate = new ModelCandidate(model, averageResidual, residuals);
        return true;
    }

    private void AnchorModelAtCenter(Affine2DGazeModel model, IReadOnlyList<StageMedian> medians)
    {
        if (!_options.AnchorCenterStage || model.OutX.Length == 0 || model.OutY.Length == 0)
        {
            return;
        }

        var center = medians.FirstOrDefault(IsCenterStage);
        if (center is null)
        {
            return;
        }

        var predicted = model.Map(center.RawX, center.RawY);
        model.OutX[0] -= predicted.X;
        model.OutY[0] -= predicted.Y;
    }

    private static bool IsCenterStage(StageMedian median) =>
        median.StageId.Equals("center", StringComparison.OrdinalIgnoreCase) ||
        median.StageId.Equals("center_confirm", StringComparison.OrdinalIgnoreCase) ||
        (Math.Abs(median.Target.X) < 0.001 && Math.Abs(median.Target.Y) < 0.001);

    private static FeatureNormalization BuildNormalization(IReadOnlyList<StageMedian> medians)
    {
        var centerX = Median(medians.Select(median => median.RawX));
        var centerY = Median(medians.Select(median => median.RawY));
        var scaleX = medians.Select(median => Math.Abs(median.RawX - centerX)).DefaultIfEmpty(1).Max();
        var scaleY = medians.Select(median => Math.Abs(median.RawY - centerY)).DefaultIfEmpty(1).Max();
        return new FeatureNormalization(centerX, centerY, Math.Max(1, scaleX), Math.Max(1, scaleY));
    }

    private static double[] BuildFeatureVector(StageMedian median, FeatureNormalization normalization, int coefficientCount)
    {
        var x = (median.RawX - normalization.CenterX) / normalization.ScaleX;
        var y = (median.RawY - normalization.CenterY) / normalization.ScaleY;
        if (coefficientCount >= 6)
        {
            return new[] { 1d, x, y, x * x, x * y, y * y };
        }

        return new[] { 1d, x, y };
    }

    private static string NormalizeModelMode(string mode)
    {
        if (mode.Equals("affine", StringComparison.OrdinalIgnoreCase) ||
            mode.Equals("affine2d", StringComparison.OrdinalIgnoreCase))
        {
            return "affine2d";
        }

        if (mode.Equals("quadratic", StringComparison.OrdinalIgnoreCase) ||
            mode.Equals("quadratic2d", StringComparison.OrdinalIgnoreCase))
        {
            return "quadratic2d";
        }

        return "auto";
    }

    private void EstimateCenterDeadzone(Affine2DGazeModel model, IReadOnlyList<CalibrationSample> samples)
    {
        var centerSamples = samples
            .Where(sample => Math.Abs(sample.Target.X) < 0.001 && Math.Abs(sample.Target.Y) < 0.001)
            .ToList();
        if (centerSamples.Count == 0)
        {
            return;
        }

        var mapped = centerSamples
            .Select(sample => model.Map(sample.RawX, sample.RawY))
            .ToList();
        var medianAbsX = Median(mapped.Select(point => Math.Abs(point.X)));
        var medianAbsY = Median(mapped.Select(point => Math.Abs(point.Y)));

        model.CenterDeadzoneX = Math.Clamp(medianAbsX * _options.CenterDeadzoneScale, _options.MinCenterDeadzone, _options.MaxCenterDeadzone);
        model.CenterDeadzoneY = Math.Clamp(medianAbsY * _options.CenterDeadzoneScale, _options.MinCenterDeadzone, _options.MaxCenterDeadzone);
    }

    private static Dictionary<string, CalibrationResidual> BuildResiduals(
        Affine2DGazeModel model,
        IReadOnlyList<StageMedian> medians)
    {
        var residuals = new Dictionary<string, CalibrationResidual>(StringComparer.OrdinalIgnoreCase);
        foreach (var median in medians)
        {
            var predicted = model.Map(median.RawX, median.RawY);
            var dx = predicted.X - median.Target.X;
            var dy = predicted.Y - median.Target.Y;
            residuals[median.StageId] = new CalibrationResidual
            {
                X = dx,
                Y = dy,
                Distance = Math.Sqrt((dx * dx) + (dy * dy))
            };
        }

        return residuals;
    }

    private static CalibrationMetrics BuildMetrics(EyeFit left, EyeFit right, int validSamples, int possibleSamples)
    {
        var metrics = new CalibrationMetrics
        {
            Model = BuildModelSummary(left, right),
            ValidPairs = validSamples / 2,
            RejectedPairs = Math.Max(0, (possibleSamples - validSamples) / 2),
            Quality = EstimateQuality(left, right)
        };

        metrics.Eyes["left"] = ToEyeMetrics(left);
        metrics.Eyes["right"] = ToEyeMetrics(right);
        foreach (var residual in AverageResiduals(left.Residuals, right.Residuals))
        {
            metrics.StageResiduals[residual.Key] = residual.Value;
        }

        metrics.CenterDrift = BuildCenterDrift(metrics.StageResiduals);
        return metrics;
    }

    private static EyeCalibrationMetrics ToEyeMetrics(EyeFit fit) =>
        new()
        {
            Succeeded = fit.Model is not null,
            Model = fit.Model?.Type ?? string.Empty,
            StageCount = fit.StageCount,
            SampleCount = fit.SampleCount,
            AverageResidual = double.IsFinite(fit.AverageResidual) ? fit.AverageResidual : 0,
            FailureReason = fit.FailureReason
        };

    private static string BuildModelSummary(EyeFit left, EyeFit right)
    {
        var leftType = left.Model?.Type;
        var rightType = right.Model?.Type;
        if (leftType is null && rightType is null)
        {
            return "none";
        }

        if (leftType is not null &&
            rightType is not null &&
            leftType.Equals(rightType, StringComparison.OrdinalIgnoreCase))
        {
            return leftType;
        }

        return $"left={leftType ?? "none"},right={rightType ?? "none"}";
    }

    private static Dictionary<string, CalibrationResidual> AverageResiduals(
        IReadOnlyDictionary<string, CalibrationResidual> left,
        IReadOnlyDictionary<string, CalibrationResidual> right)
    {
        var result = new Dictionary<string, CalibrationResidual>(StringComparer.OrdinalIgnoreCase);
        foreach (var stageId in left.Keys.Concat(right.Keys).Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var values = new List<CalibrationResidual>();
            if (left.TryGetValue(stageId, out var leftValue))
            {
                values.Add(leftValue);
            }

            if (right.TryGetValue(stageId, out var rightValue))
            {
                values.Add(rightValue);
            }

            result[stageId] = new CalibrationResidual
            {
                X = values.Average(value => value.X),
                Y = values.Average(value => value.Y),
                Distance = values.Average(value => value.Distance)
            };
        }

        return result;
    }

    private static CalibrationResidual BuildCenterDrift(IReadOnlyDictionary<string, CalibrationResidual> residuals)
    {
        if (residuals.TryGetValue("center_confirm", out var centerConfirm))
        {
            return centerConfirm;
        }

        return residuals.TryGetValue("center", out var center)
            ? center
            : new CalibrationResidual();
    }

    private static string EstimateQuality(EyeFit left, EyeFit right)
    {
        var fits = new[] { left, right }.Where(fit => fit.Model is not null).ToList();
        if (fits.Count == 0)
        {
            return "poor";
        }

        var averageResidual = fits.Average(fit => fit.AverageResidual);
        if (averageResidual <= 0.15)
        {
            return "good";
        }

        return averageResidual <= 0.28 ? "fair" : "poor";
    }

    private static bool TrySolve(double[,] matrix, double[] values, out double[] result)
    {
        var size = values.Length;
        var augmented = new double[size, size + 1];
        for (var row = 0; row < size; row++)
        {
            for (var column = 0; column < size; column++)
            {
                augmented[row, column] = matrix[row, column];
            }

            augmented[row, size] = values[row];
        }

        for (var pivot = 0; pivot < size; pivot++)
        {
            var bestRow = pivot;
            var best = Math.Abs(augmented[pivot, pivot]);
            for (var row = pivot + 1; row < size; row++)
            {
                var candidate = Math.Abs(augmented[row, pivot]);
                if (candidate > best)
                {
                    best = candidate;
                    bestRow = row;
                }
            }

            if (best < 1e-9)
            {
                result = Array.Empty<double>();
                return false;
            }

            if (bestRow != pivot)
            {
                for (var column = pivot; column <= size; column++)
                {
                    (augmented[pivot, column], augmented[bestRow, column]) = (augmented[bestRow, column], augmented[pivot, column]);
                }
            }

            var divisor = augmented[pivot, pivot];
            for (var column = pivot; column <= size; column++)
            {
                augmented[pivot, column] /= divisor;
            }

            for (var row = 0; row < size; row++)
            {
                if (row == pivot)
                {
                    continue;
                }

                var factor = augmented[row, pivot];
                for (var column = pivot; column <= size; column++)
                {
                    augmented[row, column] -= factor * augmented[pivot, column];
                }
            }
        }

        result = Enumerable.Range(0, size).Select(row => augmented[row, size]).ToArray();
        return true;
    }

    private static double Median(IEnumerable<double> values)
    {
        var sorted = values.OrderBy(value => value).ToList();
        if (sorted.Count == 0)
        {
            return 0;
        }

        var middle = sorted.Count / 2;
        return sorted.Count % 2 == 1
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2;
    }

    private static double Distance(double x1, double y1, double x2, double y2)
    {
        var dx = x1 - x2;
        var dy = y1 - y2;
        return Math.Sqrt((dx * dx) + (dy * dy));
    }

    private sealed record CalibrationSample(
        string StageId,
        CalibrationTarget Target,
        long Sequence,
        double RawX,
        double RawY,
        double Confidence,
        double Openness);

    private sealed record StageMedian(
        string StageId,
        CalibrationTarget Target,
        double RawX,
        double RawY,
        int Count);

    private sealed record FeatureNormalization(
        double CenterX,
        double CenterY,
        double ScaleX,
        double ScaleY);

    private sealed record ModelCandidate(
        Affine2DGazeModel Model,
        double AverageResidual,
        IReadOnlyDictionary<string, CalibrationResidual> Residuals)
    {
        public ModelCandidate(Affine2DGazeModel model, double averageResidual)
            : this(model, averageResidual, new Dictionary<string, CalibrationResidual>())
        {
        }
    }

    private sealed record EyeFit(
        Affine2DGazeModel? Model,
        int SampleCount,
        int StageCount,
        double AverageResidual,
        IReadOnlyDictionary<string, CalibrationResidual> Residuals,
        string FailureReason)
    {
        public static EyeFit Failed(int sampleCount, int stageCount, string reason) =>
            new(null, sampleCount, stageCount, double.NaN, new Dictionary<string, CalibrationResidual>(), reason);
    }
}
