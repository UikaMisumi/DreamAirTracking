using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeBinocularGazeFilter
{
    private readonly BridgeBinocularGazeFilterOptions _options;
    private bool _hasPrevious;
    private double _previousX;
    private double _previousY;
    private int _missingFrames;

    public BridgeBinocularGazeFilter(BridgeBinocularGazeFilterOptions? options = null)
    {
        _options = options ?? new BridgeBinocularGazeFilterOptions();
    }

    public BridgeGazeState Update(BridgeGazeState monocular, NormalizedEyePoint leftInput, NormalizedEyePoint rightInput)
    {
        var leftValid = IsUsable(leftInput);
        var rightValid = IsUsable(rightInput);

        if (!leftValid && !rightValid)
        {
            return HoldOrRecenterMissing(monocular);
        }

        _missingFrames = 0;
        var fused = leftValid && rightValid
            ? FuseBoth(monocular, leftInput.Confidence, rightInput.Confidence)
            : leftValid
                ? (monocular.LeftX, monocular.LeftY)
                : (monocular.RightX, monocular.RightY);

        _previousX = ClampSigned(fused.Item1);
        _previousY = ClampSigned(fused.Item2);
        _hasPrevious = true;
        return SameForBoth(_previousX, _previousY);
    }

    private (double X, double Y) FuseBoth(BridgeGazeState monocular, double leftConfidence, double rightConfidence)
    {
        var xDisagrees = HasAxisDisagreement(monocular.LeftX, monocular.RightX, _options.MaxCoherentXDifference);
        var yDisagrees = HasAxisDisagreement(monocular.LeftY, monocular.RightY, _options.MaxCoherentYDifference);
        if (!xDisagrees && !yDisagrees)
        {
            return WeightedAverage(
                monocular.LeftX,
                monocular.LeftY,
                leftConfidence,
                monocular.RightX,
                monocular.RightY,
                rightConfidence);
        }

        var weighted = WeightedAverage(
            monocular.LeftX,
            monocular.LeftY,
            leftConfidence,
            monocular.RightX,
            monocular.RightY,
            rightConfidence);
        var x = xDisagrees
            ? ResolveAxis(monocular.LeftX, monocular.RightX, leftConfidence, rightConfidence, _previousX, Axis.Horizontal)
            : weighted.X;
        var y = yDisagrees
            ? ResolveAxis(monocular.LeftY, monocular.RightY, leftConfidence, rightConfidence, _previousY, Axis.Vertical)
            : weighted.Y;
        if (yDisagrees)
        {
            y = ClampVerticalDriftDuringHorizontalDominance(y, x);
        }

        if (xDisagrees)
        {
            x = ClampHorizontalDriftDuringVerticalDominance(x, y);
        }

        return (x, y);
    }

    private bool HasAxisDisagreement(double leftValue, double rightValue, double maxCoherentDifference)
        => Math.Abs(leftValue - rightValue) > maxCoherentDifference ||
            AreOpposite(leftValue, rightValue);

    private bool AreOpposite(double a, double b)
    {
        return Math.Abs(a) >= _options.OppositeDirectionMagnitude &&
            Math.Abs(b) >= _options.OppositeDirectionMagnitude &&
            Math.Sign(a) != Math.Sign(b);
    }

    private double ResolveAxis(
        double leftValue,
        double rightValue,
        double leftConfidence,
        double rightConfidence,
        double previousValue,
        Axis axis)
    {
        if (axis == Axis.Horizontal)
        {
            var preferred = TryGetPreferredHorizontalValue(leftValue, rightValue, leftConfidence, rightConfidence);
            if (preferred is not null)
            {
                return preferred.Value;
            }

            var magnitudeDominant = TryGetHorizontalMagnitudeDominantValue(leftValue, rightValue, leftConfidence, rightConfidence);
            if (magnitudeDominant is not null)
            {
                return magnitudeDominant.Value;
            }
        }

        if (axis == Axis.Vertical)
        {
            var magnitudeDominant = TryGetMagnitudeDominantValue(leftValue, rightValue, leftConfidence, rightConfidence);
            if (magnitudeDominant is not null)
            {
                return magnitudeDominant.Value;
            }
        }

        var source = PickDisagreementSource(leftValue, rightValue, leftConfidence, rightConfidence, previousValue);
        return source == EyeSource.Left
            ? leftValue
            : source == EyeSource.Right
                ? rightValue
                : WeightedAverage(leftValue, 0, leftConfidence, rightValue, 0, rightConfidence).X;
    }

    private double? TryGetPreferredHorizontalValue(
        double leftValue,
        double rightValue,
        double leftConfidence,
        double rightConfidence)
    {
        if (_options.PreferredHorizontalEye == BridgePreferredEye.Left &&
            leftConfidence >= _options.PreferredEyeMinConfidence)
        {
            return leftValue;
        }

        if (_options.PreferredHorizontalEye == BridgePreferredEye.Right &&
            rightConfidence >= _options.PreferredEyeMinConfidence)
        {
            return rightValue;
        }

        return null;
    }

    private double? TryGetMagnitudeDominantValue(
        double leftValue,
        double rightValue,
        double leftConfidence,
        double rightConfidence)
    {
        var leftAbs = Math.Abs(leftValue);
        var rightAbs = Math.Abs(rightValue);
        if (leftConfidence >= _options.MagnitudeDominanceMinConfidence &&
            leftAbs >= rightAbs + _options.MagnitudeDominanceMargin)
        {
            return leftValue;
        }

        if (rightConfidence >= _options.MagnitudeDominanceMinConfidence &&
            rightAbs >= leftAbs + _options.MagnitudeDominanceMargin)
        {
            return rightValue;
        }

        return null;
    }

    private double? TryGetHorizontalMagnitudeDominantValue(
        double leftValue,
        double rightValue,
        double leftConfidence,
        double rightConfidence)
    {
        var leftAbs = Math.Abs(leftValue);
        var rightAbs = Math.Abs(rightValue);
        if (AreOpposite(leftValue, rightValue))
        {
            return null;
        }

        if (leftConfidence >= _options.HorizontalMagnitudeDominanceMinConfidence &&
            leftAbs >= _options.HorizontalMagnitudeDominanceMinAbs &&
            leftAbs >= rightAbs + _options.HorizontalMagnitudeDominanceMargin &&
            rightAbs <= _options.HorizontalMagnitudeDominanceNearCenterAbs)
        {
            return leftValue;
        }

        if (rightConfidence >= _options.HorizontalMagnitudeDominanceMinConfidence &&
            rightAbs >= _options.HorizontalMagnitudeDominanceMinAbs &&
            rightAbs >= leftAbs + _options.HorizontalMagnitudeDominanceMargin &&
            leftAbs <= _options.HorizontalMagnitudeDominanceNearCenterAbs)
        {
            return rightValue;
        }

        return null;
    }

    private double ClampHorizontalDriftDuringVerticalDominance(double x, double y)
    {
        var absY = Math.Abs(y);
        if (absY <= _options.VerticalDominantStart)
        {
            return x;
        }

        var denominator = Math.Max(_options.VerticalDominantFull - _options.VerticalDominantStart, 0.001);
        var t = Math.Clamp((absY - _options.VerticalDominantStart) / denominator, 0, 1);
        var clampAtStart = y < 0
            ? _options.DownwardDominantXClampAtStart
            : _options.VerticalDominantXClampAtStart;
        var clampAtFull = y < 0
            ? _options.DownwardDominantXClampAtFull
            : _options.VerticalDominantXClampAtFull;
        var maxAbsX = clampAtStart + (clampAtFull - clampAtStart) * t;
        return Math.Clamp(x, -Math.Abs(maxAbsX), Math.Abs(maxAbsX));
    }

    private double ClampVerticalDriftDuringHorizontalDominance(double y, double x)
    {
        var absX = Math.Abs(x);
        if (absX <= _options.HorizontalDominantStart)
        {
            return y;
        }

        var denominator = Math.Max(_options.HorizontalDominantFull - _options.HorizontalDominantStart, 0.001);
        var t = Math.Clamp((absX - _options.HorizontalDominantStart) / denominator, 0, 1);
        var maxAbsY = _options.HorizontalDominantYClampAtStart +
            (_options.HorizontalDominantYClampAtFull - _options.HorizontalDominantYClampAtStart) * t;
        return Math.Clamp(y, -Math.Abs(maxAbsY), Math.Abs(maxAbsY));
    }

    private EyeSource PickDisagreementSource(
        double leftValue,
        double rightValue,
        double leftConfidence,
        double rightConfidence,
        double previousValue)
    {
        if (leftConfidence + _options.ConfidenceEpsilon >= rightConfidence * _options.ConfidenceDominanceRatio)
        {
            return EyeSource.Left;
        }

        if (rightConfidence + _options.ConfidenceEpsilon >= leftConfidence * _options.ConfidenceDominanceRatio)
        {
            return EyeSource.Right;
        }

        if (AreOpposite(leftValue, rightValue))
        {
            return EyeSource.Fused;
        }

        if (_hasPrevious)
        {
            var leftDistance = Math.Abs(leftValue - previousValue);
            var rightDistance = Math.Abs(rightValue - previousValue);
            if (Math.Abs(leftDistance - rightDistance) >= _options.PreviousDistanceMargin)
            {
                return leftDistance < rightDistance ? EyeSource.Left : EyeSource.Right;
            }
        }

        return EyeSource.Fused;
    }

    private bool IsUsable(NormalizedEyePoint input)
        => input.Found && input.Confidence >= _options.MinUsableConfidence;

    private static (double X, double Y) WeightedAverage(
        double leftX,
        double leftY,
        double leftConfidence,
        double rightX,
        double rightY,
        double rightConfidence)
    {
        var leftWeight = Math.Max(leftConfidence, 0.001);
        var rightWeight = Math.Max(rightConfidence, 0.001);
        var total = leftWeight + rightWeight;
        return (
            ((leftX * leftWeight) + (rightX * rightWeight)) / total,
            ((leftY * leftWeight) + (rightY * rightWeight)) / total);
    }

    private static BridgeGazeState SameForBoth(double x, double y)
        => new(x, y, x, y);

    private static double ClampSigned(double value) => Math.Clamp(value, -1.0, 1.0);

    private BridgeGazeState HoldOrRecenterMissing(BridgeGazeState monocular)
    {
        if (!_hasPrevious)
        {
            return monocular;
        }

        _missingFrames++;
        if (_missingFrames <= _options.MissingHoldFrames)
        {
            return SameForBoth(_previousX, _previousY);
        }

        var alpha = Math.Clamp(_options.MissingRecenterAlpha, 0, 1);
        _previousX = ClampSigned(_previousX + ((0 - _previousX) * alpha));
        _previousY = ClampSigned(_previousY + ((0 - _previousY) * alpha));
        if (Math.Abs(_previousX) < 0.001)
        {
            _previousX = 0;
        }

        if (Math.Abs(_previousY) < 0.001)
        {
            _previousY = 0;
        }

        return SameForBoth(_previousX, _previousY);
    }

    private enum Axis
    {
        Horizontal,
        Vertical
    }

    private enum EyeSource
    {
        Fused,
        Left,
        Right
    }
}
