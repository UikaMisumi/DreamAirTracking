using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeGazeFilter
{
    private readonly BridgeGazeFilterOptions _options;
    private readonly EyeState _left = new();
    private readonly EyeState _right = new();

    public BridgeGazeFilter(BridgeGazeFilterOptions? options = null)
    {
        _options = options ?? new BridgeGazeFilterOptions();
    }

    public BridgeGazeState Update(NormalizedEyePoint left, NormalizedEyePoint right)
    {
        var filteredLeft = UpdateEye(_left, left);
        var filteredRight = UpdateEye(_right, right);
        return new BridgeGazeState(filteredLeft.X, filteredLeft.Y, filteredRight.X, filteredRight.Y);
    }

    private (double X, double Y) UpdateEye(EyeState state, NormalizedEyePoint input)
    {
        if (!input.Found || input.Confidence < _options.MinUsableConfidence)
        {
            return HoldOrRecenterMissing(state);
        }

        state.MissingFrames = 0;
        var targetX = ClampSigned(input.X);
        var targetY = ClampSigned(input.Y);
        if (!state.Initialized)
        {
            state.X = targetX;
            state.Y = targetY;
            state.Initialized = true;
            return (state.X, state.Y);
        }

        var dx = targetX - state.X;
        var dy = targetY - state.Y;
        var distance = Math.Sqrt(dx * dx + dy * dy);
        var isRecentering = IsRecenterTarget(state, input.Confidence, targetX, targetY);
        if (!isRecentering && ShouldHoldLowConfidenceJump(state, input.Confidence, targetX, targetY, distance))
        {
            return (state.X, state.Y);
        }

        if (!isRecentering && input.Confidence < _options.HighConfidence && distance > _options.LowConfidenceMaxStep)
        {
            var scale = _options.LowConfidenceMaxStep / distance;
            targetX = state.X + dx * scale;
            targetY = state.Y + dy * scale;
            distance = _options.LowConfidenceMaxStep;
        }

        var confidenceT = Math.Clamp(input.Confidence / Math.Max(_options.HighConfidence, 0.001), 0, 1);
        var alpha = _options.SlowAlpha + (_options.FastAlpha - _options.SlowAlpha) * confidenceT;
        if (distance >= _options.FastDistance && input.Confidence >= _options.HighConfidence)
        {
            alpha = _options.FastAlpha;
        }

        if (isRecentering)
        {
            state.ResetPendingLowConfidenceTarget();
            alpha = Math.Max(alpha, _options.RecenterAlpha);
        }

        alpha = Math.Clamp(alpha, 0, 1);
        state.X = ClampSigned(state.X + (targetX - state.X) * alpha);
        state.Y = ClampSigned(state.Y + (targetY - state.Y) * alpha);
        return (state.X, state.Y);
    }

    private bool IsRecenterTarget(EyeState state, double confidence, double targetX, double targetY)
    {
        if (confidence < _options.RecenterMinConfidence)
        {
            return false;
        }

        if (Math.Abs(targetX) <= _options.RecenterTargetX &&
            Math.Abs(targetY) <= _options.RecenterTargetY)
        {
            return true;
        }

        var currentRadius = Math.Sqrt((state.X * state.X) + (state.Y * state.Y));
        var targetRadius = Math.Sqrt((targetX * targetX) + (targetY * targetY));
        return currentRadius >= _options.RecenterCurrentRadius &&
            targetRadius <= _options.RecenterTargetRadius &&
            currentRadius - targetRadius >= _options.RecenterRadiusDrop;
    }

    private bool ShouldHoldLowConfidenceJump(
        EyeState state,
        double confidence,
        double targetX,
        double targetY,
        double distanceFromCurrent)
    {
        if (confidence >= _options.HighConfidence ||
            distanceFromCurrent <= _options.LowConfidenceJumpDistance)
        {
            state.ResetPendingLowConfidenceTarget();
            return false;
        }

        var pendingDistance = state.HasPendingLowConfidenceTarget
            ? Distance(targetX, targetY, state.PendingLowConfidenceX, state.PendingLowConfidenceY)
            : double.PositiveInfinity;

        if (!state.HasPendingLowConfidenceTarget ||
            pendingDistance > _options.LowConfidenceStableDistance)
        {
            state.PendingLowConfidenceX = targetX;
            state.PendingLowConfidenceY = targetY;
            state.PendingLowConfidenceFrames = 1;
            return true;
        }

        state.PendingLowConfidenceX = Average(state.PendingLowConfidenceX, targetX);
        state.PendingLowConfidenceY = Average(state.PendingLowConfidenceY, targetY);
        state.PendingLowConfidenceFrames++;
        return state.PendingLowConfidenceFrames <= _options.LowConfidenceConfirmFrames;
    }

    private static double ClampSigned(double value) => Math.Clamp(value, -1.0, 1.0);

    private static double Distance(double ax, double ay, double bx, double by)
    {
        var dx = ax - bx;
        var dy = ay - by;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    private static double Average(double a, double b) => (a + b) * 0.5;

    private (double X, double Y) HoldOrRecenterMissing(EyeState state)
    {
        state.ResetPendingLowConfidenceTarget();
        if (!state.Initialized)
        {
            return (0, 0);
        }

        state.MissingFrames++;
        if (state.MissingFrames <= _options.MissingHoldFrames)
        {
            return (state.X, state.Y);
        }

        var alpha = Math.Clamp(_options.MissingRecenterAlpha, 0, 1);
        state.X = ClampSigned(state.X + ((0 - state.X) * alpha));
        state.Y = ClampSigned(state.Y + ((0 - state.Y) * alpha));
        if (Math.Abs(state.X) < 0.001)
        {
            state.X = 0;
        }

        if (Math.Abs(state.Y) < 0.001)
        {
            state.Y = 0;
        }

        return (state.X, state.Y);
    }

    private sealed class EyeState
    {
        public bool Initialized { get; set; }
        public double X { get; set; }
        public double Y { get; set; }
        public int MissingFrames { get; set; }
        public bool HasPendingLowConfidenceTarget => PendingLowConfidenceFrames > 0;
        public double PendingLowConfidenceX { get; set; }
        public double PendingLowConfidenceY { get; set; }
        public int PendingLowConfidenceFrames { get; set; }

        public void ResetPendingLowConfidenceTarget()
        {
            PendingLowConfidenceFrames = 0;
            PendingLowConfidenceX = 0;
            PendingLowConfidenceY = 0;
        }
    }
}
