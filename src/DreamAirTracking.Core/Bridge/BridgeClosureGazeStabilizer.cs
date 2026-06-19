namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeClosureGazeStabilizer
{
    private readonly BridgeClosureGazeStabilizerOptions _options;
    private bool _initialized;
    private BridgeGazeState _previous = new(0, 0, 0, 0);
    private double _previousAverageOpenness = 1;
    private int _holdFrames;

    public BridgeClosureGazeStabilizer(BridgeClosureGazeStabilizerOptions? options = null)
    {
        _options = options ?? new BridgeClosureGazeStabilizerOptions();
    }

    public BridgeGazeState Update(BridgeGazeState input, BridgeOpennessState openness)
    {
        var averageOpenness = Clamp01((openness.Left + openness.Right) * 0.5);
        if (!_initialized)
        {
            _initialized = true;
            _previous = Clamp(input);
            _previousAverageOpenness = averageOpenness;
            return _previous;
        }

        var closingDrop = _previousAverageOpenness - averageOpenness >= _options.ClosingDropThreshold &&
            averageOpenness <= _options.ClosingDropMaxOpenness;
        var closing = closingDrop || averageOpenness <= _options.ClosingOpennessThreshold;
        var closed = averageOpenness <= _options.ClosedOpennessThreshold;
        if (closing)
        {
            _holdFrames = Math.Max(_holdFrames, Math.Max(0, _options.HoldFramesAfterClosing));
        }

        BridgeGazeState output;
        if (closed || (_holdFrames > 0 && averageOpenness < _options.ReopenOpennessThreshold))
        {
            output = _previous;
            DecrementHold();
        }
        else if (_holdFrames > 0)
        {
            output = Lerp(_previous, input, Math.Clamp(_options.ReopenReleaseAlpha, 0, 1));
            _previous = output;
            DecrementHold();
        }
        else
        {
            output = Clamp(input);
            _previous = output;
        }

        _previousAverageOpenness = averageOpenness;
        return output;
    }

    private void DecrementHold()
    {
        if (_holdFrames > 0)
        {
            _holdFrames--;
        }
    }

    private static BridgeGazeState Lerp(BridgeGazeState previous, BridgeGazeState input, double alpha) =>
        Clamp(new BridgeGazeState(
            previous.LeftX + ((input.LeftX - previous.LeftX) * alpha),
            previous.LeftY + ((input.LeftY - previous.LeftY) * alpha),
            previous.RightX + ((input.RightX - previous.RightX) * alpha),
            previous.RightY + ((input.RightY - previous.RightY) * alpha)));

    private static BridgeGazeState Clamp(BridgeGazeState state) =>
        new(
            ClampSigned(state.LeftX),
            ClampSigned(state.LeftY),
            ClampSigned(state.RightX),
            ClampSigned(state.RightY));

    private static double ClampSigned(double value) => Math.Clamp(value, -1.0, 1.0);

    private static double Clamp01(double value) => Math.Clamp(value, 0.0, 1.0);
}
