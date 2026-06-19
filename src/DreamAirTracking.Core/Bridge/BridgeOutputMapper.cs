namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeOutputMapper
{
    private readonly BridgeOutputMapperOptions _options;
    private bool _centerLocked;

    public BridgeOutputMapper(BridgeOutputMapperOptions? options = null)
    {
        _options = options ?? new BridgeOutputMapperOptions();
    }

    public BridgeGazeState Update(BridgeGazeState input)
    {
        var x = (input.LeftX + input.RightX) * 0.5;
        var y = (input.LeftY + input.RightY) * 0.5;

        if (!_centerLocked && IsInsideEllipse(x, y, _options.CenterLockEnterX, _options.CenterLockEnterY))
        {
            _centerLocked = true;
        }

        if (_centerLocked)
        {
            if (IsInsideEllipse(x, y, _options.CenterLockReleaseX, _options.CenterLockReleaseY))
            {
                return SameForBoth(0, 0);
            }

            _centerLocked = false;
        }

        var mappedX = ApplyAxis(x, _options.CenterDeadzoneX, _options.OutputGainX, _options.OutputOffsetX);
        var mappedY = ApplyAxis(y, _options.CenterDeadzoneY, _options.OutputGainY, _options.OutputOffsetY);
        return SameForBoth(mappedX, mappedY);
    }

    private static BridgeGazeState SameForBoth(double x, double y) => new(x, y, x, y);

    private static double ApplyAxis(double value, double deadzone, double gain, double offset)
    {
        var abs = Math.Abs(value);
        var sign = Math.Sign(value);
        var normalized = abs <= deadzone
            ? 0
            : (abs - deadzone) / Math.Max(0.001, 1 - deadzone);
        return Math.Clamp((normalized * sign * gain) + offset, -1, 1);
    }

    private static bool IsInsideEllipse(double x, double y, double radiusX, double radiusY)
    {
        if (radiusX <= 0 || radiusY <= 0)
        {
            return false;
        }

        var distance = ((x * x) / (radiusX * radiusX)) + ((y * y) / (radiusY * radiusY));
        return distance <= 1;
    }
}
