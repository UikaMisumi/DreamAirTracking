namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeOutputMapperOptions
{
    public double CenterDeadzoneX { get; set; } = 0.12;
    public double CenterDeadzoneY { get; set; } = 0.22;
    public double CenterLockEnterX { get; set; } = 0.16;
    public double CenterLockEnterY { get; set; } = 0.42;
    public double CenterLockReleaseX { get; set; } = 0.26;
    public double CenterLockReleaseY { get; set; } = 0.62;
    public double OutputGainX { get; set; } = 1.75;
    public double OutputGainY { get; set; } = 1.75;
    public double OutputOffsetX { get; set; }
    public double OutputOffsetY { get; set; }
}
