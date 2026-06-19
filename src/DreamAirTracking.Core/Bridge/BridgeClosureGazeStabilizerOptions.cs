namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeClosureGazeStabilizerOptions
{
    public double ClosingOpennessThreshold { get; set; } = 0.50;
    public double ClosedOpennessThreshold { get; set; } = 0.20;
    public double ReopenOpennessThreshold { get; set; } = 0.70;
    public double ClosingDropThreshold { get; set; } = 0.28;
    public double ClosingDropMaxOpenness { get; set; } = 0.65;
    public int HoldFramesAfterClosing { get; set; } = 4;
    public double ReopenReleaseAlpha { get; set; } = 0.35;
}
