namespace DreamAirTracking.Core.Calibration;

public sealed class CalibrationLiveGazeHighlighterOptions
{
    public double AxisThreshold { get; init; } = 0.30;
    public double AxisReleaseThreshold { get; init; } = 0.12;
    public double ClosedOpennessThreshold { get; init; } = 0.18;
    public double MinConfidence { get; init; } = 0.04;
    public double MinEyeOpenness { get; init; } = 0.08;
}
