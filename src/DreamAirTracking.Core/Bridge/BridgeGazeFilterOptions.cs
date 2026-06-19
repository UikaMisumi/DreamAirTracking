namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeGazeFilterOptions
{
    public double MinUsableConfidence { get; set; } = 0.045;
    public double HighConfidence { get; set; } = 0.14;
    public double SlowAlpha { get; set; } = 0.22;
    public double FastAlpha { get; set; } = 0.58;
    public double FastDistance { get; set; } = 0.45;
    public double LowConfidenceMaxStep { get; set; } = 0.18;
    public double LowConfidenceJumpDistance { get; set; } = 0.35;
    public double LowConfidenceStableDistance { get; set; } = 0.22;
    public int LowConfidenceConfirmFrames { get; set; } = 2;
    public double RecenterTargetX { get; set; } = 0.14;
    public double RecenterTargetY { get; set; } = 0.36;
    public double RecenterMinConfidence { get; set; } = 0.05;
    public double RecenterAlpha { get; set; } = 0.82;
    public double RecenterCurrentRadius { get; set; } = 0.55;
    public double RecenterTargetRadius { get; set; } = 0.68;
    public double RecenterRadiusDrop { get; set; } = 0.18;
    public int MissingHoldFrames { get; set; } = 2;
    public double MissingRecenterAlpha { get; set; } = 0.35;
}
