namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeBinocularGazeFilterOptions
{
    public double MinUsableConfidence { get; set; } = 0.05;
    public double MaxCoherentXDifference { get; set; } = 0.30;
    public double MaxCoherentYDifference { get; set; } = 0.55;
    public double OppositeDirectionMagnitude { get; set; } = 0.22;
    public double PreviousDistanceMargin { get; set; } = 0.12;
    public double ConfidenceDominanceRatio { get; set; } = 1.8;
    public double ConfidenceEpsilon { get; set; } = 0.001;
    public double MagnitudeDominanceMargin { get; set; } = 0.24;
    public double MagnitudeDominanceMinConfidence { get; set; } = 0.045;
    public double HorizontalMagnitudeDominanceMargin { get; set; } = 0.45;
    public double HorizontalMagnitudeDominanceMinAbs { get; set; } = 0.60;
    public double HorizontalMagnitudeDominanceNearCenterAbs { get; set; } = 0.25;
    public double HorizontalMagnitudeDominanceMinConfidence { get; set; } = 0.07;
    public BridgePreferredEye PreferredHorizontalEye { get; set; } = BridgePreferredEye.None;
    public double PreferredEyeMinConfidence { get; set; } = 0.08;
    public double VerticalDominantStart { get; set; } = 0.35;
    public double VerticalDominantFull { get; set; } = 0.60;
    public double VerticalDominantXClampAtStart { get; set; } = 0.28;
    public double VerticalDominantXClampAtFull { get; set; } = 0.12;
    public double DownwardDominantXClampAtStart { get; set; } = 0.18;
    public double DownwardDominantXClampAtFull { get; set; } = 0.04;
    public double HorizontalDominantStart { get; set; } = 0.30;
    public double HorizontalDominantFull { get; set; } = 0.65;
    public double HorizontalDominantYClampAtStart { get; set; } = 0.20;
    public double HorizontalDominantYClampAtFull { get; set; } = 0.08;
    public int MissingHoldFrames { get; set; } = 2;
    public double MissingRecenterAlpha { get; set; } = 0.35;
}

public enum BridgePreferredEye
{
    None,
    Left,
    Right
}
