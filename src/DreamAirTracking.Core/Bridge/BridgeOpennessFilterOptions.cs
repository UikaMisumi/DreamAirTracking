namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeOpennessFilterOptions
{
    public double ClosedThreshold { get; set; } = 0.20;
    public double OtherEyeOpenThreshold { get; set; } = 0.60;
    public double ClosedPeakFraction { get; set; } = 0.18;
    public double InitialOpenPeakFraction { get; set; } = 0.35;
    public double MinOpenBaselinePeakFraction { get; set; } = 0.30;
    public int MinOpenBaselineApertureHeight { get; set; } = 18;
    public double OpenBaselineAlpha { get; set; } = 0.08;
    public double OpenBaselineDownAlpha { get; set; } = 0.01;
    public double PeakOpennessWeight { get; set; } = 0.70;
    public double OpenEvidenceDetectorThreshold { get; set; } = 0.90;
    public double OpenEvidenceFloor { get; set; } = 0.75;
    public double OpenHoldThreshold { get; set; } = 0.80;
    public double SingleEyeDropThreshold { get; set; } = 0.70;
    public double OtherEyeFullyOpenHoldThreshold { get; set; } = 0.78;
    public int AsymmetricDropConfirmFrames { get; set; } = 2;
    public double ClosingCurveGamma { get; set; } = 1.35;
    public double RiseAlpha { get; set; } = 0.75;
    public double FallAlpha { get; set; } = 0.55;
    public double BothClosedFallAlpha { get; set; } = 0.90;
}
