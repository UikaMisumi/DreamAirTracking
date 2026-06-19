namespace DreamAirTracking.Core.Bridge;

public sealed record BridgeEyeState(
    bool Found,
    double Confidence,
    double RawX,
    double RawY,
    double NormalizedX,
    double NormalizedY,
    double Openness)
{
    public double PupilOpenness { get; init; }
    public double ApertureOpenness { get; init; }
    public double CalibratedOpenness { get; init; }
    public double OutputOpenness { get; init; }
    public double Wide { get; init; }
    public double Squint { get; init; }
    public double AperturePeakDarkFraction { get; init; }
    public int ApertureHeight { get; init; }
    public string ApertureReason { get; init; } = string.Empty;
    public double RawNormalizedX { get; init; }
    public double RawNormalizedY { get; init; }
    public double MonocularNormalizedX { get; init; }
    public double MonocularNormalizedY { get; init; }
    public bool PupilDiameterFound { get; init; }
    public bool PupilDiameterQuality { get; init; }
    public double PupilDiameterConfidence { get; init; }
    public double PupilDiameterPx { get; init; }
    public double PupilDiameterNormalized { get; init; }
    public double PupilExpressionNormalized { get; init; } = 0.5;
    public double PupilGeometryRadius { get; init; }
    public double PupilDiameterAxisRatio { get; init; }
    public string PupilDiameterReason { get; init; } = string.Empty;
    public bool NormalizationFound { get; init; }
    public double NormalizationConfidence { get; init; }
    public double NormalizationShiftX { get; init; }
    public double NormalizationShiftY { get; init; }
    public double NormalizationScale { get; init; } = 1.0;
    public double NormalizationPupilX { get; init; } = 0.5;
    public double NormalizationPupilY { get; init; } = 0.54;
    public string? NormalizationDropReason { get; init; }
}
