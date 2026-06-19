namespace DreamAirTracking.Core.Tracking;

public sealed class PupilDiameterCalibrationProfile
{
    public int TailPairs { get; set; }
    public double MinConfidence { get; set; } = 0.28;
    public double MaxAxisRatio { get; set; } = 1.8;
    public double MinOpenness { get; set; } = 0.75;
    public PupilDiameterEyeCalibrationProfile Left { get; set; } = new();
    public PupilDiameterEyeCalibrationProfile Right { get; set; } = new();

    public PupilDiameterEyeCalibrationProfile GetEye(EyeSide side) =>
        side == EyeSide.Left ? Left : Right;
}

public sealed class PupilDiameterEyeCalibrationProfile
{
    public string Side { get; set; } = string.Empty;
    public bool Valid { get; set; }
    public double BrightPx { get; set; }
    public double DarkPx { get; set; }
    public double RangePx { get; set; }
    public double RelativeRange { get; set; }
    public bool RepeatabilityWarning { get; set; }
}
