namespace DreamAirTracking.Core.Calibration;

public sealed class CalibrationStage
{
    public int Order { get; set; }
    public string StageId { get; set; } = string.Empty;
    public CalibrationTarget Target { get; set; }
    public double CaptureSeconds { get; set; } = 1.8;
    public string DisplayName { get; set; } = string.Empty;
}
