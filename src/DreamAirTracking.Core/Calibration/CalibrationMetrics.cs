namespace DreamAirTracking.Core.Calibration;

public sealed class CalibrationMetrics
{
    public string Model { get; set; } = "affine2d";
    public int ValidPairs { get; set; }
    public int RejectedPairs { get; set; }
    public CalibrationResidual CenterDrift { get; set; } = new();
    public Dictionary<string, CalibrationResidual> StageResiduals { get; set; } = new();
    public Dictionary<string, EyeCalibrationMetrics> Eyes { get; set; } = new();
    public string Quality { get; set; } = "unknown";
}

public sealed class EyeCalibrationMetrics
{
    public bool Succeeded { get; set; }
    public string Model { get; set; } = string.Empty;
    public int StageCount { get; set; }
    public int SampleCount { get; set; }
    public double AverageResidual { get; set; }
    public string FailureReason { get; set; } = string.Empty;
}

public sealed class CalibrationResidual
{
    public double X { get; set; }
    public double Y { get; set; }
    public double Distance { get; set; }
}
