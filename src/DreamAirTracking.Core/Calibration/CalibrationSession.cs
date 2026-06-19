namespace DreamAirTracking.Core.Calibration;

public sealed class CalibrationSession
{
    public string SchemaVersion { get; set; } = "0.1";
    public string SessionId { get; set; } = string.Empty;
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.Now;
    public CalibrationDeviceInfo Device { get; set; } = new();
    public CalibrationOperatorInfo Operator { get; set; } = new();
    public string ProfileInput { get; set; } = string.Empty;
    public string Notes { get; set; } = string.Empty;
    public List<CalibrationStage> Stages { get; set; } = new();
}

public sealed class CalibrationDeviceInfo
{
    public string Source { get; set; } = "BrokenEye HTTP";
    public string Host { get; set; } = "127.0.0.1";
    public int Port { get; set; } = 5555;
}

public sealed class CalibrationOperatorInfo
{
    public string DisplayName { get; set; } = string.Empty;
}
