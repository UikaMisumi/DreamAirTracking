namespace DreamAirTracking.Core.Calibration;

public sealed class CalibrationProfile
{
    public EyeCalibration Left { get; init; } = new();
    public EyeCalibration Right { get; init; } = new();

    public static CalibrationProfile Default => new();
}

public sealed class EyeCalibration
{
    public double CenterX { get; set; } = 100;
    public double CenterY { get; set; } = 100;
    public double HorizontalRange { get; set; } = 40;
    public double VerticalRange { get; set; } = 30;
    public double HorizontalOutputSign { get; set; } = 1;
    public double VerticalOutputSign { get; set; } = 1;
    public double HorizontalOutputOffset { get; set; }
    public double VerticalOutputOffset { get; set; }
    public double OpenThreshold { get; set; } = 0.35;
    public double ClosedThreshold { get; set; } = 0.15;
}
