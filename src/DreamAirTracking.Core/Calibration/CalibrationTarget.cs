namespace DreamAirTracking.Core.Calibration;

public readonly record struct CalibrationTarget(double X, double Y)
{
    public static CalibrationTarget Center => new(0, 0);
}
