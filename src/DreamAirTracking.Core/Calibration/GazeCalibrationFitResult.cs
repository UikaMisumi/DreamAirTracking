namespace DreamAirTracking.Core.Calibration;

public sealed class GazeCalibrationFitResult
{
    public Affine2DGazeModel? LeftModel { get; init; }
    public Affine2DGazeModel? RightModel { get; init; }
    public CalibrationMetrics Metrics { get; init; } = new();

    public bool Succeeded => LeftModel is not null || RightModel is not null;
}
