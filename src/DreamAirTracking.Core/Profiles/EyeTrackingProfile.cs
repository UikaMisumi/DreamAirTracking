using DreamAirTracking.Core.Calibration;
using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Profiles;

public sealed class EyeTrackingProfile
{
    public PixelRect? Roi { get; set; }
    public EyeCalibration Calibration { get; set; } = new();
    public Affine2DGazeModel? GazeModel { get; set; }
    public RawInputTransform InputTransform { get; set; } = new();
}
