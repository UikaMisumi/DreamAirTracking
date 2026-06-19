using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Core.Profiles;

public sealed class TrackingProfile
{
    public string Version { get; set; } = "0.1";
    public BridgeOutputMapperOptions Output { get; set; } = new();
    public EyeTrackingProfile Left { get; set; } = new();
    public EyeTrackingProfile Right { get; set; } = new();

    public EyeTrackingProfile GetEye(EyeSide side) => side == EyeSide.Left ? Left : Right;

    public PupilDetectorOptions CreateDetectorOptions(EyeSide side, PupilDetection? previous = null)
    {
        var eye = GetEye(side);
        return new PupilDetectorOptions
        {
            Roi = eye.Roi,
            ExpectedCenterX = previous is { Found: true } ? previous.CenterX : eye.Calibration.CenterX,
            ExpectedCenterY = previous is { Found: true } ? previous.CenterY : eye.Calibration.CenterY,
            PreviousCenterWeight = previous is { Found: true } ? 2.0 : 1.2
        };
    }
}
