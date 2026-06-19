namespace DreamAirTracking.Core.Tracking;

public sealed class EyeTracker
{
    private readonly PupilDetector _detector = new();
    private readonly DreamAirTracking.Core.Profiles.TrackingProfile _profile;
    private PupilDetection? _previousLeft;
    private PupilDetection? _previousRight;

    public EyeTracker(DreamAirTracking.Core.Profiles.TrackingProfile? profile = null)
    {
        _profile = profile ?? new Profiles.TrackingProfile();
    }

    public PupilDetection Process(EyeFrame frame)
    {
        var previous = frame.Side == EyeSide.Left ? _previousLeft : _previousRight;
        var options = _profile.CreateDetectorOptions(frame.Side, previous);

        var detection = _detector.Detect(frame, options);
        if (frame.Side == EyeSide.Left)
        {
            _previousLeft = detection;
        }
        else
        {
            _previousRight = detection;
        }

        return detection;
    }
}
