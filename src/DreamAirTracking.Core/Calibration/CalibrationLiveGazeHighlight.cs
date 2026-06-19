namespace DreamAirTracking.Core.Calibration;

public sealed record CalibrationLiveGazeHighlight(
    string StageId,
    CalibrationTarget Target,
    bool IsPaused,
    double Openness,
    bool HasUsableEye,
    string Reason);
