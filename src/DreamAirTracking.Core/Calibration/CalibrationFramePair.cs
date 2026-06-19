namespace DreamAirTracking.Core.Calibration;

public sealed record CalibrationFramePair(
    long Sequence,
    string StageId,
    string LeftFile,
    string RightFile,
    double DeltaMs,
    bool LeftFound,
    double LeftRawX,
    double LeftRawY,
    double LeftConfidence,
    double LeftOpenness,
    bool RightFound,
    double RightRawX,
    double RightRawY,
    double RightConfidence,
    double RightOpenness);
