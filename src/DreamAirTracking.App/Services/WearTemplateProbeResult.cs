namespace DreamAirTracking.App.Services;

public sealed record WearTemplateProbeResult(
    string Status,
    string Action,
    string? TemplateId,
    double? Distance,
    int SampleCount,
    bool RequirementsMet,
    string Reason,
    double? CenterOffsetX = null,
    double? CenterOffsetY = null,
    double? XGain = null,
    double? YGain = null,
    string? RuntimeCalibrationPath = null)
{
    public bool HasRuntimeCalibration =>
        CenterOffsetX is not null
        && CenterOffsetY is not null
        && XGain is not null
        && YGain is not null;

    public static WearTemplateProbeResult NotRun(string reason) =>
        new("not_run", "none", null, null, 0, false, reason);
}
