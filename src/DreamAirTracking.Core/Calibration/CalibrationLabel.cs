using System.Text.Json.Serialization;

namespace DreamAirTracking.Core.Calibration;

public sealed class CalibrationLabel
{
    public string StageId { get; set; } = string.Empty;
    public CalibrationTarget Target { get; set; }
    public long FrameStart { get; set; }
    public long FrameEnd { get; set; }
    public bool Accepted { get; set; } = true;
    public bool Edited { get; set; }
    public List<long> BadFrames { get; set; } = new();
    public string OperatorVerdict { get; set; } = "good";
    public string Notes { get; set; } = string.Empty;
    public CalibrationTarget? OperatorOverrideTarget { get; set; }

    [JsonIgnore]
    public CalibrationTarget EffectiveTarget => OperatorOverrideTarget ?? Target;
}
