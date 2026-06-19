using System.Text.Json;
using DreamAirTracking.Core.Bridge;

namespace DreamAirTracking.Tests;

public sealed class BridgeContractTests
{
    [Fact]
    public void SerializesLegacyAndDiagnosticOpennessFields()
    {
        var eye = new BridgeEyeState(true, 0.8, 101, 92, 0.2, -0.1, 1.0)
        {
            PupilOpenness = 1.0,
            ApertureOpenness = 0.42,
            CalibratedOpenness = 0.72,
            OutputOpenness = 0.95,
            Wide = 0.34,
            Squint = 0.12,
            AperturePeakDarkFraction = 0.31,
            ApertureHeight = 18,
            ApertureReason = "ok",
            RawNormalizedX = 0.7,
            RawNormalizedY = -0.6,
            MonocularNormalizedX = 0.25,
            MonocularNormalizedY = -0.15,
            PupilDiameterFound = true,
            PupilDiameterQuality = true,
            PupilDiameterConfidence = 0.4,
            PupilDiameterPx = 42.5,
            PupilDiameterNormalized = 0.65,
            PupilExpressionNormalized = 0.65,
            PupilGeometryRadius = 0.11,
            PupilDiameterAxisRatio = 1.2,
            NormalizationFound = true,
            NormalizationConfidence = 0.33,
            NormalizationShiftX = -0.04,
            NormalizationShiftY = 0.02,
            NormalizationScale = 1.08,
            NormalizationPupilX = 0.46,
            NormalizationPupilY = 0.56
        };

        var json = JsonSerializer.Serialize(eye, new JsonSerializerOptions(JsonSerializerDefaults.Web));

        Assert.Contains("\"openness\":1", json);
        Assert.Contains("\"apertureOpenness\":0.42", json);
        Assert.Contains("\"calibratedOpenness\":0.72", json);
        Assert.Contains("\"outputOpenness\":0.95", json);
        Assert.Contains("\"wide\":0.34", json);
        Assert.Contains("\"squint\":0.12", json);
        Assert.Contains("\"apertureHeight\":18", json);
        Assert.Contains("\"rawNormalizedX\":0.7", json);
        Assert.Contains("\"monocularNormalizedX\":0.25", json);
        Assert.Contains("\"pupilDiameterQuality\":true", json);
        Assert.Contains("\"pupilDiameterNormalized\":0.65", json);
        Assert.Contains("\"pupilExpressionNormalized\":0.65", json);
        Assert.Contains("\"pupilGeometryRadius\":0.11", json);
        Assert.Contains("\"normalizationFound\":true", json);
        Assert.Contains("\"normalizationConfidence\":0.33", json);
        Assert.Contains("\"normalizationShiftX\":-0.04", json);
        Assert.Contains("\"normalizationScale\":1.08", json);
    }

    [Fact]
    public void LegacyEyeDtoIgnoresDiagnosticFields()
    {
        var eye = new BridgeEyeState(true, 0.8, 101, 92, 0.2, -0.1, 1.0)
        {
            ApertureOpenness = 0.42,
            ApertureHeight = 18
        };
        var json = JsonSerializer.Serialize(eye, new JsonSerializerOptions(JsonSerializerDefaults.Web));

        var legacy = JsonSerializer.Deserialize<LegacyEyeDto>(json, new JsonSerializerOptions(JsonSerializerDefaults.Web));

        Assert.NotNull(legacy);
        Assert.Equal(1.0, legacy.Openness);
        Assert.Equal(0.2, legacy.NormalizedX);
    }

    [Fact]
    public void SerializesBridgeOutputSwitches()
    {
        var eye = new BridgeEyeState(true, 0.8, 101, 92, 0.2, -0.1, 1.0);
        var state = new BridgeTrackingState(1, DateTimeOffset.UnixEpoch, 2.5, eye, eye)
        {
            EyeTrackingEnabled = false,
            EyeExpressionEnabled = true,
            EyeExpressionMode = "steamlink_eye_shapes",
            PupilDiameterEnabled = true,
            PupilDiameterMode = "expression_constrict_on_wide",
            NormalizationMode = "pupil_center"
        };

        var json = JsonSerializer.Serialize(state, new JsonSerializerOptions(JsonSerializerDefaults.Web));

        Assert.Contains("\"eyeTrackingEnabled\":false", json);
        Assert.Contains("\"eyeExpressionEnabled\":true", json);
        Assert.Contains("\"eyeExpressionMode\":\"steamlink_eye_shapes\"", json);
        Assert.Contains("\"pupilDiameterEnabled\":true", json);
        Assert.Contains("\"pupilDiameterMode\":\"expression_constrict_on_wide\"", json);
        Assert.Contains("\"normalizationMode\":\"pupil_center\"", json);
    }

    private sealed class LegacyEyeDto
    {
        public bool Found { get; set; }
        public double NormalizedX { get; set; }
        public double Openness { get; set; }
    }
}
