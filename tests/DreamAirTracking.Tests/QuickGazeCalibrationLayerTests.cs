using DreamAirTracking.Core.Bridge;

namespace DreamAirTracking.Tests;

public sealed class QuickGazeCalibrationLayerTests
{
    [Fact]
    public void AppliesAffineCorrectionAndClamps()
    {
        var layer = new QuickGazeCalibrationLayer(
            a00: 2.0,
            a01: 0.5,
            a10: -0.25,
            a11: 1.5,
            b0: 0.1,
            b1: -0.2,
            session: "test");

        var corrected = layer.Apply(new BridgeGazeState(0.2, 0.4, 0.9, 0.9));

        Assert.Equal(0.7, corrected.LeftX, 3);
        Assert.Equal(0.35, corrected.LeftY, 3);
        Assert.Equal(1.0, corrected.RightX, 3);
        Assert.Equal(0.925, corrected.RightY, 3);
    }

    [Fact]
    public void LoadsRequestedSessionFromEvaluatorArtifact()
    {
        var json = """
        {
          "schema": "dream_air_tracking.quick_gaze_calibration_layer.v1",
          "raw_source": "bridge_output_xy",
          "sessions": {
            "unused": {
              "used": false
            },
            "session_b": {
              "used": true,
              "A": [[1.0, 0.2], [-0.1, 1.0]],
              "b": [0.3, -0.4]
            }
          }
        }
        """;

        var layer = QuickGazeCalibrationLayer.FromJson(json, "session_b");
        var corrected = layer.Apply(new BridgeGazeState(0.5, 0.25, 0.5, 0.25));

        Assert.Equal("session_b", layer.Session);
        Assert.Equal(0.85, corrected.LeftX, 3);
        Assert.Equal(-0.2, corrected.LeftY, 3);
    }

    [Fact]
    public void RejectsMissingRequestedSession()
    {
        var json = """
        {
          "raw_source": "bridge_output_xy",
          "sessions": {
            "session_a": {
              "used": true,
              "A": [[1.0, 0.0], [0.0, 1.0]],
              "b": [0.0, 0.0]
            }
          }
        }
        """;

        Assert.Throws<InvalidDataException>(() => QuickGazeCalibrationLayer.FromJson(json, "missing"));
    }

    [Fact]
    public void RejectsModelPredictionLayerForBridgeRuntime()
    {
        var json = """
        {
          "raw_source": "model_prediction_xy",
          "sessions": {
            "session_a": {
              "used": true,
              "A": [[1.0, 0.0], [0.0, 1.0]],
              "b": [0.0, 0.0]
            }
          }
        }
        """;

        Assert.Throws<InvalidDataException>(() => QuickGazeCalibrationLayer.FromJson(json));
    }
}
