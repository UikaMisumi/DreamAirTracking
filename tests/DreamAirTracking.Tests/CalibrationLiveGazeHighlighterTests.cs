using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class CalibrationLiveGazeHighlighterTests
{
    private readonly CalibrationLiveGazeHighlighter _highlighter = new();

    [Fact]
    public void MapsLiveGazeToNearestNinePoint()
    {
        var state = State(Eye(0.82, 0.70), Eye(0.78, 0.74));

        var highlight = _highlighter.Evaluate(state);

        Assert.False(highlight.IsPaused);
        Assert.Equal("right_up", highlight.StageId);
        Assert.InRange(highlight.Target.X, 0.79, 0.83);
        Assert.InRange(highlight.Target.Y, 0.71, 0.75);
    }

    [Fact]
    public void KeepsSmallOffsetsInCenterCell()
    {
        var state = State(Eye(0.22, -0.18), Eye(0.18, -0.12));

        var highlight = _highlighter.Evaluate(state);

        Assert.False(highlight.IsPaused);
        Assert.Equal("center", highlight.StageId);
    }

    [Fact]
    public void UsesUsableEyeWhenOtherEyeIsLowConfidence()
    {
        var state = State(Eye(-0.75, -0.08), Eye(0.90, 0.80, confidence: 0.01));

        var highlight = _highlighter.Evaluate(state);

        Assert.False(highlight.IsPaused);
        Assert.True(highlight.HasUsableEye);
        Assert.Equal("left", highlight.StageId);
        Assert.InRange(highlight.Target.X, -0.78, -0.72);
    }

    [Fact]
    public void EntersLeftCellAtModerateLeftGaze()
    {
        var state = State(Eye(-0.32, 0.02), Eye(-0.30, 0.01));

        var highlight = _highlighter.Evaluate(state, "center", CalibrationTarget.Center);

        Assert.False(highlight.IsPaused);
        Assert.Equal("left", highlight.StageId);
    }

    [Fact]
    public void KeepsLeftCellThroughThresholdJitter()
    {
        var state = State(Eye(-0.24, 0.02), Eye(-0.22, 0.01));

        var highlight = _highlighter.Evaluate(state, "left", new CalibrationTarget(-0.34, 0.0));

        Assert.False(highlight.IsPaused);
        Assert.Equal("left", highlight.StageId);
    }

    [Fact]
    public void KeepsLeftDownCellThroughThresholdJitter()
    {
        var state = State(Eye(-0.24, -0.24), Eye(-0.22, -0.22));

        var highlight = _highlighter.Evaluate(state, "left_down", new CalibrationTarget(-0.35, -0.35));

        Assert.False(highlight.IsPaused);
        Assert.Equal("left_down", highlight.StageId);
    }

    [Fact]
    public void HoldsPreviousHighlightWhenEyesClose()
    {
        var previousTarget = new CalibrationTarget(-0.70, 0.05);
        var state = State(Eye(0.90, 0.90, openness: 0.04), Eye(0.85, 0.80, openness: 0.03));

        var highlight = _highlighter.Evaluate(state, "left", previousTarget);

        Assert.True(highlight.IsPaused);
        Assert.Equal("closed", highlight.Reason);
        Assert.Equal("left", highlight.StageId);
        Assert.Equal(previousTarget, highlight.Target);
    }

    [Fact]
    public void PausesWhenNoEyeIsUsable()
    {
        var state = State(Eye(-0.90, 0.80, found: false), Eye(0.90, -0.80, confidence: 0.01));

        var highlight = _highlighter.Evaluate(state, "up", new CalibrationTarget(0, 0.80));

        Assert.True(highlight.IsPaused);
        Assert.Equal("low_confidence", highlight.Reason);
        Assert.Equal("up", highlight.StageId);
    }

    private static BridgeTrackingState State(BridgeEyeState left, BridgeEyeState right) =>
        new(1, DateTimeOffset.UnixEpoch, 0, left, right);

    private static BridgeEyeState Eye(
        double x,
        double y,
        double confidence = 0.80,
        double openness = 1.0,
        bool found = true) =>
        new(found, confidence, 0, 0, x, y, openness);
}
