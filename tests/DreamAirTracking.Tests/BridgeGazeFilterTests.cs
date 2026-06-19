using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Tests;

public sealed class BridgeGazeFilterTests
{
    [Fact]
    public void HoldsSingleFrameLowConfidenceSpike()
    {
        var filter = new BridgeGazeFilter();
        filter.Update(Point(0, 0, 0.2), Point(0, 0, 0.2));

        var state = filter.Update(Point(1, 0, 0.05), Point(0, 0, 0.2));

        Assert.Equal(0, state.LeftX, 3);
    }

    [Fact]
    public void AllowsStableLowConfidenceMovementAfterConfirmation()
    {
        var filter = new BridgeGazeFilter();
        filter.Update(Point(0, 0, 0.2), Point(0, 0, 0.2));

        filter.Update(Point(1, 0, 0.05), Point(0, 0, 0.2));
        filter.Update(Point(1, 0, 0.05), Point(0, 0, 0.2));
        var state = filter.Update(Point(1, 0, 0.05), Point(0, 0, 0.2));

        Assert.InRange(state.LeftX, 0.01, 0.20);
    }

    [Fact]
    public void AllowsHighConfidenceMovement()
    {
        var filter = new BridgeGazeFilter();
        filter.Update(Point(0, 0, 0.2), Point(0, 0, 0.2));

        var state = filter.Update(Point(1, 0, 0.3), Point(0, 0, 0.2));

        Assert.True(state.LeftX > 0.5);
    }

    [Fact]
    public void RecentersLowConfidenceCenterTargetWithoutConfirmationDelay()
    {
        var filter = new BridgeGazeFilter();
        filter.Update(Point(1, 0, 0.2), Point(-1, 0, 0.2));

        var state = filter.Update(Point(0.08, 0.04, 0.055), Point(-0.06, -0.03, 0.055));

        Assert.InRange(state.LeftX, -0.25, 0.25);
        Assert.InRange(state.RightX, -0.25, 0.25);
    }

    [Fact]
    public void ContractsFromExtremeWhenLowConfidenceTargetMovesTowardCenter()
    {
        var filter = new BridgeGazeFilter();
        filter.Update(Point(1, 0, 0.2), Point(1, 0, 0.2));

        var state = filter.Update(Point(-0.28, 0.02, 0.055), Point(-0.28, 0.02, 0.055));

        Assert.InRange(state.LeftX, -0.20, 0.20);
        Assert.InRange(state.RightX, -0.20, 0.20);
    }

    [Fact]
    public void HoldsBriefMissingPointThenRecenters()
    {
        var filter = new BridgeGazeFilter();
        filter.Update(Point(0.5, -0.25, 0.2), Point(0, 0, 0.2));

        var state = filter.Update(new NormalizedEyePoint(false, 0, 0, 0), Point(0, 0, 0.2));

        Assert.Equal(0.5, state.LeftX, 3);
        Assert.Equal(-0.25, state.LeftY, 3);

        filter.Update(new NormalizedEyePoint(false, 0, 0, 0), Point(0, 0, 0.2));
        state = filter.Update(new NormalizedEyePoint(false, 0, 0, 0), Point(0, 0, 0.2));

        Assert.InRange(state.LeftX, 0.30, 0.40);
        Assert.InRange(state.LeftY, -0.17, -0.10);
    }

    private static NormalizedEyePoint Point(double x, double y, double confidence) =>
        new(true, confidence, x, y);
}
