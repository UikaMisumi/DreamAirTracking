using DreamAirTracking.Core.Bridge;

namespace DreamAirTracking.Tests;

public sealed class BridgeClosureGazeStabilizerTests
{
    [Fact]
    public void HoldsPreviousGazeWhenEyesStartClosing()
    {
        var stabilizer = new BridgeClosureGazeStabilizer();
        stabilizer.Update(Gaze(0.1, -0.1), Open(1.0));

        var state = stabilizer.Update(Gaze(0.9, 0.8), Open(0.60));

        Assert.Equal(0.1, state.LeftX, 3);
        Assert.Equal(-0.1, state.LeftY, 3);
        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.Equal(state.LeftY, state.RightY, 3);
    }

    [Fact]
    public void KeepsHoldingWhileEyesAreClosed()
    {
        var stabilizer = new BridgeClosureGazeStabilizer();
        stabilizer.Update(Gaze(-0.2, 0.3), Open(1.0));
        stabilizer.Update(Gaze(0.8, -0.8), Open(0.60));

        var state = stabilizer.Update(Gaze(1.0, -1.0), Open(0.05));

        Assert.Equal(-0.2, state.LeftX, 3);
        Assert.Equal(0.3, state.LeftY, 3);
    }

    [Fact]
    public void ReleasesSmoothlyAfterReopen()
    {
        var stabilizer = new BridgeClosureGazeStabilizer(
            new BridgeClosureGazeStabilizerOptions
            {
                HoldFramesAfterClosing = 3,
                ReopenReleaseAlpha = 0.5
            });
        stabilizer.Update(Gaze(0, 0), Open(1.0));
        stabilizer.Update(Gaze(1, 1), Open(0.60));

        var state = stabilizer.Update(Gaze(1, 1), Open(1.0));

        Assert.Equal(0.5, state.LeftX, 3);
        Assert.Equal(0.5, state.LeftY, 3);
    }

    [Fact]
    public void DoesNotHoldNormalOpenGazeMovement()
    {
        var stabilizer = new BridgeClosureGazeStabilizer();
        stabilizer.Update(Gaze(0, 0), Open(1.0));

        var state = stabilizer.Update(Gaze(0.6, -0.4), Open(0.96));

        Assert.Equal(0.6, state.LeftX, 3);
        Assert.Equal(-0.4, state.LeftY, 3);
    }

    [Fact]
    public void DoesNotHoldPartialOpenOpennessDip()
    {
        var stabilizer = new BridgeClosureGazeStabilizer();
        stabilizer.Update(Gaze(0, 0), Open(1.0));

        var state = stabilizer.Update(Gaze(0.7, 0.4), Open(0.78));

        Assert.Equal(0.7, state.LeftX, 3);
        Assert.Equal(0.4, state.LeftY, 3);
    }

    private static BridgeGazeState Gaze(double x, double y) => new(x, y, x, y);

    private static BridgeOpennessState Open(double value) => new(value, value);
}
