using DreamAirTracking.Core.Bridge;

namespace DreamAirTracking.Tests;

public sealed class BridgeOutputMapperTests
{
    [Fact]
    public void LocksSmallCenterJitterToZero()
    {
        var mapper = new BridgeOutputMapper();

        var centered = mapper.Update(new BridgeGazeState(0.12, -0.08, 0.10, -0.06));
        var jitter = mapper.Update(new BridgeGazeState(-0.20, 0.02, -0.18, 0.04));

        Assert.Equal(0, centered.LeftX, 3);
        Assert.Equal(0, centered.LeftY, 3);
        Assert.Equal(0, jitter.LeftX, 3);
        Assert.Equal(0, jitter.LeftY, 3);
    }

    [Fact]
    public void ReleasesCenterLockForIntentionalSideGaze()
    {
        var mapper = new BridgeOutputMapper();
        mapper.Update(new BridgeGazeState(0.08, 0.02, 0.09, 0.01));

        var side = mapper.Update(new BridgeGazeState(0.32, 0.02, 0.30, 0.02));

        Assert.True(side.LeftX > 0.35);
        Assert.Equal(0, side.LeftY, 3);
    }

    [Fact]
    public void ReleasesCenterLockForIntentionalLeftGaze()
    {
        var mapper = new BridgeOutputMapper();
        mapper.Update(new BridgeGazeState(0.08, 0.02, 0.09, 0.01));

        var side = mapper.Update(new BridgeGazeState(-0.32, 0.02, -0.30, 0.02));

        Assert.True(side.LeftX < -0.35);
        Assert.Equal(0, side.LeftY, 3);
    }
}
