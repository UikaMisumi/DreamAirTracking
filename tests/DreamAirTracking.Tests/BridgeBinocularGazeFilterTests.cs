using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Tests;

public sealed class BridgeBinocularGazeFilterTests
{
    [Fact]
    public void FusesCoherentEyesForBothOutputs()
    {
        var filter = new BridgeBinocularGazeFilter();

        var state = filter.Update(
            new BridgeGazeState(-0.4, 0.1, -0.2, 0.2),
            Point(-0.4, 0.1, 0.2),
            Point(-0.2, 0.2, 0.1));

        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.Equal(state.LeftY, state.RightY, 3);
        Assert.InRange(state.LeftX, -0.34, -0.33);
    }

    [Fact]
    public void UsesDominantConfidenceDuringOppositeDirectionDisagreement()
    {
        var filter = new BridgeBinocularGazeFilter();
        filter.Update(
            new BridgeGazeState(-0.3, 0, -0.3, 0),
            Point(-0.3, 0, 0.2),
            Point(-0.3, 0, 0.2));

        var state = filter.Update(
            new BridgeGazeState(-0.35, 0, 0.55, 0),
            Point(-0.35, 0, 0.07),
            Point(0.55, 0, 0.2));

        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.InRange(state.LeftX, 0.54, 0.56);
    }

    [Fact]
    public void FusesOppositeDirectionInsteadOfStickingToPreviousExtreme()
    {
        var filter = new BridgeBinocularGazeFilter();
        filter.Update(
            new BridgeGazeState(1, 0, 1, 0),
            Point(1, 0, 0.2),
            Point(1, 0, 0.2));

        var state = filter.Update(
            new BridgeGazeState(0.60, 0, -0.50, 0),
            Point(0.60, 0, 0.10),
            Point(-0.50, 0, 0.09));

        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.InRange(state.LeftX, -0.10, 0.20);
    }

    [Fact]
    public void PreferredHorizontalEyeCanOverrideBiasedOtherEye()
    {
        var filter = new BridgeBinocularGazeFilter(new BridgeBinocularGazeFilterOptions
        {
            PreferredHorizontalEye = BridgePreferredEye.Left
        });

        var state = filter.Update(
            new BridgeGazeState(0.45, 0.1, -0.35, 0.2),
            Point(0.45, 0.1, 0.16),
            Point(-0.35, 0.2, 0.14));

        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.Equal(0.45, state.LeftX, 3);
        Assert.InRange(state.LeftY, 0.14, 0.16);
    }

    [Fact]
    public void IgnoresPreferredHorizontalEyeWhenConfidenceIsTooLow()
    {
        var filter = new BridgeBinocularGazeFilter(new BridgeBinocularGazeFilterOptions
        {
            PreferredHorizontalEye = BridgePreferredEye.Left
        });

        var state = filter.Update(
            new BridgeGazeState(0.8, 0, -0.4, 0),
            Point(0.8, 0, 0.05),
            Point(-0.4, 0, 0.2));

        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.Equal(-0.4, state.LeftX, 3);
    }

    [Fact]
    public void ClampsHorizontalDriftDuringStrongVerticalGaze()
    {
        var filter = new BridgeBinocularGazeFilter();

        var state = filter.Update(
            new BridgeGazeState(0.45, -0.7, -0.5, -0.6),
            Point(0.45, -0.7, 0.16),
            Point(-0.5, -0.6, 0.14));

        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.InRange(state.LeftX, -0.05, 0.05);
        Assert.True(state.LeftY < -0.60);
    }

    [Fact]
    public void KeepsUpwardVerticalClampLessAggressive()
    {
        var filter = new BridgeBinocularGazeFilter();

        var state = filter.Update(
            new BridgeGazeState(0.45, 0.7, -0.5, 0.6),
            Point(0.45, 0.7, 0.16),
            Point(-0.5, 0.6, 0.14));

        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.InRange(state.LeftX, -0.05, 0.05);
        Assert.True(state.LeftY > 0.60);
    }

    [Fact]
    public void FusesConflictingHorizontalGazeNearVerticalCenter()
    {
        var filter = new BridgeBinocularGazeFilter();

        var state = filter.Update(
            new BridgeGazeState(0.45, 0.1, -0.35, 0.2),
            Point(0.45, 0.1, 0.16),
            Point(-0.35, 0.2, 0.14));

        Assert.InRange(state.LeftX, 0.05, 0.10);
    }

    [Fact]
    public void ClampsVerticalDriftDuringStrongHorizontalGaze()
    {
        var filter = new BridgeBinocularGazeFilter();

        var state = filter.Update(
            new BridgeGazeState(-0.5, 0.0, -0.5, 0.7),
            Point(-0.5, 0.0, 0.1),
            Point(-0.5, 0.7, 0.3));

        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.True(state.LeftX < -0.45);
        Assert.InRange(state.LeftY, -0.14, 0.14);
    }

    [Fact]
    public void KeepsDiagonalGazeWhenBothEyesAgreeVertically()
    {
        var filter = new BridgeBinocularGazeFilter();

        var state = filter.Update(
            new BridgeGazeState(-0.5, 0.55, -0.45, 0.60),
            Point(-0.5, 0.55, 0.2),
            Point(-0.45, 0.60, 0.2));

        Assert.True(state.LeftX < -0.45);
        Assert.True(state.LeftY > 0.55);
    }

    [Fact]
    public void UsesHorizontalMagnitudeDominanceWhenOneEyeStaysNearCenter()
    {
        var filter = new BridgeBinocularGazeFilter();

        var state = filter.Update(
            new BridgeGazeState(0.08, 0.02, 0.88, 0.04),
            Point(0.08, 0.02, 0.22),
            Point(0.88, 0.04, 0.12));

        Assert.True(state.LeftX > 0.80);
        Assert.Equal(state.LeftX, state.RightX, 3);
    }

    [Fact]
    public void UsesMagnitudeDominantAxisWhenOneEyeCarriesTheDirection()
    {
        var filter = new BridgeBinocularGazeFilter();

        var state = filter.Update(
            new BridgeGazeState(-0.05, -0.72, -0.12, -0.08),
            Point(-0.05, -0.72, 0.05),
            Point(-0.12, -0.08, 0.22));

        Assert.True(state.LeftY < -0.65);
        Assert.Equal(state.LeftY, state.RightY, 3);
    }

    [Fact]
    public void FallsBackToSingleValidEyeForBothOutputs()
    {
        var filter = new BridgeBinocularGazeFilter();

        var state = filter.Update(
            new BridgeGazeState(0.8, -0.1, -0.4, 0.2),
            Point(0.8, -0.1, 0.2),
            new NormalizedEyePoint(false, 0, 0, 0));

        Assert.Equal(0.8, state.LeftX, 3);
        Assert.Equal(0.8, state.RightX, 3);
        Assert.Equal(-0.1, state.LeftY, 3);
        Assert.Equal(-0.1, state.RightY, 3);
    }

    [Fact]
    public void HoldsBriefMissingBothEyesThenRecentersPreviousGaze()
    {
        var filter = new BridgeBinocularGazeFilter();
        filter.Update(
            new BridgeGazeState(1, 0.4, 1, 0.4),
            Point(1, 0.4, 0.2),
            Point(1, 0.4, 0.2));

        var missing = new NormalizedEyePoint(false, 0, 0, 0);
        var state = filter.Update(new BridgeGazeState(0, 0, 0, 0), missing, missing);

        Assert.Equal(1, state.LeftX, 3);
        Assert.Equal(0.4, state.LeftY, 3);

        filter.Update(new BridgeGazeState(0, 0, 0, 0), missing, missing);
        state = filter.Update(new BridgeGazeState(0, 0, 0, 0), missing, missing);

        Assert.InRange(state.LeftX, 0.60, 0.70);
        Assert.InRange(state.LeftY, 0.20, 0.30);
        Assert.Equal(state.LeftX, state.RightX, 3);
        Assert.Equal(state.LeftY, state.RightY, 3);
    }

    private static NormalizedEyePoint Point(double x, double y, double confidence) =>
        new(true, confidence, x, y);
}
