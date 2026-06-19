using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Tests;

public sealed class BridgeOpennessFilterTests
{
    [Fact]
    public void HoldsSingleLowEyeWhenOtherEyeIsOpen()
    {
        var filter = new BridgeOpennessFilter();
        filter.Update(Open(), Open());

        var state = filter.Update(Closed(), Open());

        Assert.True(state.Left > 0.95);
        Assert.True(state.Right > 0.95);
    }

    [Fact]
    public void ClosesBothEyesQuicklyWhenBothAreLow()
    {
        var filter = new BridgeOpennessFilter();
        filter.Update(Open(), Open());

        var state = filter.Update(Closed(), Closed());

        Assert.True(state.Left < 0.15);
        Assert.True(state.Right < 0.15);
    }

    [Fact]
    public void ReopensQuicklyAfterBlink()
    {
        var filter = new BridgeOpennessFilter();
        filter.Update(Open(), Open());
        filter.Update(Closed(), Closed());

        var state = filter.Update(Open(), Open());

        Assert.True(state.Left > 0.75);
        Assert.True(state.Right > 0.75);
    }

    [Fact]
    public void TreatsLowerPeakOpenEyeAsOpenWhenApertureIsPresent()
    {
        var filter = new BridgeOpennessFilter();

        var state = filter.Update(Detection(openness: 0.2, apertureHeight: 22, peak: 0.31), Open());

        Assert.True(state.Left > 0.75);
    }

    [Fact]
    public void KeepsSmallLowPeakApertureClosed()
    {
        var filter = new BridgeOpennessFilter();
        filter.Update(Open(), Open());

        var state = filter.Update(Detection(openness: 0.3, apertureHeight: 4, peak: 0.26), Detection(openness: 0.3, apertureHeight: 4, peak: 0.26));

        Assert.True(state.Left < 0.15);
        Assert.True(state.Right < 0.15);
    }

    [Fact]
    public void HoldsAsymmetricPartialDropWhenOtherEyeStaysOpen()
    {
        var filter = new BridgeOpennessFilter();
        filter.Update(Open(), Open());

        var state = filter.Update(Partial(), Open());

        Assert.True(state.Left > 0.95);
        Assert.True(state.Right > 0.95);
    }

    [Fact]
    public void KeepsSustainedAsymmetricDropHeldWhenOtherEyeIsFullyOpen()
    {
        var filter = new BridgeOpennessFilter();
        filter.Update(Open(), Open());

        var state = filter.Update(Closed(), Open());
        for (var i = 0; i < 5; i++)
        {
            state = filter.Update(Closed(), Open());
        }

        Assert.True(state.Left > 0.95);
        Assert.True(state.Right > 0.95);
    }

    [Fact]
    public void AllowsSlowBilateralCloseWhenOtherEyeIsPartlyClosing()
    {
        var filter = new BridgeOpennessFilter();
        filter.Update(Open(), Open());

        var state = filter.Update(Closed(), HalfClosing());
        for (var i = 0; i < 3; i++)
        {
            state = filter.Update(Closed(), HalfClosing());
        }

        Assert.True(state.Left < 0.55);
        Assert.InRange(state.Right, 0.45, 0.85);
    }

    [Fact]
    public void AllowsBilateralPartialDrop()
    {
        var filter = new BridgeOpennessFilter();
        filter.Update(Open(), Open());

        var state = filter.Update(Partial(), Partial());

        Assert.InRange(state.Left, 0.55, 0.90);
        Assert.InRange(state.Right, 0.55, 0.90);
    }

    [Fact]
    public void KeepsStrongOpenEvidenceAboveHalfClosedRange()
    {
        var filter = new BridgeOpennessFilter();
        filter.Update(TallOpen(), TallOpen());

        var state = filter.Update(Detection(openness: 1.0, apertureHeight: 18, peak: 0.30), Detection(openness: 1.0, apertureHeight: 18, peak: 0.30));

        Assert.True(state.Left > 0.85);
        Assert.True(state.Right > 0.85);
    }

    private static EyeOpennessDetection Open() => Detection(1.0);

    private static EyeOpennessDetection TallOpen() => Detection(openness: 1.0, apertureHeight: 60, peak: 0.50);

    private static EyeOpennessDetection Closed() => Detection(0.0);

    private static EyeOpennessDetection Partial() => Detection(openness: 0.45, apertureHeight: 20, peak: 0.26);

    private static EyeOpennessDetection HalfClosing() => Detection(openness: 0.60, apertureHeight: 18, peak: 0.27);

    private static EyeOpennessDetection Detection(double openness) =>
        new(true, openness, 10, 30, 21, openness, openness * 0.5, 90, "test");

    private static EyeOpennessDetection Detection(double openness, int apertureHeight, double peak) =>
        new(true, openness, 10, 10 + apertureHeight, apertureHeight, peak, peak * 0.5, 90, "test");
}
