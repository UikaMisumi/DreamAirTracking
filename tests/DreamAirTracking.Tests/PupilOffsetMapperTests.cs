using DreamAirTracking.Core;
using DreamAirTracking.Core.Geometry;
using DreamAirTracking.Core.Profiles;
using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Tests;

public sealed class PupilOffsetMapperTests
{
    [Fact]
    public void MapsDetectionToNormalizedOffset()
    {
        var profile = new TrackingProfile();
        profile.Left.Calibration.CenterX = 100;
        profile.Left.Calibration.CenterY = 80;
        profile.Left.Calibration.HorizontalRange = 20;
        profile.Left.Calibration.VerticalRange = 10;
        var detection = new PupilDetection(true, 0.8, 110, 75, 80, new PixelRect(0, 0, 10, 10), 100, 0.7);

        var mapped = new PupilOffsetMapper().Map(EyeSide.Left, detection, profile);

        Assert.True(mapped.Found);
        Assert.Equal(0.8, mapped.Confidence);
        Assert.Equal(0.5, mapped.X, 3);
        Assert.Equal(-0.5, mapped.Y, 3);
    }

    [Fact]
    public void ClampsLargeOffsets()
    {
        var profile = new TrackingProfile();
        profile.Right.Calibration.CenterX = 100;
        profile.Right.Calibration.CenterY = 100;
        profile.Right.Calibration.HorizontalRange = 10;
        profile.Right.Calibration.VerticalRange = 10;
        var detection = new PupilDetection(true, 0.6, 160, 40, 80, new PixelRect(0, 0, 10, 10), 100, 0.7);

        var mapped = new PupilOffsetMapper().Map(EyeSide.Right, detection, profile);

        Assert.Equal(1, mapped.X);
        Assert.Equal(-1, mapped.Y);
    }

    [Fact]
    public void AppliesPerEyeOutputSigns()
    {
        var profile = new TrackingProfile();
        profile.Left.Calibration.CenterX = 100;
        profile.Left.Calibration.CenterY = 100;
        profile.Left.Calibration.HorizontalRange = 20;
        profile.Left.Calibration.VerticalRange = 20;
        profile.Left.Calibration.HorizontalOutputSign = -1;
        var detection = new PupilDetection(true, 0.8, 110, 90, 80, new PixelRect(0, 0, 10, 10), 100, 0.7);

        var mapped = new PupilOffsetMapper().Map(EyeSide.Left, detection, profile);

        Assert.Equal(-0.5, mapped.X, 3);
        Assert.Equal(-0.5, mapped.Y, 3);
    }

    [Fact]
    public void AppliesPerEyeOutputOffsets()
    {
        var profile = new TrackingProfile();
        profile.Left.Calibration.CenterX = 100;
        profile.Left.Calibration.CenterY = 100;
        profile.Left.Calibration.HorizontalRange = 20;
        profile.Left.Calibration.VerticalRange = 20;
        profile.Left.Calibration.VerticalOutputOffset = -0.08;
        var detection = new PupilDetection(true, 0.8, 100, 100, 80, new PixelRect(0, 0, 10, 10), 100, 0.7);

        var mapped = new PupilOffsetMapper().Map(EyeSide.Left, detection, profile);

        Assert.Equal(0, mapped.X, 3);
        Assert.Equal(-0.08, mapped.Y, 3);
    }

    [Fact]
    public void AppliesRawInputTransformBeforeMapping()
    {
        var profile = new TrackingProfile();
        profile.Left.Calibration.CenterX = 100;
        profile.Left.Calibration.CenterY = 80;
        profile.Left.Calibration.HorizontalRange = 20;
        profile.Left.Calibration.VerticalRange = 10;
        profile.Left.InputTransform = new RawInputTransform
        {
            Enabled = true,
            CurrentCenterX = 130,
            CurrentCenterY = 70,
            TargetCenterX = 100,
            TargetCenterY = 80,
            ScaleX = 0.5,
            ScaleY = 0.5
        };
        var detection = new PupilDetection(true, 0.8, 170, 70, 80, new PixelRect(0, 0, 10, 10), 100, 0.7);

        var mapped = new PupilOffsetMapper().Map(EyeSide.Left, detection, profile);

        Assert.Equal(1, mapped.X, 3);
        Assert.Equal(0, mapped.Y, 3);
    }

    [Fact]
    public void UsesAffineGazeModelWhenPresent()
    {
        var profile = new TrackingProfile();
        profile.Left.Calibration.CenterX = 100;
        profile.Left.Calibration.CenterY = 100;
        profile.Left.Calibration.HorizontalRange = 20;
        profile.Left.Calibration.VerticalRange = 20;
        profile.Left.GazeModel = new()
        {
            OutX = new[] { -2d, 0.02, 0d },
            OutY = new[] { -1d, 0d, 0.02 }
        };
        var detection = new PupilDetection(true, 0.8, 125, 75, 80, new PixelRect(0, 0, 10, 10), 100, 0.7);

        var mapped = new PupilOffsetMapper().Map(EyeSide.Left, detection, profile);

        Assert.Equal(0.5, mapped.X, 3);
        Assert.Equal(0.5, mapped.Y, 3);
    }

    [Fact]
    public void UsesQuadraticGazeModelWhenPresent()
    {
        var profile = new TrackingProfile();
        profile.Left.GazeModel = new()
        {
            Type = "quadratic2d",
            OutX = new[] { 0d, 1d, 0d, 0d, 0.25, 0d },
            OutY = new[] { 0d, 0d, 1d, 0d, 0d, 0d }
        };
        profile.Left.Calibration.CenterX = 100;
        profile.Left.Calibration.CenterY = 100;
        profile.Left.Calibration.HorizontalRange = 20;
        profile.Left.Calibration.VerticalRange = 20;
        var detection = new PupilDetection(true, 0.8, 0.5, 0.5, 80, new PixelRect(0, 0, 10, 10), 100, 0.7);

        var mapped = new PupilOffsetMapper().Map(EyeSide.Left, detection, profile);

        Assert.Equal(0.5625, mapped.X, 3);
        Assert.Equal(0.5, mapped.Y, 3);
    }
}
