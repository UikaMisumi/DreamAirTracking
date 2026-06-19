using DreamAirTracking.Core.Calibration;
using DreamAirTracking.Core.Geometry;
using DreamAirTracking.Core.Profiles;

namespace DreamAirTracking.Tests;

public sealed class GazeCalibrationProfileBuilderTests
{
    [Fact]
    public void BuildsVersionedProfileWithoutMutatingSource()
    {
        var source = new TrackingProfile();
        source.Version = "0.1";
        source.Left.Roi = new PixelRect(1, 2, 3, 4);
        source.Left.Calibration.CenterX = 123;
        source.Output.OutputOffsetX = 0.56;
        source.Output.OutputOffsetY = -0.25;
        var fit = new GazeCalibrationFitResult
        {
            LeftModel = new Affine2DGazeModel { OutX = new[] { 1d, 2d, 3d } },
            RightModel = new Affine2DGazeModel { OutY = new[] { 4d, 5d, 6d } }
        };

        var profile = new GazeCalibrationProfileBuilder().BuildProfile(source, fit);

        Assert.Equal("0.2", profile.Version);
        Assert.Equal(new PixelRect(1, 2, 3, 4), profile.Left.Roi);
        Assert.Equal(123, profile.Left.Calibration.CenterX);
        Assert.NotNull(profile.Left.GazeModel);
        Assert.NotNull(profile.Right.GazeModel);
        Assert.Equal(0, profile.Output.OutputOffsetX);
        Assert.Equal(0, profile.Output.OutputOffsetY);
        Assert.Null(source.Left.GazeModel);
        Assert.Equal(0.56, source.Output.OutputOffsetX);
        Assert.Equal(-0.25, source.Output.OutputOffsetY);
        Assert.Equal("0.1", source.Version);
    }

}
