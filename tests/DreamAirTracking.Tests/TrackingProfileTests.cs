using DreamAirTracking.Core;
using DreamAirTracking.Core.Calibration;
using DreamAirTracking.Core.Geometry;
using DreamAirTracking.Core.Profiles;

namespace DreamAirTracking.Tests;

public sealed class TrackingProfileTests
{
    [Fact]
    public async Task SavesAndLoadsRoiAndCalibration()
    {
        var path = Path.Combine(Path.GetTempPath(), $"dream-air-profile-{Guid.NewGuid():N}.json");
        try
        {
            var profile = new TrackingProfile();
            profile.Output.CenterLockEnterY = 0.32;
            profile.Left.Roi = new PixelRect(10, 20, 80, 90);
            profile.Left.Calibration.CenterX = 112.5;
            profile.Left.Calibration.CenterY = 95.25;
            profile.Left.GazeModel = new Affine2DGazeModel
            {
                OutX = new[] { 1d, 2d, 3d },
                OutY = new[] { 4d, 5d, 6d },
                CenterDeadzoneX = 0.08,
                CenterDeadzoneY = 0.1
            };

            await TrackingProfileStore.SaveAsync(profile, path);
            var loaded = await TrackingProfileStore.LoadOrDefaultAsync(path);

            Assert.Equal(new PixelRect(10, 20, 80, 90), loaded.Left.Roi);
            Assert.Equal(112.5, loaded.Left.Calibration.CenterX);
            Assert.Equal(95.25, loaded.Left.Calibration.CenterY);
            Assert.Equal(0.32, loaded.Output.CenterLockEnterY);
            Assert.Equal(100, loaded.Right.Calibration.CenterX);
            Assert.NotNull(loaded.Left.GazeModel);
            Assert.Equal(3, loaded.Left.GazeModel!.OutX[2]);
            Assert.Equal(0.1, loaded.Left.GazeModel.CenterDeadzoneY);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public void CreatesDetectorOptionsForSide()
    {
        var profile = new TrackingProfile();
        profile.Right.Roi = new PixelRect(70, 50, 60, 80);
        profile.Right.Calibration.CenterX = 88;
        profile.Right.Calibration.CenterY = 79;

        var options = profile.CreateDetectorOptions(EyeSide.Right);

        Assert.Equal(new PixelRect(70, 50, 60, 80), options.Roi);
        Assert.Equal(88, options.ExpectedCenterX);
        Assert.Equal(79, options.ExpectedCenterY);
    }
}
