using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Profiles;

namespace DreamAirTracking.Core.Calibration;

public sealed class GazeCalibrationProfileBuilder
{
    public TrackingProfile BuildProfile(TrackingProfile source, GazeCalibrationFitResult fit)
    {
        var profile = Clone(source);
        profile.Version = "0.2";

        if (fit.LeftModel is not null)
        {
            profile.Left.GazeModel = fit.LeftModel;
        }

        if (fit.RightModel is not null)
        {
            profile.Right.GazeModel = fit.RightModel;
        }

        profile.Output.OutputOffsetX = 0;
        profile.Output.OutputOffsetY = 0;

        StabilizeWeakFit(profile, fit);
        return profile;
    }

    private static void StabilizeWeakFit(TrackingProfile profile, GazeCalibrationFitResult fit)
    {
        if (!fit.Metrics.Quality.Equals("poor", StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        profile.Output.CenterDeadzoneX = Math.Max(profile.Output.CenterDeadzoneX, 0.20);
        profile.Output.CenterDeadzoneY = Math.Max(profile.Output.CenterDeadzoneY, 0.24);
        profile.Output.CenterLockEnterX = Math.Max(profile.Output.CenterLockEnterX, 0.28);
        profile.Output.CenterLockEnterY = Math.Max(profile.Output.CenterLockEnterY, 0.42);
        profile.Output.CenterLockReleaseX = Math.Max(profile.Output.CenterLockReleaseX, 0.42);
        profile.Output.CenterLockReleaseY = Math.Max(profile.Output.CenterLockReleaseY, 0.58);
        profile.Output.OutputGainX = Math.Min(profile.Output.OutputGainX, 1.65);
        profile.Output.OutputGainY = Math.Min(profile.Output.OutputGainY, 1.55);

        StabilizeModel(profile.Left.GazeModel);
        StabilizeModel(profile.Right.GazeModel);
    }

    private static void StabilizeModel(Affine2DGazeModel? model)
    {
        if (model is null)
        {
            return;
        }

        model.CenterDeadzoneX = Math.Max(model.CenterDeadzoneX, 0.16);
        model.CenterDeadzoneY = Math.Max(model.CenterDeadzoneY, 0.20);
        model.CenterDeadzoneSoftness = Math.Max(model.CenterDeadzoneSoftness, 0.45);
    }

    private static TrackingProfile Clone(TrackingProfile source) =>
        new()
        {
            Version = source.Version,
            Output = Clone(source.Output),
            Left = Clone(source.Left),
            Right = Clone(source.Right)
        };

    private static BridgeOutputMapperOptions Clone(BridgeOutputMapperOptions source) =>
        new()
        {
            CenterDeadzoneX = source.CenterDeadzoneX,
            CenterDeadzoneY = source.CenterDeadzoneY,
            CenterLockEnterX = source.CenterLockEnterX,
            CenterLockEnterY = source.CenterLockEnterY,
            CenterLockReleaseX = source.CenterLockReleaseX,
            CenterLockReleaseY = source.CenterLockReleaseY,
            OutputGainX = source.OutputGainX,
            OutputGainY = source.OutputGainY,
            OutputOffsetX = source.OutputOffsetX,
            OutputOffsetY = source.OutputOffsetY
        };

    private static EyeTrackingProfile Clone(EyeTrackingProfile source) =>
        new()
        {
            Roi = source.Roi,
            Calibration = new EyeCalibration
            {
                CenterX = source.Calibration.CenterX,
                CenterY = source.Calibration.CenterY,
                HorizontalRange = source.Calibration.HorizontalRange,
                VerticalRange = source.Calibration.VerticalRange,
                HorizontalOutputSign = source.Calibration.HorizontalOutputSign,
                VerticalOutputSign = source.Calibration.VerticalOutputSign,
                HorizontalOutputOffset = source.Calibration.HorizontalOutputOffset,
                VerticalOutputOffset = source.Calibration.VerticalOutputOffset,
                OpenThreshold = source.Calibration.OpenThreshold,
                ClosedThreshold = source.Calibration.ClosedThreshold
            },
            GazeModel = Clone(source.GazeModel),
            InputTransform = Clone(source.InputTransform)
        };

    private static Affine2DGazeModel? Clone(Affine2DGazeModel? source) =>
        source is null
            ? null
            : new Affine2DGazeModel
            {
                Type = source.Type,
                InputCenterX = source.InputCenterX,
                InputCenterY = source.InputCenterY,
                InputScaleX = source.InputScaleX,
                InputScaleY = source.InputScaleY,
                OutX = source.OutX.ToArray(),
                OutY = source.OutY.ToArray(),
                CenterDeadzoneX = source.CenterDeadzoneX,
                CenterDeadzoneY = source.CenterDeadzoneY,
                CenterDeadzoneSoftness = source.CenterDeadzoneSoftness
            };

    private static RawInputTransform Clone(RawInputTransform source) =>
        new()
        {
            Enabled = source.Enabled,
            CurrentCenterX = source.CurrentCenterX,
            CurrentCenterY = source.CurrentCenterY,
            TargetCenterX = source.TargetCenterX,
            TargetCenterY = source.TargetCenterY,
            ScaleX = source.ScaleX,
            ScaleY = source.ScaleY
        };
}
