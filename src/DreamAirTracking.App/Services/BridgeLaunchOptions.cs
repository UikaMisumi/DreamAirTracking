namespace DreamAirTracking.App.Services;

public sealed class BridgeLaunchOptions
{
    public bool EnableEyeTracking { get; set; } = true;
    public bool EnablePupilAssist { get; set; } = true;
    public bool EnablePupilDiameter { get; set; }
    public bool EnableVrcftEyeExpressions { get; set; }
    public bool AutoStartEyeTracking { get; set; } = true;
    public string RuntimeModelType { get; set; } = "multitask";
    // "native" = in-process C# ONNX runtime (E11, default — validated on-headset); "python" = legacy predict_live_multitask.py
    public string RuntimeEngine { get; set; } = "native";
    public bool EnableMultitaskVrcftOutput { get; set; } = true;
    public double VrcftOutputXOffset { get; set; }
    public double VrcftOutputYOffset { get; set; }
    public double VrcftOutputXGain { get; set; } = 1.0;
    public double VrcftOutputYGain { get; set; } = 1.0;
    public double VrcftOutputXSign { get; set; } = 1.0;
    public double VrcftOutputYSign { get; set; } = 1.0;
    public string? ProfilePath { get; set; }
    public string? PupilCalibrationPath { get; set; }
    public string? QuickGazeCalibrationPath { get; set; }
    public string? QuickGazeCalibrationSession { get; set; }
    public string? OnnxPath { get; set; }
    public string? MetadataPath { get; set; }
    public string MultitaskModelPreset { get; set; } = BridgeProcessService.DefaultMultitaskModelPreset;
    public string? MultitaskOnnxPath { get; set; }
    public string? MultitaskMetadataPath { get; set; }
    public string? ExpressionOnnxPath { get; set; }
    public string? ExpressionMetadataPath { get; set; }
    public double CenterOffsetX { get; set; }
    public double CenterOffsetY { get; set; }
    public double XGain { get; set; } = 1.0;
    public double YGain { get; set; } = 1.0;
    public double Clamp { get; set; } = 0.95;
    public string ClampMode { get; set; } = "soft";
    public double SoftKnee { get; set; } = 0.72;
    public double CenterBias { get; set; } = 0.10;
    public double CenterBiasStart { get; set; } = 0.25;
    public string OutputMapMode { get; set; } = "off";
    public double OutputDeadzone { get; set; } = 0.015;
    public double OutputCurveGamma { get; set; } = 1.0;
    public double OutputCenterRadius { get; set; } = 0.08;
    public double OutputCenterTau { get; set; } = 6.0;
    public double OutputCenterMaxStep { get; set; } = 0.025;
    public double EmaAlpha { get; set; } = 0.35;
    public double MaxStep { get; set; }
    public string OpennessMode { get; set; } = "image";
    // "dual_path" = S1 velocity-gated hover curve (blink stays snappy, slow half-open reads linear)
    public string OpennessCurveMode { get; set; } = "dual_path";
    public double OpennessHoverEnterVelocity { get; set; } = 0.06;
    public double OpennessHoverExitVelocity { get; set; } = 0.02;
    public int OpennessHoverHoldFrames { get; set; } = 6;
    public string? OpennessCalibrationPath { get; set; }
    public string? OpennessPerEyeCalibrationPath { get; set; }
    public bool EnableTrackingStateMachine { get; set; } = true;
    public bool EnableClosureGazeHold { get; set; } = true;
    // P0-3: closure needs N consecutive frames below threshold before it reads closed (transient
    // shake dips never confirm); unconfirmed closures are floored; half-open band gets a deadband.
    public int TrackingCloseConfirmFrames { get; set; } = 3;
    public double TrackingCloseConfirmFloor { get; set; } = 0.22;
    public double TrackingPartialDeadband { get; set; } = 0.04;
    public double OpennessBoostGamma { get; set; } = 1.25;
    public double OpennessFullOpenThreshold { get; set; } = 0.90;
    public double OpennessBoostKnee { get; set; } = 0.28;
    public double EyeShapeWideScale { get; set; } = 0.60;
    public double EyeShapeSquintScale { get; set; } = 0.55;
    public double EyeShapeGamma { get; set; } = 1.15;
    public double EyeShapeDeadzone { get; set; } = 0.03;
    // Suppress EyeWide by upward gaze so looking up doesn't fire wide (only a raised lid at a
    // forward gaze = genuine widen). 0 = off; Sign flips which gaze-Y direction counts as "up".
    public double WideGazeUpSuppress { get; set; } = 0.85;
    public double WideGazeUpSign { get; set; } = 1.0;
    public double PupilWideEnterThreshold { get; set; } = 0.90;
    public double PupilWideExitThreshold { get; set; } = 0.86;
    public int PupilWideHoldFrames { get; set; } = 8;
    public double PupilWideEmaAlpha { get; set; } = 0.45;
    public string NormalizationMode { get; set; } = "probe_only";
    public double NormalizationTargetX { get; set; } = 0.50;
    public double NormalizationTargetY { get; set; } = 0.54;
    public double NormalizationLeftTargetX { get; set; } = 0.81;
    public double NormalizationRightTargetX { get; set; } = 0.20;
    public double NormalizationLeftTargetY { get; set; } = 0.54;
    public double NormalizationRightTargetY { get; set; } = 0.54;
    public double NormalizationMaxShiftX { get; set; } = 0.18;
    public double NormalizationMaxShiftY { get; set; } = 0.16;
    public double NormalizationScaleMin { get; set; } = 0.86;
    public double NormalizationScaleMax { get; set; } = 1.18;
    public int NormalizationHoldFrames { get; set; } = 5;
}
