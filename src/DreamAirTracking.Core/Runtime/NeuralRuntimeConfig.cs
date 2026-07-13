using DreamAirTracking.Core.Runtime.PostProcess;

namespace DreamAirTracking.Core.Runtime;

/// <summary>
/// All runtime constants for <see cref="NeuralEyeRuntime"/>. Defaults mirror the values the
/// app passes to predict_live_multitask.py today (BridgeProcessService.AddRuntimeArguments),
/// so the native runtime reproduces the shipped behavior. The App maps BridgeLaunchOptions
/// onto this in E11-4.
/// </summary>
public sealed class NeuralRuntimeConfig
{
    // input
    public string BrokenEyeHost { get; set; } = "127.0.0.1";
    public int BrokenEyePort { get; set; } = 5555;
    public double MaxDeltaMs { get; set; } = 30.0;

    // models
    public string MainOnnxPath { get; set; } = "";
    public int ImageSize { get; set; } = 128;
    public string? ExpressionOnnxPath { get; set; }
    public int? ExpressionImageSize { get; set; }
    public int ExpressionEveryNFrames { get; set; } = 3;
    public double ExpressionEmaAlpha { get; set; } = 0.65;
    public int OnnxThreads { get; set; } = 2;

    // openness curve
    public string OpennessCurveMode { get; set; } = "blink_s_curve";
    public double OpennessFullOpenThreshold { get; set; } = 0.90;
    public double OpennessBoostKnee { get; set; } = 0.28;
    public double OpennessBoostGamma { get; set; } = 1.25;
    // optional per-eye calibration (v2 open_p95/closed_p05); null => identity
    public EyePair? OpenP95 { get; set; }
    public EyePair? ClosedP05 { get; set; }

    // gaze map
    public double CenterOffsetX { get; set; }
    public double CenterOffsetY { get; set; }
    public double XGain { get; set; } = 1.0;
    public double YGain { get; set; } = 1.0;
    public double Clamp { get; set; } = 1.0;
    public string ClampMode { get; set; } = "soft";
    public double SoftKnee { get; set; } = 0.72;
    public double EmaAlpha { get; set; } = 0.35;
    public double MaxStep { get; set; }

    // output gaze mapper
    public string OutputMapMode { get; set; } = "off";
    public double OutputDeadzone { get; set; } = 0.015;
    public double OutputCurveGamma { get; set; } = 1.0;
    public double OutputCenterRadius { get; set; } = 0.08;
    public double OutputCenterTau { get; set; } = 6.0;
    public double OutputCenterMaxStep { get; set; } = 0.025;

    // tracking state machine (shipped: both on)
    public bool TrackingOpennessMachine { get; set; } = true;
    public bool TrackingGazeHold { get; set; } = true;
    // P0-3 closure confirmation + half-open deadband (see TrackingStateParams)
    public int TrackingCloseConfirmFrames { get; set; }
    public double TrackingCloseConfirmFloor { get; set; } = 0.22;
    public double TrackingPartialDeadband { get; set; }

    // expression shaping (VRCFT wide/squint)
    public string WideSource { get; set; } = "model_head";
    public double EyeShapeWideScale { get; set; } = 0.60;
    public double EyeShapeSquintScale { get; set; } = 0.55;
    public double EyeShapeGamma { get; set; } = 1.15;
    public double EyeShapeDeadzone { get; set; } = 0.03;

    // pupil-wide hold (drives pupil constriction)
    public double PupilWideEnterThreshold { get; set; } = 0.90;
    public double PupilWideExitThreshold { get; set; } = 0.86;
    public int PupilWideHoldFrames { get; set; } = 8;
    public double PupilWideEmaAlpha { get; set; } = 0.45;

    // output modes
    public string PupilOutputMode { get; set; } = "off";      // off | expression_constrict_on_wide | model_radius (+aliases)
    public string EyeExpressionMode { get; set; } = "off";    // off | steamlink_eye_shapes

    // udp
    public string UdpHost { get; set; } = "127.0.0.1";
    public int UdpPort { get; set; } = 9400;
    public string MonitorUdpHost { get; set; } = "127.0.0.1";
    public int MonitorUdpPort { get; set; } = 9401;

    public TrackingStateParams BuildTrackingParams() => new()
    {
        OpennessMachine = TrackingOpennessMachine,
        GazeHold = TrackingGazeHold,
        CloseConfirmFrames = TrackingCloseConfirmFrames,
        CloseConfirmFloor = TrackingCloseConfirmFloor,
        PartialDeadband = TrackingPartialDeadband,
    };
}
