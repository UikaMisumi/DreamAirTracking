using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using System.Threading.Channels;
using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Input;
using DreamAirTracking.Core.Runtime.PostProcess;

namespace DreamAirTracking.Core.Runtime;

/// <summary>
/// Native in-process eye-tracking runtime. Faithful port of predict_live_multitask.py:
/// MJPEG input -> pair/staleness -> preprocess -> ONNX (main + expression) -> post-processing
/// -> BridgeTrackingState -> UDP 9400/9401. Replaces the Python runtime (E11). The processing
/// core is <see cref="ProcessPair"/> (stateful, deterministic) so it can be driven directly
/// by golden-parity tests; <see cref="RunAsync"/> wraps it with live BrokenEye input.
/// </summary>
public sealed class NeuralEyeRuntime : IDisposable
{
    private readonly NeuralRuntimeConfig _cfg;
    private readonly MultitaskEyeModel _model;
    private readonly OutputGazeMapper _outputMapper;
    private readonly TrackingStateMachine _tracking;
    private readonly PupilWideHold _pupilWide;
    private readonly string _pupilMode;
    private readonly JsonSerializerOptions _json = new(JsonSerializerDefaults.Web);

    private long _sequence;
    private Vec2? _smooth;
    private EyePair? _lastWide, _lastSquint;
    private long _lastExprSeq;

    public NeuralEyeRuntime(NeuralRuntimeConfig cfg)
    {
        _cfg = cfg;
        _model = new MultitaskEyeModel(cfg.MainOnnxPath, cfg.ImageSize, cfg.OnnxThreads, cfg.ExpressionOnnxPath, cfg.ExpressionImageSize);
        _outputMapper = new OutputGazeMapper(cfg.OutputMapMode, cfg.OutputDeadzone, cfg.OutputCurveGamma,
            cfg.Clamp > 0 ? cfg.Clamp : 1.0, cfg.OutputCenterRadius, cfg.OutputCenterTau, cfg.OutputCenterMaxStep);
        _tracking = new TrackingStateMachine(cfg.BuildTrackingParams());
        _pupilWide = new PupilWideHold();
        _pupilMode = PupilOutput.NormalizeMode(cfg.PupilOutputMode);
    }

    private sealed record Frame(string Side, double ReceivedAt, byte[] Jpeg);

    /// <summary>Process one paired left/right frame. Advances runtime state. Returns the emitted state.</summary>
    public BridgeTrackingState ProcessPair(byte[] leftJpeg, byte[] rightJpeg, double frameTimestamp, double deltaMs)
    {
        float[] leftBuf = EyeTensorPreprocessor.Preprocess(leftJpeg, _cfg.ImageSize);
        float[] rightBuf = EyeTensorPreprocessor.Preprocess(rightJpeg, _cfg.ImageSize);
        EyeModelOutputs o = _model.RunMain(leftBuf, rightBuf);

        EyePair modelWide = o.Wide, modelSquint = o.Squint;
        if (_model.HasExpressionModel)
        {
            long nextSeq = _sequence + 1;
            bool shouldUpdate = _lastWide is null || (nextSeq - _lastExprSeq >= _cfg.ExpressionEveryNFrames);
            if (shouldUpdate)
            {
                float[] eL = leftBuf, eR = rightBuf;
                if (_model.ExpressionImageSize != _cfg.ImageSize)
                {
                    eL = EyeTensorPreprocessor.Preprocess(leftJpeg, _model.ExpressionImageSize);
                    eR = EyeTensorPreprocessor.Preprocess(rightJpeg, _model.ExpressionImageSize);
                }
                var (w, s) = _model.RunExpression(eL, eR);
                if (_lastWide is EyePair lw && _lastSquint is EyePair ls && _cfg.ExpressionEmaAlpha < 1.0)
                {
                    double a = _cfg.ExpressionEmaAlpha;
                    w = new EyePair(a * w.Left + (1.0 - a) * lw.Left, a * w.Right + (1.0 - a) * lw.Right);
                    s = new EyePair(a * s.Left + (1.0 - a) * ls.Left, a * s.Right + (1.0 - a) * ls.Right);
                }
                _lastWide = w; _lastSquint = s; _lastExprSeq = nextSeq;
            }
            if (_lastWide is EyePair) { modelWide = _lastWide.Value; modelSquint = _lastSquint!.Value; }
        }

        // openness chain
        EyePair modelOpenness = o.Openness;
        EyePair normOpenness = (_cfg.OpenP95 is EyePair hi && _cfg.ClosedP05 is EyePair lo)
            ? PerEyeOpennessNormalizer.Normalize(modelOpenness, hi, lo)
            : modelOpenness;
        EyePair curved = OpennessCurve.Apply(normOpenness, _cfg.OpennessCurveMode,
            _cfg.OpennessFullOpenThreshold, _cfg.OpennessBoostKnee, _cfg.OpennessBoostGamma);

        double pairConf = o.Confidence.Length > 2 ? o.Confidence[2] : Math.Min(o.Confidence[0], o.Confidence[1]);

        // gaze chain
        Vec2 raw = o.Gaze;
        Vec2 mapped = RuntimeGazeMap.Apply(raw, new Vec2(_cfg.CenterOffsetX, _cfg.CenterOffsetY),
            new Vec2(_cfg.XGain, _cfg.YGain), _cfg.Clamp, _cfg.ClampMode, _cfg.SoftKnee);
        mapped = _outputMapper.Update(mapped, frameTimestamp, pairConf, (curved.Left + curved.Right) / 2.0);
        _smooth = GazeSmoother.Update(_smooth, mapped, _cfg.EmaAlpha, _cfg.MaxStep);
        TrackingResult tr = _tracking.Update(curved, pairConf, _smooth.Value);
        EyePair openness = tr.Openness;
        _smooth = tr.Gaze;

        _sequence += 1;

        // expression + pupil
        float[] pupil = o.Pupil;
        // De-conflate wide from looking up: a raised upper lid drives wide, but the lid also raises
        // when the eye looks up. Suppress wide by upward gaze so only a raised lid at a forward gaze
        // (a genuine widen) survives. gate=1 forward, ->(1-suppress) fully up.
        double wideGate = 1.0;
        if (_cfg.WideGazeUpSuppress > 0.0)
        {
            double up = EyeMath.Clamp(_cfg.WideGazeUpSign * _smooth.Value.Y, 0.0, 1.0);
            wideGate = 1.0 - EyeMath.Clamp(_cfg.WideGazeUpSuppress, 0.0, 1.0) * up;
        }
        EyePair modelWideC = new(Clip01(modelWide.Left) * wideGate, Clip01(modelWide.Right) * wideGate);
        EyePair modelSquintC = new(Clip01(modelSquint.Left), Clip01(modelSquint.Right));
        EyePair shapeWide = EyeShapeCurve.Apply(modelWideC, _cfg.EyeShapeWideScale, _cfg.EyeShapeGamma, _cfg.EyeShapeDeadzone);
        EyePair shapeSquint = EyeShapeCurve.Apply(modelSquintC, _cfg.EyeShapeSquintScale, _cfg.EyeShapeGamma, _cfg.EyeShapeDeadzone);
        EyePair pw = _pupilWide.Update(modelWideC, _cfg.PupilWideEnterThreshold, _cfg.PupilWideExitThreshold,
            _cfg.PupilWideHoldFrames, _cfg.PupilWideEmaAlpha);
        PupilResult leftPo = PupilOutput.Compute(_pupilMode, openness.Left, pw.Left, pupil[2], o.Confidence[0]);
        PupilResult rightPo = PupilOutput.Compute(_pupilMode, openness.Right, pw.Right, pupil[5], o.Confidence[1]);

        Vec2 smooth = _smooth.Value;
        var state = new BridgeTrackingState(_sequence, DateTimeOffset.UtcNow, deltaMs,
            BuildEye(raw, mapped, smooth, modelOpenness.Left, openness.Left, shapeWide.Left, shapeSquint.Left, pupil[0], pupil[1], pupil[2], o.Confidence[0], leftPo),
            BuildEye(raw, mapped, smooth, modelOpenness.Right, openness.Right, shapeWide.Right, shapeSquint.Right, pupil[3], pupil[4], pupil[5], o.Confidence[1], rightPo))
        {
            EyeTrackingEnabled = true,
            EyeExpressionEnabled = _cfg.EyeExpressionMode != "off",
            EyeExpressionMode = _cfg.EyeExpressionMode,
            PupilDiameterEnabled = _pupilMode != "off",
            PupilDiameterMode = _pupilMode,
            NormalizationMode = "multitask_diagnostic",
        };
        return state;
    }

    /// <summary>Run against a live BrokenEye MJPEG feed until cancelled. Sends UDP + invokes onState.</summary>
    public async Task RunAsync(Action<BridgeTrackingState>? onState, CancellationToken ct)
    {
        var channel = Channel.CreateBounded<Frame>(new BoundedChannelOptions(80)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = false,
        });
        var sw = Stopwatch.StartNew();
        using var httpClient = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        var mjpeg = new MjpegStreamClient(httpClient);

        Task Producer(string side) => Task.Run(async () =>
        {
            var uri = new Uri($"http://{_cfg.BrokenEyeHost}:{_cfg.BrokenEyePort}/eye/{side}");
            try
            {
                await foreach (var jpeg in mjpeg.ReadJpegFramesAsync(uri, ct))
                    channel.Writer.TryWrite(new Frame(side, sw.Elapsed.TotalSeconds, jpeg));
            }
            catch (OperationCanceledException) { }
            catch { /* stream ended/errored; producer stops */ }
        }, ct);

        var producers = new[] { Producer("left"), Producer("right") };
        _ = Task.WhenAll(producers).ContinueWith(_ => channel.Writer.TryComplete(), TaskScheduler.Default);

        using var udp = new UdpClient();
        var udpEp = new IPEndPoint(IPAddress.Parse(_cfg.UdpHost), _cfg.UdpPort);
        var monEp = new IPEndPoint(IPAddress.Parse(_cfg.MonitorUdpHost), _cfg.MonitorUdpPort);

        var leftFrames = new List<Frame>();
        var rightFrames = new List<Frame>();

        try
        {
            await foreach (var frame in channel.Reader.ReadAllAsync(ct))
            {
                if (frame.Side == "left") { leftFrames.Add(frame); Trim(leftFrames); }
                else { rightFrames.Add(frame); Trim(rightFrames); }
                if (leftFrames.Count == 0 || rightFrames.Count == 0) continue;

                Frame left = leftFrames.MinBy(f => Math.Abs(f.ReceivedAt - frame.ReceivedAt))!;
                Frame right = rightFrames.MinBy(f => Math.Abs(f.ReceivedAt - left.ReceivedAt))!;
                double deltaMs = Math.Abs(left.ReceivedAt - right.ReceivedAt) * 1000.0;
                if (deltaMs > _cfg.MaxDeltaMs) continue;

                double frameTs = Math.Max(left.ReceivedAt, right.ReceivedAt);
                BridgeTrackingState state = ProcessPair(left.Jpeg, right.Jpeg, frameTs, deltaMs);

                byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(state, _json);
                if (_cfg.UdpPort > 0) udp.Send(bytes, bytes.Length, udpEp);
                if (_cfg.MonitorUdpPort > 0) udp.Send(bytes, bytes.Length, monEp);
                onState?.Invoke(state);
            }
        }
        catch (OperationCanceledException) { }
    }

    private static void Trim(List<Frame> frames)
    {
        if (frames.Count > 8) frames.RemoveRange(0, frames.Count - 8);
    }

    private static double Clip01(double v) => Math.Min(1.0, Math.Max(0.0, v));

    private static BridgeEyeState BuildEye(Vec2 raw, Vec2 mapped, Vec2 smooth, double modelOpen, double openness,
        double wide, double squint, double pupilX, double pupilY, double pupilRadius, double confidence, PupilResult po)
        => new(true, confidence, raw.X, raw.Y, smooth.X, smooth.Y, openness)
        {
            PupilOpenness = openness,
            ApertureOpenness = modelOpen,
            CalibratedOpenness = modelOpen,
            OutputOpenness = openness,
            Wide = EyeMath.Clamp01(wide),
            Squint = EyeMath.Clamp01(squint),
            AperturePeakDarkFraction = 0.0,
            ApertureHeight = 0,
            ApertureReason = "multitask-openness-head+runtime-curve",
            RawNormalizedX = raw.X,
            RawNormalizedY = raw.Y,
            MonocularNormalizedX = mapped.X,
            MonocularNormalizedY = mapped.Y,
            PupilDiameterFound = po.Found,
            PupilDiameterQuality = po.Quality,
            PupilDiameterConfidence = po.Confidence,
            PupilDiameterPx = 0.0,
            PupilDiameterNormalized = po.Normalized,
            PupilExpressionNormalized = po.ExpressionNormalized,
            PupilGeometryRadius = pupilRadius,
            PupilDiameterAxisRatio = 1.0,
            PupilDiameterReason = po.Reason,
            NormalizationFound = true,
            NormalizationConfidence = confidence,
            NormalizationShiftX = 0.0,
            NormalizationShiftY = 0.0,
            NormalizationScale = Math.Max(0.05, pupilRadius),
            NormalizationPupilX = pupilX,
            NormalizationPupilY = pupilY,
            NormalizationDropReason = "multitask-pupil-head",
        };

    public void Dispose() => _model.Dispose();
}
