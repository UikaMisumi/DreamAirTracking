using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using VRCFaceTracking.Core.Library;
using VRCFaceTracking.Core.Params.Expressions;
using VRCFaceTracking.Core.Types;

namespace VRCFaceTracking;

public sealed class DreamAirTrackingModule : ExtTrackingModule
{
    private const int DefaultUdpPort = 9400;
    private const float DefaultPupilDiameterMm = 5.0f;
    private const float DefaultPupilMinDilationMm = 0.0f;
    private const float DefaultPupilMaxDilationMm = 10.0f;
    private const float MinConfidence = 0.05f;
    private const float MinGazeOpenness = 0.08f;
    private const float DefaultXGain = 1.0f;
    private const float DefaultYGain = 1.0f;
    private const float DefaultYOffset = 0.0f;
    private const float DefaultPupilMinMm = 3.0f;
    private const float DefaultPupilMaxMm = 7.0f;
    private const float DefaultPupilSmoothing = 0.35f;
    private static readonly TimeSpan StaleAfter = TimeSpan.FromMilliseconds(300);
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    private UdpClient? _udp;
    private BridgeTrackingState? _latest;
    private DateTimeOffset _latestReceivedAt;
    private float _xSign = 1.0f;
    private float _ySign = 1.0f;
    private float _xGain = 1.0f;
    private float _yGain = 1.0f;
    private float _xOffset;
    private float _yOffset = DefaultYOffset;
    private bool _enableEyeTracking = true;
    private bool _enableEyeExpression;
    private bool _enablePupilDiameter = true;
    private bool _logPupilDiagnostics;
    private float _wideOpennessBoost = 1.0f;
    private float _pupilMinMm = DefaultPupilMinMm;
    private float _pupilMaxMm = DefaultPupilMaxMm;
    private float _pupilSmoothing = DefaultPupilSmoothing;
    private float _lastLeftPupilMm = DefaultPupilDiameterMm;
    private float _lastRightPupilMm = DefaultPupilDiameterMm;
    private DateTimeOffset _lastPupilDiagnosticLogAt;
    private string _outputOptionsPath = string.Empty;
    private DateTimeOffset _lastOutputOptionsRefreshAt;
    private DateTime _lastOutputOptionsWriteTimeUtc = DateTime.MinValue;

    public override (bool SupportsEye, bool SupportsExpression) Supported =>
        (ReadBoolEnv("DREAMAIR_VRCFT_ENABLE_EYE_TRACKING", true), IsEyeExpressionEnabledByConfig());

    public override (bool eyeSuccess, bool expressionSuccess) Initialize(bool eyeAvailable, bool expressionAvailable)
    {
        ModuleInformation = new ModuleMetadata
        {
            Name = "DreamAirTracking",
            StaticImages = new List<Stream>()
        };

        _enableEyeTracking = ReadBoolEnv("DREAMAIR_VRCFT_ENABLE_EYE_TRACKING", true);
        var requestedEyeExpression = IsEyeExpressionEnabledByConfig();
        _enableEyeExpression = requestedEyeExpression && expressionAvailable;
        _enablePupilDiameter = ReadBoolEnv("DREAMAIR_VRCFT_ENABLE_PUPIL_DIAMETER", true);
        _logPupilDiagnostics = ReadBoolEnv("DREAMAIR_VRCFT_LOG_PUPIL", false);
        // D: on strong "eye wide", nudge openness toward fully-open (Eye bucket = ours; no SR conflict).
        // 0 disables; >1 makes it snap harder (push = clamp(wide * boost, 0, 1)).
        _wideOpennessBoost = Math.Clamp(ReadFloatEnv("DREAMAIR_VRCFT_WIDE_OPENNESS_BOOST", 1.0f), 0.0f, 4.0f);
        if (!_enableEyeTracking)
        {
            Logger?.LogInformation("DreamAirTracking eye output is disabled by DREAMAIR_VRCFT_ENABLE_EYE_TRACKING.");
            Status = ModuleState.Uninitialized;
            return (false, false);
        }

        if (!eyeAvailable)
        {
            Logger?.LogInformation("DreamAirTracking skipped because another eye module is already active.");
            return (false, false);
        }

        if (requestedEyeExpression && !expressionAvailable)
        {
            Logger?.LogInformation("DreamAirTracking eye expression output was requested, but another expression module is already active.");
        }

        var port = ReadIntEnv("DREAMAIR_VRCFT_UDP_PORT", DefaultUdpPort);
        _xSign = ReadFloatEnv("DREAMAIR_VRCFT_X_SIGN", 1.0f);
        _ySign = ReadFloatEnv("DREAMAIR_VRCFT_Y_SIGN", 1.0f);
        _xGain = ReadFloatEnv("DREAMAIR_VRCFT_X_GAIN", DefaultXGain);
        _yGain = ReadFloatEnv("DREAMAIR_VRCFT_Y_GAIN", DefaultYGain);
        _xOffset = ReadFloatEnv("DREAMAIR_VRCFT_X_OFFSET", 0.0f);
        _yOffset = ReadFloatEnv("DREAMAIR_VRCFT_Y_OFFSET", DefaultYOffset);
        _outputOptionsPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "DreamAirTracking",
            "vrcft_output_options.json");
        RefreshOutputOptions(force: true);
        _pupilMinMm = ReadFloatEnv("DREAMAIR_VRCFT_PUPIL_MIN_MM", DefaultPupilMinMm);
        _pupilMaxMm = ReadFloatEnv("DREAMAIR_VRCFT_PUPIL_MAX_MM", DefaultPupilMaxMm);
        if (_pupilMaxMm <= _pupilMinMm)
        {
            _pupilMinMm = DefaultPupilMinMm;
            _pupilMaxMm = DefaultPupilMaxMm;
        }

        _pupilSmoothing = Math.Clamp(ReadFloatEnv("DREAMAIR_VRCFT_PUPIL_SMOOTHING", DefaultPupilSmoothing), 0.0f, 1.0f);
        _lastLeftPupilMm = DefaultPupilDiameterMm;
        _lastRightPupilMm = DefaultPupilDiameterMm;
        _lastPupilDiagnosticLogAt = DateTimeOffset.MinValue;

        try
        {
            _udp = new UdpClient(new IPEndPoint(IPAddress.Loopback, port));
            _udp.Client.Blocking = false;
            Status = ModuleState.Active;
            Logger?.LogInformation(
                "DreamAirTracking listening for bridge packets on 127.0.0.1:{port}. pupilDiameter={pupilDiameter} eyeExpression={eyeExpression}",
                port,
                _enablePupilDiameter,
                _enableEyeExpression);
            return (true, _enableEyeExpression);
        }
        catch (Exception ex)
        {
            Logger?.LogError(ex, "DreamAirTracking failed to bind UDP receiver.");
            Status = ModuleState.Uninitialized;
            return (false, false);
        }
    }

    public override void Update()
    {
        ReceiveAvailablePackets();
        RefreshOutputOptions(force: false);

        var state = _latest;
        if (state is not null && DateTimeOffset.UtcNow - _latestReceivedAt <= StaleAfter)
        {
            WriteEyeData(state);
        }
        else
        {
            SetNeutralEyeData();
            if (_enableEyeExpression)
            {
                SetNeutralEyeExpressionData();
            }
        }

        Thread.Sleep(1);
    }

    public override void Teardown()
    {
        _udp?.Dispose();
        _udp = null;
        _enableEyeExpression = false;
        Status = ModuleState.Uninitialized;
    }

    private void ReceiveAvailablePackets()
    {
        if (_udp is null)
        {
            return;
        }

        while (_udp.Available > 0)
        {
            try
            {
                var endpoint = new IPEndPoint(IPAddress.Any, 0);
                var payload = _udp.Receive(ref endpoint);
                var packet = JsonSerializer.Deserialize<BridgeTrackingState>(payload, JsonOptions);
                if (packet is not null)
                {
                    _latest = packet;
                    _latestReceivedAt = DateTimeOffset.UtcNow;
                }
            }
            catch (SocketException)
            {
                return;
            }
            catch (JsonException ex)
            {
                Logger?.LogDebug(ex, "DreamAirTracking ignored a malformed bridge packet.");
            }
        }
    }

    private void WriteEyeData(BridgeTrackingState state)
    {
        if (!state.EyeTrackingEnabled)
        {
            SetNeutralEyeData();
            return;
        }

        var leftValid = IsValid(state.Left);
        var rightValid = IsValid(state.Right);
        var left = leftValid ? state.Left : rightValid ? state.Right : null;
        var right = rightValid ? state.Right : leftValid ? state.Left : null;

        UnifiedTracking.Data.Eye.Left.Gaze = ToGaze(left);
        UnifiedTracking.Data.Eye.Right.Gaze = ToGaze(right);
        UnifiedTracking.Data.Eye.Left.Openness = ToOpenness(state.Left);
        UnifiedTracking.Data.Eye.Right.Openness = ToOpenness(state.Right);
        SetStablePupilRange();
        var shouldPublishPupil = _enablePupilDiameter && state.PupilDiameterEnabled;
        UnifiedTracking.Data.Eye.Left.PupilDiameter_MM = shouldPublishPupil
            ? TryGetPupilDiameterMm(state.Left, state.PupilDiameterMode, ref _lastLeftPupilMm, out var leftPupilMm)
                ? leftPupilMm
                : DefaultPupilDiameterMm
            : DefaultPupilDiameterMm;
        UnifiedTracking.Data.Eye.Right.PupilDiameter_MM = shouldPublishPupil
            ? TryGetPupilDiameterMm(state.Right, state.PupilDiameterMode, ref _lastRightPupilMm, out var rightPupilMm)
                ? rightPupilMm
                : DefaultPupilDiameterMm
            : DefaultPupilDiameterMm;
        if (_enableEyeExpression)
        {
            if (state.EyeExpressionEnabled)
            {
                WriteEyeExpressionData(state);
            }
            else
            {
                SetNeutralEyeExpressionData();
            }
        }
        LogPupilDiagnosticsIfEnabled(state);
    }

    private static void SetNeutralEyeData()
    {
        UnifiedTracking.Data.Eye.Left.Gaze = Vector2.zero;
        UnifiedTracking.Data.Eye.Right.Gaze = Vector2.zero;
        UnifiedTracking.Data.Eye.Left.Openness = 1.0f;
        UnifiedTracking.Data.Eye.Right.Openness = 1.0f;
        SetStablePupilRange();
        UnifiedTracking.Data.Eye.Left.PupilDiameter_MM = DefaultPupilDiameterMm;
        UnifiedTracking.Data.Eye.Right.PupilDiameter_MM = DefaultPupilDiameterMm;
    }

    private static void WriteEyeExpressionData(BridgeTrackingState state)
    {
        UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeWideLeft].Weight = ToShape(state.Left.Wide);
        UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeWideRight].Weight = ToShape(state.Right.Wide);
        UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeSquintLeft].Weight = ToShape(state.Left.Squint);
        UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeSquintRight].Weight = ToShape(state.Right.Squint);
    }

    private static void SetNeutralEyeExpressionData()
    {
        UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeWideLeft].Weight = 0.0f;
        UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeWideRight].Weight = 0.0f;
        UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeSquintLeft].Weight = 0.0f;
        UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeSquintRight].Weight = 0.0f;
    }

    private Vector2 ToGaze(BridgeEyeState? eye)
    {
        if (eye is null)
        {
            return Vector2.zero;
        }

        return new Vector2(
            ClampSigned(((float)eye.NormalizedX * _xGain * _xSign) + _xOffset),
            ClampSigned(((float)eye.NormalizedY * _yGain * _ySign) + _yOffset));
    }

    private float ToOpenness(BridgeEyeState eye)
    {
        float o = Math.Clamp((float)eye.Openness, 0.0f, 1.0f);
        if (_wideOpennessBoost > 0.0f)
        {
            // D: push openness toward 1.0 proportional to the shaped "eye wide" so wide eyes read fully open.
            float push = Math.Clamp((float)eye.Wide * _wideOpennessBoost, 0.0f, 1.0f);
            o = Math.Clamp(o + (1.0f - o) * push, 0.0f, 1.0f);
        }
        return o;
    }

    private bool TryGetPupilDiameterMm(BridgeEyeState eye, string mode, ref float lastMm, out float pupilMm)
    {
        pupilMm = DefaultPupilDiameterMm;
        if (!eye.PupilDiameterFound ||
            !eye.PupilDiameterQuality ||
            !double.IsFinite(eye.PupilDiameterNormalized))
        {
            return false;
        }

        var normalized = Math.Clamp((float)eye.PupilDiameterNormalized, 0.0f, 1.0f);
        var expressionMode = mode.Equals("expression_constrict_on_wide", StringComparison.OrdinalIgnoreCase)
            || mode.Equals("eye_wide_assist", StringComparison.OrdinalIgnoreCase)
            || mode.Equals("assist", StringComparison.OrdinalIgnoreCase);
        var minMm = expressionMode ? DefaultPupilMinDilationMm : _pupilMinMm;
        var maxMm = expressionMode ? DefaultPupilMaxDilationMm : _pupilMaxMm;
        var target = minMm + ((maxMm - minMm) * normalized);
        lastMm = Math.Clamp(
            lastMm + ((target - lastMm) * _pupilSmoothing),
            minMm,
            maxMm);
        pupilMm = lastMm;
        return true;
    }

    private static void SetStablePupilRange()
    {
        UnifiedTracking.Data.Eye._minDilation = DefaultPupilMinDilationMm;
        UnifiedTracking.Data.Eye._maxDilation = DefaultPupilMaxDilationMm;
    }

    private void LogPupilDiagnosticsIfEnabled(BridgeTrackingState state)
    {
        if (!_logPupilDiagnostics || DateTimeOffset.UtcNow - _lastPupilDiagnosticLogAt < TimeSpan.FromSeconds(2))
        {
            return;
        }

        _lastPupilDiagnosticLogAt = DateTimeOffset.UtcNow;
        Logger?.LogInformation(
            "DreamAirTracking pupil: packetEnabled={packetEnabled} mode={mode} left=({leftFound},{leftQuality},{leftNorm:0.000}->{leftMm:0.00}mm) right=({rightFound},{rightQuality},{rightNorm:0.000}->{rightMm:0.00}mm)",
            state.PupilDiameterEnabled,
            state.PupilDiameterMode,
            state.Left.PupilDiameterFound,
            state.Left.PupilDiameterQuality,
            state.Left.PupilDiameterNormalized,
            _lastLeftPupilMm,
            state.Right.PupilDiameterFound,
            state.Right.PupilDiameterQuality,
            state.Right.PupilDiameterNormalized,
            _lastRightPupilMm);
    }

    private static bool IsValid(BridgeEyeState eye)
        => eye.Found && eye.Confidence > MinConfidence && eye.Openness >= MinGazeOpenness;

    private static float ClampSigned(float value)
        => Math.Clamp(value, -1.0f, 1.0f);

    private static float ToShape(double value)
        => double.IsFinite(value) ? Math.Clamp((float)value, 0.0f, 1.0f) : 0.0f;

    private void RefreshOutputOptions(bool force)
    {
        if (string.IsNullOrWhiteSpace(_outputOptionsPath) || !File.Exists(_outputOptionsPath))
        {
            return;
        }

        var now = DateTimeOffset.UtcNow;
        if (!force && now - _lastOutputOptionsRefreshAt < TimeSpan.FromSeconds(1))
        {
            return;
        }

        _lastOutputOptionsRefreshAt = now;
        try
        {
            var writeTime = File.GetLastWriteTimeUtc(_outputOptionsPath);
            if (!force && writeTime <= _lastOutputOptionsWriteTimeUtc)
            {
                return;
            }

            var json = File.ReadAllText(_outputOptionsPath);
            var options = JsonSerializer.Deserialize<VrcftOutputOptions>(json, JsonOptions);
            if (options is null)
            {
                return;
            }

            _xOffset = FiniteOrDefault(options.XOffset, _xOffset);
            _yOffset = FiniteOrDefault(options.YOffset, _yOffset);
            _xGain = NonZeroFiniteOrDefault(options.XGain, _xGain);
            _yGain = NonZeroFiniteOrDefault(options.YGain, _yGain);
            _xSign = options.XSign < 0 ? -1.0f : 1.0f;
            _ySign = options.YSign < 0 ? -1.0f : 1.0f;
            _lastOutputOptionsWriteTimeUtc = writeTime;
            Logger?.LogInformation(
                "DreamAirTracking VRCFT output trim: x=({xSign:0.0} * input * {xGain:0.000}) + {xOffset:0.000}, y=({ySign:0.0} * input * {yGain:0.000}) + {yOffset:0.000}",
                _xSign,
                _xGain,
                _xOffset,
                _ySign,
                _yGain,
                _yOffset);
        }
        catch (Exception ex)
        {
            Logger?.LogDebug(ex, "DreamAirTracking could not refresh VRCFT output trim.");
        }
    }

    private static float FiniteOrDefault(double value, float fallback)
        => double.IsFinite(value) ? (float)value : fallback;

    private static float NonZeroFiniteOrDefault(double value, float fallback)
        => double.IsFinite(value) && Math.Abs(value) >= 0.000001 ? (float)value : fallback;

    private static int ReadIntEnv(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out var value) ? value : fallback;

    private static float ReadFloatEnv(string name, float fallback)
        => float.TryParse(Environment.GetEnvironmentVariable(name), out var value) ? value : fallback;

    private static bool ReadBoolEnv(string name, bool fallback)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return fallback;
        }

        return raw.Trim().ToLowerInvariant() switch
        {
            "1" or "true" or "yes" or "on" => true,
            "0" or "false" or "no" or "off" => false,
            _ => fallback
        };
    }

    private static bool IsEyeExpressionEnabledByConfig()
        => ReadBoolEnv("DREAMAIR_VRCFT_ENABLE_EYE_EXPRESSION", ReadOutputOptionBool("enableEyeExpression", false));

    private static bool ReadOutputOptionBool(string name, bool fallback)
    {
        try
        {
            var path = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "DreamAirTracking",
                "vrcft_output_options.json");
            if (!File.Exists(path))
            {
                return fallback;
            }

            using var document = JsonDocument.Parse(File.ReadAllText(path));
            if (!document.RootElement.TryGetProperty(name, out var value))
            {
                return fallback;
            }

            return value.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.String => value.GetString()?.Trim().ToLowerInvariant() switch
                {
                    "1" or "true" or "yes" or "on" => true,
                    "0" or "false" or "no" or "off" => false,
                    _ => fallback
                },
                _ => fallback
            };
        }
        catch
        {
            return fallback;
        }
    }

    private sealed class BridgeTrackingState
    {
        public long Sequence { get; set; }
        public DateTimeOffset Timestamp { get; set; }
        public double DeltaMs { get; set; }
        public bool EyeTrackingEnabled { get; set; } = true;
        public bool EyeExpressionEnabled { get; set; }
        public string EyeExpressionMode { get; set; } = "off";
        public bool PupilDiameterEnabled { get; set; }
        public string PupilDiameterMode { get; set; } = "off";
        public BridgeEyeState Left { get; set; } = new();
        public BridgeEyeState Right { get; set; } = new();
    }

    private sealed class BridgeEyeState
    {
        public bool Found { get; set; }
        public double Confidence { get; set; }
        public double RawX { get; set; }
        public double RawY { get; set; }
        public double NormalizedX { get; set; }
        public double NormalizedY { get; set; }
        public double Openness { get; set; }
        public double Wide { get; set; }
        public double Squint { get; set; }
        public bool PupilDiameterFound { get; set; }
        public bool PupilDiameterQuality { get; set; }
        public double PupilDiameterNormalized { get; set; }
        public double PupilExpressionNormalized { get; set; }
        public double PupilGeometryRadius { get; set; }
    }

    private sealed class VrcftOutputOptions
    {
        public double XOffset { get; set; }
        public double YOffset { get; set; }
        public double XGain { get; set; } = 1.0;
        public double YGain { get; set; } = 1.0;
        public double XSign { get; set; } = 1.0;
        public double YSign { get; set; } = 1.0;
    }
}
