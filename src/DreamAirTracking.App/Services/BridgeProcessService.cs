using System.Diagnostics;
using System.Globalization;
using System.Net.Http;
using System.Text.Json;
using DreamAirTracking.Core.Models;
using DreamAirTracking.Core.Runtime;

namespace DreamAirTracking.App.Services;

public sealed class BridgeProcessService
{
    public static BridgeProcessService Instance { get; } = new();

    private const int BrokenEyePort = 5555;
    private const string LegacyBridgeProcessName = "DreamAirTracking.Bridge";
    public const string DefaultMultitaskModelPreset = "round2_round8_expression_20260618";
    public const string SplitMultitaskModelPreset = "split_round2_round8_20260618";
    public const string PreviousMultitaskModelPreset = "round2_openness_tune_20260617";
    public const string LegacyMultitaskModelPreset = "legacy_twochannel_retarget070";
    private const string PreferredMultitaskRun = "eye_multitask_siamese_v5_app9_rightdown_geometry_round9_20260618";
    private const string SplitMultitaskRun = "eye_multitask_split_round2_geometry_round8_expression_20260618";
    private const string PreviousMultitaskRun = "eye_multitask_siamese_per_eye_openness_tune_round2_20260617";
    private const string LegacyMultitaskRun = "eye_multitask_mobilenetv3_fullparam_round1_twochannel_retarget070";
    private const string DefaultExpressionRun = "eye_multitask_siamese_v4_round8_headmask_freezebn_20260618";

    private Process? _process;
    private readonly NativeRuntimeHost _nativeHost = new();
    private readonly string _optionsPath;
    private readonly string _launchStatusPath;
    private readonly string _vrcftOutputOptionsPath;
    private readonly WearTemplateService _wearTemplateService = new();
    private long _launchAttempt;

    private BridgeProcessService()
    {
        var appData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "DreamAirTracking");
        Directory.CreateDirectory(appData);
        _optionsPath = Path.Combine(appData, "bridge_launch_options.json");
        _launchStatusPath = Path.Combine(appData, "runtime_launch_status.json");
        _vrcftOutputOptionsPath = Path.Combine(appData, "vrcft_output_options.json");
        Options = LoadOptions();
        SaveOptions();
    }

    public event EventHandler? StateChanged;

    public bool IsRunning => IsOwnedProcessRunning || _nativeHost.IsRunning || IsLegacyBridgeRunning();

    public bool IsExternalRunning => !IsOwnedProcessRunning && !_nativeHost.IsRunning && IsLegacyBridgeRunning();

    public string? ProfilePath => ResolveOnnxPath(FindRepoRoot());

    public string LastOutput { get; private set; } = "Eye tracking runtime has not been started.";

    public string LastRuntimeCommand { get; private set; } = string.Empty;

    public WearTemplateProbeResult? LatestWearTemplateProbe { get; private set; }

    public string StatusText => IsExternalRunning
        ? "Legacy Bridge is running. Stop it before using the model runtime."
        : LastOutput;

    public BridgeLaunchOptions Options { get; private set; }

    /// <summary>Switch between the Python runtime and the in-process native ONNX runtime (E11).</summary>
    public void SetRuntimeEngine(string engine)
    {
        var normalized = string.Equals(engine, "native", StringComparison.OrdinalIgnoreCase) ? "native" : "python";
        if (string.Equals(Options.RuntimeEngine, normalized, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        Options.RuntimeEngine = normalized;
        SaveOptions();
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    public string CurrentMultitaskModelPreset => NormalizeMultitaskModelPreset(Options.MultitaskModelPreset);

    public string CurrentMultitaskModelLabel => ModelLabel(CurrentMultitaskModelPreset);

    public IReadOnlyList<BridgeModelChoice> GetMultitaskModelChoices()
    {
        var registry = LoadModelRegistry(FindRepoRoot());
        var registryChoices = registry?.Models
            .Where(model =>
                model.DeviceFamily.Equals("Dream Air", StringComparison.OrdinalIgnoreCase) &&
                model.Runtime.Equals("predict_live_multitask", StringComparison.OrdinalIgnoreCase) &&
                model.Role.Equals("main", StringComparison.OrdinalIgnoreCase) &&
                IsCompleteModelPackage(model))
            .OrderByDescending(model => model.Default)
            .ThenBy(model => model.DisplayName, StringComparer.OrdinalIgnoreCase)
            .Select(model => new BridgeModelChoice(model.Id, model.DisplayName, true))
            .ToArray();
        if (registryChoices is { Length: > 0 })
        {
            return registryChoices;
        }

        return Array.Empty<BridgeModelChoice>();
    }

    public void UpdateOptions(BridgeLaunchOptions options)
    {
        NormalizeLoadedOptions(options);
        Options = options;
        SaveOptions();
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    public void SelectMultitaskModelPreset(string presetId)
    {
        var options = Options;
        options.RuntimeModelType = "multitask";
        options.EnableMultitaskVrcftOutput = true;
        options.MultitaskModelPreset = NormalizeMultitaskModelPreset(presetId);
        ApplyMultitaskModelPresetPaths(options, FindRepoRoot());
        UpdateOptions(options);
    }

    public void SetPreferredProfile(string profilePath)
    {
        Options.ProfilePath = profilePath;
        SaveOptions();
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    public void ApplyRuntimeCalibration(
        double centerOffsetX,
        double centerOffsetY,
        double xGain,
        double yGain,
        string? calibrationPath = null,
        string? calibrationSession = null)
    {
        Options.CenterOffsetX = centerOffsetX;
        Options.CenterOffsetY = centerOffsetY;
        Options.XGain = xGain;
        Options.YGain = yGain;
        if (!string.IsNullOrWhiteSpace(calibrationPath))
        {
            Options.QuickGazeCalibrationPath = calibrationPath;
        }

        if (!string.IsNullOrWhiteSpace(calibrationSession))
        {
            Options.QuickGazeCalibrationSession = calibrationSession;
        }

        SaveOptions();
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    public bool BindRuntimeCalibrationToLatestWearTemplate(
        string runtimeCalibrationPath,
        double centerOffsetX,
        double centerOffsetY,
        double xGain,
        double yGain,
        int sampleCount)
    {
        var probe = LatestWearTemplateProbe;
        if (probe?.TemplateId is null)
        {
            return false;
        }

        return _wearTemplateService.BindRuntimeCalibration(
            FindRepoRoot(),
            probe.TemplateId,
            runtimeCalibrationPath,
            centerOffsetX,
            centerOffsetY,
            xGain,
            yGain,
            sampleCount);
    }

    public async Task<bool> BindRuntimeCalibrationToCurrentWearTemplateAsync(
        string runtimeCalibrationPath,
        double centerOffsetX,
        double centerOffsetY,
        double xGain,
        double yGain,
        int sampleCount)
    {
        var repoRoot = FindRepoRoot();
        var templateId = LatestWearTemplateProbe?.TemplateId;
        if (string.IsNullOrWhiteSpace(templateId))
        {
            try
            {
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(3.5));
                var probe = await _wearTemplateService.ProbeAsync(repoRoot, timeout.Token);
                LatestWearTemplateProbe = probe;
                templateId = probe.TemplateId;
            }
            catch
            {
                templateId = null;
            }
        }

        if (string.IsNullOrWhiteSpace(templateId))
        {
            return false;
        }

        var updated = _wearTemplateService.BindRuntimeCalibration(
            repoRoot,
            templateId,
            runtimeCalibrationPath,
            centerOffsetX,
            centerOffsetY,
            xGain,
            yGain,
            sampleCount);
        if (updated)
        {
            StateChanged?.Invoke(this, EventArgs.Empty);
        }

        return updated;
    }

    public async Task StartAsync(string? profilePath = null, bool restartIfRunning = false)
    {
        if (IsOwnedProcessRunning || _nativeHost.IsRunning)
        {
            if (!restartIfRunning)
            {
                LastOutput = "Eye tracking runtime is already running.";
                StateChanged?.Invoke(this, EventArgs.Empty);
                return;
            }

            Stop();
        }

        if (IsLegacyBridgeRunning())
        {
            Stop();
        }

        var repoRoot = FindRepoRoot();
        BridgeMonitorService.Instance.Start();
        LatestWearTemplateProbe = null;
        _launchAttempt++;
        LastOutput = IsMultitaskRuntime
            ? "Checking MobileNetV3 multitask runtime files..."
            : "Checking MobileNetV3 runtime files...";
        StateChanged?.Invoke(this, EventArgs.Empty);
        WriteLaunchStatus("checking_model", repoRoot, null, null, null);
        var onnxPath = ResolveOnnxPath(repoRoot);
        var metadataPath = ResolveMetadataPath(repoRoot, onnxPath);
        var expressionOnnxPath = ResolveExpressionOnnxPath(repoRoot);
        var expressionMetadataPath = ResolveExpressionMetadataPath(repoRoot, expressionOnnxPath);
        if (string.IsNullOrWhiteSpace(onnxPath) || !File.Exists(onnxPath))
        {
            LastOutput = $"ONNX model not found. Expected {onnxPath}.";
            WriteLaunchStatus("failed", repoRoot, onnxPath, metadataPath, LastOutput);
            StateChanged?.Invoke(this, EventArgs.Empty);
            return;
        }

        LastOutput = $"Checking BrokenEye eye streams on 127.0.0.1:{BrokenEyePort}...";
        StateChanged?.Invoke(this, EventArgs.Empty);
        WriteLaunchStatus("checking_brokeneye", repoRoot, onnxPath, metadataPath, null);
        var brokenEyeReady = await ProbeBrokenEyeHttpAsync();
        if (!brokenEyeReady)
        {
            LastOutput = $"BrokenEye HTTP is not ready on 127.0.0.1:{BrokenEyePort}. Open BrokenEye and start its eye streams before starting eye tracking.";
            WriteLaunchStatus("waiting_for_brokeneye", repoRoot, onnxPath, metadataPath, LastOutput);
            StateChanged?.Invoke(this, EventArgs.Empty);
            return;
        }

        if (IsMultitaskRuntime && string.Equals(Options.RuntimeEngine, "native", StringComparison.OrdinalIgnoreCase))
        {
            var nativeConfig = BuildNativeConfig(onnxPath, metadataPath, expressionOnnxPath, expressionMetadataPath);
            LastRuntimeCommand = "native (in-process Microsoft.ML.OnnxRuntime)";
            var nativeStarted = _nativeHost.Start(
                nativeConfig,
                onStopped: () =>
                {
                    LastOutput = "Eye tracking runtime stopped.";
                    StateChanged?.Invoke(this, EventArgs.Empty);
                },
                onError: ex =>
                {
                    LastOutput = $"Native runtime error: {ex.Message}";
                    StateChanged?.Invoke(this, EventArgs.Empty);
                });
            if (!nativeStarted)
            {
                LastOutput = $"{LastOutput} (native runtime failed to initialize)";
                WriteLaunchStatus("failed", repoRoot, onnxPath, metadataPath, LastOutput);
                StateChanged?.Invoke(this, EventArgs.Empty);
                return;
            }
            LastOutput = "Eye tracking runtime starting (native)...";
            WriteLaunchStatus("started", repoRoot, onnxPath, metadataPath, null);
            StateChanged?.Invoke(this, EventArgs.Empty);
            _ = ProbeWearTemplateAsync(repoRoot, _launchAttempt);
            return;
        }

        var scriptName = IsMultitaskRuntime ? "predict_live_multitask.py" : "predict_live.py";
        var scriptPath = Path.Combine(repoRoot, "scripts", "ml", scriptName);
        if (!File.Exists(scriptPath))
        {
            LastOutput = $"Runtime script not found: {scriptPath}";
            WriteLaunchStatus("failed", repoRoot, onnxPath, metadataPath, LastOutput);
            StateChanged?.Invoke(this, EventArgs.Empty);
            return;
        }

        var outputDir = Path.Combine(repoRoot, "runs", "app_eye_tracking_runtime");
        var startInfo = new ProcessStartInfo
        {
            FileName = "python",
            WorkingDirectory = repoRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        AddRuntimeArguments(startInfo, scriptPath, onnxPath, metadataPath, expressionOnnxPath, expressionMetadataPath, outputDir);
        LastRuntimeCommand = BuildCommandPreview(startInfo);

        var process = new Process
        {
            StartInfo = startInfo,
            EnableRaisingEvents = true
        };
        process.OutputDataReceived += (_, args) => HandleOutput(args.Data);
        process.ErrorDataReceived += (_, args) => HandleOutput(args.Data);
        process.Exited += (_, _) =>
        {
            var exitCode = process.ExitCode;
            LastOutput = exitCode == 0
                ? "Eye tracking runtime stopped."
                : $"{LastOutput} Eye tracking runtime stopped with exit code {exitCode}.";
            StateChanged?.Invoke(this, EventArgs.Empty);
        };

        bool started;
        try
        {
            started = process.Start();
        }
        catch (Exception ex)
        {
            LastOutput = $"Eye tracking runtime failed to start: {ex.Message}";
            WriteLaunchStatus("failed", repoRoot, onnxPath, metadataPath, LastOutput);
            StateChanged?.Invoke(this, EventArgs.Empty);
            return;
        }

        if (!started)
        {
            LastOutput = "Eye tracking runtime failed to start.";
            WriteLaunchStatus("failed", repoRoot, onnxPath, metadataPath, LastOutput);
            StateChanged?.Invoke(this, EventArgs.Empty);
            return;
        }

        _process = process;
        LastOutput = "Eye tracking runtime starting...";
        WriteLaunchStatus("started", repoRoot, onnxPath, metadataPath, null);
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        StateChanged?.Invoke(this, EventArgs.Empty);
        _ = ProbeWearTemplateAsync(repoRoot, _launchAttempt);
    }

    public void Stop()
    {
        var stoppedAny = false;
        if (_nativeHost.IsRunning)
        {
            _nativeHost.Stop();
            stoppedAny = true;
        }
        if (_process is not null)
        {
            try
            {
                if (!_process.HasExited)
                {
                    _process.Kill(entireProcessTree: true);
                    stoppedAny = true;
                }
            }
            catch (InvalidOperationException)
            {
            }

            _process.Dispose();
            _process = null;
        }

        foreach (var process in FindLegacyBridgeProcesses())
        {
            using (process)
            {
                try
                {
                    if (!process.HasExited)
                    {
                        process.Kill(entireProcessTree: true);
                        stoppedAny = true;
                    }
                }
                catch (InvalidOperationException)
                {
                }
                catch (Exception ex)
                {
                    LastOutput = $"Could not stop legacy Bridge {process.Id}: {ex.Message}";
                }
            }
        }

        LastOutput = stoppedAny ? "Eye tracking runtime stopped." : "No eye tracking runtime was running.";
        LatestWearTemplateProbe = null;
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    private async Task ProbeWearTemplateAsync(string repoRoot, long launchAttempt)
    {
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(3.5));
            var result = await _wearTemplateService.ProbeAsync(repoRoot, timeout.Token);
            LatestWearTemplateProbe = result;
            LastOutput = $"{LastOutput} Wear template: {FormatWearTemplateProbe(result)}";
            StateChanged?.Invoke(this, EventArgs.Empty);
            if (ShouldApplyWearTemplateCalibration(result))
            {
                ApplyRuntimeCalibration(
                    result.CenterOffsetX!.Value,
                    result.CenterOffsetY!.Value,
                    result.XGain!.Value,
                    result.YGain!.Value,
                    result.RuntimeCalibrationPath,
                    result.TemplateId);
                LastOutput = $"{LastOutput} Applied wear-template runtime calibration.";
                StateChanged?.Invoke(this, EventArgs.Empty);
                if (IsOwnedProcessRunning && launchAttempt == _launchAttempt)
                {
                    await StartAsync(restartIfRunning: true);
                }
            }
        }
        catch (Exception ex)
        {
            LatestWearTemplateProbe = WearTemplateProbeResult.NotRun(ex.Message);
            StateChanged?.Invoke(this, EventArgs.Empty);
        }
    }

    private bool ShouldApplyWearTemplateCalibration(WearTemplateProbeResult result)
    {
        if (!string.Equals(result.Status, "good", StringComparison.OrdinalIgnoreCase)
            || !result.HasRuntimeCalibration
            || !IsFinite(result.CenterOffsetX!.Value)
            || !IsFinite(result.CenterOffsetY!.Value)
            || !IsFinite(result.XGain!.Value)
            || !IsFinite(result.YGain!.Value))
        {
            return false;
        }

        const double epsilon = 0.000001;
        return Math.Abs(Options.CenterOffsetX - result.CenterOffsetX.Value) > epsilon
            || Math.Abs(Options.CenterOffsetY - result.CenterOffsetY.Value) > epsilon
            || Math.Abs(Options.XGain - result.XGain.Value) > epsilon
            || Math.Abs(Options.YGain - result.YGain.Value) > epsilon
            || (!string.IsNullOrWhiteSpace(result.RuntimeCalibrationPath)
                && !string.Equals(Options.QuickGazeCalibrationPath, result.RuntimeCalibrationPath, StringComparison.OrdinalIgnoreCase));
    }

    private static string FormatWearTemplateProbe(WearTemplateProbeResult result)
    {
        var distance = result.Distance is null ? "n/a" : result.Distance.Value.ToString("0.000", CultureInfo.InvariantCulture);
        var template = string.IsNullOrWhiteSpace(result.TemplateId) ? "none" : result.TemplateId;
        var calibration = result.HasRuntimeCalibration ? "calibration=template" : "calibration=none";
        return $"{result.Status}, action={result.Action}, template={template}, distance={distance}, {calibration}, {result.Reason}.";
    }

    private void AddRuntimeArguments(
        ProcessStartInfo startInfo,
        string scriptPath,
        string onnxPath,
        string? metadataPath,
        string? expressionOnnxPath,
        string? expressionMetadataPath,
        string outputDir)
    {
        startInfo.ArgumentList.Add(scriptPath);
        startInfo.ArgumentList.Add("--duration-seconds");
        startInfo.ArgumentList.Add("0");
        startInfo.ArgumentList.Add("--onnx");
        startInfo.ArgumentList.Add(onnxPath);
        if (!string.IsNullOrWhiteSpace(metadataPath) && File.Exists(metadataPath))
        {
            startInfo.ArgumentList.Add("--metadata");
            startInfo.ArgumentList.Add(metadataPath);
        }

        if (IsMultitaskRuntime
            && !string.IsNullOrWhiteSpace(expressionOnnxPath)
            && File.Exists(expressionOnnxPath))
        {
            startInfo.ArgumentList.Add("--expression-onnx");
            startInfo.ArgumentList.Add(expressionOnnxPath);
            startInfo.ArgumentList.Add("--expression-every-n-frames");
            startInfo.ArgumentList.Add("3");
            startInfo.ArgumentList.Add("--expression-ema-alpha");
            startInfo.ArgumentList.Add("0.65");
            if (!string.IsNullOrWhiteSpace(expressionMetadataPath) && File.Exists(expressionMetadataPath))
            {
                startInfo.ArgumentList.Add("--expression-metadata");
                startInfo.ArgumentList.Add(expressionMetadataPath);
            }
        }

        startInfo.ArgumentList.Add("--udp-port");
        startInfo.ArgumentList.Add("9400");
        startInfo.ArgumentList.Add("--monitor-udp-port");
        startInfo.ArgumentList.Add("9401");
        startInfo.ArgumentList.Add("--output-dir");
        startInfo.ArgumentList.Add(outputDir);
        startInfo.ArgumentList.Add("--center-offset-x");
        startInfo.ArgumentList.Add(Format(Options.CenterOffsetX));
        startInfo.ArgumentList.Add("--center-offset-y");
        startInfo.ArgumentList.Add(Format(Options.CenterOffsetY));
        startInfo.ArgumentList.Add("--x-gain");
        startInfo.ArgumentList.Add(Format(Options.XGain));
        startInfo.ArgumentList.Add("--y-gain");
        startInfo.ArgumentList.Add(Format(Options.YGain));
        startInfo.ArgumentList.Add("--clamp");
        startInfo.ArgumentList.Add(Format(Options.Clamp));
        startInfo.ArgumentList.Add("--clamp-mode");
        startInfo.ArgumentList.Add(string.IsNullOrWhiteSpace(Options.ClampMode) ? "soft" : Options.ClampMode);
        startInfo.ArgumentList.Add("--soft-knee");
        startInfo.ArgumentList.Add(Format(Options.SoftKnee));
        startInfo.ArgumentList.Add("--output-map-mode");
        startInfo.ArgumentList.Add(string.IsNullOrWhiteSpace(Options.OutputMapMode) ? "off" : Options.OutputMapMode);
        startInfo.ArgumentList.Add("--output-deadzone");
        startInfo.ArgumentList.Add(Format(Options.OutputDeadzone));
        startInfo.ArgumentList.Add("--output-curve-gamma");
        startInfo.ArgumentList.Add(Format(Options.OutputCurveGamma));
        startInfo.ArgumentList.Add("--output-center-radius");
        startInfo.ArgumentList.Add(Format(Options.OutputCenterRadius));
        startInfo.ArgumentList.Add("--output-center-tau");
        startInfo.ArgumentList.Add(Format(Options.OutputCenterTau));
        startInfo.ArgumentList.Add("--output-center-max-step");
        startInfo.ArgumentList.Add(Format(Options.OutputCenterMaxStep));
        if (IsMultitaskRuntime)
        {
            startInfo.ArgumentList.Add("--pupil-output-mode");
            startInfo.ArgumentList.Add(ResolvePupilOutputMode());
            startInfo.ArgumentList.Add("--pupil-wide-enter-threshold");
            startInfo.ArgumentList.Add(Format(Options.PupilWideEnterThreshold));
            startInfo.ArgumentList.Add("--pupil-wide-exit-threshold");
            startInfo.ArgumentList.Add(Format(Options.PupilWideExitThreshold));
            startInfo.ArgumentList.Add("--pupil-wide-hold-frames");
            startInfo.ArgumentList.Add(Math.Max(0, Options.PupilWideHoldFrames).ToString(CultureInfo.InvariantCulture));
            startInfo.ArgumentList.Add("--pupil-wide-ema-alpha");
            startInfo.ArgumentList.Add(Format(Options.PupilWideEmaAlpha));
            startInfo.ArgumentList.Add("--eye-expression-mode");
            startInfo.ArgumentList.Add(Options.EnableVrcftEyeExpressions ? "steamlink_eye_shapes" : "off");
            startInfo.ArgumentList.Add("--wide-source");
            startInfo.ArgumentList.Add("model_head");
            startInfo.ArgumentList.Add("--wide-model-threshold");
            startInfo.ArgumentList.Add("0.93");
            startInfo.ArgumentList.Add("--wide-output-threshold");
            startInfo.ArgumentList.Add("0.98");
            startInfo.ArgumentList.Add("--eye-shape-wide-scale");
            startInfo.ArgumentList.Add(Format(Options.EyeShapeWideScale));
            startInfo.ArgumentList.Add("--eye-shape-squint-scale");
            startInfo.ArgumentList.Add(Format(Options.EyeShapeSquintScale));
            startInfo.ArgumentList.Add("--eye-shape-gamma");
            startInfo.ArgumentList.Add(Format(Options.EyeShapeGamma));
            startInfo.ArgumentList.Add("--eye-shape-deadzone");
            startInfo.ArgumentList.Add(Format(Options.EyeShapeDeadzone));
            startInfo.ArgumentList.Add("--wide-gaze-up-suppress");
            startInfo.ArgumentList.Add(Format(Options.WideGazeUpSuppress));
            startInfo.ArgumentList.Add("--wide-gaze-up-sign");
            startInfo.ArgumentList.Add(Format(Options.WideGazeUpSign));
            startInfo.ArgumentList.Add("--openness-curve-mode");
            startInfo.ArgumentList.Add(string.IsNullOrWhiteSpace(Options.OpennessCurveMode) ? "soft_open_plateau" : Options.OpennessCurveMode);
            startInfo.ArgumentList.Add("--openness-hover-enter-velocity");
            startInfo.ArgumentList.Add(Format(Options.OpennessHoverEnterVelocity));
            startInfo.ArgumentList.Add("--openness-hover-exit-velocity");
            startInfo.ArgumentList.Add(Format(Options.OpennessHoverExitVelocity));
            startInfo.ArgumentList.Add("--openness-hover-hold-frames");
            startInfo.ArgumentList.Add(Math.Max(1, Options.OpennessHoverHoldFrames).ToString(CultureInfo.InvariantCulture));
            startInfo.ArgumentList.Add("--openness-full-open-threshold");
            startInfo.ArgumentList.Add(Format(Options.OpennessFullOpenThreshold));
            startInfo.ArgumentList.Add("--openness-boost-knee");
            startInfo.ArgumentList.Add(Format(Options.OpennessBoostKnee));
            startInfo.ArgumentList.Add("--openness-boost-gamma");
            startInfo.ArgumentList.Add(Format(Options.OpennessBoostGamma));
            startInfo.ArgumentList.Add("--tracking-openness-machine");
            startInfo.ArgumentList.Add(Options.EnableTrackingStateMachine ? "on" : "off");
            startInfo.ArgumentList.Add(
                Options.EnableTrackingStateMachine && Options.EnableClosureGazeHold
                    ? "--tracking-gaze-hold"
                    : "--no-tracking-gaze-hold");
            startInfo.ArgumentList.Add("--tracking-close-confirm-frames");
            startInfo.ArgumentList.Add(Math.Max(0, Options.TrackingCloseConfirmFrames).ToString(CultureInfo.InvariantCulture));
            startInfo.ArgumentList.Add("--tracking-close-confirm-floor");
            startInfo.ArgumentList.Add(Format(Options.TrackingCloseConfirmFloor));
            startInfo.ArgumentList.Add("--tracking-partial-deadband");
            startInfo.ArgumentList.Add(Format(Options.TrackingPartialDeadband));
            var perEyeCalibrationPath = ResolveOpennessPerEyeCalibrationPath(FindRepoRoot());
            if (!string.IsNullOrWhiteSpace(perEyeCalibrationPath))
            {
                startInfo.ArgumentList.Add("--openness-per-eye-calibration");
                startInfo.ArgumentList.Add(perEyeCalibrationPath);
            }
            startInfo.ArgumentList.Add("--ema-alpha");
            startInfo.ArgumentList.Add(Format(Options.EmaAlpha));
            startInfo.ArgumentList.Add("--max-step");
            startInfo.ArgumentList.Add(Format(Options.MaxStep));
            startInfo.ArgumentList.Add("--print-every");
            startInfo.ArgumentList.Add("30");
            startInfo.ArgumentList.Add("--snapshot-every");
            startInfo.ArgumentList.Add("0");
            return;
        }

        startInfo.ArgumentList.Add("--center-bias");
        startInfo.ArgumentList.Add(Format(Options.CenterBias));
        startInfo.ArgumentList.Add("--center-bias-start");
        startInfo.ArgumentList.Add(Format(Options.CenterBiasStart));
        startInfo.ArgumentList.Add("--ema-alpha");
        startInfo.ArgumentList.Add(Format(Options.EmaAlpha));
        startInfo.ArgumentList.Add("--max-step");
        startInfo.ArgumentList.Add(Format(Options.MaxStep));
        startInfo.ArgumentList.Add("--confidence");
        startInfo.ArgumentList.Add(Options.EnableEyeTracking ? "1.0" : "0.0");
        startInfo.ArgumentList.Add("--openness");
        startInfo.ArgumentList.Add("1.0");
        startInfo.ArgumentList.Add("--openness-mode");
        startInfo.ArgumentList.Add(string.IsNullOrWhiteSpace(Options.OpennessMode) ? "image" : Options.OpennessMode);
        startInfo.ArgumentList.Add("--openness-boost-gamma");
        startInfo.ArgumentList.Add(Format(Options.OpennessBoostGamma));
        startInfo.ArgumentList.Add("--openness-full-open-threshold");
        startInfo.ArgumentList.Add(Format(Options.OpennessFullOpenThreshold));
        startInfo.ArgumentList.Add("--openness-boost-knee");
        startInfo.ArgumentList.Add(Format(Options.OpennessBoostKnee));
        startInfo.ArgumentList.Add("--normalization-mode");
        startInfo.ArgumentList.Add(string.IsNullOrWhiteSpace(Options.NormalizationMode) ? "probe_only" : Options.NormalizationMode);
        startInfo.ArgumentList.Add("--normalization-target-x");
        startInfo.ArgumentList.Add(Format(Options.NormalizationTargetX));
        startInfo.ArgumentList.Add("--normalization-target-y");
        startInfo.ArgumentList.Add(Format(Options.NormalizationTargetY));
        startInfo.ArgumentList.Add("--normalization-left-target-x");
        startInfo.ArgumentList.Add(Format(Options.NormalizationLeftTargetX));
        startInfo.ArgumentList.Add("--normalization-right-target-x");
        startInfo.ArgumentList.Add(Format(Options.NormalizationRightTargetX));
        startInfo.ArgumentList.Add("--normalization-left-target-y");
        startInfo.ArgumentList.Add(Format(Options.NormalizationLeftTargetY));
        startInfo.ArgumentList.Add("--normalization-right-target-y");
        startInfo.ArgumentList.Add(Format(Options.NormalizationRightTargetY));
        startInfo.ArgumentList.Add("--normalization-max-shift-x");
        startInfo.ArgumentList.Add(Format(Options.NormalizationMaxShiftX));
        startInfo.ArgumentList.Add("--normalization-max-shift-y");
        startInfo.ArgumentList.Add(Format(Options.NormalizationMaxShiftY));
        startInfo.ArgumentList.Add("--normalization-scale-min");
        startInfo.ArgumentList.Add(Format(Options.NormalizationScaleMin));
        startInfo.ArgumentList.Add("--normalization-scale-max");
        startInfo.ArgumentList.Add(Format(Options.NormalizationScaleMax));
        startInfo.ArgumentList.Add("--normalization-hold-frames");
        startInfo.ArgumentList.Add(Options.NormalizationHoldFrames.ToString(CultureInfo.InvariantCulture));
        var opennessCalibrationPath = ResolveOpennessCalibrationPath(FindRepoRoot());
        if (!string.IsNullOrWhiteSpace(opennessCalibrationPath))
        {
            startInfo.ArgumentList.Add("--openness-calibration");
            startInfo.ArgumentList.Add(opennessCalibrationPath);
        }
        startInfo.ArgumentList.Add("--print-every");
        startInfo.ArgumentList.Add("30");
        startInfo.ArgumentList.Add("--snapshot-every");
        startInfo.ArgumentList.Add("0");
    }

    private void HandleOutput(string? line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return;
        }

        LastOutput = line;
        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    private void WriteLaunchStatus(
        string state,
        string repoRoot,
        string? onnxPath,
        string? metadataPath,
        string? error)
    {
        try
        {
            var payload = new
            {
                state,
                error,
                attempt = _launchAttempt,
                updatedAt = DateTimeOffset.Now,
                repoRoot,
                onnx = onnxPath,
                metadata = metadataPath,
                command = LastRuntimeCommand,
                options = Options
            };
            var json = JsonSerializer.Serialize(payload, AppJsonOptions.Web(writeIndented: true));
            File.WriteAllText(_launchStatusPath, json);
        }
        catch
        {
        }
    }

    private static string BuildCommandPreview(ProcessStartInfo startInfo)
        => startInfo.FileName + " " + string.Join(" ", startInfo.ArgumentList.Select(QuoteArgument));

    private static string QuoteArgument(string argument)
        => argument.Contains(' ') ? $"\"{argument}\"" : argument;

    private static async Task<bool> ProbeBrokenEyeHttpAsync()
    {
        try
        {
            using var http = new HttpClient
            {
                Timeout = TimeSpan.FromMilliseconds(1200)
            };

            return await ProbeBrokenEyeEyeAsync(http, "left") && await ProbeBrokenEyeEyeAsync(http, "right");
        }
        catch
        {
            return false;
        }
    }

    private static async Task<bool> ProbeBrokenEyeEyeAsync(HttpClient http, string eye)
    {
        try
        {
            using var response = await http.GetAsync(
                $"http://127.0.0.1:{BrokenEyePort}/eye/{eye}",
                HttpCompletionOption.ResponseHeadersRead);
            var mediaType = response.Content.Headers.ContentType?.MediaType ?? string.Empty;
            return response.IsSuccessStatusCode
                && mediaType.Contains("multipart", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "DreamAirTracking.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "DreamAirTracking.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return AppContext.BaseDirectory;
    }

    private string? ResolveOnnxPath(string repoRoot)
    {
        if (IsMultitaskRuntime)
        {
            if (!string.IsNullOrWhiteSpace(Options.MultitaskOnnxPath) && File.Exists(Options.MultitaskOnnxPath))
            {
                return Options.MultitaskOnnxPath;
            }

            var registryModel = ResolveRegistryMainModel(repoRoot);
            if (registryModel is not null)
            {
                return registryModel.ResolvedOnnx;
            }

            return FindDefaultMultitaskOnnxPath(repoRoot);
        }

        if (!string.IsNullOrWhiteSpace(Options.OnnxPath) && File.Exists(Options.OnnxPath))
        {
            return Options.OnnxPath;
        }

        return FindDefaultOnnxPath(repoRoot);
    }

    private string? ResolveMetadataPath(string repoRoot, string? onnxPath)
    {
        if (IsMultitaskRuntime)
        {
            if (!string.IsNullOrWhiteSpace(Options.MultitaskMetadataPath) && File.Exists(Options.MultitaskMetadataPath))
            {
                return Options.MultitaskMetadataPath;
            }

            if (!string.IsNullOrWhiteSpace(onnxPath))
            {
                var metadata = Path.ChangeExtension(onnxPath, ".metadata.json");
                if (File.Exists(metadata))
                {
                    return metadata;
                }
            }

            var registryModel = ResolveRegistryMainModel(repoRoot);
            if (registryModel is not null)
            {
                return registryModel.ResolvedMetadata;
            }

            var multitaskMetadata = Path.Combine(repoRoot, "runs", PreferredMultitaskRun, "eye_multitask.metadata.json");
            return File.Exists(multitaskMetadata) ? multitaskMetadata : null;
        }

        if (!string.IsNullOrWhiteSpace(Options.MetadataPath) && File.Exists(Options.MetadataPath))
        {
            return Options.MetadataPath;
        }

        if (!string.IsNullOrWhiteSpace(onnxPath))
        {
            var metadata = Path.ChangeExtension(onnxPath, ".metadata.json");
            if (File.Exists(metadata))
            {
                return metadata;
            }
        }

        var defaultMetadata = Path.Combine(repoRoot, "runs", "gaze_mobilenetv3_small_live_round1_round2", "gaze_baseline.metadata.json");
        return File.Exists(defaultMetadata) ? defaultMetadata : null;
    }

    private string? ResolveExpressionOnnxPath(string repoRoot)
    {
        if (!IsMultitaskRuntime)
        {
            return null;
        }

        if (!string.IsNullOrWhiteSpace(Options.ExpressionOnnxPath) && File.Exists(Options.ExpressionOnnxPath))
        {
            return Options.ExpressionOnnxPath;
        }

        var registryExpression = LoadModelRegistry(repoRoot)?.FindDefaultExpression();
        if (registryExpression is not null)
        {
            return registryExpression.ResolvedOnnx;
        }

        var defaultExpression = Path.Combine(repoRoot, "runs", DefaultExpressionRun, "eye_multitask.onnx");
        return File.Exists(defaultExpression) ? defaultExpression : null;
    }

    private string? ResolveExpressionMetadataPath(string repoRoot, string? expressionOnnxPath)
    {
        if (!IsMultitaskRuntime)
        {
            return null;
        }

        if (!string.IsNullOrWhiteSpace(Options.ExpressionMetadataPath) && File.Exists(Options.ExpressionMetadataPath))
        {
            return Options.ExpressionMetadataPath;
        }

        if (!string.IsNullOrWhiteSpace(expressionOnnxPath))
        {
            var metadata = Path.ChangeExtension(expressionOnnxPath, ".metadata.json");
            if (File.Exists(metadata))
            {
                return metadata;
            }
        }

        var registryExpression = LoadModelRegistry(repoRoot)?.FindDefaultExpression();
        if (registryExpression is not null)
        {
            return registryExpression.ResolvedMetadata;
        }

        var defaultMetadata = Path.Combine(repoRoot, "runs", DefaultExpressionRun, "eye_multitask.metadata.json");
        return File.Exists(defaultMetadata) ? defaultMetadata : null;
    }

    private ModelRegistryEntry? ResolveRegistryMainModel(string repoRoot)
    {
        var registry = LoadModelRegistry(repoRoot);
        if (registry is null)
        {
            return null;
        }

        return FindUsableMainModel(registry, Options.MultitaskModelPreset);
    }

    private static string? FindDefaultOnnxPath(string repoRoot)
    {
        var preferred = Path.Combine(repoRoot, "runs", "gaze_mobilenetv3_small_live_round1_round2", "gaze_baseline.onnx");
        if (File.Exists(preferred))
        {
            return preferred;
        }

        return Directory.EnumerateFiles(repoRoot, "gaze_baseline.onnx", SearchOption.AllDirectories)
            .Select(path => new FileInfo(path))
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .Select(file => file.FullName)
            .FirstOrDefault();
    }

    private static string? FindDefaultMultitaskOnnxPath(string repoRoot)
    {
        var preferred = Path.Combine(repoRoot, "runs", PreferredMultitaskRun, "eye_multitask.onnx");
        if (File.Exists(preferred))
        {
            return preferred;
        }

        return Directory.EnumerateFiles(repoRoot, "eye_multitask.onnx", SearchOption.AllDirectories)
            .Select(path => new FileInfo(path))
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .Select(file => file.FullName)
            .FirstOrDefault();
    }

    private bool IsMultitaskRuntime =>
        string.Equals(Options.RuntimeModelType, "multitask", StringComparison.OrdinalIgnoreCase);

    private bool IsOwnedProcessRunning => _process is { HasExited: false };

    private bool IsLegacyBridgeRunning()
        => IsBridgeMutexHeld(9400) || FindLegacyBridgeProcesses().Any(process =>
        {
            using (process)
            {
                try
                {
                    return !process.HasExited;
                }
                catch (InvalidOperationException)
                {
                    return false;
                }
                catch (System.ComponentModel.Win32Exception)
                {
                    return false;
                }
                catch (UnauthorizedAccessException)
                {
                    return false;
                }
            }
        });

    private IEnumerable<Process> FindLegacyBridgeProcesses()
    {
        foreach (var process in Process.GetProcessesByName(LegacyBridgeProcessName))
        {
            yield return process;
        }
    }

    private static bool IsBridgeMutexHeld(int udpPort)
    {
        var mutexName = $"DreamAirTracking.Bridge.Udp.{udpPort}";
        try
        {
            using var mutex = new Mutex(false, mutexName);
            if (mutex.WaitOne(0))
            {
                mutex.ReleaseMutex();
                return false;
            }

            return true;
        }
        catch (AbandonedMutexException)
        {
            return false;
        }
        catch
        {
            return false;
        }
    }

    public string? FindLatestPupilCalibrationPath()
    {
        var repoRoot = FindRepoRoot();
        return Directory.EnumerateFiles(repoRoot, "pupil_diameter_calibration.json", SearchOption.AllDirectories)
            .Select(path => new FileInfo(path))
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .Select(file => file.FullName)
            .FirstOrDefault();
    }

    public string? FindLatestQuickGazeCalibrationPath()
    {
        var repoRoot = FindRepoRoot();
        return Directory.EnumerateFiles(repoRoot, "runtime_gaze_calibration.json", SearchOption.AllDirectories)
            .Select(path => new FileInfo(path))
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .Select(file => file.FullName)
            .FirstOrDefault();
    }

    public string? FindLatestOpennessCalibrationPath()
    {
        var repoRoot = FindRepoRoot();
        return Directory.EnumerateFiles(repoRoot, "openness_calibration.json", SearchOption.AllDirectories)
            .Select(path => new FileInfo(path))
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .Select(file => file.FullName)
            .FirstOrDefault();
    }

    private string? ResolveOpennessCalibrationPath(string repoRoot)
    {
        if (!string.IsNullOrWhiteSpace(Options.OpennessCalibrationPath) && File.Exists(Options.OpennessCalibrationPath))
        {
            return Options.OpennessCalibrationPath;
        }

        return Directory.EnumerateFiles(repoRoot, "openness_calibration.json", SearchOption.AllDirectories)
            .Select(path => new FileInfo(path))
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .Select(file => file.FullName)
            .FirstOrDefault();
    }

    private string? ResolveOpennessPerEyeCalibrationPath(string repoRoot)
    {
        if (!string.IsNullOrWhiteSpace(Options.OpennessPerEyeCalibrationPath) && File.Exists(Options.OpennessPerEyeCalibrationPath))
        {
            return Options.OpennessPerEyeCalibrationPath;
        }

        // Per-eye normalization needs a v2 calibration (open_p95/closed_p05). Find the newest
        // openness_calibration.json under the repo that is v2; ignore legacy v1 files (for the
        // dead image-openness branch) so a stale pinned v1 never masks a fresh v2 calibration.
        try
        {
            return Directory.EnumerateFiles(repoRoot, "openness_calibration.json", SearchOption.AllDirectories)
                .Select(path => new FileInfo(path))
                .OrderByDescending(file => file.LastWriteTimeUtc)
                .Where(file => IsV2OpennessCalibration(file.FullName))
                .Select(file => file.FullName)
                .FirstOrDefault();
        }
        catch
        {
            return null;
        }
    }

    private static bool IsV2OpennessCalibration(string path)
    {
        try
        {
            return File.ReadAllText(path).Contains("\"open_p95\"", StringComparison.Ordinal);
        }
        catch
        {
            return false;
        }
    }

    private BridgeLaunchOptions LoadOptions()
    {
        try
        {
            if (File.Exists(_optionsPath))
            {
                var json = File.ReadAllText(_optionsPath);
                var options = JsonSerializer.Deserialize<BridgeLaunchOptions>(json, AppJsonOptions.Web())
                    ?? new BridgeLaunchOptions();
                NormalizeLoadedOptions(options);
                return options;
            }
        }
        catch
        {
        }

        return new BridgeLaunchOptions();
    }

    private void SaveOptions()
    {
        var json = JsonSerializer.Serialize(Options, AppJsonOptions.Web(writeIndented: true));
        File.WriteAllText(_optionsPath, json);
        WriteVrcftOutputOptions();
    }

    private void WriteVrcftOutputOptions()
    {
        try
        {
            var payload = new
            {
                updatedAt = DateTimeOffset.Now,
                xOffset = Options.VrcftOutputXOffset,
                yOffset = Options.VrcftOutputYOffset,
                xGain = Options.VrcftOutputXGain,
                yGain = Options.VrcftOutputYGain,
                xSign = Options.VrcftOutputXSign,
                ySign = Options.VrcftOutputYSign,
                enableEyeExpression = Options.EnableVrcftEyeExpressions
            };
            var json = JsonSerializer.Serialize(payload, AppJsonOptions.Web(writeIndented: true));
            File.WriteAllText(_vrcftOutputOptionsPath, json);
        }
        catch
        {
        }
    }

    private static string Format(double value) => value.ToString("0.########", CultureInfo.InvariantCulture);

    private string ResolvePupilOutputMode()
    {
        if (Options.EnablePupilAssist)
        {
            return "expression_constrict_on_wide";
        }

        return Options.EnablePupilDiameter ? "model_radius" : "off";
    }

    // Maps BridgeLaunchOptions onto the native runtime config, matching exactly what
    // AddRuntimeArguments passes to predict_live_multitask.py so native == python behavior.
    private NeuralRuntimeConfig BuildNativeConfig(string onnxPath, string? metadataPath,
        string? expressionOnnxPath, string? expressionMetadataPath)
    {
        var cfg = new NeuralRuntimeConfig
        {
            MainOnnxPath = onnxPath,
            ImageSize = EyeModelMetadata.ReadImageSize(metadataPath, 128),
            ExpressionOnnxPath = !string.IsNullOrWhiteSpace(expressionOnnxPath) && File.Exists(expressionOnnxPath) ? expressionOnnxPath : null,
            ExpressionImageSize = EyeModelMetadata.ReadImageSize(expressionMetadataPath, 128),
            ExpressionEveryNFrames = 3,
            ExpressionEmaAlpha = 0.65,
            OnnxThreads = 2,
            OpennessCurveMode = string.IsNullOrWhiteSpace(Options.OpennessCurveMode) ? "soft_open_plateau" : Options.OpennessCurveMode,
            OpennessFullOpenThreshold = Options.OpennessFullOpenThreshold,
            OpennessBoostKnee = Options.OpennessBoostKnee,
            OpennessBoostGamma = Options.OpennessBoostGamma,
            CenterOffsetX = Options.CenterOffsetX,
            CenterOffsetY = Options.CenterOffsetY,
            XGain = Options.XGain,
            YGain = Options.YGain,
            Clamp = Options.Clamp,
            ClampMode = string.IsNullOrWhiteSpace(Options.ClampMode) ? "soft" : Options.ClampMode,
            SoftKnee = Options.SoftKnee,
            EmaAlpha = Options.EmaAlpha,
            MaxStep = Options.MaxStep,
            OutputMapMode = string.IsNullOrWhiteSpace(Options.OutputMapMode) ? "off" : Options.OutputMapMode,
            OutputDeadzone = Options.OutputDeadzone,
            OutputCurveGamma = Options.OutputCurveGamma,
            OutputCenterRadius = Options.OutputCenterRadius,
            OutputCenterTau = Options.OutputCenterTau,
            OutputCenterMaxStep = Options.OutputCenterMaxStep,
            TrackingOpennessMachine = Options.EnableTrackingStateMachine,
            TrackingGazeHold = Options.EnableTrackingStateMachine && Options.EnableClosureGazeHold,
            TrackingCloseConfirmFrames = Math.Max(0, Options.TrackingCloseConfirmFrames),
            TrackingCloseConfirmFloor = Options.TrackingCloseConfirmFloor,
            TrackingPartialDeadband = Options.TrackingPartialDeadband,
            WideSource = "model_head",
            EyeShapeWideScale = Options.EyeShapeWideScale,
            EyeShapeSquintScale = Options.EyeShapeSquintScale,
            EyeShapeGamma = Options.EyeShapeGamma,
            EyeShapeDeadzone = Options.EyeShapeDeadzone,
            WideGazeUpSuppress = Options.WideGazeUpSuppress,
            WideGazeUpSign = Options.WideGazeUpSign,
            PupilWideEnterThreshold = Options.PupilWideEnterThreshold,
            PupilWideExitThreshold = Options.PupilWideExitThreshold,
            PupilWideHoldFrames = Math.Max(0, Options.PupilWideHoldFrames),
            PupilWideEmaAlpha = Options.PupilWideEmaAlpha,
            PupilOutputMode = ResolvePupilOutputMode(),
            EyeExpressionMode = Options.EnableVrcftEyeExpressions ? "steamlink_eye_shapes" : "off",
            UdpPort = 9400,
            MonitorUdpPort = 9401,
        };
        var perEyeCalPath = ResolveOpennessPerEyeCalibrationPath(FindRepoRoot());
        if (EyeModelMetadata.TryLoadPerEyeCalibration(perEyeCalPath, out var openP95, out var closedP05, out var halfP50))
        {
            cfg.OpenP95 = openP95;
            cfg.ClosedP05 = closedP05;
            cfg.HalfP50 = halfP50;
        }
        cfg.OpennessHoverEnterVelocity = Options.OpennessHoverEnterVelocity;
        cfg.OpennessHoverExitVelocity = Options.OpennessHoverExitVelocity;
        cfg.OpennessHoverHoldFrames = Math.Max(1, Options.OpennessHoverHoldFrames);
        return cfg;
    }

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    private static void NormalizeLoadedOptions(BridgeLaunchOptions options)
    {
        var loadedPreset = options.MultitaskModelPreset;
        var loadedMultitaskOnnxPath = options.MultitaskOnnxPath;
        var loadedMultitaskMetadataPath = options.MultitaskMetadataPath;
        options.MultitaskModelPreset = NormalizeMultitaskModelPreset(options.MultitaskModelPreset);
        var migratedToDefaultMultitask =
            options.MultitaskModelPreset == DefaultMultitaskModelPreset
            && ((!string.IsNullOrWhiteSpace(loadedPreset)
                    && !string.Equals(loadedPreset, DefaultMultitaskModelPreset, StringComparison.OrdinalIgnoreCase))
                || IsNonDefaultMultitaskPath(loadedMultitaskOnnxPath)
                || IsNonDefaultMultitaskPath(loadedMultitaskMetadataPath));

        if (string.IsNullOrWhiteSpace(options.NormalizationMode) || options.NormalizationMode == "off")
        {
            options.NormalizationMode = "probe_only";
        }

        if (!string.Equals(options.RuntimeEngine, "python", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(options.RuntimeEngine, "native", StringComparison.OrdinalIgnoreCase))
        {
            options.RuntimeEngine = "native"; // empty/invalid -> native (E11, on-headset validated); explicit python/native preserved
        }

        if (string.IsNullOrWhiteSpace(options.RuntimeModelType)
            || string.Equals(options.RuntimeModelType, "gaze_only", StringComparison.OrdinalIgnoreCase))
        {
            options.RuntimeModelType = "multitask";
            options.EnableMultitaskVrcftOutput = true;
        }
        else if (!string.Equals(options.RuntimeModelType, "multitask", StringComparison.OrdinalIgnoreCase))
        {
            options.RuntimeModelType = "multitask";
            options.EnableMultitaskVrcftOutput = true;
        }
        else
        {
            options.EnableMultitaskVrcftOutput = true;
        }

        // Eye shapes (EyeWide/EyeSquint) are user-controlled. With the fused VRCFT module
        // DreamAirTracking owns the Expression slot (and proxies the SRanipal mouth into it), so
        // eye-wide is ours to drive — honor the loaded/UI value instead of forcing it off.

        if (options.MultitaskModelPreset == DefaultMultitaskModelPreset
            && IsNonDefaultMultitaskPath(options.MultitaskOnnxPath))
        {
            options.MultitaskOnnxPath = null;
        }

        if (options.MultitaskModelPreset == DefaultMultitaskModelPreset
            && IsNonDefaultMultitaskPath(options.MultitaskMetadataPath))
        {
            options.MultitaskMetadataPath = null;
        }

        if (migratedToDefaultMultitask)
        {
            options.QuickGazeCalibrationPath = null;
            options.QuickGazeCalibrationSession = null;
            options.CenterOffsetX = 0.0;
            options.CenterOffsetY = 0.0;
            options.XGain = 1.0;
            options.YGain = 1.0;
        }

        options.OutputMapMode = "off";

        if (IsNear(options.OutputDeadzone, 0.04)
            && IsNear(options.OutputCurveGamma, 1.20)
            && IsNear(options.OutputCenterRadius, 0.32)
            && IsNear(options.OutputCenterTau, 2.5)
            && IsNear(options.OutputCenterMaxStep, 0.08))
        {
            options.OutputDeadzone = 0.015;
            options.OutputCurveGamma = 1.0;
            options.OutputCenterRadius = 0.08;
            options.OutputCenterTau = 6.0;
            options.OutputCenterMaxStep = 0.025;
        }

        options.OutputDeadzone = Clamp(options.OutputDeadzone, 0.0, 0.25, 0.015);
        options.OutputCurveGamma = Clamp(options.OutputCurveGamma, 0.35, 3.0, 1.0);
        options.OutputCenterRadius = Clamp(options.OutputCenterRadius, 0.02, 0.80, 0.08);
        options.OutputCenterTau = Clamp(options.OutputCenterTau, 0.25, 10.0, 6.0);
        options.OutputCenterMaxStep = Clamp(options.OutputCenterMaxStep, 0.0, 0.50, 0.025);

        if ((IsNear(options.OpennessFullOpenThreshold, 0.82)
                && IsNear(options.OpennessBoostKnee, 0.55)
                && IsNear(options.OpennessBoostGamma, 1.8))
            || (IsNear(options.OpennessFullOpenThreshold, 0.82)
                && IsNear(options.OpennessBoostKnee, 0.25)
                && IsNear(options.OpennessBoostGamma, 0.75))
            || (IsNear(options.OpennessFullOpenThreshold, 0.90)
                && IsNear(options.OpennessBoostKnee, 0.28)
                && IsNear(options.OpennessBoostGamma, 1.25)))
        {
            options.OpennessCurveMode = "blink_s_curve";
            options.OpennessFullOpenThreshold = 0.90;
            options.OpennessBoostKnee = 0.28;
            options.OpennessBoostGamma = 1.25;
        }

        if (string.Equals(options.OpennessCurveMode, "soft_open_plateau", StringComparison.OrdinalIgnoreCase)
            && IsNear(options.OpennessFullOpenThreshold, 0.93)
            && IsNear(options.OpennessBoostKnee, 0.28)
            && IsNear(options.OpennessBoostGamma, 1.0))
        {
            options.OpennessCurveMode = "blink_s_curve";
            options.OpennessFullOpenThreshold = 0.90;
            options.OpennessBoostKnee = 0.28;
            options.OpennessBoostGamma = 1.25;
        }

        if (string.Equals(options.OpennessCurveMode, "blink_s_curve", StringComparison.OrdinalIgnoreCase)
            && IsNear(options.OpennessFullOpenThreshold, 0.90)
            && IsNear(options.OpennessBoostKnee, 0.80)
            && IsNear(options.OpennessBoostGamma, 1.0))
        {
            options.OpennessBoostKnee = 0.28;
            options.OpennessBoostGamma = 1.25;
        }

        if (string.IsNullOrWhiteSpace(options.OpennessCurveMode))
        {
            options.OpennessCurveMode = "blink_s_curve";
        }

        options.OpennessFullOpenThreshold = Clamp(options.OpennessFullOpenThreshold, 0.55, 1.0, 0.90);
        options.OpennessBoostKnee = Clamp(
            options.OpennessBoostKnee,
            0.0,
            Math.Max(0.001, options.OpennessFullOpenThreshold - 0.001),
            0.28);
        options.OpennessBoostGamma = Clamp(options.OpennessBoostGamma, 0.05, 4.0, 1.25);
        options.EyeShapeWideScale = Clamp(options.EyeShapeWideScale, 0.0, 1.0, 0.60);
        options.EyeShapeSquintScale = Clamp(options.EyeShapeSquintScale, 0.0, 1.0, 0.55);
        options.EyeShapeGamma = Clamp(options.EyeShapeGamma, 0.05, 4.0, 1.15);
        options.EyeShapeDeadzone = Clamp(options.EyeShapeDeadzone, 0.0, 0.95, 0.03);
        if (IsNear(options.PupilWideEnterThreshold, 0.75)
            && IsNear(options.PupilWideExitThreshold, 0.40)
            && options.PupilWideHoldFrames == 12
            && IsNear(options.PupilWideEmaAlpha, 0.35))
        {
            options.PupilWideEnterThreshold = 0.90;
            options.PupilWideExitThreshold = 0.86;
            options.PupilWideHoldFrames = 8;
            options.PupilWideEmaAlpha = 0.45;
        }

        options.PupilWideEnterThreshold = Clamp(options.PupilWideEnterThreshold, 0.0, 1.0, 0.90);
        options.PupilWideExitThreshold = Clamp(
            options.PupilWideExitThreshold,
            0.0,
            Math.Min(1.0, options.PupilWideEnterThreshold),
            0.86);
        options.PupilWideHoldFrames = Math.Clamp(options.PupilWideHoldFrames, 0, 90);
        options.PupilWideEmaAlpha = Clamp(options.PupilWideEmaAlpha, 0.01, 1.0, 0.45);

        if (!IsFinite(options.VrcftOutputXGain) || Math.Abs(options.VrcftOutputXGain) < 0.000001)
        {
            options.VrcftOutputXGain = 1.0;
        }

        if (!IsFinite(options.VrcftOutputYGain) || Math.Abs(options.VrcftOutputYGain) < 0.000001)
        {
            options.VrcftOutputYGain = 1.0;
        }

        options.VrcftOutputXSign = options.VrcftOutputXSign < 0 ? -1.0 : 1.0;
        options.VrcftOutputYSign = options.VrcftOutputYSign < 0 ? -1.0 : 1.0;
        if (options.EnablePupilAssist && options.EnablePupilDiameter)
        {
            options.EnablePupilDiameter = false;
        }
    }

    private static void ApplyMultitaskModelPresetPaths(BridgeLaunchOptions options, string repoRoot)
    {
        var preset = NormalizeMultitaskModelPreset(options.MultitaskModelPreset);
        var registry = LoadModelRegistry(repoRoot);
        var registryModel = FindUsableMainModel(registry, preset);
        if (registryModel is not null)
        {
            options.MultitaskModelPreset = registryModel.Id;
            options.MultitaskOnnxPath = registryModel.ResolvedOnnx;
            options.MultitaskMetadataPath = registryModel.ResolvedMetadata;
            var expression = registry?.FindDefaultExpression();
            options.ExpressionOnnxPath = expression?.ResolvedOnnx;
            options.ExpressionMetadataPath = expression?.ResolvedMetadata;
            return;
        }

        var runDirectory = MultitaskRunDirectory(preset);
        var onnxPath = Path.Combine(repoRoot, "runs", runDirectory, "eye_multitask.onnx");
        var metadataPath = Path.Combine(repoRoot, "runs", runDirectory, "eye_multitask.metadata.json");
        options.MultitaskOnnxPath = onnxPath;
        options.MultitaskMetadataPath = metadataPath;
        if (preset == DefaultMultitaskModelPreset)
        {
            options.ExpressionOnnxPath = Path.Combine(repoRoot, "runs", DefaultExpressionRun, "eye_multitask.onnx");
            options.ExpressionMetadataPath = Path.Combine(repoRoot, "runs", DefaultExpressionRun, "eye_multitask.metadata.json");
        }
        else
        {
            options.ExpressionOnnxPath = null;
            options.ExpressionMetadataPath = null;
        }
    }

    private static string NormalizeMultitaskModelPreset(string? presetId)
    {
        if (string.Equals(presetId, SplitMultitaskModelPreset, StringComparison.OrdinalIgnoreCase)
            || string.Equals(presetId, SplitMultitaskRun, StringComparison.OrdinalIgnoreCase))
        {
            return SplitMultitaskModelPreset;
        }

        if (string.Equals(presetId, LegacyMultitaskModelPreset, StringComparison.OrdinalIgnoreCase)
            || string.Equals(presetId, LegacyMultitaskRun, StringComparison.OrdinalIgnoreCase))
        {
            return LegacyMultitaskModelPreset;
        }

        if (string.Equals(presetId, PreviousMultitaskModelPreset, StringComparison.OrdinalIgnoreCase)
            || string.Equals(presetId, PreviousMultitaskRun, StringComparison.OrdinalIgnoreCase))
        {
            return PreviousMultitaskModelPreset;
        }

        if (!string.IsNullOrWhiteSpace(presetId))
        {
            return presetId.Trim();
        }

        return DefaultMultitaskModelPreset;
    }

    private static string MultitaskRunDirectory(string presetId)
        => NormalizeMultitaskModelPreset(presetId) switch
        {
            SplitMultitaskModelPreset => SplitMultitaskRun,
            LegacyMultitaskModelPreset => LegacyMultitaskRun,
            PreviousMultitaskModelPreset => PreviousMultitaskRun,
            _ => PreferredMultitaskRun
        };

    private static string ModelLabel(string presetId)
    {
        var normalized = NormalizeMultitaskModelPreset(presetId);
        var registryModel = FindUsableMainModel(LoadModelRegistry(FindRepoRoot()), normalized);
        if (registryModel is not null)
        {
            return registryModel.DisplayName;
        }

        return normalized switch
        {
            _ => "No model package"
        };
    }

    private static ModelRegistry? LoadModelRegistry(string repoRoot) =>
        ModelRegistry.TryLoadUserOrRepo(repoRoot);

    private static ModelRegistryEntry? FindUsableMainModel(ModelRegistry? registry, string? preferredId = null)
    {
        if (registry is null)
        {
            return null;
        }

        if (!string.IsNullOrWhiteSpace(preferredId))
        {
            var preferred = registry.FindById(preferredId);
            if (preferred is not null &&
                preferred.Role.Equals("main", StringComparison.OrdinalIgnoreCase) &&
                IsCompleteModelPackage(preferred))
            {
                return preferred;
            }
        }

        return registry.Models
            .Where(model =>
                model.DeviceFamily.Equals("Dream Air", StringComparison.OrdinalIgnoreCase) &&
                model.Runtime.Equals("predict_live_multitask", StringComparison.OrdinalIgnoreCase) &&
                model.Role.Equals("main", StringComparison.OrdinalIgnoreCase) &&
                IsCompleteModelPackage(model))
            .OrderByDescending(model => model.Default)
            .ThenBy(model => model.DisplayName, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
    }

    private static bool IsCompleteModelPackage(ModelRegistryEntry model)
        => File.Exists(model.ResolvedOnnx) &&
           File.Exists(model.ResolvedMetadata) &&
           File.Exists(model.ResolvedRuntimeDefaults) &&
           File.Exists(model.ResolvedAcceptance);

    private static double Clamp(double value, double min, double max, double fallback)
    {
        if (!IsFinite(value))
        {
            return fallback;
        }

        return Math.Max(min, Math.Min(max, value));
    }

    private static bool IsNonDefaultMultitaskPath(string? path)
        => !string.IsNullOrWhiteSpace(path)
            && (path.Contains("eye_multitask_mobilenetv3_v2", StringComparison.OrdinalIgnoreCase)
                || path.Contains("eye_multitask_siamese_per_eye_wide_squint_round3_20260617", StringComparison.OrdinalIgnoreCase)
                || path.Contains("eye_multitask_siamese_per_eye_openness_tune_round2_20260617", StringComparison.OrdinalIgnoreCase)
                || path.Contains("eye_multitask_siamese_expression_balanced_round4", StringComparison.OrdinalIgnoreCase)
                || path.Contains("eye_multitask_siamese_round5_eye_control_guard_20260617", StringComparison.OrdinalIgnoreCase)
                || path.Contains("eye_multitask_siamese_round6_gaze_preserve_20260618", StringComparison.OrdinalIgnoreCase)
                || path.Contains(SplitMultitaskRun, StringComparison.OrdinalIgnoreCase)
                || path.Contains(LegacyMultitaskRun, StringComparison.OrdinalIgnoreCase));

    private static bool IsNear(double value, double expected) =>
        IsFinite(value) && Math.Abs(value - expected) < 0.000001;
}

public sealed record BridgeModelChoice(string Id, string DisplayName, bool FromRegistry);
