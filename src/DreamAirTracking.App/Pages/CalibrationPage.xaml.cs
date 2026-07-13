using DreamAirTracking.App.Services;
using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Calibration;
using DreamAirTracking.Core.Profiles;
using Microsoft.UI;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using IOPath = System.IO.Path;

namespace DreamAirTracking.App.Pages;

public sealed partial class CalibrationPage : Page
{
    private const int MonitorUdpPort = 9401;
    private const int BrokenEyePort = 5555;
    private const int StageMaxPairs = 30;
    private const int StageMaxAutoRetries = 2;
    private const int RuntimeMinSamplesPerStage = 8;
    private static readonly TimeSpan StageCaptureDuration = TimeSpan.FromSeconds(1.5);
    private static readonly TimeSpan StreamGracePeriod = TimeSpan.FromSeconds(2);
    private readonly CalibrationLiveGazeHighlighter _liveHighlighter = new();
    private readonly BrokenEyeCalibrationCaptureService _captureService = new();
    private readonly PipelineDiagnosticsService _diagnosticsService = new();
    private IReadOnlyList<CalibrationStage> _stages = CalibrationStageCatalog.CreateFivePointGazeStages();
    private readonly Dictionary<string, Button> _stageButtons = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _stageStatuses = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, int> _stageAutoRetries = new(StringComparer.OrdinalIgnoreCase);
    private int _stageIndex;
    private string? _liveStageId;
    private CalibrationTarget? _liveTarget;
    private bool _hasLiveData;
    private bool _livePaused;
    private bool _workflowStarted;
    private bool _isBusy;
    private string? _sessionDirectory;
    private TrackingProfile? _profile;
    private StageCaptureResult? _currentCapture;
    private readonly List<CalibrationFramePair> _pairs = new();
    private readonly List<CalibrationLabel> _labels = new();
    private readonly List<RuntimeGazeCalibrationSample> _acceptedRuntimeSamples = new();
    private readonly List<RuntimeGazeCalibrationSample> _pendingRuntimeSamples = new();
    private readonly HashSet<long> _pendingRuntimeSequences = new();
    private long _nextSequence = 1;
    private string? _runtimeCaptureStageId;
    private CancellationTokenSource? _workflowCts;
    private CancellationTokenSource? _diagnosticsCts;
    private BridgeTrackingState? _latestBridgeState;
    private DateTimeOffset? _latestBridgeStateAt;
    private PipelineDiagnosticsSnapshot? _latestDiagnosticsSnapshot;
    private IReadOnlyList<CalibrationWeakStageSummary> _weakStageGuidance = Array.Empty<CalibrationWeakStageSummary>();
    private bool _loadingBridgeOptions;
    private string _calibrationSection = "eye";

    public CalibrationPage()
    {
        _loadingBridgeOptions = true;
        InitializeComponent();
        PupilAssistToggle.OnContent = "On";
        PupilAssistToggle.OffContent = "Off";
        PupilDiameterToggle.OnContent = "On";
        PupilDiameterToggle.OffContent = "Off";
        VrcftEyeExpressionToggle.OnContent = "On";
        VrcftEyeExpressionToggle.OffContent = "Off";
        NinePointToggle.OnContent = "9 points";
        NinePointToggle.OffContent = "5 points";
        _loadingBridgeOptions = false;
    }

    private void Page_Loaded(object sender, RoutedEventArgs e)
    {
        if (App.CurrentWindow is not null)
        {
            App.CurrentWindow.CalibrationOverlayCancelRequested -= MainWindow_CalibrationOverlayCancelRequested;
            App.CurrentWindow.CalibrationOverlayCancelRequested += MainWindow_CalibrationOverlayCancelRequested;
        }

        EnsureStageButtons();
        SyncStageListWithToggle(resetProgress: false);
        UpdateCalibrationSection();
        LoadBridgeOptions();
        BridgeProcessService.Instance.StateChanged -= Bridge_StateChanged;
        BridgeProcessService.Instance.StateChanged += Bridge_StateChanged;
        BridgeMonitorService.Instance.StateReceived -= BridgeMonitor_StateReceived;
        BridgeMonitorService.Instance.StateReceived += BridgeMonitor_StateReceived;
        BridgeMonitorService.Instance.Start();
        if (BridgeMonitorService.Instance.LatestState is not null)
        {
            UpdateLiveGaze(BridgeMonitorService.Instance.LatestState);
        }

        _ = LoadWeakStageGuidanceAsync();
        UpdateStage();
        SetActionButtonsEnabled(true);
        StartPipelineDiagnostics();
    }

    private void Page_Unloaded(object sender, RoutedEventArgs e)
    {
        if (App.CurrentWindow is not null)
        {
            App.CurrentWindow.CalibrationOverlayCancelRequested -= MainWindow_CalibrationOverlayCancelRequested;
            App.CurrentWindow.HideCalibrationOverlay();
        }

        StopPipelineDiagnostics();
        BridgeMonitorService.Instance.StateReceived -= BridgeMonitor_StateReceived;
        BridgeProcessService.Instance.StateChanged -= Bridge_StateChanged;
    }

    private void StageButton_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: string stageId })
        {
            var index = _stages.ToList().FindIndex(stage => stage.StageId.Equals(stageId, StringComparison.OrdinalIgnoreCase));
            if (index >= 0)
            {
                _stageIndex = index;
                UpdateStage();
            }
        }
    }

    private void Accept_Click(object sender, RoutedEventArgs e)
    {
        _ = AcceptCurrentStageAsync();
    }

    private void Retry_Click(object sender, RoutedEventArgs e)
    {
        if (!_workflowStarted)
        {
            ReviewInfoBar.Severity = InfoBarSeverity.Informational;
            ReviewInfoBar.Title = "Ready";
            ReviewInfoBar.Message = "Start calibration before retrying a captured point.";
            return;
        }

        _ = RunCurrentStageAsync();
    }

    private void MarkBad_Click(object sender, RoutedEventArgs e)
    {
        _ = MarkCurrentStageBadAsync();
    }

    private void StartCalibration_Click(object sender, RoutedEventArgs e)
    {
        _ = StartCurrentModuleCalibrationAsync();
    }

    private async void ExportTrainingPackage_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(_sessionDirectory) || !Directory.Exists(_sessionDirectory))
        {
            ReviewInfoBar.Severity = InfoBarSeverity.Warning;
            ReviewInfoBar.Title = "Nothing to export";
            ReviewInfoBar.Message = "Finish a gaze calibration first, then export the training package.";
            return;
        }

        try
        {
            ExportTrainingPackageButton.IsEnabled = false;
            var outputDirectory = IOPath.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "DreamAirTracking",
                "capture_packages");
            var result = await new CalibrationCapturePackageExporter().ExportAsync(new CalibrationCapturePackageExportOptions
            {
                SessionDirectory = _sessionDirectory,
                OutputDirectory = outputDirectory,
                RuntimeModelId = BridgeProcessService.Instance.CurrentMultitaskModelPreset
            });

            ReviewInfoBar.Severity = InfoBarSeverity.Success;
            ReviewInfoBar.Title = "Training package exported";
            ReviewInfoBar.Message = result.OutputZipPath;
            AutoMapText.Text = $"Upload this zip yourself and share the link in a GitHub issue: {result.OutputZipPath}";
        }
        catch (Exception ex)
        {
            ReviewInfoBar.Severity = InfoBarSeverity.Error;
            ReviewInfoBar.Title = "Export failed";
            ReviewInfoBar.Message = ex.Message;
        }
        finally
        {
            ExportTrainingPackageButton.IsEnabled = true;
        }
    }

    private void CalibrationMenu_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: string section })
        {
            SetCalibrationSection(section);
        }
    }

    private void BridgeOption_Toggled(object sender, RoutedEventArgs e)
    {
        SaveBridgeOptions();
    }

    private void PupilOutputModeBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        SyncPupilTogglesFromOutputMode();
        SaveBridgeOptions();
    }

    private void NinePointToggle_Toggled(object sender, RoutedEventArgs e)
    {
        SyncStageListWithToggle(resetProgress: true);
    }

    private Task StartCurrentModuleCalibrationAsync()
    {
        return _calibrationSection switch
        {
            "eyelid" => StartEyelidCalibrationAsync(),
            "pupil" => StartPupilCalibrationAsync(),
            _ => StartCalibrationAsync()
        };
    }

    private Task StartEyelidCalibrationAsync()
    {
        StartOpennessProcess(
            outputSubdirectory: IOPath.Combine("runs", "fullparam_capture", $"eyelid_app_{DateTime.Now:yyyyMMdd_HHmmss}"),
            extraArguments: "--preset training --settle-seconds 0.8 --stage-seconds 2.0 --save-training-images --validation-every 999 --beep --model-openness",
            title: "Eyelid calibration",
            message: "Follow the prompts. The result becomes the current eyelid calibration and exports a training package automatically.");
        return Task.CompletedTask;
    }

    private async Task StartPupilCalibrationAsync()
    {
        try
        {
            SaveBridgeOptions();
            if (!BridgeProcessService.Instance.IsRunning)
            {
                await BridgeProcessService.Instance.StartAsync();
            }

            ReviewInfoBar.Severity = InfoBarSeverity.Informational;
            ReviewInfoBar.Title = "Pupil calibration";
            ReviewInfoBar.Message = "Pupil uses the selected model package and current runtime gates. Live quality appears here when monitor packets arrive.";
            AlgorithmText.Text = "Pupil: model output + runtime quality gate";
            AutoMapText.Text = "Training rows from pupil capture must use pupil_valid only when openness and confidence gates pass.";
        }
        catch (Exception ex)
        {
            ReviewInfoBar.Severity = InfoBarSeverity.Error;
            ReviewInfoBar.Title = "Pupil calibration failed";
            ReviewInfoBar.Message = ex.Message;
        }
    }

    private void PupilCalibrationPathBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        SaveBridgeOptions();
    }

    private void QuickGazeCalibrationPathBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        SaveBridgeOptions();
    }

    private void OpennessCalibrationPathBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        SaveBridgeOptions();
    }

    private void UseLatestPupilCalibration_Click(object sender, RoutedEventArgs e)
    {
        var latest = BridgeProcessService.Instance.FindLatestPupilCalibrationPath();
        if (!string.IsNullOrWhiteSpace(latest))
        {
            PupilCalibrationPathBox.Text = latest;
            SaveBridgeOptions();
        }

        UpdateBridgeOptionsStatus();
    }

    private void UseLatestQuickGazeCalibration_Click(object sender, RoutedEventArgs e)
    {
        var latest = BridgeProcessService.Instance.FindLatestQuickGazeCalibrationPath();
        if (!string.IsNullOrWhiteSpace(latest))
        {
            QuickGazeCalibrationPathBox.Text = latest;
            SaveBridgeOptions();
        }

        UpdateBridgeOptionsStatus();
    }

    private void UseLatestOpennessCalibration_Click(object sender, RoutedEventArgs e)
    {
        var latest = BridgeProcessService.Instance.FindLatestOpennessCalibrationPath();
        if (!string.IsNullOrWhiteSpace(latest))
        {
            OpennessCalibrationPathBox.Text = latest;
            SaveBridgeOptions();
        }

        UpdateBridgeOptionsStatus();
    }

    private void StartOpennessCalibration_Click(object sender, RoutedEventArgs e)
    {
        StartOpennessProcess(
            outputSubdirectory: IOPath.Combine("runs", "live_openness_calibration_app"),
            extraArguments: "--settle-seconds 1.5 --stage-seconds 4 --beep --model-openness",
            title: "Eyelid calibration",
            message: "A terminal was opened. Follow the beep prompts, then use latest eyelid calibration.");
    }

    private void StartOpennessTrainingCapture_Click(object sender, RoutedEventArgs e)
    {
        StartOpennessProcess(
            outputSubdirectory: IOPath.Combine("runs", "fullparam_capture", $"eyelid_app_{DateTime.Now:yyyyMMdd_HHmmss}"),
            extraArguments: "--preset training --settle-seconds 0.8 --stage-seconds 2.0 --save-training-images --validation-every 999 --beep --model-openness",
            title: "Eyelid training capture",
            message: "A terminal was opened. Follow the beep prompts for open, wide, half, squint, closed, and open again.");
    }

    private void StartOpennessProcess(string outputSubdirectory, string extraArguments, string title, string message)
    {
        SaveBridgeOptions();
        var repoRoot = FindRepoRoot();
        var scriptPath = IOPath.Combine(repoRoot, "scripts", "ml", "live_openness_calibration.py");
        if (!File.Exists(scriptPath))
        {
            BridgeOptionsText.Text = $"Eyelid calibration script not found: {scriptPath}";
            return;
        }

        var outputDir = IOPath.Combine(repoRoot, outputSubdirectory);
        Directory.CreateDirectory(outputDir);
        var command = $"python {QuoteCmd(scriptPath)} --output-dir {QuoteCmd(outputDir)} {extraArguments}";
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = $"/k {command}",
                WorkingDirectory = repoRoot,
                UseShellExecute = true,
                WindowStyle = ProcessWindowStyle.Normal
            });
            ReviewInfoBar.Severity = InfoBarSeverity.Informational;
            ReviewInfoBar.Title = title;
            ReviewInfoBar.Message = message;
            BridgeOptionsText.Text = $"{title} output: {outputDir}";
        }
        catch (Exception ex)
        {
            BridgeOptionsText.Text = $"Could not start {title.ToLowerInvariant()}: {ex.Message}";
        }
    }

    private async void StartBridgeFromCalibration_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            SaveBridgeOptions();
            await BridgeProcessService.Instance.StartAsync();
        }
        catch (Exception ex)
        {
            BridgeOptionsText.Text = $"Eye tracking runtime failed to start: {ex.Message}";
        }

        UpdateBridgeOptionsStatus();
    }

    private void StopBridgeFromCalibration_Click(object sender, RoutedEventArgs e)
    {
        BridgeProcessService.Instance.Stop();
        UpdateBridgeOptionsStatus();
    }

    private void Bridge_StateChanged(object? sender, EventArgs e)
    {
        DispatcherQueue.TryEnqueue(UpdateBridgeOptionsStatus);
    }

    private void BridgeMonitor_StateReceived(object? sender, BridgeTrackingState state)
    {
        DispatcherQueue.TryEnqueue(() => UpdateLiveGaze(state));
    }

    private void CancelWorkflow_Click(object sender, RoutedEventArgs e)
    {
        CancelWorkflow();
    }

    private void MainWindow_CalibrationOverlayCancelRequested(object? sender, EventArgs e)
    {
        CancelWorkflow();
    }

    private void CancelWorkflow()
    {
        _workflowCts?.Cancel();
        _workflowStarted = false;
        HideOverlay();
        SetActionButtonsEnabled(true);
        _isBusy = false;
    }

    private void SyncStageListWithToggle(bool resetProgress)
    {
        if (_workflowStarted)
        {
            NinePointToggle.IsOn = _stages.Count == 9;
            return;
        }

        _stages = NinePointToggle.IsOn
            ? CalibrationStageCatalog.CreateNinePointGazeStages()
            : CalibrationStageCatalog.CreateFivePointGazeStages();
        if (resetProgress)
        {
            _stageIndex = 0;
            _stageStatuses.Clear();
            _stageAutoRetries.Clear();
            _currentCapture = null;
            _pairs.Clear();
            _labels.Clear();
            _acceptedRuntimeSamples.Clear();
            _pendingRuntimeSamples.Clear();
            _pendingRuntimeSequences.Clear();
            ReviewInfoBar.Severity = InfoBarSeverity.Informational;
            ReviewInfoBar.Title = "Ready";
            ReviewInfoBar.Message = $"Capture integration is ready. Start calibration to run the {_stages.Count}-point flow.";
        }

        EnsureStageButtons();
        UpdateStageButtonVisibility();
        UpdateStage();
        SetActionButtonsEnabled(true);
    }

    private void AdvanceStage()
    {
        _stageIndex = Math.Min(_stageIndex + 1, _stages.Count - 1);
        UpdateStage();
    }

    private void MarkCurrentStage(string status)
    {
        var stage = CurrentStage;
        _stageStatuses[stage.StageId] = status;
        UpdateButtonVisualStates();
    }

    private void UpdateStage()
    {
        AlgorithmText.Text = _hasLiveData && _liveStageId is not null
            ? $"Live gaze: {FormatStageName(_liveStageId)}"
            : $"Live gaze: waiting for runtime monitor UDP {MonitorUdpPort}";
        UpdateCurrentStageGuidance();
        if (!_hasLiveData)
        {
            ReviewInfoBar.Severity = InfoBarSeverity.Informational;
            ReviewInfoBar.Title = "Waiting";
            ReviewInfoBar.Message = $"Start eye tracking to receive live gaze on 127.0.0.1:{MonitorUdpPort}.";
        }

        UpdateButtonVisualStates();

    }

    private void UpdateCurrentStageGuidance()
    {
        if (_isBusy)
        {
            return;
        }

        AutoMapText.Text = CalibrationStageGuidanceAdvisor.BuildStageGuidance(
            CurrentStage.StageId,
            _weakStageGuidance);
    }

    private async Task LoadWeakStageGuidanceAsync()
    {
        try
        {
            var summary = await CalibrationSessionAuditor.AuditRootAsync();
            _weakStageGuidance = summary.WeakStages;
            DispatcherQueue.TryEnqueue(UpdateCurrentStageGuidance);
        }
        catch
        {
            _weakStageGuidance = Array.Empty<CalibrationWeakStageSummary>();
        }
    }

    private CalibrationStage CurrentStage => _stages[_stageIndex];

    private async Task StartCalibrationAsync()
    {
        if (_isBusy)
        {
            return;
        }

        SetCalibrationSection("eye");
        ExportTrainingPackageButton.Visibility = Visibility.Collapsed;

        if (!BridgeProcessService.Instance.IsRunning)
        {
            await BridgeProcessService.Instance.StartAsync();
        }

        var profilePath = FindProfilePath() ?? "tracking_profile_final.json";
        _profile = await TrackingProfileStore.LoadOrDefaultAsync(profilePath);

        var preflightFailure = await GetCalibrationPreflightFailureAsync(_profile);
        if (preflightFailure is not null)
        {
            ReviewInfoBar.Severity = InfoBarSeverity.Error;
            ReviewInfoBar.Title = "Preflight failed";
            ReviewInfoBar.Message = preflightFailure;
            AutoMapText.Text = "No calibration session was created. Fix the preflight issue, then press Start calibration again.";
            return;
        }

        if (!GazeWatcherProcessService.Instance.IsRunning)
        {
            GazeWatcherProcessService.Instance.Start(runImmediately: false);
        }

        _workflowCts?.Cancel();
        _workflowCts?.Dispose();
        _workflowCts = new CancellationTokenSource();
        _workflowStarted = true;
        _stageIndex = 0;
        _nextSequence = 1;
        _pairs.Clear();
        _labels.Clear();
        _acceptedRuntimeSamples.Clear();
        _pendingRuntimeSamples.Clear();
        _pendingRuntimeSequences.Clear();
        _runtimeCaptureStageId = null;
        _stageStatuses.Clear();
        _stageAutoRetries.Clear();
        _currentCapture = null;

        _sessionDirectory = CreateSessionDirectory();
        var session = new CalibrationSession
        {
            SessionId = IOPath.GetFileName(_sessionDirectory),
            CreatedAt = DateTimeOffset.Now,
            ProfileInput = profilePath,
            Device = new CalibrationDeviceInfo
            {
                Source = "BrokenEye HTTP",
                Host = "127.0.0.1",
                Port = BrokenEyePort
            },
            Stages = _stages.Select(stage => new CalibrationStage
            {
                Order = stage.Order,
                StageId = stage.StageId,
                Target = stage.Target,
                CaptureSeconds = StageCaptureDuration.TotalSeconds,
                DisplayName = stage.DisplayName
            }).ToList()
        };
        await CalibrationSessionStore.SaveSessionAsync(session, _sessionDirectory);

        SessionPathText.Text = $"Session: {_sessionDirectory}";
        ReviewInfoBar.Severity = InfoBarSeverity.Informational;
        ReviewInfoBar.Title = "Calibration";
        ReviewInfoBar.Message = $"Starting automatic {_stages.Count}-point capture. Watcher is waiting for this session.";
        UpdateStage();
        await RunCurrentStageAsync();
    }

    private async Task<string?> GetCalibrationPreflightFailureAsync(TrackingProfile profile)
    {
        PipelineDiagnosticsSnapshot? snapshot;
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(2.5));
            snapshot = await _diagnosticsService.CaptureAsync(
                BridgeMonitorService.Instance.LatestState ?? _latestBridgeState,
                BridgeMonitorService.Instance.LatestStateAt ?? _latestBridgeStateAt,
                timeout.Token);
            UpdatePipelineDiagnostics(snapshot);
        }
        catch
        {
            snapshot = _latestDiagnosticsSnapshot;
        }

        if (snapshot is null)
        {
            return $"Could not verify BrokenEye streams on 127.0.0.1:{BrokenEyePort}.";
        }

        if (!snapshot.BrokenEyeLeftLive || !snapshot.BrokenEyeRightLive)
        {
            return $"BrokenEye streams are not ready: left={FormatBool(snapshot.BrokenEyeLeftLive)} right={FormatBool(snapshot.BrokenEyeRightLive)}.";
        }

        var tempDirectory = IOPath.Combine(IOPath.GetTempPath(), $"dream-air-preflight-{Guid.NewGuid():N}");
        try
        {
            var center = _stages.First(stage => stage.StageId.Equals("center", StringComparison.OrdinalIgnoreCase));
            var options = new StageCaptureOptions(
                "127.0.0.1",
                BrokenEyePort,
                TimeSpan.FromSeconds(0.8),
                TimeSpan.FromSeconds(1),
                10,
                20,
                0.05,
                0.08);
            var result = await _captureService.CaptureStageAsync(
                center,
                profile,
                tempDirectory,
                1,
                options);

            if (result.QuickCheck.Quality.Equals("poor", StringComparison.OrdinalIgnoreCase)
                && !HasRecentRuntimeEyeState())
            {
                return $"Preflight eye sample is poor ({result.QuickCheck.ValidPairCount}/{result.QuickCheck.PairCount} usable). {result.QuickCheck.ReviewReason} {result.QuickCheck.ImprovementPlan}";
            }

            return null;
        }
        catch (Exception ex)
        {
            return $"Could not capture preflight eye sample: {ex.Message}";
        }
        finally
        {
            try
            {
                if (Directory.Exists(tempDirectory))
                {
                    Directory.Delete(tempDirectory, recursive: true);
                }
            }
            catch
            {
            }
        }
    }

    private async Task RunCurrentStageAsync()
    {
        if (_isBusy || !_workflowStarted || _workflowCts is null)
        {
            return;
        }

        _isBusy = true;
        SetActionButtonsEnabled(false);
        _currentCapture = null;
        UpdateButtonVisualStates();
        var token = _workflowCts.Token;
        var stage = CurrentStage;
        var shouldAdvanceAutomatically = false;
        var shouldRetryAutomatically = false;

        try
        {
            ShowOverlay(stage, "Look at this point.", "3", "Get ready", 0, showProgress: false);
            for (var count = 3; count >= 1; count--)
            {
                ShowOverlay(stage, "Look at this point.", count.ToString(), "Hold your gaze steady", 0, showProgress: false);
                await Task.Delay(TimeSpan.FromSeconds(1), token);
            }

            ShowOverlay(stage, "Capturing synchronized eye frames.", string.Empty, "Recording", 0, showProgress: true);
            var options = new StageCaptureOptions(
                "127.0.0.1",
                BrokenEyePort,
                StageCaptureDuration,
                StreamGracePeriod,
                StageMaxPairs,
                20,
                0.05,
                0.08);
            var progress = new Progress<StageCaptureProgress>(UpdateCaptureProgress);
            _pendingRuntimeSamples.Clear();
            _pendingRuntimeSequences.Clear();
            _runtimeCaptureStageId = stage.StageId;
            var result = await _captureService.CaptureStageAsync(
                stage,
                _profile ?? new TrackingProfile(),
                _sessionDirectory ?? CreateSessionDirectory(),
                _nextSequence,
                options,
                progress,
                token);

            _currentCapture = result;
            _pairs.AddRange(result.Pairs);
            _nextSequence += result.Pairs.Count;
            if (_sessionDirectory is not null)
            {
                await CalibrationSessionStore.SavePairsAsync(_pairs, _sessionDirectory, token);
            }

            HideOverlay();
            ShowQuickCheck(result);
            if (ShouldAutoAccept(result))
            {
                _stageAutoRetries.Remove(result.StageId);
                MarkCurrentStage("Accepted");
                AddLabel(accepted: true, verdict: AutoAcceptVerdict(result));
                await SaveLabelsAsync();
                ReviewInfoBar.Severity = result.QuickCheck.Quality == "good"
                    ? InfoBarSeverity.Success
                    : InfoBarSeverity.Warning;
                ReviewInfoBar.Title = "Saved";
                ReviewInfoBar.Message = $"{FormatStageName(result.StageId)} saved automatically with {_pendingRuntimeSamples.Count} runtime samples. Moving to the next point.";
                shouldAdvanceAutomatically = true;
            }
            else if (ShouldAutoRetry(result))
            {
                var retryCount = IncrementAutoRetry(result.StageId);
                MarkCurrentStage("Retrying");
                ReviewInfoBar.Severity = InfoBarSeverity.Warning;
                ReviewInfoBar.Title = "Retrying";
                ReviewInfoBar.Message = $"{FormatStageName(result.StageId)} was unstable ({result.QuickCheck.ValidPairCount}/{result.QuickCheck.PairCount} usable). {result.QuickCheck.ReviewReason} Automatic retry {retryCount}/{StageMaxAutoRetries}.";
                AutoMapText.Text = result.QuickCheck.ImprovementPlan;
                shouldRetryAutomatically = true;
            }
            else
            {
                _stageAutoRetries.Remove(result.StageId);
                MarkCurrentStage("Review");
                ReviewInfoBar.Severity = InfoBarSeverity.Warning;
                ReviewInfoBar.Title = "Manual review required";
                ReviewInfoBar.Message = $"{FormatStageName(result.StageId)} stayed unstable after retries. {result.QuickCheck.ReviewReason}";
                AutoMapText.Text = $"{result.QuickCheck.ImprovementPlan} The workflow will keep stable stages and ask for a fresh calibration when quality is too low.";
            }
        }
        catch (OperationCanceledException)
        {
            HideOverlay();
            ReviewInfoBar.Severity = InfoBarSeverity.Warning;
            ReviewInfoBar.Title = "Cancelled";
            ReviewInfoBar.Message = "Calibration capture was cancelled.";
        }
        catch (Exception ex)
        {
            HideOverlay();
            ReviewInfoBar.Severity = InfoBarSeverity.Error;
            ReviewInfoBar.Title = "Capture failed";
            ReviewInfoBar.Message = $"Could not capture synchronized eye frames from BrokenEye HTTP on 127.0.0.1:{BrokenEyePort}.";
            AlgorithmText.Text = "Capture failed before this point could be reviewed.";
            AutoMapText.Text = $"Check BrokenEye /eye/left and /eye/right streams, then retry. Detail: {ex.Message}";
        }
        finally
        {
            _runtimeCaptureStageId = null;
            _isBusy = false;
            SetActionButtonsEnabled(true);
        }

        if (shouldAdvanceAutomatically && _workflowStarted && _workflowCts is not null && !_workflowCts.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(TimeSpan.FromMilliseconds(700), _workflowCts.Token);
                await AdvanceOrCompleteAsync();
            }
            catch (OperationCanceledException)
            {
            }
        }
        else if (shouldRetryAutomatically && _workflowStarted && _workflowCts is not null && !_workflowCts.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(TimeSpan.FromMilliseconds(700), _workflowCts.Token);
                await RunCurrentStageAsync();
            }
            catch (OperationCanceledException)
            {
            }
        }
    }

    private async Task AcceptCurrentStageAsync()
    {
        if (_isBusy)
        {
            return;
        }

        MarkCurrentStage("Accepted");
        if (_workflowStarted && _currentCapture is not null)
        {
            AddLabel(accepted: true, verdict: "good");
            await SaveLabelsAsync();
        }

        await AdvanceOrCompleteAsync();
    }

    private async Task MarkCurrentStageBadAsync()
    {
        if (_isBusy)
        {
            return;
        }

        MarkCurrentStage("Bad");
        if (_workflowStarted && _currentCapture is not null)
        {
            AddLabel(accepted: false, verdict: "bad");
            await SaveLabelsAsync();
        }

        await AdvanceOrCompleteAsync();
    }

    private async Task AdvanceOrCompleteAsync()
    {
        if (_stageIndex >= _stages.Count - 1)
        {
            await CompleteCalibrationAsync();
            return;
        }

        AdvanceStage();
        if (_workflowStarted)
        {
            await RunCurrentStageAsync();
        }
    }

    private async Task CompleteCalibrationAsync()
    {
        _workflowStarted = false;
        _currentCapture = null;
        HideOverlay();

        if (_sessionDirectory is null)
        {
            return;
        }

        await SaveLabelsAsync();
        await CalibrationSessionStore.SavePairsAsync(_pairs, _sessionDirectory);
        var legacyFit = new GazeCalibrationFitter().Fit(_pairs, _labels);
        await CalibrationSessionStore.SaveMetricsAsync(legacyFit.Metrics, _sessionDirectory);
        var outputProfilePath = IOPath.Combine(_sessionDirectory, "generated_profile.json");
        var normalizedProfilePath = IOPath.Combine(_sessionDirectory, "mask_normalized_profile.json");
        if (legacyFit.Succeeded)
        {
            var generated = new GazeCalibrationProfileBuilder().BuildProfile(_profile ?? new TrackingProfile(), legacyFit);
            await TrackingProfileStore.SaveAsync(generated, outputProfilePath);
        }

        var normalized = new RawCoordinateNormalizationProfileBuilder().BuildProfile(
            _profile ?? new TrackingProfile(),
            _pairs,
            _labels);
        if (normalized.Succeeded)
        {
            await TrackingProfileStore.SaveAsync(normalized.Profile, normalizedProfilePath);
        }

        var runtimeFit = new RuntimeGazeCalibrationFitter().Fit(_acceptedRuntimeSamples);
        var runtimeCalibrationPath = IOPath.Combine(_sessionDirectory, "runtime_gaze_calibration.json");
        await SaveRuntimeCalibrationAsync(runtimeCalibrationPath, runtimeFit);
        var runtimeWasRunning = BridgeProcessService.Instance.IsRunning;
        var runtimeRestarted = false;
        var wearTemplateBound = false;
        if (runtimeFit.Succeeded)
        {
            BridgeProcessService.Instance.ApplyRuntimeCalibration(
                runtimeFit.CenterOffsetX,
                runtimeFit.CenterOffsetY,
                runtimeFit.XGain,
                runtimeFit.YGain,
                runtimeCalibrationPath,
                IOPath.GetFileName(_sessionDirectory));
            SyncQuickGazeCalibrationFields(
                runtimeCalibrationPath,
                IOPath.GetFileName(_sessionDirectory));
            wearTemplateBound = await BridgeProcessService.Instance.BindRuntimeCalibrationToCurrentWearTemplateAsync(
                runtimeCalibrationPath,
                runtimeFit.CenterOffsetX,
                runtimeFit.CenterOffsetY,
                runtimeFit.XGain,
                runtimeFit.YGain,
                runtimeFit.SampleCount);
            if (runtimeWasRunning)
            {
                await BridgeProcessService.Instance.StartAsync(restartIfRunning: true);
                runtimeRestarted = BridgeProcessService.Instance.IsRunning;
            }
        }

        var reportPath = IOPath.Combine(_sessionDirectory, "report.md");
        await File.WriteAllTextAsync(
            reportPath,
            BuildCalibrationReport(legacyFit, runtimeFit, outputProfilePath, normalizedProfilePath, runtimeCalibrationPath),
            Encoding.UTF8);

        ReviewInfoBar.Severity = runtimeFit.Succeeded
            ? InfoBarSeverity.Success
            : InfoBarSeverity.Warning;
        ReviewInfoBar.Title = "Calibration complete";
        ReviewInfoBar.Message = runtimeFit.Succeeded
            ? $"Runtime offset applied: x={runtimeFit.CenterOffsetX:0.000}, y={runtimeFit.CenterOffsetY:0.000}. Wear template bound: {wearTemplateBound}."
            : $"Captured data, but runtime calibration was not applied. {runtimeFit.Reason}";
        AlgorithmText.Text = $"Session complete: {_labels.Count(label => label.Accepted)} accepted stages.";
        AutoMapText.Text = runtimeFit.Succeeded
            ? runtimeRestarted
                ? $"Runtime calibration: {runtimeCalibrationPath}. Eye tracking restarted with center-stage offset."
                : $"Runtime calibration: {runtimeCalibrationPath}. Start eye tracking when ready."
            : $"Review output: {reportPath}. Missing runtime samples: {string.Join(", ", runtimeFit.MissingStages)}";
        ExportTrainingPackageButton.Visibility = Visibility.Visible;
        SetActionButtonsEnabled(true);
        UpdateBridgeOptionsStatus();
    }

    private string BuildCalibrationReport(
        GazeCalibrationFitResult legacyFit,
        RuntimeGazeCalibrationFit runtimeFit,
        string generatedProfilePath,
        string normalizedProfilePath,
        string runtimeCalibrationPath)
    {
        var builder = new StringBuilder();
        builder.AppendLine("# Calibration Review Report");
        builder.AppendLine();
        builder.AppendLine($"- Session: {_sessionDirectory}");
        builder.AppendLine($"- Accepted stages: {_labels.Count(label => label.Accepted)} / {_stages.Count}");
        builder.AppendLine($"- Runtime calibration applied: {runtimeFit.Succeeded}");
        builder.AppendLine($"- Runtime reason: {runtimeFit.Reason}");
        builder.AppendLine($"- Runtime samples: {runtimeFit.SampleCount}");
        builder.AppendLine($"- Runtime offset: x={runtimeFit.CenterOffsetX:0.000000}, y={runtimeFit.CenterOffsetY:0.000000}");
        builder.AppendLine($"- Runtime gain used: x={runtimeFit.XGain:0.000000}, y={runtimeFit.YGain:0.000000}");
        builder.AppendLine($"- Runtime fitted gain diagnostic: x={runtimeFit.FittedXGain:0.000000}, y={runtimeFit.FittedYGain:0.000000}");
        builder.AppendLine($"- Runtime calibration: {runtimeCalibrationPath}");
        builder.AppendLine($"- Legacy fit succeeded: {legacyFit.Succeeded}");
        builder.AppendLine($"- Legacy fit quality: {legacyFit.Metrics.Quality}");
        builder.AppendLine($"- Legacy fit model: {legacyFit.Metrics.Model}");
        builder.AppendLine($"- Legacy generated profile: {generatedProfilePath}");
        builder.AppendLine($"- Legacy normalized profile: {normalizedProfilePath}");
        builder.AppendLine();
        builder.AppendLine("## Runtime Stage Medians");
        builder.AppendLine();
        builder.AppendLine("| Stage | Count | Raw X | Raw Y |");
        builder.AppendLine("|---|---:|---:|---:|");
        foreach (var item in runtimeFit.StageMedians.OrderBy(item => StageOrder(item.Key)))
        {
            builder.AppendLine($"| {item.Key} | {item.Value.Count} | {item.Value.X:0.000000} | {item.Value.Y:0.000000} |");
        }

        builder.AppendLine();
        builder.AppendLine("## Stage Review");
        builder.AppendLine();
        builder.AppendLine("| Stage | Accepted | Verdict | Frames | Notes |");
        builder.AppendLine("|---|---:|---|---|---|");
        foreach (var label in _labels.OrderBy(label => StageOrder(label.StageId)))
        {
            var frames = $"{label.FrameStart}-{label.FrameEnd}";
            builder.AppendLine(
                $"| {label.StageId} | {label.Accepted} | {label.OperatorVerdict} | {frames} | {EscapeMarkdownTableCell(label.Notes)} |");
        }

        builder.AppendLine();
        builder.AppendLine("## Residuals");
        builder.AppendLine();
        builder.AppendLine("| Stage | Distance | X | Y |");
        builder.AppendLine("|---|---:|---:|---:|");
        foreach (var item in legacyFit.Metrics.StageResiduals.OrderBy(item => StageOrder(item.Key)))
        {
            builder.AppendLine(
                $"| {item.Key} | {item.Value.Distance:0.000} | {item.Value.X:0.000} | {item.Value.Y:0.000} |");
        }

        return builder.ToString();
    }

    private int StageOrder(string stageId)
    {
        for (var index = 0; index < _stages.Count; index++)
        {
            if (_stages[index].StageId.Equals(stageId, StringComparison.OrdinalIgnoreCase))
            {
                return index;
            }
        }

        return int.MaxValue;
    }

    private static string EscapeMarkdownTableCell(string value) =>
        value.Replace("|", "/", StringComparison.Ordinal).ReplaceLineEndings(" ");

    private void LoadBridgeOptions()
    {
        _loadingBridgeOptions = true;
        try
        {
            var options = BridgeProcessService.Instance.Options;
            PupilAssistToggle.IsOn = options.EnablePupilAssist;
            PupilDiameterToggle.IsOn = options.EnablePupilDiameter;
            VrcftEyeExpressionToggle.IsOn = options.EnableVrcftEyeExpressions;
            SetPupilOutputMode(options.EnablePupilAssist
                ? "eye_wide_assist"
                : options.EnablePupilDiameter
                    ? "diameter"
                    : "off");
            PupilCalibrationPathBox.Text = options.PupilCalibrationPath
                ?? BridgeProcessService.Instance.FindLatestPupilCalibrationPath()
                ?? string.Empty;
            QuickGazeCalibrationPathBox.Text = options.QuickGazeCalibrationPath ?? string.Empty;
            QuickGazeCalibrationSessionBox.Text = options.QuickGazeCalibrationSession ?? string.Empty;
            OpennessCalibrationPathBox.Text = options.OpennessCalibrationPath
                ?? BridgeProcessService.Instance.FindLatestOpennessCalibrationPath()
                ?? string.Empty;
        }
        finally
        {
            _loadingBridgeOptions = false;
        }

        SaveBridgeOptions();
        UpdateBridgeOptionsStatus();
    }

    private void SaveBridgeOptions()
    {
        if (_loadingBridgeOptions || !AreBridgeOptionControlsReady())
        {
            return;
        }

        var existing = BridgeProcessService.Instance.Options;
        var pupilOutputMode = GetPupilOutputMode();
        BridgeProcessService.Instance.UpdateOptions(new BridgeLaunchOptions
        {
            EnableEyeTracking = true,
            EnablePupilAssist = pupilOutputMode.Equals("eye_wide_assist", StringComparison.OrdinalIgnoreCase),
            EnablePupilDiameter = pupilOutputMode.Equals("diameter", StringComparison.OrdinalIgnoreCase),
            EnableVrcftEyeExpressions = VrcftEyeExpressionToggle.IsOn,
            AutoStartEyeTracking = existing.AutoStartEyeTracking,
            RuntimeModelType = existing.RuntimeModelType,
            EnableMultitaskVrcftOutput = existing.EnableMultitaskVrcftOutput,
            VrcftOutputXOffset = existing.VrcftOutputXOffset,
            VrcftOutputYOffset = existing.VrcftOutputYOffset,
            VrcftOutputXGain = existing.VrcftOutputXGain,
            VrcftOutputYGain = existing.VrcftOutputYGain,
            VrcftOutputXSign = existing.VrcftOutputXSign,
            VrcftOutputYSign = existing.VrcftOutputYSign,
            ProfilePath = existing.ProfilePath,
            OnnxPath = existing.OnnxPath,
            MetadataPath = existing.MetadataPath,
            MultitaskModelPreset = existing.MultitaskModelPreset,
            MultitaskOnnxPath = existing.MultitaskOnnxPath,
            MultitaskMetadataPath = existing.MultitaskMetadataPath,
            ExpressionOnnxPath = existing.ExpressionOnnxPath,
            ExpressionMetadataPath = existing.ExpressionMetadataPath,
            CenterOffsetX = existing.CenterOffsetX,
            CenterOffsetY = existing.CenterOffsetY,
            XGain = existing.XGain,
            YGain = existing.YGain,
            Clamp = existing.Clamp,
            ClampMode = existing.ClampMode,
            SoftKnee = existing.SoftKnee,
            CenterBias = existing.CenterBias,
            CenterBiasStart = existing.CenterBiasStart,
            OutputMapMode = existing.OutputMapMode,
            OutputDeadzone = existing.OutputDeadzone,
            OutputCurveGamma = existing.OutputCurveGamma,
            OutputCenterRadius = existing.OutputCenterRadius,
            OutputCenterTau = existing.OutputCenterTau,
            OutputCenterMaxStep = existing.OutputCenterMaxStep,
            EmaAlpha = existing.EmaAlpha,
            MaxStep = existing.MaxStep,
            OpennessMode = existing.OpennessMode,
            OpennessCurveMode = existing.OpennessCurveMode,
            OpennessCalibrationPath = string.IsNullOrWhiteSpace(OpennessCalibrationPathBox.Text)
                ? null
                : OpennessCalibrationPathBox.Text.Trim(),
            OpennessBoostGamma = existing.OpennessBoostGamma,
            OpennessFullOpenThreshold = existing.OpennessFullOpenThreshold,
            OpennessBoostKnee = existing.OpennessBoostKnee,
            EyeShapeWideScale = existing.EyeShapeWideScale,
            EyeShapeSquintScale = existing.EyeShapeSquintScale,
            EyeShapeGamma = existing.EyeShapeGamma,
            EyeShapeDeadzone = existing.EyeShapeDeadzone,
            PupilWideEnterThreshold = existing.PupilWideEnterThreshold,
            PupilWideExitThreshold = existing.PupilWideExitThreshold,
            PupilWideHoldFrames = existing.PupilWideHoldFrames,
            PupilWideEmaAlpha = existing.PupilWideEmaAlpha,
            NormalizationMode = existing.NormalizationMode,
            NormalizationTargetX = existing.NormalizationTargetX,
            NormalizationTargetY = existing.NormalizationTargetY,
            NormalizationLeftTargetX = existing.NormalizationLeftTargetX,
            NormalizationRightTargetX = existing.NormalizationRightTargetX,
            NormalizationLeftTargetY = existing.NormalizationLeftTargetY,
            NormalizationRightTargetY = existing.NormalizationRightTargetY,
            NormalizationMaxShiftX = existing.NormalizationMaxShiftX,
            NormalizationMaxShiftY = existing.NormalizationMaxShiftY,
            NormalizationScaleMin = existing.NormalizationScaleMin,
            NormalizationScaleMax = existing.NormalizationScaleMax,
            NormalizationHoldFrames = existing.NormalizationHoldFrames,
            PupilCalibrationPath = string.IsNullOrWhiteSpace(PupilCalibrationPathBox.Text)
                ? null
                : PupilCalibrationPathBox.Text.Trim(),
            QuickGazeCalibrationPath = string.IsNullOrWhiteSpace(QuickGazeCalibrationPathBox.Text)
                ? existing.QuickGazeCalibrationPath
                : QuickGazeCalibrationPathBox.Text.Trim(),
            QuickGazeCalibrationSession = string.IsNullOrWhiteSpace(QuickGazeCalibrationSessionBox.Text)
                ? existing.QuickGazeCalibrationSession
                : QuickGazeCalibrationSessionBox.Text.Trim()
        });
        UpdateBridgeOptionsStatus();
    }

    private void SyncQuickGazeCalibrationFields(string path, string? session)
    {
        if (!AreBridgeOptionControlsReady())
        {
            return;
        }

        _loadingBridgeOptions = true;
        try
        {
            QuickGazeCalibrationPathBox.Text = path;
            QuickGazeCalibrationSessionBox.Text = session ?? string.Empty;
        }
        finally
        {
            _loadingBridgeOptions = false;
        }
    }

    private bool AreBridgeOptionControlsReady() =>
        PupilAssistToggle is not null
        && PupilDiameterToggle is not null
        && VrcftEyeExpressionToggle is not null
        && PupilOutputModeBox is not null
        && PupilCalibrationPathBox is not null
        && QuickGazeCalibrationPathBox is not null
        && QuickGazeCalibrationSessionBox is not null
        && OpennessCalibrationPathBox is not null;

    private string GetPupilOutputMode()
    {
        if (PupilOutputModeBox?.SelectedItem is ComboBoxItem { Tag: string tag })
        {
            return tag;
        }

        return PupilAssistToggle?.IsOn == true
            ? "eye_wide_assist"
            : PupilDiameterToggle?.IsOn == true
                ? "diameter"
                : "off";
    }

    private void SetPupilOutputMode(string mode)
    {
        if (PupilOutputModeBox is null)
        {
            return;
        }

        var target = mode.Equals("eye_wide_assist", StringComparison.OrdinalIgnoreCase)
            ? "eye_wide_assist"
            : mode.Equals("diameter", StringComparison.OrdinalIgnoreCase)
                ? "diameter"
                : "off";
        for (var index = 0; index < PupilOutputModeBox.Items.Count; index++)
        {
            if (PupilOutputModeBox.Items[index] is ComboBoxItem { Tag: string tag }
                && tag.Equals(target, StringComparison.OrdinalIgnoreCase))
            {
                PupilOutputModeBox.SelectedIndex = index;
                break;
            }
        }

        SyncPupilTogglesFromOutputMode();
    }

    private void SyncPupilTogglesFromOutputMode()
    {
        if (PupilAssistToggle is null || PupilDiameterToggle is null)
        {
            return;
        }

        var mode = GetPupilOutputMode();
        PupilAssistToggle.IsOn = mode.Equals("eye_wide_assist", StringComparison.OrdinalIgnoreCase);
        PupilDiameterToggle.IsOn = mode.Equals("diameter", StringComparison.OrdinalIgnoreCase);
    }

    private void UpdateBridgeOptionsStatus()
    {
        var bridge = BridgeProcessService.Instance;
        StartBridgeFromCalibrationButton.IsEnabled = !bridge.IsRunning;
        StopBridgeFromCalibrationButton.IsEnabled = bridge.IsRunning;
        var pupilPath = string.IsNullOrWhiteSpace(PupilCalibrationPathBox.Text)
            ? "not selected"
            : IOPath.GetFileName(PupilCalibrationPathBox.Text);
        var quickPath = string.IsNullOrWhiteSpace(QuickGazeCalibrationPathBox.Text)
            ? "off"
            : IOPath.GetFileName(QuickGazeCalibrationPathBox.Text);
        var quickSession = string.IsNullOrWhiteSpace(QuickGazeCalibrationSessionBox.Text)
            ? "auto"
            : QuickGazeCalibrationSessionBox.Text.Trim();
        var opennessPath = string.IsNullOrWhiteSpace(OpennessCalibrationPathBox.Text)
            ? "auto"
            : IOPath.GetFileName(OpennessCalibrationPathBox.Text);
        var rawState = GetPupilOutputMode();
        var normalization = string.IsNullOrWhiteSpace(bridge.Options.NormalizationMode)
            ? "off"
            : bridge.Options.NormalizationMode;
        var applyHint = bridge.IsRunning ? " Changes apply after runtime restart." : string.Empty;
        var eyeShapeHint = bridge.Options.EnableVrcftEyeExpressions ? " Restart VRCFT after changing eye shapes." : string.Empty;
        BridgeOptionsText.Text = bridge.IsRunning
            ? $"Eye tracking runtime running. Face: SR, pupil: {rawState}, eye shapes: {FormatBool(bridge.Options.EnableVrcftEyeExpressions)}, normalization: {normalization}, calibration: {pupilPath}, runtime gaze: {quickPath} ({quickSession}), eyelid: {opennessPath}. Offset=({bridge.Options.CenterOffsetX:0.000},{bridge.Options.CenterOffsetY:0.000}).{applyHint}{eyeShapeHint}"
            : $"Eye tracking runtime stopped. Face: SR, pupil: {rawState}, eye shapes: {FormatBool(bridge.Options.EnableVrcftEyeExpressions)}, normalization: {normalization}, calibration: {pupilPath}, runtime gaze: {quickPath} ({quickSession}), eyelid: {opennessPath}. Offset=({bridge.Options.CenterOffsetX:0.000},{bridge.Options.CenterOffsetY:0.000}).{eyeShapeHint}";
    }

    private void AddLabel(bool accepted, string verdict)
    {
        if (_currentCapture is null || _currentCapture.Pairs.Count == 0)
        {
            return;
        }

        var stage = CurrentStage;
        _labels.Add(new CalibrationLabel
        {
            StageId = stage.StageId,
            Target = stage.Target,
            FrameStart = _currentCapture.Pairs.Min(pair => pair.Sequence),
            FrameEnd = _currentCapture.Pairs.Max(pair => pair.Sequence),
            Accepted = accepted,
            OperatorVerdict = verdict,
            Notes = FormatLabelNotes(_currentCapture.QuickCheck)
        });

        if (accepted)
        {
            _acceptedRuntimeSamples.AddRange(_pendingRuntimeSamples.Select(sample => sample with { Accepted = true }));
        }
    }

    private static string FormatLabelNotes(StageQuickCheck check) =>
        $"quality={check.Quality}; usable={check.ValidPairCount}/{check.PairCount}; " +
        $"low_openness={check.LowOpennessPairs}; low_confidence={check.LowConfidencePairs}; " +
        $"not_found={check.NotFoundPairs}; sync_late={check.SyncLatePairs}; " +
        $"reason={check.ReviewReason}; improvement={check.ImprovementPlan}";

    private async Task SaveLabelsAsync()
    {
        if (_sessionDirectory is not null)
        {
            await CalibrationSessionStore.SaveLabelsAsync(_labels, _sessionDirectory);
        }
    }

    private async Task SaveRuntimeCalibrationAsync(string path, RuntimeGazeCalibrationFit fit)
    {
        var payload = new
        {
            schema = "dreamair.runtime_gaze_calibration.v1",
            createdAt = DateTimeOffset.Now,
            session = _sessionDirectory,
            fit.Succeeded,
            fit.Reason,
            fit.MissingStages,
            fit.SampleCount,
            fit.OffsetMode,
            fit.ApplyFittedGain,
            centerOffsetX = fit.CenterOffsetX,
            centerOffsetY = fit.CenterOffsetY,
            xGain = fit.XGain,
            yGain = fit.YGain,
            fittedXGain = fit.FittedXGain,
            fittedYGain = fit.FittedYGain,
            xSpan = fit.XSpan,
            ySpan = fit.YSpan,
            stageMedians = fit.StageMedians,
            samples = _acceptedRuntimeSamples
        };
        var json = JsonSerializer.Serialize(payload, AppJsonOptions.Web(writeIndented: true));
        await File.WriteAllTextAsync(path, json, Encoding.UTF8);
    }

    private void ShowQuickCheck(StageCaptureResult result)
    {
        var check = result.QuickCheck;
        var stageName = FormatStageName(result.StageId);
        if (check.PairCount == 0)
        {
            ReviewInfoBar.Severity = InfoBarSeverity.Error;
            ReviewInfoBar.Title = "No frames";
            ReviewInfoBar.Message = $"No synchronized eye-frame pairs were captured for {stageName}. Check BrokenEye HTTP on 127.0.0.1:{BrokenEyePort}, then Retry.";
            AlgorithmText.Text = "Captured 0 synchronized pairs.";
            AutoMapText.Text = "Quick check: retry this point after BrokenEye is streaming.";
            CapturedPairsText.Text = "Captured pairs: 0";
            ValidPairsText.Text = "Valid pairs: 0";
            AverageDeltaText.Text = "Average sync delta: n/a";
            AverageConfidenceText.Text = "Average confidence: n/a";
            LowOpennessText.Text = "Low openness pairs: n/a";
            return;
        }

        ReviewInfoBar.Severity = check.Quality == "good"
            ? InfoBarSeverity.Success
            : check.Quality == "fair"
                ? InfoBarSeverity.Warning
                : InfoBarSeverity.Error;
        ReviewInfoBar.Title = "Review";
        ReviewInfoBar.Message = $"{stageName}: {check.ValidPairCount}/{check.PairCount} usable pairs. {check.ReviewReason} Suggested: {check.SuggestedAction}.";
        AlgorithmText.Text = $"Captured {check.PairCount} pairs, avg dt={check.AverageDeltaMs:0.0} ms.";
        AutoMapText.Text = $"Quick check: {check.Quality}. {check.ImprovementPlan}";
        CapturedPairsText.Text = $"Captured pairs: {check.PairCount}";
        ValidPairsText.Text = $"Valid pairs: {check.ValidPairCount}";
        AverageDeltaText.Text = $"Average sync delta: {check.AverageDeltaMs:0.0} ms";
        AverageConfidenceText.Text = $"Average confidence: {check.AverageConfidence:0.00}";
        LowOpennessText.Text = FormatQuickCheckDetails(check);
    }

    private static string FormatQuickCheckDetails(StageQuickCheck check) =>
        $"Low openness: {check.LowOpennessPairs}; low confidence: {check.LowConfidencePairs}; not found: {check.NotFoundPairs}; sync late: {check.SyncLatePairs}. {check.ImprovementPlan}";

    private bool ShouldAutoAccept(StageCaptureResult result)
    {
        if (result.Pairs.Count == 0)
        {
            return false;
        }

        if (HasEnoughRuntimeSamplesForCurrentStage())
        {
            return true;
        }

        return false;
    }

    private string AutoAcceptVerdict(StageCaptureResult result) =>
        result.QuickCheck.Quality.Equals("poor", StringComparison.OrdinalIgnoreCase)
            ? "runtime"
            : result.QuickCheck.Quality;

    private bool HasEnoughRuntimeSamplesForCurrentStage() =>
        _runtimeCaptureStageId is not null
        && _pendingRuntimeSamples.Count(sample => sample.StageId.Equals(_runtimeCaptureStageId, StringComparison.OrdinalIgnoreCase)) >= RuntimeMinSamplesPerStage;

    private bool HasRecentRuntimeEyeState()
    {
        var state = BridgeMonitorService.Instance.LatestState ?? _latestBridgeState;
        var receivedAt = BridgeMonitorService.Instance.LatestStateAt ?? _latestBridgeStateAt;
        return state is not null
            && receivedAt is not null
            && DateTimeOffset.Now - receivedAt.Value <= TimeSpan.FromSeconds(2)
            && state.EyeTrackingEnabled
            && TryGetRuntimeRawPoint(state, out _, out _);
    }

    private bool ShouldAutoRetry(StageCaptureResult result)
    {
        _stageAutoRetries.TryGetValue(result.StageId, out var retryCount);
        return retryCount < StageMaxAutoRetries;
    }

    private int IncrementAutoRetry(string stageId)
    {
        _stageAutoRetries.TryGetValue(stageId, out var retryCount);
        retryCount++;
        _stageAutoRetries[stageId] = retryCount;
        return retryCount;
    }

    private void ShowOverlay(
        CalibrationStage stage,
        string instruction,
        string countdown,
        string status,
        double progressValue,
        bool showProgress)
    {
        App.CurrentWindow?.ShowCalibrationOverlay(
            FormatStageName(stage.StageId),
            instruction,
            countdown,
            status,
            progressValue,
            showProgress);
    }

    private void UpdateCaptureProgress(StageCaptureProgress progress)
    {
        var progressValue = Math.Max(progress.ElapsedFraction, progress.TargetPairs <= 0 ? 0 : progress.CapturedPairs / (double)progress.TargetPairs);
        App.CurrentWindow?.UpdateCalibrationOverlayProgress(progressValue, $"{progress.CapturedPairs} / {progress.TargetPairs} pairs");
    }

    private static void HideOverlay()
    {
        App.CurrentWindow?.HideCalibrationOverlay();
    }

    private void SetActionButtonsEnabled(bool enabled)
    {
        StartButton.IsEnabled = enabled;
        UpdateStartButtonLabel();
        AcceptButton.IsEnabled = enabled && _workflowStarted && _currentCapture?.Pairs.Count > 0;
        RetryButton.IsEnabled = enabled && _workflowStarted;
        MarkBadButton.IsEnabled = enabled && _workflowStarted;
        NinePointToggle.IsEnabled = enabled && !_workflowStarted;
        StartOpennessCalibrationButton.IsEnabled = enabled && !_workflowStarted;
        StartOpennessTrainingCaptureButton.IsEnabled = enabled && !_workflowStarted;
        ExportTrainingPackageButton.IsEnabled = enabled && !_workflowStarted && !string.IsNullOrWhiteSpace(_sessionDirectory);
    }

    private void SetCalibrationSection(string section)
    {
        _calibrationSection = section switch
        {
            "eyelid" => "eyelid",
            "pupil" => "pupil",
            _ => "eye"
        };
        UpdateCalibrationSection();
    }

    private void UpdateCalibrationSection()
    {
        if (EyeSection is null || EyelidSection is null || PupilSection is null)
        {
            return;
        }

        EyeSection.Visibility = _calibrationSection == "eye" ? Visibility.Visible : Visibility.Collapsed;
        EyelidSection.Visibility = _calibrationSection == "eyelid" ? Visibility.Visible : Visibility.Collapsed;
        PupilSection.Visibility = _calibrationSection == "pupil" ? Visibility.Visible : Visibility.Collapsed;
        UpdateCalibrationMenuButton(EyeMenuButton, _calibrationSection == "eye");
        UpdateCalibrationMenuButton(EyelidMenuButton, _calibrationSection == "eyelid");
        UpdateCalibrationMenuButton(PupilMenuButton, _calibrationSection == "pupil");
        UpdateStartButtonLabel();
        NinePointToggle.Visibility = _calibrationSection == "eye" ? Visibility.Visible : Visibility.Collapsed;
        UpdateCurrentModuleParameterText();
    }

    private void UpdateStartButtonLabel()
    {
        StartButtonText.Text = _calibrationSection switch
        {
            "eyelid" => "Start eyelid calibration",
            "pupil" => "Start pupil calibration",
            _ => _workflowStarted ? "Restart gaze calibration" : "Start gaze calibration"
        };
    }

    private void UpdateCurrentModuleParameterText()
    {
        var bridge = BridgeProcessService.Instance;
        switch (_calibrationSection)
        {
            case "eyelid":
                AlgorithmText.Text = string.IsNullOrWhiteSpace(bridge.Options.OpennessCalibrationPath)
                    ? "Eyelid: no saved calibration"
                    : $"Eyelid: {IOPath.GetFileName(bridge.Options.OpennessCalibrationPath)}";
                AutoMapText.Text = $"Curve {bridge.Options.OpennessCurveMode}, full-open {bridge.Options.OpennessFullOpenThreshold:0.00}, knee {bridge.Options.OpennessBoostKnee:0.00}, gamma {bridge.Options.OpennessBoostGamma:0.00}";
                break;
            case "pupil":
                AlgorithmText.Text = $"Pupil: {GetPupilOutputMode()}";
                AutoMapText.Text = string.IsNullOrWhiteSpace(bridge.Options.PupilCalibrationPath)
                    ? "Pupil: no saved diameter calibration"
                    : $"Pupil: {IOPath.GetFileName(bridge.Options.PupilCalibrationPath)}";
                AutoMapText.Text += $"; wide enter {bridge.Options.PupilWideEnterThreshold:0.00}, exit {bridge.Options.PupilWideExitThreshold:0.00}";
                break;
            default:
                AlgorithmText.Text = _hasLiveData && _liveStageId is not null
                    ? $"Gaze: live {FormatStageName(_liveStageId)}"
                    : $"Gaze: waiting for runtime monitor UDP {MonitorUdpPort}";
                AutoMapText.Text = $"Mode: {(_stages.Count == 9 ? "9-point" : "5-point")}; offset=({bridge.Options.CenterOffsetX:0.000},{bridge.Options.CenterOffsetY:0.000}); gain=({bridge.Options.XGain:0.00},{bridge.Options.YGain:0.00})";
                break;
        }
    }

    private static void UpdateCalibrationMenuButton(Button button, bool active)
    {
        button.Opacity = active ? 1.0 : 0.72;
        button.Background = active ? CreateBrush(Colors.DeepSkyBlue, 0.22) : CreateBrush(Colors.Transparent);
        button.BorderBrush = active ? CreateBrush(Colors.DeepSkyBlue, 0.80) : CreateBrush(Colors.Gray, 0.40);
    }

    private static string CreateSessionDirectory()
    {
        var root = IOPath.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "DreamAirTracking",
            "calibration_data");
        return IOPath.Combine(root, DateTime.Now.ToString("yyyyMMdd_HHmmss"));
    }

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (directory is not null)
        {
            if (File.Exists(IOPath.Combine(directory.FullName, "DreamAirTracking.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(IOPath.Combine(directory.FullName, "DreamAirTracking.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return AppContext.BaseDirectory;
    }

    private static string QuoteCmd(string value) =>
        "\"" + value.Replace("\"", "\\\"", StringComparison.Ordinal) + "\"";

    private static string? FindProfilePath()
    {
        var names = new[] { "tracking_profile_final.json", "tracking_profile.json" };
        var directory = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (directory is not null)
        {
            foreach (var name in names)
            {
                var candidate = IOPath.Combine(directory.FullName, name);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            directory = directory.Parent;
        }

        directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            foreach (var name in names)
            {
                var candidate = IOPath.Combine(directory.FullName, name);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            directory = directory.Parent;
        }

        return null;
    }

    private void EnsureStageButtons()
    {
        if (_stageButtons.Count == 0)
        {
            _stageButtons["left_up"] = LeftUpButton;
            _stageButtons["up"] = UpButton;
            _stageButtons["right_up"] = RightUpButton;
            _stageButtons["left"] = LeftButton;
            _stageButtons["center"] = CenterButton;
            _stageButtons["right"] = RightButton;
            _stageButtons["left_down"] = LeftDownButton;
            _stageButtons["down"] = DownButton;
            _stageButtons["right_down"] = RightDownButton;
        }

        foreach (var stage in CalibrationStageCatalog.CreateNinePointGazeStages())
        {
            if (_stageButtons.TryGetValue(stage.StageId, out var button))
            {
                button.Content = CreateStageButtonContent(stage.DisplayName);
            }
        }

        UpdateStageButtonVisibility();
    }

    private void UpdateStageButtonVisibility()
    {
        var active = _stages
            .Select(stage => stage.StageId)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var (stageId, button) in _stageButtons)
        {
            button.Visibility = active.Contains(stageId)
                ? Visibility.Visible
                : Visibility.Collapsed;
        }
    }

    private void StartPipelineDiagnostics()
    {
        if (_diagnosticsCts is not null)
        {
            return;
        }

        _diagnosticsCts = new CancellationTokenSource();
        var token = _diagnosticsCts.Token;
        _ = Task.Run(() => DiagnosticsLoopAsync(token), token);
    }

    private void StopPipelineDiagnostics()
    {
        _diagnosticsCts?.Cancel();
        _diagnosticsCts?.Dispose();
        _diagnosticsCts = null;
    }

    private async Task DiagnosticsLoopAsync(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            PipelineDiagnosticsSnapshot? snapshot = null;
            try
            {
                snapshot = await _diagnosticsService.CaptureAsync(
                    BridgeMonitorService.Instance.LatestState ?? _latestBridgeState,
                    BridgeMonitorService.Instance.LatestStateAt ?? _latestBridgeStateAt,
                    token);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch
            {
            }

            if (snapshot is not null)
            {
                DispatcherQueue.TryEnqueue(() => UpdatePipelineDiagnostics(snapshot));
            }

            try
            {
                await Task.Delay(TimeSpan.FromSeconds(2), token);
            }
            catch (OperationCanceledException)
            {
                return;
            }
        }
    }

    private void UpdateLiveGaze(BridgeTrackingState state)
    {
        _latestBridgeState = state;
        _latestBridgeStateAt = DateTimeOffset.Now;
        CaptureRuntimeCalibrationSample(state);
        UpdatePupilGauges(state);
        UpdateEyelidGauges(state);

        var previousLiveTarget = _liveTarget;
        var highlight = _liveHighlighter.Evaluate(state, _liveStageId, previousLiveTarget);
        _hasLiveData = true;
        _livePaused = highlight.IsPaused;
        if (_livePaused)
        {
            _liveStageId = highlight.StageId;
            _liveTarget = previousLiveTarget is null ? null : highlight.Target;
            AlgorithmText.Text = _liveStageId is null
                ? $"Live gaze: paused, openness={highlight.Openness:0.00}"
                : $"Live gaze: holding {FormatStageName(_liveStageId)}, openness={highlight.Openness:0.00}";
            ReviewInfoBar.Severity = InfoBarSeverity.Warning;
            ReviewInfoBar.Title = "Holding";
            ReviewInfoBar.Message = "Low openness or confidence; highlight is held instead of jumping.";

            UpdateButtonVisualStates();
            return;
        }

        _liveTarget = highlight.Target;
        _liveStageId = highlight.StageId;

        var targetMatch = _liveStageId.Equals(CurrentStage.StageId, StringComparison.OrdinalIgnoreCase);
        AlgorithmText.Text = $"Live gaze: {FormatStageName(_liveStageId)} ({highlight.Target.X:0.00}, {highlight.Target.Y:0.00})";
        ReviewInfoBar.Severity = targetMatch ? InfoBarSeverity.Success : InfoBarSeverity.Informational;
        ReviewInfoBar.Title = targetMatch ? "Matched" : "Live";
        ReviewInfoBar.Message = $"Receiving runtime monitor UDP {MonitorUdpPort}. Retry if the highlighted point is clearly unstable.";

        UpdateButtonVisualStates();
    }

    private void CaptureRuntimeCalibrationSample(BridgeTrackingState state)
    {
        if (_runtimeCaptureStageId is null || !state.EyeTrackingEnabled || _pendingRuntimeSequences.Contains(state.Sequence))
        {
            return;
        }

        if (!TryGetRuntimeRawPoint(state, out var x, out var y))
        {
            return;
        }

        _pendingRuntimeSequences.Add(state.Sequence);
        _pendingRuntimeSamples.Add(new RuntimeGazeCalibrationSample(
            _runtimeCaptureStageId,
            x,
            y,
            state.Sequence,
            state.Timestamp));
    }

    private static bool TryGetRuntimeRawPoint(BridgeTrackingState state, out double x, out double y)
    {
        x = 0;
        y = 0;
        var leftValid = IsRuntimeEyeValid(state.Left);
        var rightValid = IsRuntimeEyeValid(state.Right);
        if (!leftValid && !rightValid)
        {
            return false;
        }

        if (leftValid && rightValid)
        {
            x = (RuntimeRawX(state.Left) + RuntimeRawX(state.Right)) * 0.5;
            y = (RuntimeRawY(state.Left) + RuntimeRawY(state.Right)) * 0.5;
            return IsFinite(x) && IsFinite(y);
        }

        var eye = leftValid ? state.Left : state.Right;
        x = RuntimeRawX(eye);
        y = RuntimeRawY(eye);
        return IsFinite(x) && IsFinite(y);
    }

    private static bool IsRuntimeEyeValid(BridgeEyeState eye)
        => eye.Found && eye.Confidence >= 0.05 && eye.Openness >= 0.08;

    private static double RuntimeRawX(BridgeEyeState eye)
        => Math.Abs(eye.RawNormalizedX) > 0.0000001 ? eye.RawNormalizedX : eye.RawX;

    private static double RuntimeRawY(BridgeEyeState eye)
        => Math.Abs(eye.RawNormalizedY) > 0.0000001 ? eye.RawNormalizedY : eye.RawY;

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    private void UpdatePupilGauges(BridgeTrackingState state)
    {
        PupilStatusText.Text = state.PupilDiameterEnabled
            ? $"Live {state.PupilDiameterMode}, sequence {state.Sequence}, dt={state.DeltaMs:0.0} ms"
            : state.PupilDiameterMode.Equals("expression_constrict_on_wide", StringComparison.OrdinalIgnoreCase)
                ? $"Live expression pupil, sequence {state.Sequence}, dt={state.DeltaMs:0.0} ms"
            : "Runtime live, pupil output disabled";
        UpdatePupilGauge(state.Left, LeftPupilInner, LeftPupilMetricText, LeftPupilQualityText, Colors.DeepSkyBlue);
        UpdatePupilGauge(state.Right, RightPupilInner, RightPupilMetricText, RightPupilQualityText, Colors.CornflowerBlue);
        PupilPipelineText.Text = state.PupilDiameterEnabled
            ? $"Output mode {state.PupilDiameterMode}. Inner circle shows VRCFT pupil expression; offset shows model pupil center."
            : state.PupilDiameterMode.Equals("expression_constrict_on_wide", StringComparison.OrdinalIgnoreCase)
                ? "Output mode expression constrict on wide. Pupil stays neutral until wide_amount is available."
            : $"Output mode {GetPupilOutputMode()}. Pupil output stays neutral until detector validation passes.";
    }

    private void UpdateEyelidGauges(BridgeTrackingState state)
    {
        var averageOpenness = (state.Left.Openness + state.Right.Openness) * 0.5;
        EyelidStatusText.Text = state.EyeTrackingEnabled
            ? $"Live openness, sequence {state.Sequence}, avg={averageOpenness:0.00}, dt={state.DeltaMs:0.0} ms"
            : "Eye tracking runtime is stopped";
        UpdateEyelidGauge(state.Left, LeftEyelidInner, LeftEyelidMetricText, LeftEyelidQualityText, Colors.MediumSeaGreen);
        UpdateEyelidGauge(state.Right, RightEyelidInner, RightEyelidMetricText, RightEyelidQualityText, Colors.CornflowerBlue);
        EyelidPipelineText.Text = "Outer circle is full open. Values show raw aperture, calibrated runtime openness, and final output after the open-eye boost.";
    }

    private static void UpdateEyelidGauge(
        BridgeEyeState eye,
        Ellipse inner,
        TextBlock metricText,
        TextBlock qualityText,
        Windows.UI.Color openColor)
    {
        var openness = Math.Clamp(eye.Openness, 0, 1);
        var size = eye.Found ? 20 + openness * 92 : 18;
        inner.Width = size;
        inner.Height = size;
        inner.Fill = eye.Found
            ? CreateBrush(OpennessColor(openness, openColor))
            : CreateBrush(Colors.Gray, 0.42);
        inner.Stroke = eye.Found ? CreateBrush(Colors.White, 0.86) : CreateBrush(Colors.Transparent);
        inner.StrokeThickness = eye.Found ? 2 : 0;

        var calibrated = eye.CalibratedOpenness > 0.0000001
            ? eye.CalibratedOpenness
            : eye.Openness;
        var output = eye.OutputOpenness > 0.0000001
            ? eye.OutputOpenness
            : eye.Openness;
        metricText.Text = eye.Found
            ? $"raw {eye.ApertureOpenness:0.00} | cal {calibrated:0.00} | out {output:0.00}"
            : "--";

        qualityText.Text = eye.Found
            ? eye.Confidence < 0.15
                ? $"weak confidence {eye.Confidence:0.00}"
                : openness < 0.18
                    ? $"closed | peak {eye.AperturePeakDarkFraction:0.00}"
                    : openness < 0.55
                        ? $"partial | height {eye.ApertureHeight}px"
                        : $"open | wide {eye.Wide:0.00} squint {eye.Squint:0.00}"
            : string.IsNullOrWhiteSpace(eye.ApertureReason) ? "no eyelid" : eye.ApertureReason;
    }

    private static Windows.UI.Color OpennessColor(double openness, Windows.UI.Color openColor)
    {
        if (openness < 0.18)
        {
            return Colors.IndianRed;
        }

        return openness < 0.55 ? Colors.Gold : openColor;
    }

    private static void UpdatePupilGauge(
        BridgeEyeState eye,
        Ellipse inner,
        TextBlock metricText,
        TextBlock qualityText,
        Windows.UI.Color liveColor)
    {
        var normalized = Math.Clamp(eye.PupilDiameterNormalized, 0, 1);
        var found = eye.PupilDiameterFound;
        var centerFound = eye.NormalizationFound || found;
        var centerX = Math.Clamp(eye.NormalizationPupilX, 0, 1);
        var centerY = Math.Clamp(eye.NormalizationPupilY, 0, 1);
        var size = found ? 20 + normalized * 78 : 18;
        inner.Width = size;
        inner.Height = size;
        inner.RenderTransform = centerFound
            ? new TranslateTransform
            {
                X = (centerX - 0.5) * 78,
                Y = (centerY - 0.5) * 78
            }
            : null;
        inner.Fill = found
            ? CreateBrush(eye.PupilDiameterQuality ? liveColor : Colors.Gold)
            : centerFound
                ? CreateBrush(Colors.Gold, 0.72)
                : CreateBrush(Colors.Gray, 0.42);
        inner.Stroke = found ? CreateBrush(Colors.White, 0.86) : CreateBrush(Colors.Transparent);
        inner.StrokeThickness = found ? 2 : 0;

        metricText.Text = found
            ? eye.PupilDiameterPx > 0
                ? $"center {centerX:0.00},{centerY:0.00} | dia {normalized:0.00}/{eye.PupilDiameterPx:0}px"
                : $"center {centerX:0.00},{centerY:0.00} | expr {eye.PupilExpressionNormalized:0.00}"
            : centerFound
                ? $"center {centerX:0.00},{centerY:0.00} | geo {eye.PupilGeometryRadius:0.00}"
                : "--";
        qualityText.Text = found
            ? eye.PupilDiameterQuality
                ? $"quality {eye.PupilDiameterConfidence:0.00} | wide {eye.Wide:0.00}"
                : $"weak {eye.PupilDiameterReason}"
            : centerFound
                ? $"normalization quality {eye.NormalizationConfidence:0.00}"
                : "no pupil";
    }

    private void UpdatePipelineDiagnostics(PipelineDiagnosticsSnapshot snapshot)
    {
        _latestDiagnosticsSnapshot = snapshot;
        BrokenEyeStatusText.Text = snapshot.BrokenEyeLeftLive && snapshot.BrokenEyeRightLive
            ? "live L/R"
            : $"missing L={FormatBool(snapshot.BrokenEyeLeftLive)} R={FormatBool(snapshot.BrokenEyeRightLive)}";

        BridgePupilStatusText.Text = snapshot.BridgeMonitorLive
            ? snapshot.BridgePupilEnabled
                ? $"monitor live, pupil on ({snapshot.BridgePupilMode})"
                : snapshot.BridgePupilMode.Equals("expression_constrict_on_wide", StringComparison.OrdinalIgnoreCase)
                    ? "monitor live, expression pupil"
                : "monitor live, pupil off"
            : $"no monitor packet on UDP {MonitorUdpPort}";
        BridgePupilStatusText.Text += $"; wear={snapshot.WearTemplateStatus}";

        VrcftStatusText.Text = snapshot.VrcftModuleProcessRunning && snapshot.VrcftUdp9400Listening
            ? "loaded, UDP 9400 listening"
            : $"app={FormatBool(snapshot.VrcftProcessRunning)} module={FormatBool(snapshot.VrcftModuleProcessRunning)} udp9400={FormatBool(snapshot.VrcftUdp9400Listening)}";

        VrchatPupilStatusText.Text = snapshot.VrchatOscQueryLive
            ? snapshot.FtPupilDilationPresent
                ? snapshot.PupilParameterSummary
                : $"pupil params: {snapshot.PupilParameterSummary}"
            : "OscQuery not found";

        AvatarGateStatusText.Text = snapshot.VrchatOscQueryLive
            ? snapshot.PupilGateAliasSummary
            : "VRChat avatar unavailable";
    }

    private void UpdateButtonVisualStates()
    {
        var currentStageId = CurrentStage.StageId;
        foreach (var (stageId, button) in _stageButtons)
        {
            var isCurrent = stageId.Equals(currentStageId, StringComparison.OrdinalIgnoreCase);
            var isLive = _hasLiveData && _liveTarget is not null && stageId.Equals(_liveStageId, StringComparison.OrdinalIgnoreCase);
            _stageStatuses.TryGetValue(stageId, out var status);
            var liveColor = _livePaused ? Colors.Gold : Colors.DeepSkyBlue;

            button.Opacity = isLive || isCurrent ? 1.0 : status is null ? 0.46 : 0.72;
            button.Background = CreateBrush(Colors.Transparent);
            button.BorderBrush = CreateBrush(Colors.Transparent);
            button.BorderThickness = new Thickness(0);
            UpdateStageButtonDot(button, isLive, _livePaused, isCurrent, status);
        }
    }

    private static Grid CreateStageButtonContent(string label)
    {
        var grid = new Grid
        {
            Width = 142,
            Height = 116
        };
        grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

        var dot = new Ellipse
        {
            Width = 54,
            Height = 54,
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            Fill = CreateBrush(Colors.Gray, 0.72),
            Stroke = CreateBrush(Colors.Transparent),
            StrokeThickness = 0
        };
        var text = new TextBlock
        {
            Text = label,
            HorizontalAlignment = HorizontalAlignment.Center,
            Foreground = CreateBrush(Colors.White, 0.62),
            TextWrapping = TextWrapping.NoWrap
        };
        Grid.SetRow(dot, 0);
        Grid.SetRow(text, 1);
        grid.Children.Add(dot);
        grid.Children.Add(text);
        return grid;
    }

    private static void UpdateStageButtonDot(Button button, bool isLive, bool isPaused, bool isCurrent, string? status)
    {
        if (button.Content is not Grid grid)
        {
            return;
        }

        foreach (var dot in grid.Children.OfType<Ellipse>())
        {
            dot.Width = isLive ? 72 : isCurrent ? 62 : 54;
            dot.Height = dot.Width;
            dot.Fill = isLive
                ? CreateBrush(isPaused ? Colors.Gold : Colors.DeepSkyBlue)
                : status == "Bad"
                    ? CreateBrush(Colors.IndianRed)
                    : status == "Retrying"
                        ? CreateBrush(Colors.Gold)
                    : status == "Review"
                        ? CreateBrush(Colors.Orange)
                    : status is not null
                        ? CreateBrush(Colors.SeaGreen)
                        : isCurrent
                            ? CreateBrush(Colors.Gray, 0.88)
                            : CreateBrush(Colors.Gray, 0.60);
            dot.Stroke = isLive
                ? CreateBrush(Colors.White)
                : isCurrent
                    ? CreateBrush(Colors.White, 0.92)
                    : status is not null
                        ? CreateBrush(Colors.White, 0.72)
                        : CreateBrush(Colors.Transparent);
            dot.StrokeThickness = isLive ? 5 : isCurrent || status is not null ? 3 : 0;
        }

        foreach (var text in grid.Children.OfType<TextBlock>())
        {
            text.Foreground = isLive
                ? CreateBrush(Colors.White)
                : isCurrent
                    ? CreateBrush(Colors.White, 0.86)
                    : CreateBrush(Colors.White, 0.54);
            text.FontWeight = isLive || isCurrent ? Microsoft.UI.Text.FontWeights.SemiBold : Microsoft.UI.Text.FontWeights.Normal;
        }
    }

    private static SolidColorBrush CreateBrush(Windows.UI.Color color, double opacity = 1.0) =>
        new(color) { Opacity = opacity };

    private static string FormatBool(bool value) => value ? "yes" : "no";

    private static string FormatGate(bool? value) => value switch
    {
        true => "on",
        false => "off",
        null => "n/a"
    };

    private static string FormatNumber(double? value) =>
        value is null ? "n/a" : value.Value.ToString("0.000");

    private static string FormatStageName(string stageId) =>
        stageId.Replace('_', ' ');

    private static string FormatApplicationDecisionDetails(CalibrationProfileApplicationDecision decision)
    {
        if (decision.MissingAcceptedStageIds.Count > 0)
        {
            return $"Retry missing stages: {string.Join(", ", decision.MissingAcceptedStageIds.Select(FormatStageName))}.";
        }

        if (decision.HighResiduals.Count > 0)
        {
            return $"Retry high-residual stages: {string.Join(", ", decision.HighResiduals.Select(item => $"{FormatStageName(item.StageId)} {item.Distance:0.00}"))}.";
        }

        return decision.Reason;
    }
}
