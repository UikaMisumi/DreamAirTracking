using System.Net.NetworkInformation;
using DreamAirTracking.App.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace DreamAirTracking.App.Pages;

public sealed partial class HomePage : Page
{
    private bool _autoStartInFlight;
    private bool _manualStopRequested;
    private bool _loadingModelPreset;
    private CancellationTokenSource? _modelDownloadCts;
    private readonly DispatcherTimer _autoStartTimer = new()
    {
        Interval = TimeSpan.FromSeconds(5)
    };

    public HomePage()
    {
        InitializeComponent();
        _autoStartTimer.Tick += AutoStartTimer_Tick;
    }

    private async void Page_Loaded(object sender, RoutedEventArgs e)
    {
        BridgeProcessService.Instance.StateChanged += Bridge_StateChanged;
        SyncModelPresetBox();
        UpdateStatus();
        _autoStartTimer.Start();
        await TryAutoStartAsync();
    }

    private void Page_Unloaded(object sender, RoutedEventArgs e)
    {
        _autoStartTimer.Stop();
        BridgeProcessService.Instance.StateChanged -= Bridge_StateChanged;
    }

    private void Bridge_StateChanged(object? sender, EventArgs e)
    {
        DispatcherQueue.TryEnqueue(UpdateStatus);
    }

    private async void StartBridge_Click(object sender, RoutedEventArgs e)
    {
        if (ModelPresetBox.Items.Count == 0)
        {
            BridgeOutputText.Text = "No model package installed. Import a Dream Air model package first.";
            UpdateStatus();
            return;
        }

        try
        {
            _manualStopRequested = false;
            await BridgeProcessService.Instance.StartAsync();
        }
        catch (Exception ex)
        {
            BridgeOutputText.Text = $"Eye tracking runtime failed to start: {ex.Message}";
        }

        UpdateStatus();
    }

    private void StopBridge_Click(object sender, RoutedEventArgs e)
    {
        _manualStopRequested = true;
        BridgeProcessService.Instance.Stop();
        UpdateStatus();
    }

    private void Refresh_Click(object sender, RoutedEventArgs e)
    {
        SyncModelPresetBox();
        UpdateStatus();
    }

    private async void InstallModel_Click(object sender, RoutedEventArgs e)
    {
        if (_modelDownloadCts is not null)
        {
            return;
        }

        _modelDownloadCts = new CancellationTokenSource();
        SetModelDownloadUi(
            InfoBarSeverity.Informational,
            "Model download",
            "Connecting to Hugging Face...",
            null,
            isDownloading: true);

        var progress = new Progress<ModelDownloadProgress>(UpdateModelDownloadProgress);
        try
        {
            var result = await HuggingFaceModelDownloadService.Instance.InstallDefaultModelAsync(
                progress,
                _modelDownloadCts.Token);
            if (result.Success)
            {
                SyncModelPresetBox();
                SetModelDownloadUi(
                    InfoBarSeverity.Success,
                    "Model installed",
                    result.Message,
                    1,
                    isDownloading: false);
            }
            else
            {
                SetModelDownloadUi(
                    InfoBarSeverity.Error,
                    "Model download failed",
                    result.Message,
                    null,
                    isDownloading: false);
            }
        }
        finally
        {
            _modelDownloadCts?.Dispose();
            _modelDownloadCts = null;
            UpdateStatus();
        }
    }

    private void CancelModelDownload_Click(object sender, RoutedEventArgs e)
    {
        _modelDownloadCts?.Cancel();
    }

    private async void ModelPresetBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_loadingModelPreset || ModelPresetBox.SelectedItem is not ComboBoxItem { Tag: string presetId })
        {
            return;
        }

        var bridge = BridgeProcessService.Instance;
        var wasRunning = bridge.IsRunning;
        bridge.SelectMultitaskModelPreset(presetId);
        ModelPresetStatusText.Text = wasRunning
            ? $"Switching to {bridge.CurrentMultitaskModelLabel}..."
            : $"{bridge.CurrentMultitaskModelLabel} selected.";

        if (wasRunning)
        {
            try
            {
                await bridge.StartAsync(restartIfRunning: true);
            }
            catch (Exception ex)
            {
                BridgeOutputText.Text = $"Eye tracking runtime failed to restart: {ex.Message}";
            }
        }

        UpdateStatus();
    }

    private async void AutoStartTimer_Tick(object? sender, object e)
    {
        await TryAutoStartAsync();
    }

    private void UpdateStatus()
    {
        var bridge = BridgeProcessService.Instance;
        var hasModelPackage = ModelPresetBox.Items.Count > 0;
        var modelType = string.Equals(bridge.Options.RuntimeModelType, "multitask", StringComparison.OrdinalIgnoreCase)
            ? bridge.CurrentMultitaskModelLabel
            : "gaze-only";
        ModelPresetStatusText.Text = hasModelPackage
            ? bridge.IsRunning
                ? $"{bridge.CurrentMultitaskModelLabel} running."
                : $"{bridge.CurrentMultitaskModelLabel} selected."
            : "No model package installed.";
        BrokenEyeStateText.Text = "Runtime reads BrokenEye HTTP /eye/left and /eye/right on 127.0.0.1:5555.";
        BridgeStateText.Text = bridge.IsRunning
            ? bridge.IsExternalRunning
                ? "Legacy Bridge running"
                : $"Running {modelType} with {Path.GetFileName(bridge.ProfilePath ?? "model")}"
            : hasModelPackage
                ? "Stopped"
                : "No model package";
        BridgeOutputText.Text = hasModelPackage
            ? bridge.LatestWearTemplateProbe is null
                ? bridge.StatusText
                : $"{bridge.StatusText} Wear template: {FormatWearTemplate(bridge.LatestWearTemplateProbe)}"
            : "Import a model package to enable eye tracking.";
        StartBridgeButton.IsEnabled = hasModelPackage && !bridge.IsRunning;
        StopBridgeButton.IsEnabled = bridge.IsRunning;
        InstallModelButton.IsEnabled = _modelDownloadCts is null;
        CancelModelDownloadButton.Visibility = _modelDownloadCts is null ? Visibility.Collapsed : Visibility.Visible;

        VrcftStateText.Text = IsUdpPortListening(9400)
            ? "DreamAirTracking VRCFT module appears to be listening on UDP 9400."
            : "No VRCFT module listener on UDP 9400. Install/enable the module and restart VRCFaceTracking.";
    }

    private async Task TryAutoStartAsync()
    {
        if (_autoStartInFlight)
        {
            return;
        }

        var bridge = BridgeProcessService.Instance;
        if (_manualStopRequested ||
            !bridge.Options.AutoStartEyeTracking ||
            bridge.IsRunning ||
            bridge.GetMultitaskModelChoices().Count == 0)
        {
            return;
        }

        try
        {
            _autoStartInFlight = true;
            await bridge.StartAsync();
        }
        catch (Exception ex)
        {
            BridgeOutputText.Text = $"Eye tracking runtime failed to start: {ex.Message}";
        }
        finally
        {
            _autoStartInFlight = false;
        }

        UpdateStatus();
    }

    private void SyncModelPresetBox()
    {
        _loadingModelPreset = true;
        try
        {
            var bridge = BridgeProcessService.Instance;
            var choices = bridge.GetMultitaskModelChoices();
            ModelPresetBox.Items.Clear();
            ModelPresetBox.PlaceholderText = "No model package";
            foreach (var choice in choices)
            {
                ModelPresetBox.Items.Add(new ComboBoxItem
                {
                    Content = choice.DisplayName,
                    Tag = choice.Id
                });
            }

            var preset = bridge.CurrentMultitaskModelPreset;
            foreach (var item in ModelPresetBox.Items.OfType<ComboBoxItem>())
            {
                if (item.Tag is string tag && tag.Equals(preset, StringComparison.OrdinalIgnoreCase))
                {
                    ModelPresetBox.SelectedItem = item;
                    return;
                }
            }

            if (ModelPresetBox.Items.Count > 0)
            {
                ModelPresetBox.SelectedIndex = 0;
                if (choices.Count > 0)
                {
                    bridge.SelectMultitaskModelPreset(choices[0].Id);
                }
            }
            else
            {
                ModelPresetBox.SelectedItem = null;
                ModelPresetBox.SelectedIndex = -1;
            }
        }
        finally
        {
            _loadingModelPreset = false;
        }
    }

    private static bool IsUdpPortListening(int port)
    {
        try
        {
            return IPGlobalProperties.GetIPGlobalProperties()
                .GetActiveUdpListeners()
                .Any(endpoint => endpoint.Port == port);
        }
        catch
        {
            return false;
        }
    }

    private static string FormatWearTemplate(WearTemplateProbeResult result)
    {
        var distance = result.Distance is null ? "n/a" : result.Distance.Value.ToString("0.000");
        var template = string.IsNullOrWhiteSpace(result.TemplateId) ? "none" : result.TemplateId;
        var calibration = result.HasRuntimeCalibration ? "calibration=template" : "calibration=none";
        return $"{result.Status}, action={result.Action}, template={template}, distance={distance}, {calibration}.";
    }

    private void UpdateModelDownloadProgress(ModelDownloadProgress progress)
    {
        var severity = progress.Stage.Equals("failed", StringComparison.OrdinalIgnoreCase)
            ? InfoBarSeverity.Error
            : progress.Stage.Equals("done", StringComparison.OrdinalIgnoreCase)
                ? InfoBarSeverity.Success
                : InfoBarSeverity.Informational;
        SetModelDownloadUi(
            severity,
            progress.Stage.Equals("done", StringComparison.OrdinalIgnoreCase) ? "Model installed" : "Model download",
            progress.Detail,
            progress.Fraction,
            isDownloading: _modelDownloadCts is not null);
    }

    private void SetModelDownloadUi(
        InfoBarSeverity severity,
        string title,
        string message,
        double? progress,
        bool isDownloading)
    {
        ModelDownloadInfoBar.IsOpen = true;
        ModelDownloadInfoBar.Severity = severity;
        ModelDownloadInfoBar.Title = title;
        ModelDownloadMessageText.Text = message;
        ModelDownloadProgressBar.IsIndeterminate = isDownloading && progress is null;
        ModelDownloadProgressBar.Value = progress ?? 0;
        ModelDownloadProgressBar.Visibility = isDownloading || progress is not null
            ? Visibility.Visible
            : Visibility.Collapsed;
        InstallModelButton.IsEnabled = !isDownloading;
        CancelModelDownloadButton.Visibility = isDownloading ? Visibility.Visible : Visibility.Collapsed;
    }
}
