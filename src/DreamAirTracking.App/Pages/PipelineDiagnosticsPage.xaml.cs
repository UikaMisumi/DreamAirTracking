using DreamAirTracking.App.Services;
using DreamAirTracking.Core.Bridge;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace DreamAirTracking.App.Pages;

public sealed partial class PipelineDiagnosticsPage : Page
{
    private const int MonitorUdpPort = 9401;

    private readonly PipelineDiagnosticsService _diagnosticsService = new();
    private CancellationTokenSource? _diagnosticsCts;

    public PipelineDiagnosticsPage()
    {
        InitializeComponent();
    }

    private void Page_Loaded(object sender, RoutedEventArgs e)
    {
        BridgeMonitorService.Instance.StateReceived -= BridgeMonitor_StateReceived;
        BridgeMonitorService.Instance.StateReceived += BridgeMonitor_StateReceived;
        BridgeMonitorService.Instance.Start();
        if (BridgeMonitorService.Instance.LatestState is not null)
        {
            UpdateBridgeMonitor(BridgeMonitorService.Instance.LatestState);
        }
        else if (!string.IsNullOrWhiteSpace(BridgeMonitorService.Instance.ErrorMessage))
        {
            BridgeMonitorText.Text = BridgeMonitorService.Instance.ErrorMessage;
        }

        StartPipelineDiagnostics();
    }

    private void Page_Unloaded(object sender, RoutedEventArgs e)
    {
        StopPipelineDiagnostics();
        BridgeMonitorService.Instance.StateReceived -= BridgeMonitor_StateReceived;
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
                    BridgeMonitorService.Instance.LatestState,
                    BridgeMonitorService.Instance.LatestStateAt,
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

    private void UpdateBridgeMonitor(BridgeTrackingState state)
    {
        BridgeMonitorText.Text = $"seq={state.Sequence}, dt={state.DeltaMs:0.0} ms, pupil={FormatBool(state.PupilDiameterEnabled)} mode={state.PupilDiameterMode}, eyeShapes={FormatBool(state.EyeExpressionEnabled)} {state.EyeExpressionMode}, normalization={state.NormalizationMode}";
        LeftEyeText.Text = FormatEye("Left", state.Left);
        RightEyeText.Text = FormatEye("Right", state.Right);
    }

    private void BridgeMonitor_StateReceived(object? sender, BridgeTrackingState state)
    {
        DispatcherQueue.TryEnqueue(() => UpdateBridgeMonitor(state));
    }

    private void UpdatePipelineDiagnostics(PipelineDiagnosticsSnapshot snapshot)
    {
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

        VrcftStatusText.Text = snapshot.VrcftModuleProcessRunning && snapshot.VrcftUdp9400Listening
            ? "loaded, UDP 9400 listening"
            : $"app={FormatBool(snapshot.VrcftProcessRunning)} module={FormatBool(snapshot.VrcftModuleProcessRunning)} udp9400={FormatBool(snapshot.VrcftUdp9400Listening)}";

        WearTemplateStatusText.Text = FormatWearTemplate(snapshot.WearTemplateStatus, snapshot.WearTemplateAction, snapshot.WearTemplateId, snapshot.WearTemplateDistance);

        OscQueryStatusText.Text = snapshot.VrchatOscQueryLive
            ? $"live on {snapshot.VrchatOscQueryPort}, avatar={snapshot.AvatarId ?? "unknown"}"
            : "not found";

        VrchatPupilStatusText.Text = snapshot.VrchatOscQueryLive
            ? snapshot.FtPupilDilationPresent
                ? snapshot.PupilParameterSummary
                : $"pupil params: {snapshot.PupilParameterSummary}"
            : "VRChat unavailable";

        AvatarGateStatusText.Text = snapshot.VrchatOscQueryLive
            ? snapshot.PupilGateAliasSummary
            : "n/a";
    }

    private static string FormatEye(string label, BridgeEyeState eye)
    {
        var pupil = eye.PupilDiameterFound
            ? $"pd={eye.PupilDiameterNormalized:0.00} expr={eye.PupilExpressionNormalized:0.00} geo={eye.PupilGeometryRadius:0.00} q={eye.PupilDiameterConfidence:0.00}"
            : $"pd=none expr={eye.PupilExpressionNormalized:0.00} geo={eye.PupilGeometryRadius:0.00}";
        var normalizationReason = string.IsNullOrWhiteSpace(eye.NormalizationDropReason)
            ? string.Empty
            : $" {eye.NormalizationDropReason}";
        var normalization = $"norm={FormatBool(eye.NormalizationFound)} q={eye.NormalizationConfidence:0.00} shift=({eye.NormalizationShiftX:0.00},{eye.NormalizationShiftY:0.00}) scale={eye.NormalizationScale:0.00}{normalizationReason}";
        return $"{label}: gaze=({eye.NormalizedX:0.00},{eye.NormalizedY:0.00}) open={eye.Openness:0.00} wide={eye.Wide:0.00} squint={eye.Squint:0.00}, {pupil}, {normalization}";
    }

    private static string FormatBool(bool value) => value ? "yes" : "no";

    private static string FormatGate(bool? value) => value switch
    {
        true => "on",
        false => "off",
        null => "n/a"
    };

    private static string FormatNumber(double? value) =>
        value is null ? "n/a" : value.Value.ToString("0.000");

    private static string FormatWearTemplate(string status, string action, string? templateId, double? distance)
    {
        var template = string.IsNullOrWhiteSpace(templateId) ? "none" : templateId;
        return $"{status}, action={action}, template={template}, distance={FormatNumber(distance)}";
    }
}
