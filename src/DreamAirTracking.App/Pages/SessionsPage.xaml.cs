using Microsoft.UI.Xaml.Controls;
using DreamAirTracking.App.Services;
using DreamAirTracking.Core.Calibration;
using Microsoft.UI.Xaml;
using System.Text.Json;

namespace DreamAirTracking.App.Pages;

public sealed partial class SessionsPage : Page
{
    public SessionsPage()
    {
        InitializeComponent();
        GazeWatcherProcessService.Instance.StateChanged += Watcher_StateChanged;
        Loaded += SessionsPage_Loaded;
        Unloaded += SessionsPage_Unloaded;
    }

    private async void SessionsPage_Loaded(object sender, RoutedEventArgs e)
    {
        await RefreshAuditAsync();
    }

    private async void Refresh_Click(object sender, RoutedEventArgs e)
    {
        await RefreshAuditAsync();
    }

    private void StartWatcher_Click(object sender, RoutedEventArgs e)
    {
        GazeWatcherProcessService.Instance.Start();
        RefreshWatcherStatus();
    }

    private void StopWatcher_Click(object sender, RoutedEventArgs e)
    {
        GazeWatcherProcessService.Instance.Stop();
        RefreshWatcherStatus();
    }

    private async void ExportTrainingPackage_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button { Tag: string sessionDirectory } || string.IsNullOrWhiteSpace(sessionDirectory))
        {
            return;
        }

        try
        {
            var outputDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "DreamAirTracking",
                "capture_packages");
            var result = await new CalibrationCapturePackageExporter().ExportAsync(new CalibrationCapturePackageExportOptions
            {
                SessionDirectory = sessionDirectory,
                OutputDirectory = outputDirectory,
                RuntimeModelId = BridgeProcessService.Instance.CurrentMultitaskModelPreset
            });
            SummaryInfoBar.Severity = InfoBarSeverity.Success;
            SummaryInfoBar.Title = "Training package exported";
            SummaryInfoBar.Message = result.OutputZipPath;
        }
        catch (Exception ex)
        {
            SummaryInfoBar.Severity = InfoBarSeverity.Error;
            SummaryInfoBar.Title = "Export failed";
            SummaryInfoBar.Message = ex.Message;
        }
    }

    private void SessionsPage_Unloaded(object sender, RoutedEventArgs e)
    {
        GazeWatcherProcessService.Instance.StateChanged -= Watcher_StateChanged;
    }

    private void Watcher_StateChanged(object? sender, EventArgs e)
    {
        DispatcherQueue.TryEnqueue(RefreshWatcherStatus);
    }

    private async Task RefreshAuditAsync()
    {
        SummaryInfoBar.Severity = InfoBarSeverity.Informational;
        SummaryInfoBar.Title = "Audit";
        SummaryInfoBar.Message = "Scanning calibration records.";
        RootText.Text = $"Root: {CalibrationSessionAuditor.DefaultCalibrationRoot}";

        try
        {
            var summary = await CalibrationSessionAuditor.AuditRootAsync();
            SessionsList.ItemsSource = summary.Sessions;
            RootText.Text = $"Root: {summary.RootDirectory}";
            SummaryInfoBar.Severity = summary.NeedsMoreDataForSessionValidation
                ? InfoBarSeverity.Warning
                : InfoBarSeverity.Success;
            SummaryInfoBar.Title = $"{summary.TotalCount} sessions, {summary.ImageTrainingUsableCount} image usable, {summary.StrictUsableCount} strict profile usable";
            SummaryInfoBar.Message =
                $"{summary.BottomLine} {summary.NextRecordingPlan} {summary.FairnessAudit.StatusText} Excluded={summary.ExcludedCount}, partial={summary.PartialCandidateCount}, unstable={summary.CompleteButUnstableCount}, incomplete={summary.IncompleteCount}, empty={summary.EmptyOrUnlabeledCount}.";
            RefreshWatcherStatus();
        }
        catch (Exception ex)
        {
            SessionsList.ItemsSource = null;
            SummaryInfoBar.Severity = InfoBarSeverity.Error;
            SummaryInfoBar.Title = "Audit failed";
            SummaryInfoBar.Message = ex.Message;
            RefreshWatcherStatus();
        }
    }

    private void RefreshWatcherStatus()
    {
        var watcher = GazeWatcherProcessService.Instance;
        StartWatcherButton.IsEnabled = !watcher.IsRunning;
        StopWatcherButton.IsEnabled = watcher.IsRunning;
        try
        {
            var statusPath = FindLatestWatchStatusPath();
            if (statusPath is null)
            {
                WatcherInfoBar.Severity = InfoBarSeverity.Informational;
                WatcherInfoBar.Title = watcher.IsRunning ? "Watcher running" : "Watcher";
                WatcherInfoBar.Message = watcher.IsRunning
                    ? watcher.LastOutput
                    : "No automatic watcher status found yet. Start watcher or run scripts\\watch_gaze_pipeline.ps1.";
                return;
            }

            using var document = JsonDocument.Parse(File.ReadAllText(statusPath));
            var root = document.RootElement;
            var state = ReadString(root, "state") ?? "unknown";
            var attemptCount = ReadInt(root, "attempt_count");
            var updatedAt = ReadString(root, "updated_at") ?? "unknown time";
            var requestPath = ReadString(root, "recording_request");
            var auditPath = ReadString(root, "latest_audit_report") ?? ReadString(root, "audit_report");
            var fixPlan = RecordingRequestSummaryParser.ReadWeakStageFixSummary(requestPath, maxRows: 2);
            var requestSummary = ReadFirstMatchingLine(requestPath, "- Weak stages to watch:*");
            var auditPlan = ReadFirstMatchingLine(auditPath, "Next recording plan:*");
            var detail = fixPlan ?? auditPlan ?? requestSummary ?? ReadString(root, "reason") ?? "No detail available.";

            WatcherInfoBar.Severity = state.Equals("complete", StringComparison.OrdinalIgnoreCase)
                ? InfoBarSeverity.Success
                : state.Equals("failed", StringComparison.OrdinalIgnoreCase)
                    ? InfoBarSeverity.Error
                    : InfoBarSeverity.Warning;
            WatcherInfoBar.Title = watcher.IsRunning ? $"Watcher running, last state {state}" : $"Watcher {state}";
            WatcherInfoBar.Message = $"Attempts={attemptCount}, updated={updatedAt}. {detail} Status: {statusPath}. {watcher.LastOutput}";
        }
        catch (Exception ex)
        {
            WatcherInfoBar.Severity = InfoBarSeverity.Error;
            WatcherInfoBar.Title = "Watcher status failed";
            WatcherInfoBar.Message = ex.Message;
        }
    }

    private static string? FindLatestWatchStatusPath()
    {
        var repoRoot = FindRepoRoot();
        var runs = Path.Combine(repoRoot, "runs");
        if (!Directory.Exists(runs))
        {
            return null;
        }

        return Directory
            .EnumerateFiles(runs, "watch_status.json", SearchOption.AllDirectories)
            .Select(path => new FileInfo(path))
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .FirstOrDefault()
            ?.FullName;
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

    private static string? ReadString(JsonElement root, string propertyName) =>
        root.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    private static int ReadInt(JsonElement root, string propertyName) =>
        root.TryGetProperty(propertyName, out var value) && value.TryGetInt32(out var result)
            ? result
            : 0;

    private static string? ReadFirstMatchingLine(string? path, string pattern)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            return null;
        }

        var prefix = pattern.TrimEnd('*');
        return File.ReadLines(path).FirstOrDefault(line => line.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
    }
}
