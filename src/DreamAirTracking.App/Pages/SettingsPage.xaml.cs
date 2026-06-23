// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System.Diagnostics;
using DreamAirTracking.App.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace DreamAirTracking.App.Pages;

public sealed partial class SettingsPage : Page
{
    private CancellationTokenSource? _modelDownloadCts;

    public SettingsPage()
    {
        InitializeComponent();
        ModelFolderTextBox.Text = HuggingFaceModelDownloadService.Instance.ModelRoot;
    }

    private async void DownloadModel_Click(object sender, RoutedEventArgs e)
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
            SetModelDownloadUi(
                result.Success ? InfoBarSeverity.Success : InfoBarSeverity.Error,
                result.Success ? "Model installed" : "Model download failed",
                result.Message,
                result.Success ? 1 : null,
                isDownloading: false);
        }
        finally
        {
            _modelDownloadCts?.Dispose();
            _modelDownloadCts = null;
            DownloadModelButton.IsEnabled = true;
            CancelModelDownloadButton.Visibility = Visibility.Collapsed;
        }
    }

    private void CancelModelDownload_Click(object sender, RoutedEventArgs e)
    {
        _modelDownloadCts?.Cancel();
    }

    private void OpenModelFolder_Click(object sender, RoutedEventArgs e)
    {
        Directory.CreateDirectory(HuggingFaceModelDownloadService.Instance.ModelRoot);
        Process.Start(new ProcessStartInfo
        {
            FileName = HuggingFaceModelDownloadService.Instance.ModelRoot,
            UseShellExecute = true
        });
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
        DownloadModelButton.IsEnabled = !isDownloading;
        CancelModelDownloadButton.Visibility = isDownloading ? Visibility.Visible : Visibility.Collapsed;
    }
}
