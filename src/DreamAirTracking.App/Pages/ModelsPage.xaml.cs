using System.Diagnostics;
using DreamAirTracking.App.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;

namespace DreamAirTracking.App.Pages;

public sealed partial class ModelsPage : Page
{
    private RemoteModelCatalog? _catalog;
    private CancellationTokenSource? _operationCts;

    public ModelsPage()
    {
        InitializeComponent();
    }

    private void Page_Loaded(object sender, RoutedEventArgs e)
    {
        RefreshInstalledList();
    }

    private async void RefreshRemote_Click(object sender, RoutedEventArgs e)
    {
        if (_operationCts is not null)
        {
            return;
        }

        _operationCts = new CancellationTokenSource();
        SetBusyUi(true);
        ShowStatus(InfoBarSeverity.Informational, "Remote models", "Loading model list from Hugging Face...", null, true);
        try
        {
            _catalog = await HuggingFaceModelDownloadService.Instance.RefreshCatalogAsync(_operationCts.Token);
            RenderRemotePackages();
            ShowStatus(
                InfoBarSeverity.Success,
                "Remote models loaded",
                $"Loaded {_catalog.Packages.Count} paired model version(s) from Hugging Face.",
                null,
                false);
        }
        catch (OperationCanceledException)
        {
            ShowStatus(InfoBarSeverity.Warning, "Remote refresh canceled", "Model list refresh was canceled.", null, false);
        }
        catch (Exception ex)
        {
            ShowStatus(
                InfoBarSeverity.Error,
                "Remote refresh failed",
                $"Could not load model list from Hugging Face. {ex.Message}",
                null,
                false);
        }
        finally
        {
            _operationCts?.Dispose();
            _operationCts = null;
            SetBusyUi(false);
        }
    }

    private async void DownloadPackage_Click(object sender, RoutedEventArgs e)
    {
        if (_operationCts is not null || sender is not Button { Tag: RemoteModelPackage package })
        {
            return;
        }

        _operationCts = new CancellationTokenSource();
        SetBusyUi(true);
        ShowStatus(InfoBarSeverity.Informational, "Model download", $"Downloading {package.DisplayName}...", null, true);
        var progress = new Progress<ModelDownloadProgress>(UpdateDownloadProgress);
        try
        {
            var result = await HuggingFaceModelDownloadService.Instance.DownloadPackageAsync(
                package,
                progress,
                _operationCts.Token);
            ShowStatus(
                result.Success ? InfoBarSeverity.Success : InfoBarSeverity.Error,
                result.Success ? "Model installed" : "Model download failed",
                result.Message,
                result.Success ? 1 : null,
                false);
            RefreshInstalledList();
            RenderRemotePackages();
        }
        finally
        {
            _operationCts?.Dispose();
            _operationCts = null;
            SetBusyUi(false);
        }
    }

    private void CancelDownload_Click(object sender, RoutedEventArgs e)
    {
        _operationCts?.Cancel();
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

    private void OpenHome_Click(object sender, RoutedEventArgs e)
    {
        Frame.Navigate(typeof(HomePage));
    }

    private void RenderRemotePackages()
    {
        RemotePackagesList.Items.Clear();
        if (_catalog is null)
        {
            RemoteSummaryText.Text = "Refresh remote list to load available Hugging Face model versions.";
            return;
        }

        RemoteSummaryText.Text =
            $"Repo {_catalog.RepoId}, revision {ShortRevision(_catalog.Revision)}, updated {FormatDate(_catalog.LastModified)}.";
        var installed = HuggingFaceModelDownloadService.Instance.LoadInstalledPackages()
            .ToDictionary(model => model.Id, StringComparer.OrdinalIgnoreCase);
        foreach (var package in _catalog.Packages)
        {
            RemotePackagesList.Items.Add(BuildRemotePackageItem(package, installed.ContainsKey(package.MainModelId)));
        }
    }

    private ListViewItem BuildRemotePackageItem(RemoteModelPackage package, bool installed)
    {
        var root = new Grid
        {
            Padding = new Thickness(0, 8, 0, 8),
            ColumnSpacing = 12
        };
        root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        root.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

        var text = new StackPanel { Spacing = 3 };
        text.Children.Add(new TextBlock
        {
            Text = package.DisplayName,
            FontWeight = Microsoft.UI.Text.FontWeights.SemiBold
        });
        // Version and date are distinct: "Version <tag>" then the publish date, kept separate.
        var versionParts = new List<string>();
        if (!string.IsNullOrWhiteSpace(package.Version))
        {
            versionParts.Add($"Version {package.Version}");
        }
        if (!string.IsNullOrWhiteSpace(package.PublishedDate))
        {
            versionParts.Add(package.PublishedDate);
        }
        if (versionParts.Count > 0)
        {
            text.Children.Add(new TextBlock
            {
                Text = string.Join("  ·  ", versionParts),
                Foreground = SecondaryBrush(),
                TextWrapping = TextWrapping.Wrap
            });
        }
        text.Children.Add(new TextBlock
        {
            Text = $"Architecture: {package.Architecture}  |  files={package.Files.Count}",
            Foreground = SecondaryBrush(),
            TextWrapping = TextWrapping.Wrap
        });

        var button = new Button
        {
            Content = installed ? "Download again" : "Download",
            Tag = package,
            MinWidth = 120
        };
        button.Click += DownloadPackage_Click;

        Grid.SetColumn(text, 0);
        Grid.SetColumn(button, 1);
        root.Children.Add(text);
        root.Children.Add(button);

        return new ListViewItem { Content = root };
    }

    private void RefreshInstalledList()
    {
        InstalledPackagesList.Items.Clear();
        var installed = HuggingFaceModelDownloadService.Instance.LoadInstalledPackages();
        InstalledSummaryText.Text = installed.Count == 0
            ? "No model package installed. Refresh remote list, then download a model."
            : $"{installed.Count} installed paired model version(s). Select one on Home before starting eye tracking.";
        foreach (var package in installed)
        {
            InstalledPackagesList.Items.Add(BuildInstalledPackageItem(package));
        }
    }

    private static ListViewItem BuildInstalledPackageItem(InstalledModelPackage package)
    {
        var root = new StackPanel
        {
            Padding = new Thickness(0, 8, 0, 8),
            Spacing = 3
        };
        root.Children.Add(new TextBlock
        {
            Text = package.DisplayName,
            FontWeight = Microsoft.UI.Text.FontWeights.SemiBold
        });
        root.Children.Add(new TextBlock
        {
            Text = $"status={(package.IsComplete ? "ready" : "missing files")}{(package.IsDefault ? "  |  default" : string.Empty)}",
            TextWrapping = TextWrapping.Wrap
        });
        root.Children.Add(new TextBlock
        {
            Text = $"main: {package.MainOnnxPath}",
            TextWrapping = TextWrapping.Wrap
        });
        if (!string.IsNullOrWhiteSpace(package.ExpressionOnnxPath))
        {
            root.Children.Add(new TextBlock
            {
                Text = $"expression: {package.ExpressionOnnxPath}",
                TextWrapping = TextWrapping.Wrap
            });
        }
        return new ListViewItem { Content = root };
    }

    private void UpdateDownloadProgress(ModelDownloadProgress progress)
    {
        var severity = progress.Stage.Equals("failed", StringComparison.OrdinalIgnoreCase)
            ? InfoBarSeverity.Error
            : progress.Stage.Equals("done", StringComparison.OrdinalIgnoreCase)
                ? InfoBarSeverity.Success
                : InfoBarSeverity.Informational;
        ShowStatus(
            severity,
            progress.Stage.Equals("done", StringComparison.OrdinalIgnoreCase) ? "Model installed" : "Model download",
            progress.Detail,
            progress.Fraction,
            _operationCts is not null);
    }

    private void ShowStatus(
        InfoBarSeverity severity,
        string title,
        string message,
        double? progress,
        bool isBusy)
    {
        StatusInfoBar.IsOpen = true;
        StatusInfoBar.Severity = severity;
        StatusInfoBar.Title = title;
        StatusMessageText.Text = message;
        DownloadProgressBar.IsIndeterminate = isBusy && progress is null;
        DownloadProgressBar.Value = progress ?? 0;
        DownloadProgressBar.Visibility = isBusy || progress is not null
            ? Visibility.Visible
            : Visibility.Collapsed;
    }

    private void SetBusyUi(bool isBusy)
    {
        RefreshRemoteButton.IsEnabled = !isBusy;
        CancelDownloadButton.Visibility = isBusy ? Visibility.Visible : Visibility.Collapsed;
        foreach (var item in RemotePackagesList.Items.OfType<ListViewItem>())
        {
            if (item.Content is Grid grid)
            {
                foreach (var button in grid.Children.OfType<Button>())
                {
                    button.IsEnabled = !isBusy;
                }
            }
        }
    }

    private Brush SecondaryBrush()
        => (Brush)Application.Current.Resources["TextFillColorSecondaryBrush"];

    private static string ShortRevision(string revision)
        => revision.Length <= 8 ? revision : revision[..8];

    private static string FormatDate(DateTimeOffset? date)
        => date is null ? "unknown" : date.Value.ToString("yyyy-MM-dd HH:mm 'UTC'");
}
