using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using DreamAirTracking.App.Pages;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace DreamAirTracking.App;

public sealed partial class MainWindow : Window
{
    public event EventHandler? CalibrationOverlayCancelRequested;

    public MainWindow()
    {
        InitializeComponent();

        NavFrame.Navigate(typeof(HomePage));
    }

    public void ShowCalibrationOverlay(
        string stage,
        string instruction,
        string countdown,
        string status,
        double progressValue,
        bool showProgress)
    {
        CalibrationOverlayStageText.Text = stage;
        CalibrationOverlayInstructionText.Text = instruction;
        CalibrationOverlayCountdownText.Text = countdown;
        CalibrationOverlayStatusText.Text = status;
        CalibrationOverlayProgressBar.Value = progressValue;
        CalibrationOverlayProgressBar.Visibility = showProgress ? Visibility.Visible : Visibility.Collapsed;
        CalibrationOverlayHost.Visibility = Visibility.Visible;
    }

    public void UpdateCalibrationOverlayProgress(double progressValue, string status)
    {
        CalibrationOverlayProgressBar.Value = progressValue;
        CalibrationOverlayStatusText.Text = status;
    }

    public void HideCalibrationOverlay()
    {
        CalibrationOverlayHost.Visibility = Visibility.Collapsed;
    }

    private void CalibrationOverlayCancel_Click(object sender, RoutedEventArgs e)
    {
        CalibrationOverlayCancelRequested?.Invoke(this, EventArgs.Empty);
    }

    private void NavButton_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not Button { Tag: string tag })
        {
            return;
        }

        switch (tag)
        {
            case "home":
                NavFrame.Navigate(typeof(HomePage));
                break;
            case "calibration":
                NavFrame.Navigate(typeof(CalibrationPage));
                break;
            case "pipeline":
                NavFrame.Navigate(typeof(PipelineDiagnosticsPage));
                break;
            case "about":
                NavFrame.Navigate(typeof(AboutPage));
                break;
            case "settings":
                NavFrame.Navigate(typeof(SettingsPage));
                break;
            default:
                throw new InvalidOperationException($"Unknown navigation item tag: {tag}");
        }
    }
}
