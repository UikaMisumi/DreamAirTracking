using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using System.Text;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace DreamAirTracking.App;

/// <summary>
/// Provides application-specific behavior to supplement the default Application class.
/// </summary>
public partial class App : Application
{
    private Window? _window;
    public static MainWindow? CurrentWindow { get; private set; }
    
    /// <summary>
    /// Initializes the singleton application object.  This is the first line of authored code
    /// executed, and as such is the logical equivalent of main() or WinMain().
    /// </summary>
    public App()
    {
        InitializeComponent();
        UnhandledException += App_UnhandledException;
        AppDomain.CurrentDomain.UnhandledException += CurrentDomain_UnhandledException;
        TaskScheduler.UnobservedTaskException += TaskScheduler_UnobservedTaskException;
    }

    /// <summary>
    /// Invoked when the application is launched.
    /// </summary>
    /// <param name="args">Details about the launch request and process.</param>
    protected override void OnLaunched(Microsoft.UI.Xaml.LaunchActivatedEventArgs args)
    {
        CurrentWindow = new MainWindow();
        _window = CurrentWindow;
        _window.Activate();
    }

    private static void App_UnhandledException(object sender, Microsoft.UI.Xaml.UnhandledExceptionEventArgs e)
    {
        WriteCrashLog("XAML unhandled exception", e.Exception);
        e.Handled = true;
    }

    private static void CurrentDomain_UnhandledException(object sender, System.UnhandledExceptionEventArgs e)
    {
        WriteCrashLog("AppDomain unhandled exception", e.ExceptionObject as Exception);
    }

    private static void TaskScheduler_UnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
    {
        WriteCrashLog("Task unobserved exception", e.Exception);
        e.SetObserved();
    }

    private static void WriteCrashLog(string source, Exception? exception)
    {
        try
        {
            var appData = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "DreamAirTracking");
            Directory.CreateDirectory(appData);
            var logPath = Path.Combine(appData, "app_crash.log");
            var builder = new StringBuilder();
            builder.AppendLine("============================================================");
            builder.AppendLine($"{DateTimeOffset.Now:O} {source}");
            builder.AppendLine(exception?.ToString() ?? "No Exception object was provided.");
            File.AppendAllText(logPath, builder.ToString());
        }
        catch
        {
        }
    }
}
