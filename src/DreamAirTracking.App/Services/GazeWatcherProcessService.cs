using System.Diagnostics;

namespace DreamAirTracking.App.Services;

public sealed class GazeWatcherProcessService
{
    public static GazeWatcherProcessService Instance { get; } = new();

    private Process? _process;

    private GazeWatcherProcessService()
    {
    }

    public event EventHandler? StateChanged;

    public bool IsRunning => _process is { HasExited: false };

    public string LastOutput { get; private set; } = "Watcher has not been started.";

    public string OutputRoot => Path.Combine(FindRepoRoot(), "runs", "gaze_watch");

    public string StatusPath => Path.Combine(OutputRoot, "watch_status.json");

    public void Start(bool runImmediately = true, int epochs = 60, int intervalSeconds = 20)
    {
        if (IsRunning)
        {
            LastOutput = "Watcher is already running.";
            StateChanged?.Invoke(this, EventArgs.Empty);
            return;
        }

        var repoRoot = FindRepoRoot();
        Directory.CreateDirectory(OutputRoot);
        var scriptPath = Path.Combine(repoRoot, "scripts", "ml", "watch_gaze_pipeline.py");
        var startInfo = new ProcessStartInfo
        {
            FileName = "python",
            WorkingDirectory = repoRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        startInfo.ArgumentList.Add(scriptPath);
        startInfo.ArgumentList.Add("--output-root");
        startInfo.ArgumentList.Add(OutputRoot);
        startInfo.ArgumentList.Add("--epochs");
        startInfo.ArgumentList.Add(epochs.ToString());
        startInfo.ArgumentList.Add("--interval-seconds");
        startInfo.ArgumentList.Add(intervalSeconds.ToString());
        if (runImmediately)
        {
            startInfo.ArgumentList.Add("--run-immediately");
        }

        var process = new Process
        {
            StartInfo = startInfo,
            EnableRaisingEvents = true
        };
        process.OutputDataReceived += (_, args) => CaptureLine(args.Data);
        process.ErrorDataReceived += (_, args) => CaptureLine(args.Data);
        process.Exited += (_, _) =>
        {
            LastOutput = process.ExitCode == 0
                ? "Watcher completed successfully."
                : $"{LastOutput} Watcher stopped with exit code {process.ExitCode}.";
            if (ReferenceEquals(_process, process))
            {
                _process = null;
            }

            StateChanged?.Invoke(this, EventArgs.Empty);
            process.Dispose();
        };

        try
        {
            if (!process.Start())
            {
                LastOutput = "Watcher failed to start.";
                StateChanged?.Invoke(this, EventArgs.Empty);
                return;
            }

            _process = process;
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            LastOutput = $"Watcher running. Status: {StatusPath}";
        }
        catch (Exception ex)
        {
            LastOutput = $"Watcher failed to start: {ex.Message}";
        }

        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    public void Stop()
    {
        if (_process is { HasExited: false })
        {
            try
            {
                _process.Kill(entireProcessTree: true);
                LastOutput = "Watcher stopped.";
            }
            catch (Exception ex)
            {
                LastOutput = $"Could not stop watcher: {ex.Message}";
            }
        }
        else
        {
            LastOutput = "No watcher process was running.";
        }

        StateChanged?.Invoke(this, EventArgs.Empty);
    }

    private void CaptureLine(string? line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return;
        }

        LastOutput = line;
        StateChanged?.Invoke(this, EventArgs.Empty);
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
}
