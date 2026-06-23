using System.IO.Compression;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using DreamAirTracking.Core.Models;

namespace DreamAirTracking.App.Services;

public sealed class HuggingFaceModelDownloadService
{
    public const string DefaultRepoId = "Sumirui/dreamairtracking-dreamair-main-current";
    public const string DefaultRevision = "main";

    private static readonly HttpClient Http = CreateHttpClient();

    public static HuggingFaceModelDownloadService Instance { get; } = new();

    private HuggingFaceModelDownloadService()
    {
    }

    public bool IsDownloading { get; private set; }

    public string ModelRoot =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "DreamAirTracking",
            "models");

    public async Task<ModelDownloadResult> InstallDefaultModelAsync(
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (IsDownloading)
        {
            return new ModelDownloadResult(false, "A model download is already running.");
        }

        IsDownloading = true;
        var operationId = DateTime.UtcNow.ToString("yyyyMMddHHmmss");
        var appData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "DreamAirTracking");
        var downloadRoot = Path.Combine(appData, "model_downloads");
        var zipPath = Path.Combine(downloadRoot, $"dreamair-model-{operationId}.zip");
        var extractPath = Path.Combine(downloadRoot, $"dreamair-model-{operationId}");

        try
        {
            Directory.CreateDirectory(downloadRoot);
            Directory.CreateDirectory(ModelRoot);
            progress?.Report(new ModelDownloadProgress("download", 0, "Connecting to Hugging Face..."));

            var url = BuildArchiveUrl(DefaultRepoId, DefaultRevision);
            await DownloadAsync(url, zipPath, progress, cancellationToken);

            progress?.Report(new ModelDownloadProgress("extract", null, "Extracting model package..."));
            if (Directory.Exists(extractPath))
            {
                Directory.Delete(extractPath, recursive: true);
            }

            ZipFile.ExtractToDirectory(zipPath, extractPath);
            var packageRoot = FindExtractedPackageRoot(extractPath);
            ValidatePackage(packageRoot);

            progress?.Report(new ModelDownloadProgress("install", null, "Installing model package..."));
            InstallPackage(packageRoot, ModelRoot);

            var statusPath = Path.Combine(appData, "model_download_status.json");
            await WriteStatusAsync(statusPath, new
            {
                state = "installed",
                repoId = DefaultRepoId,
                revision = DefaultRevision,
                installedRegistryPath = Path.Combine(ModelRoot, "model_registry.json"),
                timeUtc = DateTime.UtcNow
            }, cancellationToken);

            progress?.Report(new ModelDownloadProgress("done", 1, "Model package installed."));
            return new ModelDownloadResult(
                true,
                "Model package installed. Select it from Model package, then start eye tracking.",
                Path.Combine(ModelRoot, "model_registry.json"));
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return await FailAsync("Model download was canceled. Your current installed model was not changed.", "canceled");
        }
        catch (OperationCanceledException)
        {
            return await FailAsync("Model download timed out. Your current installed model was not changed.", "timeout");
        }
        catch (HttpRequestException ex)
        {
            var message = ex.StatusCode switch
            {
                HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden =>
                    "Model download failed. This Hugging Face model source requires access permission.",
                HttpStatusCode.NotFound =>
                    "Model download failed. The Hugging Face model source or revision was not found.",
                _ =>
                    "Model download failed. Could not connect to Hugging Face. Your current installed model was not changed."
            };
            return await FailAsync($"{message} Details: {ex.Message}", "network");
        }
        catch (InvalidDataException ex)
        {
            return await FailAsync($"Model package is invalid. {ex.Message}", "invalid_package");
        }
        catch (IOException ex)
        {
            return await FailAsync($"Model install failed because a file could not be written. {ex.Message}", "io");
        }
        catch (Exception ex)
        {
            return await FailAsync($"Model install failed. {ex.Message}", "unknown");
        }
        finally
        {
            TryDelete(zipPath);
            TryDeleteDirectory(extractPath);
            IsDownloading = false;
        }

        async Task<ModelDownloadResult> FailAsync(string message, string errorCode)
        {
            var statusPath = Path.Combine(appData, "model_download_status.json");
            await WriteStatusAsync(statusPath, new
            {
                state = "failed",
                repoId = DefaultRepoId,
                revision = DefaultRevision,
                errorCode,
                message,
                currentModelPreserved = true,
                timeUtc = DateTime.UtcNow
            }, CancellationToken.None);
            progress?.Report(new ModelDownloadProgress("failed", null, message));
            return new ModelDownloadResult(false, message);
        }
    }

    private static async Task DownloadAsync(
        string url,
        string zipPath,
        IProgress<ModelDownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        using var response = await Http.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength;
        await using var remote = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var local = File.Create(zipPath);

        var buffer = new byte[1024 * 128];
        long readTotal = 0;
        while (true)
        {
            var read = await remote.ReadAsync(buffer, cancellationToken);
            if (read == 0)
            {
                break;
            }

            await local.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
            readTotal += read;
            double? fraction = totalBytes is > 0 ? (double)readTotal / totalBytes.Value : null;
            progress?.Report(new ModelDownloadProgress("download", fraction, $"Downloading model package {FormatBytes(readTotal)}..."));
        }

        if (readTotal == 0)
        {
            throw new InvalidDataException("Downloaded file is empty.");
        }
    }

    private static void ValidatePackage(string packageRoot)
    {
        var registryPath = Path.Combine(packageRoot, "model_registry.json");
        if (!File.Exists(registryPath))
        {
            throw new InvalidDataException("Missing model_registry.json.");
        }

        var registry = ModelRegistry.TryLoad(registryPath, Directory.GetParent(packageRoot)?.FullName);
        if (registry is null || registry.Models.Count == 0)
        {
            throw new InvalidDataException("Downloaded model_registry.json is invalid or empty.");
        }

        foreach (var model in registry.Models)
        {
            RequirePackageFile(packageRoot, model.Onnx, "model.onnx");
            RequirePackageFile(packageRoot, model.Metadata, "metadata.json");
            RequirePackageFile(packageRoot, model.RuntimeDefaults, "runtime_defaults.json");
            RequirePackageFile(packageRoot, model.Acceptance, "acceptance.json");
        }
    }

    private static void RequirePackageFile(string packageRoot, string registryPath, string label)
    {
        if (string.IsNullOrWhiteSpace(registryPath))
        {
            throw new InvalidDataException($"Model registry is missing {label} path.");
        }

        var relative = StripModelsPrefix(registryPath);
        var filePath = Path.Combine(packageRoot, relative.Replace('/', Path.DirectorySeparatorChar));
        if (!File.Exists(filePath))
        {
            throw new InvalidDataException($"Missing {label}: {relative}.");
        }
    }

    private static void InstallPackage(string packageRoot, string modelRoot)
    {
        Directory.CreateDirectory(modelRoot);
        var registrySource = Path.Combine(packageRoot, "model_registry.json");
        var registryTarget = Path.Combine(modelRoot, "model_registry.json");
        var registryTemp = Path.Combine(modelRoot, "model_registry.json.tmp");

        foreach (var entry in Directory.EnumerateFileSystemEntries(packageRoot))
        {
            var name = Path.GetFileName(entry);
            if (name.Equals("model_registry.json", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var target = Path.Combine(modelRoot, name);
            if (Directory.Exists(entry))
            {
                CopyDirectory(entry, target);
            }
            else
            {
                File.Copy(entry, target, overwrite: true);
            }
        }

        File.Copy(registrySource, registryTemp, overwrite: true);
        File.Copy(registryTemp, registryTarget, overwrite: true);
        File.Delete(registryTemp);
    }

    private static void CopyDirectory(string source, string target)
    {
        Directory.CreateDirectory(target);
        foreach (var file in Directory.EnumerateFiles(source))
        {
            File.Copy(file, Path.Combine(target, Path.GetFileName(file)), overwrite: true);
        }

        foreach (var directory in Directory.EnumerateDirectories(source))
        {
            CopyDirectory(directory, Path.Combine(target, Path.GetFileName(directory)));
        }
    }

    private static string FindExtractedPackageRoot(string extractPath)
    {
        if (File.Exists(Path.Combine(extractPath, "model_registry.json")))
        {
            return extractPath;
        }

        var child = Directory.EnumerateDirectories(extractPath).FirstOrDefault(directory =>
            File.Exists(Path.Combine(directory, "model_registry.json")));
        if (child is null)
        {
            throw new InvalidDataException("Downloaded archive does not contain model_registry.json.");
        }

        return child;
    }

    private static string StripModelsPrefix(string path)
    {
        var normalized = path.Replace('\\', '/').TrimStart('/');
        return normalized.StartsWith("models/", StringComparison.OrdinalIgnoreCase)
            ? normalized["models/".Length..]
            : normalized;
    }

    private static async Task WriteStatusAsync(string path, object payload, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var json = JsonSerializer.Serialize(payload, AppJsonOptions.Web());
        await File.WriteAllTextAsync(path, json, cancellationToken);
    }

    private static string BuildArchiveUrl(string repoId, string revision)
        => $"https://huggingface.co/{repoId}/archive/{Uri.EscapeDataString(revision)}.zip";

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(8)
        };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("DreamAirTracking/0.1");
        return client;
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes >= 1024L * 1024L)
        {
            return $"{bytes / 1024d / 1024d:0.0} MB";
        }

        if (bytes >= 1024)
        {
            return $"{bytes / 1024d:0.0} KB";
        }

        return $"{bytes} B";
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
        }
    }

    private static void TryDeleteDirectory(string path)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
            }
        }
        catch
        {
        }
    }
}
