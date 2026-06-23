using System.Net;
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

    public async Task<RemoteModelCatalog> RefreshCatalogAsync(CancellationToken cancellationToken = default)
    {
        using var metadataDocument = await GetJsonAsync(
            $"https://huggingface.co/api/models/{DefaultRepoId}",
            cancellationToken);
        var metadata = metadataDocument.RootElement;
        var revision = metadata.TryGetProperty("sha", out var shaProperty)
            ? shaProperty.GetString()
            : null;
        if (string.IsNullOrWhiteSpace(revision))
        {
            throw new InvalidDataException("Hugging Face model metadata did not include a commit revision.");
        }

        DateTimeOffset? lastModified = null;
        if (metadata.TryGetProperty("lastModified", out var lastModifiedProperty) &&
            DateTimeOffset.TryParse(lastModifiedProperty.GetString(), out var parsedLastModified))
        {
            lastModified = parsedLastModified;
        }

        var registry = await DownloadRegistryAsync(revision, cancellationToken);
        var siblingNames = ReadSiblingNames(metadata);
        var packages = registry.Models
            .Where(model =>
                model.DeviceFamily.Equals("Dream Air", StringComparison.OrdinalIgnoreCase) &&
                model.Runtime.Equals("predict_live_multitask", StringComparison.OrdinalIgnoreCase))
            .Select(model => CreateRemotePackage(model, revision, lastModified, siblingNames))
            .ToArray();

        if (packages.Length == 0)
        {
            throw new InvalidDataException("No Dream Air model packages were listed in the Hugging Face model registry.");
        }

        return new RemoteModelCatalog(DefaultRepoId, revision, lastModified, packages);
    }

    public IReadOnlyList<InstalledModelPackage> LoadInstalledPackages()
    {
        var registry = ModelRegistry.TryLoad(ModelRegistry.DefaultUserRegistryPath());
        if (registry is null)
        {
            return Array.Empty<InstalledModelPackage>();
        }

        return registry.Models
            .Where(model =>
                model.DeviceFamily.Equals("Dream Air", StringComparison.OrdinalIgnoreCase) &&
                model.Runtime.Equals("predict_live_multitask", StringComparison.OrdinalIgnoreCase))
            .Select(model => new InstalledModelPackage(
                model.Id,
                model.DisplayName,
                model.Role,
                model.Runtime,
                model.ResolvedOnnx,
                IsInstalledModelComplete(model),
                model.Default))
            .ToArray();
    }

    public async Task<ModelDownloadResult> DownloadPackageAsync(
        RemoteModelPackage package,
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
        var downloadRoot = Path.Combine(appData, "model_downloads", operationId);

        try
        {
            Directory.CreateDirectory(downloadRoot);
            Directory.CreateDirectory(ModelRoot);

            progress?.Report(new ModelDownloadProgress("download", 0, $"Downloading {package.DisplayName}..."));
            var remoteRegistry = await DownloadRegistryAsync(package.Revision, cancellationToken);
            var selectedEntry = remoteRegistry.FindById(package.Id)
                ?? throw new InvalidDataException($"Remote registry no longer contains {package.Id}.");

            var files = package.Files.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
            for (var index = 0; index < files.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var file = files[index];
                var fraction = files.Length == 0 ? 0 : (double)index / files.Length;
                progress?.Report(new ModelDownloadProgress("download", fraction, $"Downloading {file}..."));
                await DownloadFileAsync(file, package.Revision, downloadRoot, cancellationToken);
            }

            ValidateDownloadedPackage(downloadRoot, selectedEntry);
            progress?.Report(new ModelDownloadProgress("install", null, "Installing model package..."));
            InstallDownloadedPackage(downloadRoot, selectedEntry);

            var statusPath = Path.Combine(appData, "model_download_status.json");
            await WriteStatusAsync(statusPath, new
            {
                state = "installed",
                repoId = DefaultRepoId,
                revision = package.Revision,
                modelId = package.Id,
                modelDisplayName = package.DisplayName,
                installedRegistryPath = Path.Combine(ModelRoot, "model_registry.json"),
                timeUtc = DateTime.UtcNow
            }, cancellationToken);

            progress?.Report(new ModelDownloadProgress("done", 1, $"{package.DisplayName} installed."));
            return new ModelDownloadResult(
                true,
                $"{package.DisplayName} installed. Select it from Home -> Model package.",
                Path.Combine(ModelRoot, "model_registry.json"));
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return await FailAsync("Model download was canceled. Current installed models were not changed.", "canceled");
        }
        catch (OperationCanceledException)
        {
            return await FailAsync("Model download timed out. Current installed models were not changed.", "timeout");
        }
        catch (HttpRequestException ex)
        {
            var message = ex.StatusCode switch
            {
                HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden =>
                    "Model download failed. This Hugging Face model source requires access permission.",
                HttpStatusCode.NotFound =>
                    "Model download failed. The selected model file or revision was not found on Hugging Face.",
                _ =>
                    "Model download failed. Could not connect to Hugging Face. Current installed models were not changed."
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
            TryDeleteDirectory(downloadRoot);
            IsDownloading = false;
        }

        async Task<ModelDownloadResult> FailAsync(string message, string errorCode)
        {
            var statusPath = Path.Combine(appData, "model_download_status.json");
            await WriteStatusAsync(statusPath, new
            {
                state = "failed",
                repoId = DefaultRepoId,
                revision = package.Revision,
                modelId = package.Id,
                errorCode,
                message,
                currentModelPreserved = true,
                timeUtc = DateTime.UtcNow
            }, CancellationToken.None);
            progress?.Report(new ModelDownloadProgress("failed", null, message));
            return new ModelDownloadResult(false, message);
        }
    }

    private static RemoteModelPackage CreateRemotePackage(
        ModelRegistryEntry model,
        string revision,
        DateTimeOffset? lastModified,
        HashSet<string> siblingNames)
    {
        var files = new List<string>
        {
            RemotePath(model.Onnx),
            RemotePath(model.Metadata),
            RemotePath(model.RuntimeDefaults),
            RemotePath(model.Acceptance)
        };
        var modelCard = Path.Combine(Path.GetDirectoryName(RemotePath(model.Onnx)) ?? string.Empty, "model_card.md")
            .Replace('\\', '/');
        if (siblingNames.Contains(modelCard))
        {
            files.Add(modelCard);
        }

        var version = lastModified is null
            ? $"revision {ShortRevision(revision)}"
            : $"{lastModified.Value:yyyy.MM.dd} ({ShortRevision(revision)})";

        return new RemoteModelPackage(
            model.Id,
            model.DisplayName,
            version,
            model.Role,
            model.Runtime,
            model.Architecture,
            revision,
            lastModified,
            model.Outputs,
            files.Where(file => !string.IsNullOrWhiteSpace(file)).Distinct(StringComparer.OrdinalIgnoreCase).ToArray());
    }

    private static async Task<JsonDocument> GetJsonAsync(string url, CancellationToken cancellationToken)
    {
        using var response = await Http.GetAsync(url, cancellationToken);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        return await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
    }

    private static async Task<ModelRegistry> DownloadRegistryAsync(string revision, CancellationToken cancellationToken)
    {
        var registryUrl = ResolveUrl("model_registry.json", revision);
        using var response = await Http.GetAsync(registryUrl, cancellationToken);
        response.EnsureSuccessStatusCode();
        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        var registry = JsonSerializer.Deserialize<ModelRegistry>(json, AppJsonOptions.Web());
        if (registry is null ||
            !registry.Schema.Equals("dream_air_tracking.model_registry.v1", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("Downloaded model_registry.json is invalid.");
        }

        return registry;
    }

    private static async Task DownloadFileAsync(
        string remotePath,
        string revision,
        string downloadRoot,
        CancellationToken cancellationToken)
    {
        var targetPath = Path.Combine(downloadRoot, remotePath.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(targetPath)!);
        using var response = await Http.GetAsync(ResolveUrl(remotePath, revision), HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();
        await using var remote = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var local = File.Create(targetPath);
        await remote.CopyToAsync(local, cancellationToken);
        if (local.Length == 0)
        {
            throw new InvalidDataException($"Downloaded file is empty: {remotePath}.");
        }
    }

    private static void ValidateDownloadedPackage(string downloadRoot, ModelRegistryEntry entry)
    {
        RequireDownloadedFile(downloadRoot, entry.Onnx, "model.onnx");
        RequireDownloadedFile(downloadRoot, entry.Metadata, "metadata.json");
        RequireDownloadedFile(downloadRoot, entry.RuntimeDefaults, "runtime_defaults.json");
        RequireDownloadedFile(downloadRoot, entry.Acceptance, "acceptance.json");
    }

    private static void RequireDownloadedFile(string downloadRoot, string registryPath, string label)
    {
        var remotePath = RemotePath(registryPath);
        if (string.IsNullOrWhiteSpace(remotePath))
        {
            throw new InvalidDataException($"Model registry is missing {label} path.");
        }

        var filePath = Path.Combine(downloadRoot, remotePath.Replace('/', Path.DirectorySeparatorChar));
        if (!File.Exists(filePath))
        {
            throw new InvalidDataException($"Missing {label}: {remotePath}.");
        }
    }

    private void InstallDownloadedPackage(string downloadRoot, ModelRegistryEntry entry)
    {
        foreach (var path in Directory.EnumerateFiles(downloadRoot, "*", SearchOption.AllDirectories))
        {
            var relative = Path.GetRelativePath(downloadRoot, path);
            var target = Path.Combine(ModelRoot, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(path, target, overwrite: true);
        }

        var registryPath = ModelRegistry.DefaultUserRegistryPath();
        var existing = ModelRegistry.TryLoad(registryPath);
        var registry = new ModelRegistry();
        if (existing is not null)
        {
            registry.Models.AddRange(existing.Models.Where(model =>
                !model.Id.Equals(entry.Id, StringComparison.OrdinalIgnoreCase)));
        }

        var installedEntry = CloneRegistryEntry(entry);
        registry.Models.Add(installedEntry);
        if (installedEntry.Role.Equals("main", StringComparison.OrdinalIgnoreCase) && installedEntry.Default)
        {
            foreach (var otherMain in registry.Models.Where(model =>
                         !model.Id.Equals(installedEntry.Id, StringComparison.OrdinalIgnoreCase) &&
                         model.Role.Equals("main", StringComparison.OrdinalIgnoreCase)))
            {
                otherMain.Default = false;
            }
        }

        if (!registry.Models.Any(model => model.Role.Equals("main", StringComparison.OrdinalIgnoreCase) && model.Default))
        {
            var firstMain = registry.Models.FirstOrDefault(model => model.Role.Equals("main", StringComparison.OrdinalIgnoreCase));
            if (firstMain is not null)
            {
                firstMain.Default = true;
            }
        }

        Directory.CreateDirectory(Path.GetDirectoryName(registryPath)!);
        var tempPath = registryPath + ".tmp";
        var json = JsonSerializer.Serialize(registry, AppJsonOptions.Web());
        File.WriteAllText(tempPath, json);
        File.Copy(tempPath, registryPath, overwrite: true);
        File.Delete(tempPath);
    }

    private static ModelRegistryEntry CloneRegistryEntry(ModelRegistryEntry entry)
        => new()
        {
            Id = entry.Id,
            DisplayName = entry.DisplayName,
            DeviceFamily = entry.DeviceFamily,
            Role = entry.Role,
            Architecture = entry.Architecture,
            Runtime = entry.Runtime,
            Onnx = entry.Onnx,
            Metadata = entry.Metadata,
            RuntimeDefaults = entry.RuntimeDefaults,
            Acceptance = entry.Acceptance,
            Outputs = entry.Outputs.ToList(),
            Default = entry.Default
        };

    private static bool IsInstalledModelComplete(ModelRegistryEntry model)
        => File.Exists(model.ResolvedOnnx) &&
           File.Exists(model.ResolvedMetadata) &&
           File.Exists(model.ResolvedRuntimeDefaults) &&
           File.Exists(model.ResolvedAcceptance);

    private static HashSet<string> ReadSiblingNames(JsonElement metadata)
    {
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (!metadata.TryGetProperty("siblings", out var siblings) || siblings.ValueKind != JsonValueKind.Array)
        {
            return names;
        }

        foreach (var sibling in siblings.EnumerateArray())
        {
            if (sibling.TryGetProperty("rfilename", out var nameProperty) &&
                !string.IsNullOrWhiteSpace(nameProperty.GetString()))
            {
                names.Add(nameProperty.GetString()!);
            }
        }

        return names;
    }

    private static string ResolveUrl(string remotePath, string revision)
        => $"https://huggingface.co/{DefaultRepoId}/resolve/{Uri.EscapeDataString(revision)}/{EscapeRemotePath(remotePath)}";

    private static string EscapeRemotePath(string remotePath)
        => string.Join("/", remotePath.Split('/', StringSplitOptions.RemoveEmptyEntries).Select(Uri.EscapeDataString));

    private static string RemotePath(string registryPath)
    {
        var normalized = registryPath.Replace('\\', '/').TrimStart('/');
        return normalized.StartsWith("models/", StringComparison.OrdinalIgnoreCase)
            ? normalized["models/".Length..]
            : normalized;
    }

    private static string ShortRevision(string revision)
        => revision.Length <= 8 ? revision : revision[..8];

    private static async Task WriteStatusAsync(string path, object payload, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var json = JsonSerializer.Serialize(payload, AppJsonOptions.Web());
        await File.WriteAllTextAsync(path, json, cancellationToken);
    }

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(8)
        };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("DreamAirTracking/0.1");
        return client;
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
