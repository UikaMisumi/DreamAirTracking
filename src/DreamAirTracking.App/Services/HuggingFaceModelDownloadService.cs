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
        var packages = CreateRemotePackages(registry, revision, lastModified, siblingNames);

        if (packages.Count == 0)
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

        var expression = registry.FindDefaultExpression();
        return registry.Models
            .Where(model =>
                model.DeviceFamily.Equals("Dream Air", StringComparison.OrdinalIgnoreCase) &&
                model.Runtime.Equals("predict_live_multitask", StringComparison.OrdinalIgnoreCase) &&
                model.Role.Equals("main", StringComparison.OrdinalIgnoreCase))
            .Select(model => new InstalledModelPackage(
                model.Id,
                VersionLabel(model),
                VersionLabel(model),
                model.Runtime,
                model.ResolvedOnnx,
                expression?.ResolvedOnnx,
                IsInstalledModelComplete(model) && expression is not null && IsInstalledModelComplete(expression),
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
            var selectedMain = remoteRegistry.FindById(package.MainModelId)
                ?? throw new InvalidDataException($"Remote registry no longer contains {package.MainModelId}.");
            var selectedExpression = string.IsNullOrWhiteSpace(package.ExpressionModelId)
                ? remoteRegistry.FindDefaultExpression()
                : remoteRegistry.FindById(package.ExpressionModelId);
            if (selectedExpression is null)
            {
                throw new InvalidDataException("Remote registry does not contain the paired expression model.");
            }

            var files = package.Files.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
            for (var index = 0; index < files.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var file = files[index];
                var fraction = files.Length == 0 ? 0 : (double)index / files.Length;
                progress?.Report(new ModelDownloadProgress("download", fraction, $"Downloading {file}..."));
                await DownloadFileAsync(file, package.Revision, downloadRoot, cancellationToken);
            }

            ValidateDownloadedPackage(downloadRoot, selectedMain);
            ValidateDownloadedPackage(downloadRoot, selectedExpression);
            progress?.Report(new ModelDownloadProgress("install", null, "Installing model package..."));
            InstallDownloadedPackage(downloadRoot, package, selectedMain, selectedExpression);

            var statusPath = Path.Combine(appData, "model_download_status.json");
            await WriteStatusAsync(statusPath, new
            {
                state = "installed",
                repoId = DefaultRepoId,
                revision = package.Revision,
                packageId = package.Id,
                modelDisplayName = package.DisplayName,
                mainModelId = selectedMain.Id,
                expressionModelId = selectedExpression.Id,
                installedRegistryPath = Path.Combine(ModelRoot, "model_registry.json"),
                timeUtc = DateTime.UtcNow
            }, cancellationToken);

            progress?.Report(new ModelDownloadProgress("done", 1, $"{package.DisplayName} installed."));
            return new ModelDownloadResult(
                true,
                $"{package.DisplayName} installed. Select this version from Home -> Model package.",
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
                packageId = package.Id,
                errorCode,
                message,
                currentModelPreserved = true,
                timeUtc = DateTime.UtcNow
            }, CancellationToken.None);
            progress?.Report(new ModelDownloadProgress("failed", null, message));
            return new ModelDownloadResult(false, message);
        }
    }

    // One downloadable package per role=main registry entry (each paired with the default
    // expression model), so alternative model versions published in the same repo are
    // individually selectable and downloadable in the app.
    private static IReadOnlyList<RemoteModelPackage> CreateRemotePackages(
        ModelRegistry registry,
        string revision,
        DateTimeOffset? lastModified,
        HashSet<string> siblingNames)
    {
        var expression = registry.FindDefaultExpression();
        if (expression is null)
        {
            return Array.Empty<RemoteModelPackage>();
        }

        var repoDate = lastModified?.ToString("yyyy-MM-dd") ?? string.Empty;

        var packages = new List<RemoteModelPackage>();
        foreach (var main in registry.Models.Where(model =>
                     model.Role.Equals("main", StringComparison.OrdinalIgnoreCase)))
        {
            var files = new List<string>();
            AddModelFiles(files, main, siblingNames);
            AddModelFiles(files, expression, siblingNames);

            // Name / version / date are three separate things: the model's own displayName,
            // its own version tag, and its own publish date — falling back to the id / the
            // repo revision / the repo commit date only when a model doesn't specify its own.
            var displayName = string.IsNullOrWhiteSpace(main.DisplayName) ? main.Id : main.DisplayName;
            var version = string.IsNullOrWhiteSpace(main.Version) ? ShortRevision(revision) : main.Version;
            var publishedDate = string.IsNullOrWhiteSpace(main.UpdatedAt) ? repoDate : main.UpdatedAt;

            packages.Add(new RemoteModelPackage(
                $"{main.Id}+{expression.Id}@{ShortRevision(revision)}",
                displayName,
                version,
                publishedDate,
                main.Id,
                expression.Id,
                main.Runtime,
                $"{main.Architecture} + expression auxiliary",
                revision,
                lastModified,
                main.Outputs.Concat(expression.Outputs).Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
                files.Where(file => !string.IsNullOrWhiteSpace(file)).Distinct(StringComparer.OrdinalIgnoreCase).ToArray()));
        }

        // default main first, then the alternates
        return packages
            .OrderByDescending(p => registry.FindById(p.MainModelId)?.Default == true)
            .ThenBy(p => p.MainModelId, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static void AddModelFiles(List<string> files, ModelRegistryEntry model, HashSet<string> siblingNames)
    {
        files.Add(RemotePath(model.Onnx));
        files.Add(RemotePath(model.Metadata));
        files.Add(RemotePath(model.RuntimeDefaults));
        files.Add(RemotePath(model.Acceptance));
        var modelCard = Path.Combine(Path.GetDirectoryName(RemotePath(model.Onnx)) ?? string.Empty, "model_card.md")
            .Replace('\\', '/');
        if (siblingNames.Contains(modelCard))
        {
            files.Add(modelCard);
        }
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

    private void InstallDownloadedPackage(
        string downloadRoot,
        RemoteModelPackage package,
        ModelRegistryEntry mainEntry,
        ModelRegistryEntry expressionEntry)
    {
        foreach (var path in Directory.EnumerateFiles(downloadRoot, "*", SearchOption.AllDirectories))
        {
            var relative = Path.GetRelativePath(downloadRoot, path);
            var target = Path.Combine(ModelRoot, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(path, target, overwrite: true);
        }

        // Merge into the existing user registry (never wipe other installed presets):
        // replace same-id entries, make the downloaded main the default, demote the rest.
        var registryPath0 = ModelRegistry.DefaultUserRegistryPath();
        var registry = ModelRegistry.TryLoad(registryPath0) ?? new ModelRegistry();
        registry.Models.RemoveAll(model =>
            model.Id.Equals(mainEntry.Id, StringComparison.OrdinalIgnoreCase) ||
            model.Id.Equals(expressionEntry.Id, StringComparison.OrdinalIgnoreCase));
        foreach (var model in registry.Models)
        {
            if (model.Role.Equals("main", StringComparison.OrdinalIgnoreCase))
            {
                model.Default = false;
            }
            else if (model.Role.Equals("expression", StringComparison.OrdinalIgnoreCase))
            {
                model.Default = false;
            }
        }

        var installedMain = CloneRegistryEntry(mainEntry);
        installedMain.Default = true;
        var installedExpression = CloneRegistryEntry(expressionEntry);
        installedExpression.Default = true;
        registry.Models.Add(installedMain);
        registry.Models.Add(installedExpression);

        var registryPath = ModelRegistry.DefaultUserRegistryPath();
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
            Version = entry.Version,
            UpdatedAt = entry.UpdatedAt,
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

    private static string VersionLabel(ModelRegistryEntry model)
        => string.IsNullOrWhiteSpace(model.DisplayName)
            ? model.Id
            : model.DisplayName;

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
