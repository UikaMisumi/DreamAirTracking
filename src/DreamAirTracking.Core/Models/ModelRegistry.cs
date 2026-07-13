using System.Text.Json;

namespace DreamAirTracking.Core.Models;

public sealed class ModelRegistry
{
    public string Schema { get; set; } = "dream_air_tracking.model_registry.v1";
    public List<ModelRegistryEntry> Models { get; set; } = new();

    public static string DefaultRegistryPath(string repoRoot) =>
        Path.Combine(repoRoot, "models", "model_registry.json");

    public static string DefaultUserRegistryPath() =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "DreamAirTracking",
            "models",
            "model_registry.json");

    public static ModelRegistry? TryLoadUserOrRepo(string repoRoot)
    {
        var userRegistry = TryLoad(DefaultUserRegistryPath());
        return userRegistry ?? TryLoadFromRepo(repoRoot);
    }

    public static ModelRegistry? TryLoadFromRepo(string repoRoot)
    {
        var path = DefaultRegistryPath(repoRoot);
        return TryLoad(path, repoRoot);
    }

    public static ModelRegistry? TryLoad(string path, string? repoRoot = null)
    {
        try
        {
            if (!File.Exists(path))
            {
                return null;
            }

            var json = File.ReadAllText(path);
            var registry = JsonSerializer.Deserialize<ModelRegistry>(json, CoreJsonOptions.CamelCase());
            if (registry is null ||
                !registry.Schema.Equals("dream_air_tracking.model_registry.v1", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            var root = repoRoot ?? Directory.GetParent(Path.GetFullPath(path))?.Parent?.FullName ?? Directory.GetCurrentDirectory();
            foreach (var model in registry.Models)
            {
                model.ResolveAgainst(root);
            }

            return registry;
        }
        catch
        {
            return null;
        }
    }

    public ModelRegistryEntry? FindById(string? id)
    {
        if (string.IsNullOrWhiteSpace(id))
        {
            return null;
        }

        return Models.FirstOrDefault(model => model.Id.Equals(id, StringComparison.OrdinalIgnoreCase));
    }

    public ModelRegistryEntry? FindDefaultMain(string runtime = "predict_live_multitask") =>
        Models
            .Where(model =>
                model.DeviceFamily.Equals("Dream Air", StringComparison.OrdinalIgnoreCase) &&
                model.Runtime.Equals(runtime, StringComparison.OrdinalIgnoreCase) &&
                model.Role.Equals("main", StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(model => model.Default)
            .ThenBy(model => model.DisplayName, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();

    public ModelRegistryEntry? FindDefaultExpression(string runtime = "predict_live_multitask") =>
        Models.FirstOrDefault(model =>
            model.DeviceFamily.Equals("Dream Air", StringComparison.OrdinalIgnoreCase) &&
            model.Runtime.Equals(runtime, StringComparison.OrdinalIgnoreCase) &&
            model.Role.Equals("expression", StringComparison.OrdinalIgnoreCase));
}

public sealed class ModelRegistryEntry
{
    public string Id { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    // Per-model version identity, distinct from the repo commit (e.g. "1.0", "1.1"). Optional.
    public string Version { get; set; } = string.Empty;
    // ISO date this specific model was trained/published (e.g. "2026-07-09"). Optional.
    public string UpdatedAt { get; set; } = string.Empty;
    public string DeviceFamily { get; set; } = "Dream Air";
    public string Role { get; set; } = "main";
    public string Architecture { get; set; } = string.Empty;
    public string Runtime { get; set; } = "predict_live_multitask";
    public string Onnx { get; set; } = string.Empty;
    public string Metadata { get; set; } = string.Empty;
    public string RuntimeDefaults { get; set; } = string.Empty;
    public string Acceptance { get; set; } = string.Empty;
    public List<string> Outputs { get; set; } = new();
    public bool Default { get; set; }

    public string ResolvedOnnx { get; private set; } = string.Empty;
    public string ResolvedMetadata { get; private set; } = string.Empty;
    public string ResolvedRuntimeDefaults { get; private set; } = string.Empty;
    public string ResolvedAcceptance { get; private set; } = string.Empty;

    internal void ResolveAgainst(string repoRoot)
    {
        ResolvedOnnx = ResolvePath(repoRoot, Onnx);
        ResolvedMetadata = ResolvePath(repoRoot, Metadata);
        ResolvedRuntimeDefaults = ResolvePath(repoRoot, RuntimeDefaults);
        ResolvedAcceptance = ResolvePath(repoRoot, Acceptance);
    }

    private static string ResolvePath(string repoRoot, string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        var normalized = path.Replace('/', Path.DirectorySeparatorChar);
        return Path.IsPathRooted(normalized)
            ? Path.GetFullPath(normalized)
            : Path.GetFullPath(Path.Combine(repoRoot, normalized));
    }
}
