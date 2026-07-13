namespace DreamAirTracking.App.Services;

public sealed record ModelDownloadProgress(
    string Stage,
    double? Fraction,
    string Detail);

public sealed record ModelDownloadResult(
    bool Success,
    string Message,
    string? InstalledRegistryPath = null);

public sealed record RemoteModelCatalog(
    string RepoId,
    string Revision,
    DateTimeOffset? LastModified,
    IReadOnlyList<RemoteModelPackage> Packages);

public sealed record RemoteModelPackage(
    string Id,
    string DisplayName,
    string Version,
    string PublishedDate,
    string MainModelId,
    string? ExpressionModelId,
    string Runtime,
    string Architecture,
    string Revision,
    DateTimeOffset? LastModified,
    IReadOnlyList<string> Outputs,
    IReadOnlyList<string> Files);

public sealed record InstalledModelPackage(
    string Id,
    string DisplayName,
    string Version,
    string Runtime,
    string MainOnnxPath,
    string? ExpressionOnnxPath,
    bool IsComplete,
    bool IsDefault);
