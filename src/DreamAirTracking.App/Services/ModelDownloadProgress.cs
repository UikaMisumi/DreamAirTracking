namespace DreamAirTracking.App.Services;

public sealed record ModelDownloadProgress(
    string Stage,
    double? Fraction,
    string Detail);

public sealed record ModelDownloadResult(
    bool Success,
    string Message,
    string? InstalledRegistryPath = null);
