using System.Text.Json;
using DreamAirTracking.Core;

namespace DreamAirTracking.Core.Profiles;

public static class TrackingProfileStore
{
    private static readonly JsonSerializerOptions JsonOptions = CoreJsonOptions.CamelCase();

    public static async Task<TrackingProfile> LoadOrDefaultAsync(string? path, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            return new TrackingProfile();
        }

        await using var stream = File.OpenRead(path);
        return await JsonSerializer.DeserializeAsync<TrackingProfile>(stream, JsonOptions, cancellationToken)
            ?? new TrackingProfile();
    }

    public static async Task SaveAsync(TrackingProfile profile, string path, CancellationToken cancellationToken = default)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        await using var stream = File.Create(path);
        await JsonSerializer.SerializeAsync(stream, profile, JsonOptions, cancellationToken);
        await stream.WriteAsync(new byte[] { (byte)'\n' }, cancellationToken);
    }
}
