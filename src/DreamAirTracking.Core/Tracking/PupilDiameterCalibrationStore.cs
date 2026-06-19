using System.Text.Json;
using DreamAirTracking.Core;

namespace DreamAirTracking.Core.Tracking;

public static class PupilDiameterCalibrationStore
{
    private static readonly JsonSerializerOptions JsonOptions = CoreJsonOptions.Web();

    public static async Task<PupilDiameterCalibrationProfile> LoadAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        await using var stream = File.OpenRead(path);
        return await JsonSerializer.DeserializeAsync<PupilDiameterCalibrationProfile>(
            stream,
            JsonOptions,
            cancellationToken) ?? new PupilDiameterCalibrationProfile();
    }
}
