using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace DreamAirTracking.App.Services;

internal static class AppJsonOptions
{
    public static JsonSerializerOptions Web(bool writeIndented = false) => new(JsonSerializerDefaults.Web)
    {
        WriteIndented = writeIndented,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver()
    };
}
