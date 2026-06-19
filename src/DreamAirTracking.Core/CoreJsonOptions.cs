using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;

namespace DreamAirTracking.Core;

internal static class CoreJsonOptions
{
    public static JsonSerializerOptions CamelCase(bool writeIndented = true) => new()
    {
        WriteIndented = writeIndented,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver(),
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
    };

    public static JsonSerializerOptions Web(bool writeIndented = false) => new(JsonSerializerDefaults.Web)
    {
        WriteIndented = writeIndented,
        PropertyNameCaseInsensitive = true,
        TypeInfoResolver = new DefaultJsonTypeInfoResolver()
    };
}
