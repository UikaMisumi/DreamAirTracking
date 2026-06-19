using System.Text.Json.Serialization;

namespace DreamAirTracking.Core.Geometry;

public readonly record struct PixelRect(int X, int Y, int Width, int Height)
{
    [JsonIgnore]
    public int Right => X + Width - 1;
    [JsonIgnore]
    public int Bottom => Y + Height - 1;
    [JsonIgnore]
    public bool IsEmpty => Width <= 0 || Height <= 0;
}
