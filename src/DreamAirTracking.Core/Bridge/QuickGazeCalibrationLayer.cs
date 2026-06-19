using System.Text.Json;

namespace DreamAirTracking.Core.Bridge;

public sealed class QuickGazeCalibrationLayer
{
    public QuickGazeCalibrationLayer(
        double a00,
        double a01,
        double a10,
        double a11,
        double b0,
        double b1,
        string rawSource = "bridge_output_xy",
        string? session = null)
    {
        A00 = a00;
        A01 = a01;
        A10 = a10;
        A11 = a11;
        B0 = b0;
        B1 = b1;
        RawSource = rawSource;
        Session = session;
    }

    public double A00 { get; }
    public double A01 { get; }
    public double A10 { get; }
    public double A11 { get; }
    public double B0 { get; }
    public double B1 { get; }
    public string RawSource { get; }
    public string? Session { get; }

    public BridgeGazeState Apply(BridgeGazeState input)
    {
        var left = Apply(input.LeftX, input.LeftY);
        var right = Apply(input.RightX, input.RightY);
        return new BridgeGazeState(left.X, left.Y, right.X, right.Y);
    }

    public static QuickGazeCalibrationLayer FromJson(string json, string? session = null)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        if (!root.TryGetProperty("sessions", out var sessions) ||
            sessions.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("quick calibration layer JSON does not contain a sessions object.");
        }

        var rawSource = root.TryGetProperty("raw_source", out var rawSourceElement)
            ? rawSourceElement.GetString() ?? string.Empty
            : string.Empty;
        if (!rawSource.Equals("bridge_output_xy", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"quick calibration layer raw_source '{rawSource}' is not supported by Bridge runtime.");
        }

        var selected = SelectSession(sessions, session);
        var payload = selected.Payload;
        var a = payload.GetProperty("A");
        var b = payload.GetProperty("b");
        return new QuickGazeCalibrationLayer(
            a[0][0].GetDouble(),
            a[0][1].GetDouble(),
            a[1][0].GetDouble(),
            a[1][1].GetDouble(),
            b[0].GetDouble(),
            b[1].GetDouble(),
            rawSource,
            selected.Session);
    }

    public static async Task<QuickGazeCalibrationLayer> LoadAsync(
        string path,
        string? session = null,
        CancellationToken cancellationToken = default)
    {
        var json = await File.ReadAllTextAsync(path, cancellationToken);
        return FromJson(json, session);
    }

    private (double X, double Y) Apply(double x, double y) =>
        (
            Math.Clamp((A00 * x) + (A01 * y) + B0, -1, 1),
            Math.Clamp((A10 * x) + (A11 * y) + B1, -1, 1)
        );

    private static (string Session, JsonElement Payload) SelectSession(JsonElement sessions, string? requestedSession)
    {
        if (!string.IsNullOrWhiteSpace(requestedSession))
        {
            if (!sessions.TryGetProperty(requestedSession, out var requested))
            {
                throw new InvalidDataException($"quick calibration layer session '{requestedSession}' was not found.");
            }

            if (!IsUsableSession(requested))
            {
                throw new InvalidDataException($"quick calibration layer session '{requestedSession}' is not marked usable.");
            }

            return (requestedSession, requested);
        }

        foreach (var session in sessions.EnumerateObject())
        {
            if (IsUsableSession(session.Value))
            {
                return (session.Name, session.Value);
            }
        }

        throw new InvalidDataException("quick calibration layer JSON does not contain a usable session.");
    }

    private static bool IsUsableSession(JsonElement session) =>
        session.ValueKind == JsonValueKind.Object &&
        session.TryGetProperty("used", out var used) &&
        used.ValueKind == JsonValueKind.True &&
        session.TryGetProperty("A", out var a) &&
        a.ValueKind == JsonValueKind.Array &&
        session.TryGetProperty("b", out var b) &&
        b.ValueKind == JsonValueKind.Array;
}
