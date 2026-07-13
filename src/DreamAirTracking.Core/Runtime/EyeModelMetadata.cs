using System.Text.Json;

namespace DreamAirTracking.Core.Runtime;

/// <summary>Reads the model metadata.json (image size) and v2 per-eye openness calibration.</summary>
public static class EyeModelMetadata
{
    public static int ReadImageSize(string? metadataPath, int fallback = 128)
    {
        if (string.IsNullOrWhiteSpace(metadataPath) || !File.Exists(metadataPath)) return fallback;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(metadataPath));
            if (doc.RootElement.TryGetProperty("image_size", out var v) && v.TryGetInt32(out var s) && s > 0)
                return s;
        }
        catch { /* fall through */ }
        return fallback;
    }

    /// <summary>P0-2: whether the model was trained on pupil-recentered inputs.</summary>
    public static bool ReadRecenteredInput(string? metadataPath)
    {
        if (string.IsNullOrWhiteSpace(metadataPath) || !File.Exists(metadataPath)) return false;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(metadataPath));
            return doc.RootElement.TryGetProperty("recentered_input", out var v) && v.ValueKind == JsonValueKind.True;
        }
        catch { return false; }
    }

    /// <summary>
    /// Loads (open_p95, closed_p05) per eye from a v2 openness_calibration.json.
    /// Mirrors Python load_per_eye_openness_calibration; returns false for missing/v1/invalid files.
    /// </summary>
    public static bool TryLoadPerEyeCalibration(string? path, out EyePair openP95, out EyePair closedP05)
        => TryLoadPerEyeCalibration(path, out openP95, out closedP05, out _);

    /// <summary>
    /// Also reads the optional S2 <c>half_p50</c> anchor (null on older v2 files) which drives the
    /// piecewise-linear normalize pinning the user's half-open to 0.5.
    /// </summary>
    public static bool TryLoadPerEyeCalibration(string? path, out EyePair openP95, out EyePair closedP05, out EyePair? halfP50)
    {
        openP95 = default;
        closedP05 = default;
        halfP50 = null;
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return false;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            if (!root.TryGetProperty("left", out var l) || !root.TryGetProperty("right", out var r)) return false;
            if (!l.TryGetProperty("open_p95", out var lo) || !r.TryGetProperty("open_p95", out var ro)) return false;
            if (!l.TryGetProperty("closed_p05", out var lc) || !r.TryGetProperty("closed_p05", out var rc)) return false;
            double lop = lo.GetDouble(), rop = ro.GetDouble(), lcp = lc.GetDouble(), rcp = rc.GetDouble();
            if (!double.IsFinite(lop) || !double.IsFinite(rop) || !double.IsFinite(lcp) || !double.IsFinite(rcp)) return false;
            openP95 = new EyePair(lop, rop);
            closedP05 = new EyePair(lcp, rcp);
            if (l.TryGetProperty("half_p50", out var lh) && r.TryGetProperty("half_p50", out var rh))
            {
                double lhp = lh.GetDouble(), rhp = rh.GetDouble();
                if (double.IsFinite(lhp) && double.IsFinite(rhp))
                    halfP50 = new EyePair(lhp, rhp);
            }
            return true;
        }
        catch { return false; }
    }
}
