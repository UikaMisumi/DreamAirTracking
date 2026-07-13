namespace DreamAirTracking.Core.Runtime;

/// <summary>
/// P0-2 pupil-recentered crop — geometric input normalization. 1:1 mirror of Python
/// <c>pupil_recenter.py</c> (keep in sync; parity-tested via fixtures).
///
/// The facial gasket / wear position moves the eye 10-17% inside the camera frame, forcing
/// the model to fit contradictory wear-appearance mappings (per-wear directional gaze
/// collapse, eyelid mid-band breakdown). This is the appearance-based-gaze
/// "data normalization" step (MPIIGaze / ETH-XGaze standard) reduced to 2D: shift the frame
/// so the eye sits at a fixed canonical position before the model sees it.
///
/// Anchor: dark centroid of the EYE REGION. The raw frames contain a static vertical dark
/// band from the headset rim (left-eye frames: right edge; right-eye frames: left edge)
/// that dominates a naive dark percentile — so the estimator scans a per-side region that
/// excludes the rim band and top/bottom shadow.
/// </summary>
public sealed class PupilRecenter
{
    // Canonical per-eye centers (fraction of FULL frame) — MUST match pupil_recenter.py.
    public static readonly (double X, double Y) CanonicalLeft = (0.69, 0.75);
    public static readonly (double X, double Y) CanonicalRight = (0.33, 0.69);

    public const double MaxShiftFrac = 0.25;
    public const int DarkDelta = 26;        // threshold = region min + DarkDelta
    public const double TrimFrac = 0.12;    // median-refinement window (fraction of frame)
    public const int MinDarkPixels = 20;

    // per-side scan region fractions (x0, x1, y0, y1) — excludes rim band + top/bottom shadow.
    // Rim measured from column-darkness profiles: left-eye frames rim at x>=0.77, right at x<=0.26.
    private static readonly double[] LeftEyeRegion = { 0.00, 0.76, 0.06, 0.94 };
    private static readonly double[] RightEyeRegion = { 0.27, 1.00, 0.06, 0.94 };

    private readonly double _emaAlpha;
    private (double X, double Y)? _center;

    public PupilRecenter(double emaAlpha = 0.1) => _emaAlpha = emaAlpha;

    /// <summary>
    /// Pupil-blob estimate inside the per-side eye region (coordinates normalized by the FULL
    /// frame). Pipeline mirrors Python exactly: min+DarkDelta threshold (the pupil is the
    /// darkest structure; percentiles sweep in soft shadows), structural rejection of
    /// full-height columns / full-width rows (rim penumbra, border shadow), per-axis MEDIAN,
    /// one ±TrimFrac refinement pass.
    /// </summary>
    public static (double Cx, double Cy, double Confidence) EstimateDarkCentroid(byte[] gray, int width, int height, bool isLeft)
    {
        var region = isLeft ? LeftEyeRegion : RightEyeRegion;
        int x0 = (int)(region[0] * width), x1 = (int)(region[1] * width);
        int y0 = (int)(region[2] * height), y1 = (int)(region[3] * height);
        int subW = x1 - x0, subH = y1 - y0;
        if (subW <= 0 || subH <= 0) return (0.5, 0.5, 0.0);

        byte min = 255;
        for (int y = y0; y < y1; y++)
        {
            int row = y * width;
            for (int x = x0; x < x1; x++)
            {
                if (gray[row + x] < min) min = gray[row + x];
            }
        }
        int thr = Math.Min(255, min + DarkDelta);

        var mask = new bool[subH * subW];
        var colCounts = new int[subW];
        var rowCounts = new int[subH];
        for (int y = 0; y < subH; y++)
        {
            int row = (y0 + y) * width;
            for (int x = 0; x < subW; x++)
            {
                if (gray[row + x0 + x] <= thr)
                {
                    mask[y * subW + x] = true;
                    colCounts[x]++;
                    rowCounts[y]++;
                }
            }
        }
        for (int x = 0; x < subW; x++)
        {
            if (colCounts[x] > subH * 0.5)
                for (int y = 0; y < subH; y++) mask[y * subW + x] = false;
        }
        for (int y = 0; y < subH; y++)
        {
            if (rowCounts[y] > subW * 0.5)
                for (int x = 0; x < subW; x++) mask[y * subW + x] = false;
        }

        (int Mx, int My)? Median(int bx0, int bx1, int by0, int by1)
        {
            var xh = new int[subW];
            var yh = new int[subH];
            int n = 0;
            for (int y = by0; y < by1; y++)
            {
                for (int x = bx0; x < bx1; x++)
                {
                    if (mask[y * subW + x]) { xh[x]++; yh[y]++; n++; }
                }
            }
            if (n < MinDarkPixels) return null;
            int half = (n + 1) / 2;
            int mx = 0, my = 0, acc = 0;
            for (int x = 0; x < subW; x++) { acc += xh[x]; if (acc >= half) { mx = x; break; } }
            acc = 0;
            for (int y = 0; y < subH; y++) { acc += yh[y]; if (acc >= half) { my = y; break; } }
            return (mx, my);
        }

        var first = Median(0, subW, 0, subH);
        if (first is null) return (0.5, 0.5, 0.0);
        int tw = (int)(TrimFrac * width), th = (int)(TrimFrac * height);
        int xl = Math.Max(0, first.Value.Mx - tw), xr = Math.Min(subW, first.Value.Mx + tw);
        int yl = Math.Max(0, first.Value.My - th), yr = Math.Min(subH, first.Value.My + th);
        var refined = Median(xl, xr, yl, yr) ?? first;
        return ((refined.Value.Mx + x0) / (double)width, (refined.Value.My + y0) / (double)height, 1.0);
    }

    /// <summary>Integer-pixel shift so center lands on canonical (clamped ±MaxShiftFrac), mean fill.</summary>
    public static byte[] RecenterGray(byte[] gray, int width, int height,
        (double X, double Y) center, (double X, double Y) canonical, double maxShiftFrac = MaxShiftFrac)
    {
        // floor(x+0.5) rounding — identical to the Python mirror (see pupil_recenter.py)
        static int R(double x) => (int)Math.Floor(x + 0.5);
        int dx = R((canonical.X - center.X) * width);
        int dy = R((canonical.Y - center.Y) * height);
        int maxDx = R(maxShiftFrac * width);
        int maxDy = R(maxShiftFrac * height);
        dx = Math.Clamp(dx, -maxDx, maxDx);
        dy = Math.Clamp(dy, -maxDy, maxDy);
        if (dx == 0 && dy == 0) return gray;

        long sum = 0;
        foreach (var b in gray) sum += b;
        byte fill = (byte)Math.Clamp(R(sum / (double)gray.Length), 0, 255);

        var output = new byte[gray.Length];
        Array.Fill(output, fill);
        int srcX0 = Math.Max(0, -dx), srcX1 = Math.Min(width, width - dx);
        int srcY0 = Math.Max(0, -dy), srcY1 = Math.Min(height, height - dy);
        int dstX0 = Math.Max(0, dx), dstY0 = Math.Max(0, dy);
        int copyW = srcX1 - srcX0;
        for (int y = srcY0; y < srcY1; y++)
        {
            Array.Copy(gray, y * width + srcX0, output, (dstY0 + (y - srcY0)) * width + dstX0, copyW);
        }
        return output;
    }

    /// <summary>Temporal update: EMA while non-degenerate, freeze while not. Null until first lock.</summary>
    public (double X, double Y)? Update(double cx, double cy, double confidence)
    {
        if (confidence > 0.5)
        {
            _center = _center is (double px, double py)
                ? (px + (cx - px) * _emaAlpha, py + (cy - py) * _emaAlpha)
                : (cx, cy);
        }
        return _center;
    }

    /// <summary>Runtime path: estimate → temporal update → shift. No-op until first lock.</summary>
    public byte[] RecenterFrame(byte[] gray, int width, int height, (double X, double Y) canonical, bool isLeft)
    {
        var (cx, cy, conf) = EstimateDarkCentroid(gray, width, height, isLeft);
        var center = Update(cx, cy, conf);
        if (center is null) return gray;
        return RecenterGray(gray, width, height, center.Value, canonical);
    }
}
