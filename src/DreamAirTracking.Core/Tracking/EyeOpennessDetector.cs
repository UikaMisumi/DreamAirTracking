using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Tracking;

public sealed class EyeOpennessDetector
{
    public EyeOpennessDetection Detect(EyeFrame frame, EyeOpennessDetectorOptions? options = null)
    {
        options ??= new EyeOpennessDetectorOptions();
        if (frame.BitsPerPixel != 8)
        {
            return EyeOpennessDetection.NotFound($"Only 8-bit grayscale frames are supported, got {frame.BitsPerPixel} bpp.");
        }

        var roi = ClampRoi(options.Roi, frame.Width, frame.Height);
        if (roi.IsEmpty)
        {
            return EyeOpennessDetection.NotFound("ROI is empty.");
        }

        var threshold = options.DarkThreshold ?? EstimateDarkThreshold(frame, roi, options);
        threshold = Math.Clamp(threshold, 0, 255);
        var rowFractions = BuildDarkRowFractions(frame, roi, (byte)threshold);
        var smoothed = Smooth(rowFractions, Math.Max(options.RowSmoothingRadius, 0));
        var peak = smoothed.Length == 0 ? 0 : smoothed.Max();
        var mean = smoothed.Length == 0 ? 0 : smoothed.Average();
        var (upper, lower, aperture) = FindAperture(smoothed, roi, options.ApertureGateFraction);
        var openness = NormalizePeak(peak, options.ClosedPeakFraction, options.OpenPeakFraction);
        if (aperture == 0 && peak < options.ClosedPeakWhenNoAperture)
        {
            openness = 0;
        }
        else if (aperture > 0 && aperture < options.MinApertureHeightForPartialOpen && peak < options.OpenPeakFraction)
        {
            openness = 0;
        }

        return new EyeOpennessDetection(
            true,
            openness,
            upper,
            lower,
            aperture,
            peak,
            mean,
            threshold,
            aperture > 0 ? "ok" : "no coherent dark aperture");
    }

    private static PixelRect ClampRoi(PixelRect? roi, int width, int height)
    {
        if (roi is null || roi.Value.IsEmpty)
        {
            return new PixelRect(0, 0, width, height);
        }

        var x = Math.Clamp(roi.Value.X, 0, width - 1);
        var y = Math.Clamp(roi.Value.Y, 0, height - 1);
        var right = Math.Clamp(roi.Value.Right, x, width - 1);
        var bottom = Math.Clamp(roi.Value.Bottom, y, height - 1);
        return new PixelRect(x, y, right - x + 1, bottom - y + 1);
    }

    private static int EstimateDarkThreshold(EyeFrame frame, PixelRect roi, EyeOpennessDetectorOptions options)
    {
        var values = new byte[roi.Width * roi.Height];
        var index = 0;
        for (var y = roi.Y; y <= roi.Bottom; y++)
        {
            for (var x = roi.X; x <= roi.Right; x++)
            {
                values[index++] = frame.Pixels[y * frame.Width + x];
            }
        }

        Array.Sort(values);
        var percentileIndex = Math.Clamp(
            (int)Math.Round((values.Length - 1) * Math.Clamp(options.AdaptivePercentile, 0, 1)),
            0,
            values.Length - 1);
        return Math.Min(values[percentileIndex] + options.AdaptiveMargin, options.MaxAdaptiveThreshold);
    }

    private static double[] BuildDarkRowFractions(EyeFrame frame, PixelRect roi, byte threshold)
    {
        var rows = new double[roi.Height];
        for (var y = roi.Y; y <= roi.Bottom; y++)
        {
            var dark = 0;
            for (var x = roi.X; x <= roi.Right; x++)
            {
                if (frame.Pixels[y * frame.Width + x] <= threshold)
                {
                    dark++;
                }
            }

            rows[y - roi.Y] = dark / (double)roi.Width;
        }

        return rows;
    }

    private static double[] Smooth(double[] values, int radius)
    {
        if (radius == 0 || values.Length == 0)
        {
            return values;
        }

        var smoothed = new double[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            var start = Math.Max(0, i - radius);
            var end = Math.Min(values.Length - 1, i + radius);
            double sum = 0;
            for (var j = start; j <= end; j++)
            {
                sum += values[j];
            }

            smoothed[i] = sum / (end - start + 1);
        }

        return smoothed;
    }

    private static (int Upper, int Lower, int Height) FindAperture(double[] rowFractions, PixelRect roi, double gate)
    {
        var upper = -1;
        var lower = -1;
        for (var i = 0; i < rowFractions.Length; i++)
        {
            if (rowFractions[i] < gate)
            {
                continue;
            }

            if (upper < 0)
            {
                upper = i;
            }

            lower = i;
        }

        if (upper < 0 || lower < upper)
        {
            return (0, 0, 0);
        }

        return (roi.Y + upper, roi.Y + lower, lower - upper + 1);
    }

    private static double NormalizePeak(double peak, double closed, double open)
    {
        if (open <= closed)
        {
            return peak >= open ? 1 : 0;
        }

        return Math.Clamp((peak - closed) / (open - closed), 0, 1);
    }
}
