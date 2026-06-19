using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Tracking;

public sealed class PupilDiameterEstimator
{
    public PupilDiameterEstimate Estimate(
        EyeFrame frame,
        PupilDetection detection,
        PupilDiameterEstimatorOptions? options = null)
    {
        options ??= new PupilDiameterEstimatorOptions();
        if (!detection.Found)
        {
            return PupilDiameterEstimate.NotFound(detection.Reason ?? "pupil not found");
        }

        if (frame.BitsPerPixel != 8)
        {
            return PupilDiameterEstimate.NotFound($"Only 8-bit grayscale frames are supported, got {frame.BitsPerPixel} bpp.");
        }

        if (detection.Confidence < options.MinPupilDetectionConfidence)
        {
            return PupilDiameterEstimate.NotFound("pupil confidence is too low");
        }

        var bounds = detection.Bounds;
        if (bounds.IsEmpty)
        {
            return PupilDiameterEstimate.NotFound("pupil bounds are empty");
        }

        if (HasFiniteComponentMoments(detection))
        {
            return BuildEstimate(
                detection.Area,
                detection.CenterX,
                detection.CenterY,
                detection.MomentXX,
                detection.MomentYY,
                detection.MomentXY,
                detection.Bounds,
                detection.Threshold,
                detection.Confidence,
                detection.FillRatio,
                options);
        }

        // Compatibility fallback for hand-built detections that predate component moments.
        return EstimateFromThresholdBounds(frame, detection, options);
    }

    private static bool HasFiniteComponentMoments(PupilDetection detection)
    {
        return double.IsFinite(detection.CenterX) &&
            double.IsFinite(detection.CenterY) &&
            double.IsFinite(detection.MomentXX) &&
            double.IsFinite(detection.MomentYY) &&
            double.IsFinite(detection.MomentXY);
    }

    private static PupilDiameterEstimate EstimateFromThresholdBounds(
        EyeFrame frame,
        PupilDetection detection,
        PupilDiameterEstimatorOptions options)
    {
        var bounds = detection.Bounds;
        var threshold = Math.Clamp(detection.Threshold + Math.Max(options.ThresholdPadding, 0), 0, 255);
        var area = 0;
        double sumX = 0;
        double sumY = 0;
        double sumXX = 0;
        double sumYY = 0;
        double sumXY = 0;
        for (var y = bounds.Y; y <= bounds.Bottom; y++)
        {
            for (var x = bounds.X; x <= bounds.Right; x++)
            {
                if (!IsDarkPupilPixel(frame, x, y, threshold))
                {
                    continue;
                }

                area++;
                sumX += x;
                sumY += y;
                sumXX += x * x;
                sumYY += y * y;
                sumXY += x * y;
            }
        }

        if (area < options.MinArea)
        {
            return PupilDiameterEstimate.NotFound("not enough dark pupil pixels");
        }

        var centerX = sumX / area;
        var centerY = sumY / area;
        var momentXX = Math.Max(0, (sumXX / area) - (centerX * centerX));
        var momentYY = Math.Max(0, (sumYY / area) - (centerY * centerY));
        var momentXY = (sumXY / area) - (centerX * centerY);

        return BuildEstimate(
            area,
            centerX,
            centerY,
            momentXX,
            momentYY,
            momentXY,
            bounds,
            threshold,
            detection.Confidence,
            detection.FillRatio,
            options);
    }

    private static PupilDiameterEstimate BuildEstimate(
        int area,
        double centerX,
        double centerY,
        double momentXX,
        double momentYY,
        double momentXY,
        PixelRect bounds,
        int threshold,
        double pupilConfidence,
        double fillRatio,
        PupilDiameterEstimatorOptions options)
    {
        if (area < options.MinArea)
        {
            return PupilDiameterEstimate.NotFound("not enough dark pupil pixels");
        }

        var trace = momentXX + momentYY;
        var determinantPart = Math.Sqrt(Math.Max(0, ((momentXX - momentYY) * (momentXX - momentYY)) + (4 * momentXY * momentXY)));
        var lambdaMajor = Math.Max(0, (trace + determinantPart) * 0.5);
        var lambdaMinor = Math.Max(0, (trace - determinantPart) * 0.5);
        var majorAxis = 4 * Math.Sqrt(lambdaMajor);
        var minorAxis = 4 * Math.Sqrt(lambdaMinor);
        if (majorAxis <= 0 || minorAxis <= 0)
        {
            return PupilDiameterEstimate.NotFound("ellipse moments are degenerate");
        }

        var axisRatio = majorAxis / minorAxis;
        if (axisRatio > options.MaxAxisRatio)
        {
            return PupilDiameterEstimate.NotFound("ellipse is too elongated to be a reliable pupil diameter");
        }

        var equivalentDiameter = 2 * Math.Sqrt(area / Math.PI);
        var axisConfidence = Math.Clamp(options.MaxAxisRatio / Math.Max(axisRatio, 1.0) / options.MaxAxisRatio, 0, 1);
        var fillConfidence = Math.Clamp(fillRatio, 0, 1);
        var confidence = Math.Clamp(pupilConfidence * 0.55 + axisConfidence * 0.25 + fillConfidence * 0.20, 0, 1);
        var orientation = 0.5 * Math.Atan2(2 * momentXY, momentXX - momentYY);

        return new PupilDiameterEstimate(
            true,
            confidence,
            equivalentDiameter,
            majorAxis,
            minorAxis,
            area,
            centerX,
            centerY,
            bounds,
            axisRatio,
            threshold,
            orientation);
    }

    private static bool IsDarkPupilPixel(EyeFrame frame, int x, int y, int threshold)
    {
        if ((uint)x >= frame.Width || (uint)y >= frame.Height)
        {
            return false;
        }

        return frame.Pixels[y * frame.Width + x] <= threshold;
    }
}
