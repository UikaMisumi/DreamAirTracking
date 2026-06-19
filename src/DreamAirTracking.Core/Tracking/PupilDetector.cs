using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Tracking;

public sealed class PupilDetector
{
    private static readonly (int Dx, int Dy)[] Neighbors =
    {
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1)
    };

    public PupilDetection Detect(EyeFrame frame, PupilDetectorOptions? options = null)
    {
        options ??= new PupilDetectorOptions();
        if (frame.BitsPerPixel != 8)
        {
            return PupilDetection.NotFound($"Only 8-bit grayscale frames are supported, got {frame.BitsPerPixel} bpp.");
        }

        var roi = ClampRoi(options.Roi, frame.Width, frame.Height);
        Candidate? best = null;

        for (var threshold = options.MinThreshold; threshold <= options.MaxThreshold; threshold += options.ThresholdStep)
        {
            var visited = new bool[frame.Width * frame.Height];

            for (var y = roi.Y; y <= roi.Bottom; y++)
            {
                for (var x = roi.X; x <= roi.Right; x++)
                {
                    var index = y * frame.Width + x;
                    if (visited[index] || frame.Pixels[index] > threshold)
                    {
                        continue;
                    }

                    var candidate = FloodComponent(frame, visited, roi, x, y, threshold, options);
                    if (candidate is null)
                    {
                        continue;
                    }

                    if (best is null || candidate.Score > best.Score)
                    {
                        best = candidate;
                    }
                }
            }
        }

        if (best is null)
        {
            return PupilDetection.NotFound("No pupil-like dark component found.");
        }

        var confidence = Math.Clamp(best.Score / Math.Max(options.MaxArea, 1), 0.05, 1.0);
        return new PupilDetection(
            true,
            confidence,
            best.CenterX,
            best.CenterY,
            best.Threshold,
            best.Bounds,
            best.Area,
            best.FillRatio,
            best.MomentXX,
            best.MomentYY,
            best.MomentXY);
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

    private static Candidate? FloodComponent(
        EyeFrame frame,
        bool[] visited,
        PixelRect roi,
        int startX,
        int startY,
        int threshold,
        PupilDetectorOptions options)
    {
        var stack = new Stack<int>();
        stack.Push(startY * frame.Width + startX);
        visited[startY * frame.Width + startX] = true;

        var area = 0;
        var minX = startX;
        var maxX = startX;
        var minY = startY;
        var maxY = startY;
        double sumX = 0;
        double sumY = 0;
        double sumXX = 0;
        double sumYY = 0;
        double sumXY = 0;

        while (stack.Count > 0)
        {
            var index = stack.Pop();
            var x = index % frame.Width;
            var y = index / frame.Width;

            area++;
            sumX += x;
            sumY += y;
            sumXX += x * x;
            sumYY += y * y;
            sumXY += x * y;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;

            foreach (var (dx, dy) in Neighbors)
            {
                var nx = x + dx;
                var ny = y + dy;
                if (nx < roi.X || nx > roi.Right || ny < roi.Y || ny > roi.Bottom)
                {
                    continue;
                }

                var nextIndex = ny * frame.Width + nx;
                if (visited[nextIndex] || frame.Pixels[nextIndex] > threshold)
                {
                    continue;
                }

                visited[nextIndex] = true;
                stack.Push(nextIndex);
            }
        }

        if (area < options.MinArea || area > options.MaxArea)
        {
            return null;
        }

        var width = maxX - minX + 1;
        var height = maxY - minY + 1;
        if (width > options.MaxWidth || height > options.MaxHeight)
        {
            return null;
        }

        if (minX <= options.BorderMargin || minY <= options.BorderMargin ||
            maxX >= frame.Width - 1 - options.BorderMargin || maxY >= frame.Height - 1 - options.BorderMargin)
        {
            return null;
        }

        var aspectRatio = width / (double)Math.Max(height, 1);
        if (aspectRatio < options.MinAspectRatio || aspectRatio > options.MaxAspectRatio)
        {
            return null;
        }

        var fillRatio = area / (double)(width * height);
        if (fillRatio < options.MinFillRatio)
        {
            return null;
        }

        var centerX = sumX / area;
        var centerY = sumY / area;
        var momentXX = Math.Max(0, (sumXX / area) - (centerX * centerX));
        var momentYY = Math.Max(0, (sumYY / area) - (centerY * centerY));
        var momentXY = (sumXY / area) - (centerX * centerY);
        var expectedX = options.ExpectedCenterX ?? frame.Width * 0.5;
        var expectedY = options.ExpectedCenterY ?? frame.Height * 0.52;
        var distancePenalty = options.PreviousCenterWeight * Math.Sqrt(
            Math.Pow(centerX - expectedX, 2) +
            Math.Pow(centerY - expectedY, 2));

        var borderPenalty = DistanceToNearestBorder(minX, minY, maxX, maxY, frame.Width, frame.Height) < 4 ? 50 : 0;
        var compactDarkScore = area * fillRatio * fillRatio;
        var score = compactDarkScore - distancePenalty - borderPenalty;

        return new Candidate(
            score,
            threshold,
            new PixelRect(minX, minY, width, height),
            area,
            fillRatio,
            centerX,
            centerY,
            momentXX,
            momentYY,
            momentXY);
    }

    private static int DistanceToNearestBorder(int minX, int minY, int maxX, int maxY, int width, int height)
    {
        return Math.Min(Math.Min(minX, minY), Math.Min(width - 1 - maxX, height - 1 - maxY));
    }

    private sealed record Candidate(
        double Score,
        int Threshold,
        PixelRect Bounds,
        int Area,
        double FillRatio,
        double CenterX,
        double CenterY,
        double MomentXX,
        double MomentYY,
        double MomentXY);
}
