using DreamAirTracking.Core.Geometry;

namespace DreamAirTracking.Core.Diagnostics;

public sealed class BrightSpotDetector
{
    private static readonly (int Dx, int Dy)[] Neighbors =
    {
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1)
    };

    public IReadOnlyList<BrightSpotCandidate> Detect(EyeFrame frame, BrightSpotDetectorOptions? options = null)
    {
        options ??= new BrightSpotDetectorOptions();
        if (frame.BitsPerPixel != 8)
        {
            return Array.Empty<BrightSpotCandidate>();
        }

        var roi = ClampRoi(options.Roi, frame.Width, frame.Height);
        var visited = new bool[frame.Width * frame.Height];
        var candidates = new List<BrightSpotCandidate>();

        for (var y = roi.Y; y <= roi.Bottom; y++)
        {
            for (var x = roi.X; x <= roi.Right; x++)
            {
                var index = y * frame.Width + x;
                if (visited[index] || frame.Pixels[index] < options.Threshold)
                {
                    continue;
                }

                var candidate = FloodComponent(frame, visited, roi, x, y, options);
                if (candidate is not null)
                {
                    candidates.Add(candidate);
                }
            }
        }

        return candidates
            .OrderByDescending(static c => c.Score)
            .ThenByDescending(static c => c.PeakIntensity)
            .Take(Math.Max(options.MaxCandidates, 0))
            .ToArray();
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

    private static BrightSpotCandidate? FloodComponent(
        EyeFrame frame,
        bool[] visited,
        PixelRect roi,
        int startX,
        int startY,
        BrightSpotDetectorOptions options)
    {
        var stack = new Stack<int>();
        stack.Push(startY * frame.Width + startX);
        visited[startY * frame.Width + startX] = true;

        var area = 0;
        var minX = startX;
        var maxX = startX;
        var minY = startY;
        var maxY = startY;
        var peak = byte.MinValue;
        var sumIntensity = 0;
        double sumX = 0;
        double sumY = 0;

        while (stack.Count > 0)
        {
            var index = stack.Pop();
            var x = index % frame.Width;
            var y = index / frame.Width;
            var intensity = frame.Pixels[index];

            area++;
            sumX += x;
            sumY += y;
            sumIntensity += intensity;
            if (intensity > peak) peak = intensity;
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
                if (visited[nextIndex] || frame.Pixels[nextIndex] < options.Threshold)
                {
                    continue;
                }

                visited[nextIndex] = true;
                stack.Push(nextIndex);
            }
        }

        var width = maxX - minX + 1;
        var height = maxY - minY + 1;
        if (area < options.MinArea || area > options.MaxArea || width > options.MaxWidth || height > options.MaxHeight)
        {
            return null;
        }

        var mean = sumIntensity / (double)area;
        var score = area * Math.Max(mean - options.Threshold + 1, 1);
        return new BrightSpotCandidate(
            new PixelRect(minX, minY, width, height),
            area,
            sumX / area,
            sumY / area,
            mean,
            peak,
            score);
    }
}
