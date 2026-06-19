namespace DreamAirTracking.Core.Calibration;

public static class RecordingRequestSummaryParser
{
    public static string? ReadWeakStageFixSummary(string? path, int maxRows = 2)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            return null;
        }

        return SummarizeWeakStageFixPlan(File.ReadLines(path), maxRows);
    }

    public static string? SummarizeWeakStageFixPlan(IEnumerable<string> lines, int maxRows = 2)
    {
        var rows = ReadWeakStageRows(lines)
            .Take(Math.Max(1, maxRows))
            .ToArray();
        if (rows.Length == 0)
        {
            return null;
        }

        return "Weak fix: " + string.Join(" | ", rows.Select(row => $"{row.Stage}: {row.LikelyCause}; {row.NextImprovement}"));
    }

    private static IEnumerable<WeakStageFixRow> ReadWeakStageRows(IEnumerable<string> lines)
    {
        var inSection = false;
        foreach (var line in lines)
        {
            if (line.Equals("### Weak Stage Fix Plan", StringComparison.OrdinalIgnoreCase))
            {
                inSection = true;
                continue;
            }

            if (!inSection)
            {
                continue;
            }

            if (line.StartsWith("## ", StringComparison.Ordinal))
            {
                yield break;
            }

            if (!line.StartsWith("| ", StringComparison.Ordinal) ||
                line.StartsWith("|---", StringComparison.Ordinal) ||
                line.Contains("| Stage |", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var cells = line
                .Trim()
                .Trim('|')
                .Split('|')
                .Select(cell => cell.Trim())
                .ToArray();
            if (cells.Length < 4)
            {
                continue;
            }

            yield return new WeakStageFixRow(
                cells[0],
                cells[2],
                ShortenImprovement(cells[3]));
        }
    }

    private static string ShortenImprovement(string text)
    {
        var first = text
            .Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .FirstOrDefault();
        return string.IsNullOrWhiteSpace(first) ? text : first;
    }

    private sealed record WeakStageFixRow(string Stage, string LikelyCause, string NextImprovement);
}
