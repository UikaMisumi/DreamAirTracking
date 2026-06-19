using System.Globalization;
using System.Text;
using System.Text.Json;
using DreamAirTracking.Core;

namespace DreamAirTracking.Core.Calibration;

public static class CalibrationSessionStore
{
    private static readonly JsonSerializerOptions JsonOptions = CoreJsonOptions.CamelCase();

    private static readonly JsonSerializerOptions JsonLineOptions = new(JsonOptions)
    {
        WriteIndented = false
    };

    public static async Task SaveSessionAsync(CalibrationSession session, string sessionDirectory, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(sessionDirectory);
        await WriteJsonAsync(Path.Combine(sessionDirectory, "session.json"), session, cancellationToken);
    }

    public static async Task<CalibrationSession> LoadSessionAsync(string sessionDirectory, CancellationToken cancellationToken = default)
    {
        await using var stream = File.OpenRead(Path.Combine(sessionDirectory, "session.json"));
        return await JsonSerializer.DeserializeAsync<CalibrationSession>(stream, JsonOptions, cancellationToken)
            ?? new CalibrationSession();
    }

    public static async Task SaveLabelsAsync(IEnumerable<CalibrationLabel> labels, string sessionDirectory, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(sessionDirectory);
        await using var stream = File.Create(Path.Combine(sessionDirectory, "labels.jsonl"));
        await using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        foreach (var label in labels)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await writer.WriteLineAsync(JsonSerializer.Serialize(label, JsonLineOptions));
        }
    }

    public static async Task<IReadOnlyList<CalibrationLabel>> LoadLabelsAsync(string sessionDirectory, CancellationToken cancellationToken = default)
    {
        var path = Path.Combine(sessionDirectory, "labels.jsonl");
        if (!File.Exists(path))
        {
            return Array.Empty<CalibrationLabel>();
        }

        var labels = new List<CalibrationLabel>();
        foreach (var line in await File.ReadAllLinesAsync(path, cancellationToken))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            labels.Add(JsonSerializer.Deserialize<CalibrationLabel>(line, JsonLineOptions) ?? new CalibrationLabel());
        }

        return labels;
    }

    public static async Task SaveMetricsAsync(CalibrationMetrics metrics, string sessionDirectory, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(sessionDirectory);
        await WriteJsonAsync(Path.Combine(sessionDirectory, "metrics.json"), metrics, cancellationToken);
    }

    public static async Task<CalibrationMetrics> LoadMetricsAsync(string sessionDirectory, CancellationToken cancellationToken = default)
    {
        await using var stream = File.OpenRead(Path.Combine(sessionDirectory, "metrics.json"));
        return await JsonSerializer.DeserializeAsync<CalibrationMetrics>(stream, JsonOptions, cancellationToken)
            ?? new CalibrationMetrics();
    }

    public static async Task SavePairsAsync(IEnumerable<CalibrationFramePair> pairs, string sessionDirectory, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(sessionDirectory);
        await using var stream = File.Create(Path.Combine(sessionDirectory, "pairs.csv"));
        await using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        await writer.WriteLineAsync("sequence,stage,left_file,right_file,delta_ms,left_found,left_raw_x,left_raw_y,left_conf,left_open,right_found,right_raw_x,right_raw_y,right_conf,right_open");

        foreach (var pair in pairs)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await writer.WriteLineAsync(string.Join(",", new[]
            {
                pair.Sequence.ToString(CultureInfo.InvariantCulture),
                CsvEscape(pair.StageId),
                CsvEscape(pair.LeftFile),
                CsvEscape(pair.RightFile),
                pair.DeltaMs.ToString(CultureInfo.InvariantCulture),
                pair.LeftFound.ToString(),
                pair.LeftRawX.ToString(CultureInfo.InvariantCulture),
                pair.LeftRawY.ToString(CultureInfo.InvariantCulture),
                pair.LeftConfidence.ToString(CultureInfo.InvariantCulture),
                pair.LeftOpenness.ToString(CultureInfo.InvariantCulture),
                pair.RightFound.ToString(),
                pair.RightRawX.ToString(CultureInfo.InvariantCulture),
                pair.RightRawY.ToString(CultureInfo.InvariantCulture),
                pair.RightConfidence.ToString(CultureInfo.InvariantCulture),
                pair.RightOpenness.ToString(CultureInfo.InvariantCulture)
            }));
        }
    }

    public static async Task<IReadOnlyList<CalibrationFramePair>> LoadPairsAsync(string sessionDirectory, CancellationToken cancellationToken = default)
    {
        var path = Path.Combine(sessionDirectory, "pairs.csv");
        if (!File.Exists(path))
        {
            return Array.Empty<CalibrationFramePair>();
        }

        var lines = await File.ReadAllLinesAsync(path, cancellationToken);
        var pairs = new List<CalibrationFramePair>();
        foreach (var line in lines.Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var columns = ParseCsvLine(line);
            if (columns.Count < 15)
            {
                continue;
            }

            pairs.Add(new CalibrationFramePair(
                long.Parse(columns[0], CultureInfo.InvariantCulture),
                columns[1],
                columns[2],
                columns[3],
                double.Parse(columns[4], CultureInfo.InvariantCulture),
                bool.Parse(columns[5]),
                double.Parse(columns[6], CultureInfo.InvariantCulture),
                double.Parse(columns[7], CultureInfo.InvariantCulture),
                double.Parse(columns[8], CultureInfo.InvariantCulture),
                double.Parse(columns[9], CultureInfo.InvariantCulture),
                bool.Parse(columns[10]),
                double.Parse(columns[11], CultureInfo.InvariantCulture),
                double.Parse(columns[12], CultureInfo.InvariantCulture),
                double.Parse(columns[13], CultureInfo.InvariantCulture),
                double.Parse(columns[14], CultureInfo.InvariantCulture)));
        }

        return pairs;
    }

    private static async Task WriteJsonAsync<T>(string path, T value, CancellationToken cancellationToken)
    {
        await using var stream = File.Create(path);
        await JsonSerializer.SerializeAsync(stream, value, JsonOptions, cancellationToken);
        await stream.WriteAsync(new byte[] { (byte)'\n' }, cancellationToken);
    }

    private static string CsvEscape(string value)
    {
        if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
    }

    private static IReadOnlyList<string> ParseCsvLine(string line)
    {
        var values = new List<string>();
        var current = new StringBuilder();
        var inQuotes = false;

        for (var i = 0; i < line.Length; i++)
        {
            var character = line[i];
            if (character == '"')
            {
                if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                {
                    current.Append('"');
                    i++;
                    continue;
                }

                inQuotes = !inQuotes;
                continue;
            }

            if (character == ',' && !inQuotes)
            {
                values.Add(current.ToString());
                current.Clear();
                continue;
            }

            current.Append(character);
        }

        values.Add(current.ToString());
        return values;
    }
}
