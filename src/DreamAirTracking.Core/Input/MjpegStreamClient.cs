using System.Runtime.CompilerServices;

namespace DreamAirTracking.Core.Input;

public sealed class MjpegStreamClient
{
    private readonly HttpClient _httpClient;

    public MjpegStreamClient(HttpClient? httpClient = null)
    {
        _httpClient = httpClient ?? new HttpClient();
    }

    public async IAsyncEnumerable<byte[]> ReadJpegFramesAsync(
        Uri streamUri,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var response = await _httpClient.GetAsync(streamUri, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);

        var chunk = new byte[8192];
        var buffer = new List<byte>(64 * 1024);

        while (!cancellationToken.IsCancellationRequested)
        {
            var read = await stream.ReadAsync(chunk, cancellationToken);
            if (read == 0)
            {
                yield break;
            }

            for (var i = 0; i < read; i++)
            {
                buffer.Add(chunk[i]);
            }

            while (TryExtractJpeg(buffer, out var jpeg))
            {
                yield return jpeg;
            }
        }
    }

    private static bool TryExtractJpeg(List<byte> buffer, out byte[] jpeg)
    {
        jpeg = Array.Empty<byte>();
        var start = FindMarker(buffer, 0xFF, 0xD8, 0);
        if (start < 0)
        {
            if (buffer.Count > 4096)
            {
                buffer.RemoveRange(0, buffer.Count - 2);
            }

            return false;
        }

        var end = FindMarker(buffer, 0xFF, 0xD9, start + 2);
        if (end < 0)
        {
            if (start > 0)
            {
                buffer.RemoveRange(0, start);
            }

            return false;
        }

        var length = end - start + 2;
        jpeg = buffer.GetRange(start, length).ToArray();
        buffer.RemoveRange(0, end + 2);
        return true;
    }

    private static int FindMarker(List<byte> buffer, byte a, byte b, int start)
    {
        for (var i = start; i < buffer.Count - 1; i++)
        {
            if (buffer[i] == a && buffer[i + 1] == b)
            {
                return i;
            }
        }

        return -1;
    }
}
