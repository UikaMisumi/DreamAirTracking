using System.Buffers.Binary;
using System.Net.Sockets;
using System.Runtime.CompilerServices;

namespace DreamAirTracking.Core.Input;

public sealed class BrokenEyeRawClient
{
    public async IAsyncEnumerable<EyeFrame> ReadFramesAsync(
        string host,
        int port,
        EyeSide side,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var requestId = side == EyeSide.Left ? (byte)0x01 : (byte)0x02;

        using var client = new TcpClient();
        await client.ConnectAsync(host, port, cancellationToken);
        await using var stream = client.GetStream();
        await stream.WriteAsync(new[] { requestId }, cancellationToken);

        var header = new byte[5];
        while (!cancellationToken.IsCancellationRequested)
        {
            await ReadExactlyAsync(stream, header, cancellationToken);
            if (header[0] != requestId)
            {
                throw new InvalidDataException($"Unexpected BrokenEye packet id 0x{header[0]:X2}, expected 0x{requestId:X2}.");
            }

            var payloadLength = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(1, 4));
            if (payloadLength < 12)
            {
                throw new InvalidDataException($"BrokenEye raw payload is too short: {payloadLength} bytes.");
            }

            var payload = new byte[payloadLength];
            await ReadExactlyAsync(stream, payload, cancellationToken);

            var width = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0, 4));
            var height = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(4, 4));
            var bitsPerPixel = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(8, 4));
            var expectedBytes = width * height * (bitsPerPixel / 8);
            if (payloadLength - 12 < expectedBytes)
            {
                throw new InvalidDataException($"BrokenEye raw payload has {payloadLength - 12} image bytes, expected {expectedBytes}.");
            }

            var pixels = new byte[expectedBytes];
            Buffer.BlockCopy(payload, 12, pixels, 0, expectedBytes);
            yield return new EyeFrame(side, width, height, bitsPerPixel, pixels, DateTimeOffset.UtcNow);
        }
    }

    private static async Task ReadExactlyAsync(Stream stream, byte[] buffer, CancellationToken cancellationToken)
    {
        var offset = 0;
        while (offset < buffer.Length)
        {
            var read = await stream.ReadAsync(buffer.AsMemory(offset), cancellationToken);
            if (read == 0)
            {
                throw new EndOfStreamException("BrokenEye stream ended unexpectedly.");
            }

            offset += read;
        }
    }
}
