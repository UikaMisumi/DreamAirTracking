namespace DreamAirTracking.Core;

public sealed class EyeFrame
{
    public EyeFrame(EyeSide side, int width, int height, int bitsPerPixel, byte[] pixels, DateTimeOffset timestamp)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (bitsPerPixel <= 0 || bitsPerPixel % 8 != 0) throw new ArgumentOutOfRangeException(nameof(bitsPerPixel));

        var expectedBytes = width * height * (bitsPerPixel / 8);
        if (pixels.Length < expectedBytes)
        {
            throw new ArgumentException($"Frame has {pixels.Length} bytes, expected at least {expectedBytes}.", nameof(pixels));
        }

        Side = side;
        Width = width;
        Height = height;
        BitsPerPixel = bitsPerPixel;
        Pixels = pixels;
        Timestamp = timestamp;
    }

    public EyeSide Side { get; }
    public int Width { get; }
    public int Height { get; }
    public int BitsPerPixel { get; }
    public byte[] Pixels { get; }
    public DateTimeOffset Timestamp { get; }

    public byte GetGray(int x, int y)
    {
        if ((uint)x >= Width) throw new ArgumentOutOfRangeException(nameof(x));
        if ((uint)y >= Height) throw new ArgumentOutOfRangeException(nameof(y));
        return Pixels[y * Width + x];
    }
}
