using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DreamAirTracking.Core.Runtime;

/// <summary>
/// Decodes a BrokenEye JPEG frame to the model input representation.
/// Port of Python <c>preprocess</c> (live_gaze_dry_run.py:77-80):
/// grayscale ("L") -> resize(size,size) BILINEAR -> float32 / 255.
///
/// The resize replicates Pillow's separable triangle (bilinear) resampler exactly —
/// antialiased support scaling, coefficient normalization, and uint8 intermediate
/// rounding — so the model sees the same pixels as the Python runtime. A generic
/// resampler (e.g. ImageSharp Triangle) diverges enough on downscale to shift the
/// gaze head by ~1-2%.
/// </summary>
public static class EyeTensorPreprocessor
{
    public static float[] Preprocess(byte[] jpeg, int size)
    {
        var (src, inW, inH) = DecodeGray(jpeg);
        return ResizeToTensor(src, inW, inH, size);
    }

    /// <summary>
    /// P0-2 path: decode -> pupil-recenter (integer shift at native resolution) -> resize.
    /// Mirrors Python <c>preprocess_recentered</c>; the shift happens BEFORE the resize so
    /// the model always sees the eye at the canonical position regardless of wear geometry.
    /// </summary>
    public static float[] PreprocessRecentered(byte[] jpeg, int size, PupilRecenter state, (double X, double Y) canonical, bool isLeft)
    {
        var (src, inW, inH) = DecodeGray(jpeg);
        src = state.RecenterFrame(src, inW, inH, canonical, isLeft);
        return ResizeToTensor(src, inW, inH, size);
    }

    private static (byte[] Pixels, int Width, int Height) DecodeGray(byte[] jpeg)
    {
        using var image = Image.Load<L8>(jpeg);
        var src = new byte[image.Width * image.Height];
        image.CopyPixelDataTo(src);
        return (src, image.Width, image.Height);
    }

    private static float[] ResizeToTensor(byte[] src, int inW, int inH, int size)
    {
        if (inW == size && inH == size)
        {
            var direct = new float[size * size];
            for (int i = 0; i < direct.Length; i++) direct[i] = src[i] / 255f;
            return direct;
        }
        return PilBilinearResize(src, inW, inH, size, size);
    }

    // --- Pillow-exact separable triangle (bilinear) resize ---

    private static (int[] bounds, double[] coeffs, int ksize) Precompute(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize;
        double filterscale = Math.Max(scale, 1.0);
        double support = 1.0 * filterscale; // triangle filter support = 1.0
        int ksize = (int)Math.Ceiling(support) * 2 + 1;
        var bounds = new int[outSize * 2];
        var coeffs = new double[outSize * ksize];
        for (int xx = 0; xx < outSize; xx++)
        {
            double center = (xx + 0.5) * scale;
            double ss = 1.0 / filterscale;
            int xmin = (int)(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = (int)(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            xmax -= xmin;
            double ww = 0.0;
            for (int x = 0; x < xmax; x++)
            {
                double t = (x + xmin - center + 0.5) * ss;
                double w = (t > -1.0 && t < 1.0) ? 1.0 - Math.Abs(t) : 0.0;
                coeffs[xx * ksize + x] = w;
                ww += w;
            }
            if (ww != 0.0)
                for (int x = 0; x < xmax; x++) coeffs[xx * ksize + x] /= ww;
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        return (bounds, coeffs, ksize);
    }

    private static byte Clip8(double x)
    {
        int v = (int)Math.Floor(x + 0.5); // Pillow rounds half up
        return (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }

    private static float[] PilBilinearResize(byte[] src, int inW, int inH, int outW, int outH)
    {
        var (hb, hc, hk) = Precompute(inW, outW);
        var (vb, vc, vk) = Precompute(inH, outH);

        // horizontal pass: [inH x inW] -> uint8 [inH x outW] (Pillow rounds to uint8 between passes)
        var tmp = new byte[inH * outW];
        for (int y = 0; y < inH; y++)
        {
            int rowOff = y * inW;
            for (int xx = 0; xx < outW; xx++)
            {
                int xmin = hb[xx * 2], xmax = hb[xx * 2 + 1];
                double s = 0.0;
                int cOff = xx * hk;
                for (int x = 0; x < xmax; x++) s += src[rowOff + xmin + x] * hc[cOff + x];
                tmp[y * outW + xx] = Clip8(s);
            }
        }

        // vertical pass: [inH x outW] -> float [outH x outW] in [0,1]
        var outp = new float[outH * outW];
        for (int yy = 0; yy < outH; yy++)
        {
            int ymin = vb[yy * 2], ymax = vb[yy * 2 + 1];
            int cOff = yy * vk;
            for (int xx = 0; xx < outW; xx++)
            {
                double s = 0.0;
                for (int y = 0; y < ymax; y++) s += tmp[(ymin + y) * outW + xx] * vc[cOff + y];
                outp[yy * outW + xx] = Clip8(s) / 255f;
            }
        }
        return outp;
    }
}
