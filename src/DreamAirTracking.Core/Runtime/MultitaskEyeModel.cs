using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DreamAirTracking.Core.Runtime;

/// <summary>Raw per-frame outputs of the multitask eye model (batch dim stripped).</summary>
public readonly record struct EyeModelOutputs(
    Vec2 Gaze,          // gaze_xy   [-1,1]
    EyePair Openness,   // openness_lr [0,1]
    EyePair Wide,       // wide_lr   [0,1]
    EyePair Squint,     // squint_lr [0,1]
    float[] Pupil,      // pupil_lr  [Lx,Ly,Lr,Rx,Ry,Rr]
    float[] Confidence); // confidence [left,right,pair]

/// <summary>
/// ONNX inference for the siamese multitask eye model, plus an optional secondary
/// expression model. Mirrors <c>predict_live_multitask.run_onnx</c> +
/// <c>make_session_options</c>. Input tensor "image" [1,2,size,size] float32 grayscale
/// [0,1] (channel 0 = left, 1 = right). CPU EP with capped threads + spinning disabled
/// (the native, permanent form of the CPU fix).
/// </summary>
public sealed class MultitaskEyeModel : IDisposable
{
    private readonly InferenceSession _main;
    private readonly InferenceSession? _expression;
    private readonly bool _mainHasWide;

    public int ImageSize { get; }
    public int ExpressionImageSize { get; }
    public bool HasExpressionModel => _expression is not null;

    public MultitaskEyeModel(string mainOnnxPath, int imageSize, int onnxThreads = 2,
        string? expressionOnnxPath = null, int? expressionImageSize = null)
    {
        ImageSize = imageSize;
        _main = new InferenceSession(mainOnnxPath, MakeSessionOptions(onnxThreads));
        var outputs = _main.OutputMetadata.Keys;
        _mainHasWide = outputs.Contains("wide_lr") && outputs.Contains("squint_lr");
        if (!outputs.Contains("gaze_xy") || !outputs.Contains("openness_lr")
            || !outputs.Contains("pupil_lr") || !outputs.Contains("confidence"))
        {
            throw new InvalidOperationException(
                $"Main ONNX missing required outputs; has [{string.Join(",", outputs)}].");
        }
        if (!string.IsNullOrWhiteSpace(expressionOnnxPath))
        {
            _expression = new InferenceSession(expressionOnnxPath, MakeSessionOptions(onnxThreads));
            var eo = _expression.OutputMetadata.Keys;
            if (!eo.Contains("wide_lr") || !eo.Contains("squint_lr"))
                throw new InvalidOperationException($"Expression ONNX must expose wide_lr and squint_lr; has [{string.Join(",", eo)}].");
            ExpressionImageSize = expressionImageSize ?? imageSize;
        }
        else
        {
            ExpressionImageSize = imageSize;
        }
    }

    /// <summary>Mirrors predict_live_multitask.make_session_options: cap threads + no spinning.</summary>
    public static SessionOptions MakeSessionOptions(int threads)
    {
        var options = new SessionOptions
        {
            IntraOpNumThreads = Math.Max(1, threads),
            InterOpNumThreads = 1,
        };
        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        return options;
    }

    private static DenseTensor<float> BuildInput(float[] left, float[] right, int size)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 2, size, size });
        var span = tensor.Buffer.Span;
        int n = size * size;
        left.AsSpan(0, n).CopyTo(span.Slice(0, n));      // channel 0 = left
        right.AsSpan(0, n).CopyTo(span.Slice(n, n));      // channel 1 = right
        return tensor;
    }

    private static float[] Get(IReadOnlyDictionary<string, float[]> d, string name, int len)
        => d.TryGetValue(name, out var v) ? v : new float[len];

    /// <summary>Run the main model. wide/squint come from the main head if present, else zeros.</summary>
    public EyeModelOutputs RunMain(float[] left, float[] right)
    {
        var input = BuildInput(left, right, ImageSize);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("image", input) };
        using var results = _main.Run(inputs);
        var d = new Dictionary<string, float[]>();
        foreach (var r in results) d[r.Name] = r.AsTensor<float>().ToArray();

        var gaze = Get(d, "gaze_xy", 2);
        var open = Get(d, "openness_lr", 2);
        var wide = _mainHasWide ? Get(d, "wide_lr", 2) : new float[2];
        var squint = _mainHasWide ? Get(d, "squint_lr", 2) : new float[2];
        return new EyeModelOutputs(
            new Vec2(gaze[0], gaze[1]),
            new EyePair(open[0], open[1]),
            new EyePair(wide[0], wide[1]),
            new EyePair(squint[0], squint[1]),
            Get(d, "pupil_lr", 6),
            Get(d, "confidence", 3));
    }

    /// <summary>Run the secondary expression model for wide_lr/squint_lr only.</summary>
    public (EyePair Wide, EyePair Squint) RunExpression(float[] left, float[] right)
    {
        if (_expression is null) throw new InvalidOperationException("No expression model configured.");
        var input = BuildInput(left, right, ExpressionImageSize);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("image", input) };
        using var results = _expression.Run(inputs);
        EyePair wide = default, squint = default;
        foreach (var r in results)
        {
            if (r.Name == "wide_lr") { var a = r.AsTensor<float>().ToArray(); wide = new EyePair(a[0], a[1]); }
            else if (r.Name == "squint_lr") { var a = r.AsTensor<float>().ToArray(); squint = new EyePair(a[0], a[1]); }
        }
        return (wide, squint);
    }

    public void Dispose()
    {
        _main.Dispose();
        _expression?.Dispose();
    }
}
