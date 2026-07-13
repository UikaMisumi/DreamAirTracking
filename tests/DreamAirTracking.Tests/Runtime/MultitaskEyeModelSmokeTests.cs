using DreamAirTracking.Core.Runtime;

namespace DreamAirTracking.Tests.Runtime;

/// <summary>
/// Plumbing smoke test for the native ONNX inference path. Validates session load,
/// input tensor layout [1,2,128,128], and output names/shapes against the REAL installed
/// model. Skips (no-op) when the model is not installed on this machine (e.g. CI).
/// End-to-end numeric parity vs Python is covered by the golden-parity step (E11-5).
/// </summary>
public sealed class MultitaskEyeModelSmokeTests
{
    private static string? InstalledMainModel()
    {
        var path = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "DreamAirTracking", "models", "dreamair-main-current", "model.onnx");
        return File.Exists(path) ? path : null;
    }

    [Fact]
    public void RunsMainAndProducesValidOutputs()
    {
        string? model = InstalledMainModel();
        if (model is null) return; // model not installed here — plumbing exercised by E11-5 instead

        using var m = new MultitaskEyeModel(model, imageSize: 128, onnxThreads: 2);
        int n = 128 * 128;
        var left = new float[n];
        var right = new float[n];
        for (int i = 0; i < n; i++) { left[i] = 0.4f; right[i] = 0.4f; }

        var o = m.RunMain(left, right);

        Assert.Equal(6, o.Pupil.Length);
        Assert.Equal(3, o.Confidence.Length);
        Assert.InRange(o.Gaze.X, -1.5, 1.5);
        Assert.InRange(o.Gaze.Y, -1.5, 1.5);
        Assert.InRange(o.Openness.Left, -0.01, 1.01);
        Assert.InRange(o.Openness.Right, -0.01, 1.01);
    }
}
