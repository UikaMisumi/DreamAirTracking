namespace DreamAirTracking.Core.Calibration;

public sealed class Affine2DGazeModel
{
    public string Type { get; set; } = "affine2d";
    public double InputCenterX { get; set; }
    public double InputCenterY { get; set; }
    public double InputScaleX { get; set; } = 1;
    public double InputScaleY { get; set; } = 1;
    public double[] OutX { get; set; } = new[] { 0d, 1d, 0d };
    public double[] OutY { get; set; } = new[] { 0d, 0d, 1d };
    public double CenterDeadzoneX { get; set; }
    public double CenterDeadzoneY { get; set; }
    public double CenterDeadzoneSoftness { get; set; } = 0.25;

    public (double X, double Y) Map(double rawX, double rawY)
    {
        var normalizedX = NormalizeInput(rawX, InputCenterX, InputScaleX);
        var normalizedY = NormalizeInput(rawY, InputCenterY, InputScaleY);
        var x = Apply(OutX, normalizedX, normalizedY);
        var y = Apply(OutY, normalizedX, normalizedY);
        (x, y) = ApplyCenterDeadzone(x, y);
        return (Math.Clamp(x, -1, 1), Math.Clamp(y, -1, 1));
    }

    private static double NormalizeInput(double value, double center, double scale)
    {
        if (!double.IsFinite(scale) || Math.Abs(scale) < 1e-9)
        {
            return value - center;
        }

        return (value - center) / scale;
    }

    private static double Apply(IReadOnlyList<double> coefficients, double rawX, double rawY)
    {
        if (coefficients.Count >= 6)
        {
            return coefficients[0] +
                (coefficients[1] * rawX) +
                (coefficients[2] * rawY) +
                (coefficients[3] * rawX * rawX) +
                (coefficients[4] * rawX * rawY) +
                (coefficients[5] * rawY * rawY);
        }

        if (coefficients.Count >= 3)
        {
            return coefficients[0] + (coefficients[1] * rawX) + (coefficients[2] * rawY);
        }

        if (coefficients.Count == 0)
        {
            return 0;
        }

        return coefficients[0];
    }

    private (double X, double Y) ApplyCenterDeadzone(double x, double y)
    {
        if (CenterDeadzoneX <= 0 || CenterDeadzoneY <= 0)
        {
            return (x, y);
        }

        var distance = Math.Sqrt(
            Math.Pow(x / CenterDeadzoneX, 2) +
            Math.Pow(y / CenterDeadzoneY, 2));
        if (distance <= 1)
        {
            return (0, 0);
        }

        if (CenterDeadzoneSoftness <= 0)
        {
            return (x, y);
        }

        var fullDistance = 1 + CenterDeadzoneSoftness;
        if (distance >= fullDistance)
        {
            return (x, y);
        }

        var t = (distance - 1) / CenterDeadzoneSoftness;
        var smooth = t * t * (3 - (2 * t));
        return (x * smooth, y * smooth);
    }
}
