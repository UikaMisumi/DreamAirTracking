namespace DreamAirTracking.Core.Profiles;

public sealed class RawInputTransform
{
    public bool Enabled { get; set; }
    public double CurrentCenterX { get; set; }
    public double CurrentCenterY { get; set; }
    public double TargetCenterX { get; set; }
    public double TargetCenterY { get; set; }
    public double ScaleX { get; set; } = 1;
    public double ScaleY { get; set; } = 1;

    public (double X, double Y) Apply(double rawX, double rawY)
    {
        if (!Enabled)
        {
            return (rawX, rawY);
        }

        var scaleX = IsUsableScale(ScaleX) ? ScaleX : 1;
        var scaleY = IsUsableScale(ScaleY) ? ScaleY : 1;
        return (
            TargetCenterX + ((rawX - CurrentCenterX) * scaleX),
            TargetCenterY + ((rawY - CurrentCenterY) * scaleY));
    }

    private static bool IsUsableScale(double value) =>
        double.IsFinite(value) && Math.Abs(value) > 1e-9;
}
