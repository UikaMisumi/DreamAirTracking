using DreamAirTracking.Core.Tracking;

namespace DreamAirTracking.Core.Bridge;

public sealed class BridgeOpennessFilter
{
    private readonly BridgeOpennessFilterOptions _options;
    private readonly EyeBaseline _leftBaseline;
    private readonly EyeBaseline _rightBaseline;
    private readonly EyeDropHoldState _leftDropHold = new();
    private readonly EyeDropHoldState _rightDropHold = new();
    private double _left = 1.0;
    private double _right = 1.0;

    public BridgeOpennessFilter(BridgeOpennessFilterOptions? options = null)
    {
        _options = options ?? new BridgeOpennessFilterOptions();
        _leftBaseline = new EyeBaseline(_options.InitialOpenPeakFraction);
        _rightBaseline = new EyeBaseline(_options.InitialOpenPeakFraction);
    }

    public BridgeOpennessState Update(EyeOpennessDetection left, EyeOpennessDetection right)
    {
        var rawLeft = NormalizeAdaptive(left, _leftBaseline);
        var rawRight = NormalizeAdaptive(right, _rightBaseline);
        var bothClosed = rawLeft <= _options.ClosedThreshold && rawRight <= _options.ClosedThreshold;

        var targetLeft = rawLeft;
        var targetRight = rawRight;

        if (!bothClosed)
        {
            targetLeft = HoldAsymmetricDrop(targetLeft, rawRight, _left, _leftDropHold);
            targetRight = HoldAsymmetricDrop(targetRight, rawLeft, _right, _rightDropHold);
        }
        else
        {
            _leftDropHold.Reset();
            _rightDropHold.Reset();
        }

        _left = Smooth(_left, targetLeft, bothClosed);
        _right = Smooth(_right, targetRight, bothClosed);
        return new BridgeOpennessState(_left, _right);
    }

    private double NormalizeAdaptive(EyeOpennessDetection detection, EyeBaseline baseline)
    {
        if (!detection.Found)
        {
            return 0;
        }

        if (detection.ApertureHeight < _options.MinOpenBaselineApertureHeight &&
            detection.PeakDarkFraction <= _options.MinOpenBaselinePeakFraction)
        {
            return 0;
        }

        if (detection.ApertureHeight >= _options.MinOpenBaselineApertureHeight &&
            detection.PeakDarkFraction >= _options.MinOpenBaselinePeakFraction)
        {
            UpdateOpenBaseline(detection, baseline);
        }

        var denominator = Math.Max(baseline.OpenPeak - _options.ClosedPeakFraction, 0.001);
        var peakOpenness = (detection.PeakDarkFraction - _options.ClosedPeakFraction) / denominator;
        var heightOpenness = baseline.OpenHeight is > 0
            ? detection.ApertureHeight / baseline.OpenHeight.Value
            : detection.Openness;
        var peakWeight = Math.Clamp(_options.PeakOpennessWeight, 0, 1);
        var raw = ApplyClosingCurve((peakOpenness * peakWeight) + (heightOpenness * (1 - peakWeight)));
        if (detection.Openness >= _options.OpenEvidenceDetectorThreshold &&
            detection.ApertureHeight >= _options.MinOpenBaselineApertureHeight)
        {
            raw = Math.Max(raw, _options.OpenEvidenceFloor);
        }

        return Clamp(raw);
    }

    private void UpdateOpenBaseline(EyeOpennessDetection detection, EyeBaseline baseline)
    {
        baseline.OpenPeak = Math.Max(
            _options.MinOpenBaselinePeakFraction,
            baseline.OpenPeak + (detection.PeakDarkFraction - baseline.OpenPeak) * Math.Clamp(_options.OpenBaselineAlpha, 0, 1));

        if (baseline.OpenHeight is null)
        {
            baseline.OpenHeight = detection.ApertureHeight;
            return;
        }

        var alpha = detection.ApertureHeight >= baseline.OpenHeight.Value
            ? _options.OpenBaselineAlpha
            : _options.OpenBaselineDownAlpha;

        baseline.OpenHeight += (detection.ApertureHeight - baseline.OpenHeight.Value) * Math.Clamp(alpha, 0, 1);
    }

    private double HoldAsymmetricDrop(double target, double otherRaw, double previous, EyeDropHoldState hold)
    {
        if (previous < _options.OpenHoldThreshold ||
            target >= _options.SingleEyeDropThreshold ||
            otherRaw < _options.OtherEyeOpenThreshold)
        {
            hold.Reset();
            return target;
        }

        hold.Frames++;
        if (otherRaw >= _options.OtherEyeFullyOpenHoldThreshold ||
            hold.Frames <= _options.AsymmetricDropConfirmFrames)
        {
            return previous;
        }

        return target;
    }

    private double Smooth(double previous, double target, bool bothClosed)
    {
        var alpha = target >= previous
            ? _options.RiseAlpha
            : bothClosed ? _options.BothClosedFallAlpha : _options.FallAlpha;

        return Clamp(previous + (target - previous) * Math.Clamp(alpha, 0, 1));
    }

    private static double Clamp(double value) => Math.Clamp(value, 0.0, 1.0);

    private double ApplyClosingCurve(double value)
    {
        var gamma = Math.Max(_options.ClosingCurveGamma, 0.001);
        return Math.Pow(Clamp(value), gamma);
    }

    private sealed class EyeBaseline
    {
        public EyeBaseline(double openPeak)
        {
            OpenPeak = openPeak;
        }

        public double OpenPeak { get; set; }
        public double? OpenHeight { get; set; }
    }

    private sealed class EyeDropHoldState
    {
        public int Frames { get; set; }

        public void Reset()
        {
            Frames = 0;
        }
    }
}
