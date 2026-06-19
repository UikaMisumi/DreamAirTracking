using DreamAirTracking.Core.Bridge;

namespace DreamAirTracking.Core.Calibration;

public sealed class CalibrationLiveGazeHighlighter
{
    private readonly CalibrationLiveGazeHighlighterOptions _options;

    public CalibrationLiveGazeHighlighter(CalibrationLiveGazeHighlighterOptions? options = null)
    {
        _options = options ?? new CalibrationLiveGazeHighlighterOptions();
    }

    public CalibrationLiveGazeHighlight Evaluate(
        BridgeTrackingState state,
        string? previousStageId = null,
        CalibrationTarget? previousTarget = null)
    {
        var leftUsable = IsUsableEye(state.Left);
        var rightUsable = IsUsableEye(state.Right);
        var hasUsableEye = leftUsable || rightUsable;
        var openness = (state.Left.Openness + state.Right.Openness) * 0.5;

        if (openness < _options.ClosedOpennessThreshold || !hasUsableEye)
        {
            var heldTarget = previousTarget ?? CalibrationTarget.Center;
            return new CalibrationLiveGazeHighlight(
                previousStageId ?? "center",
                heldTarget,
                IsPaused: true,
                openness,
                hasUsableEye,
                openness < _options.ClosedOpennessThreshold ? "closed" : "low_confidence");
        }

        var target = AverageGaze(state, leftUsable, rightUsable);
        return new CalibrationLiveGazeHighlight(
            StageIdFromTarget(
                target,
                _options.AxisThreshold,
                previousStageId,
                _options.AxisReleaseThreshold),
            target,
            IsPaused: false,
            openness,
            HasUsableEye: true,
            Reason: "live");
    }

    public static string StageIdFromTarget(CalibrationTarget target, double axisThreshold = 0.35)
    {
        var x = QuantizeAxis(target.X, axisThreshold);
        var y = QuantizeAxis(target.Y, axisThreshold);
        return (x, y) switch
        {
            (-1, 1) => "left_up",
            (0, 1) => "up",
            (1, 1) => "right_up",
            (-1, 0) => "left",
            (1, 0) => "right",
            (-1, -1) => "left_down",
            (0, -1) => "down",
            (1, -1) => "right_down",
            _ => "center"
        };
    }

    private static string StageIdFromTarget(
        CalibrationTarget target,
        double axisEnterThreshold,
        string? previousStageId,
        double axisReleaseThreshold)
    {
        var (previousX, previousY) = AxesFromStageId(previousStageId);
        var x = QuantizeAxisWithHysteresis(target.X, previousX, axisEnterThreshold, axisReleaseThreshold);
        var y = QuantizeAxisWithHysteresis(target.Y, previousY, axisEnterThreshold, axisReleaseThreshold);
        return (x, y) switch
        {
            (-1, 1) => "left_up",
            (0, 1) => "up",
            (1, 1) => "right_up",
            (-1, 0) => "left",
            (1, 0) => "right",
            (-1, -1) => "left_down",
            (0, -1) => "down",
            (1, -1) => "right_down",
            _ => "center"
        };
    }

    private CalibrationTarget AverageGaze(BridgeTrackingState state, bool leftUsable, bool rightUsable)
    {
        var totalWeight = 0.0;
        var x = 0.0;
        var y = 0.0;
        AddEye(state.Left, leftUsable);
        AddEye(state.Right, rightUsable);
        return totalWeight <= 0
            ? CalibrationTarget.Center
            : new CalibrationTarget(Math.Clamp(x / totalWeight, -1, 1), Math.Clamp(y / totalWeight, -1, 1));

        void AddEye(BridgeEyeState eye, bool usable)
        {
            if (!usable)
            {
                return;
            }

            var weight = Math.Max(0.01, eye.Confidence);
            x += eye.NormalizedX * weight;
            y += eye.NormalizedY * weight;
            totalWeight += weight;
        }
    }

    private bool IsUsableEye(BridgeEyeState eye) =>
        eye.Found && eye.Confidence >= _options.MinConfidence && eye.Openness >= _options.MinEyeOpenness;

    private static int QuantizeAxis(double value, double axisThreshold)
    {
        if (value <= -axisThreshold)
        {
            return -1;
        }

        return value >= axisThreshold ? 1 : 0;
    }

    private static int QuantizeAxisWithHysteresis(
        double value,
        int previousAxis,
        double enterThreshold,
        double releaseThreshold)
    {
        if (previousAxis < 0 && value <= -releaseThreshold)
        {
            return -1;
        }

        if (previousAxis > 0 && value >= releaseThreshold)
        {
            return 1;
        }

        return QuantizeAxis(value, enterThreshold);
    }

    private static (int X, int Y) AxesFromStageId(string? stageId) =>
        stageId?.ToLowerInvariant() switch
        {
            "left_up" => (-1, 1),
            "up" => (0, 1),
            "right_up" => (1, 1),
            "left" => (-1, 0),
            "right" => (1, 0),
            "left_down" => (-1, -1),
            "down" => (0, -1),
            "right_down" => (1, -1),
            _ => (0, 0)
        };
}
