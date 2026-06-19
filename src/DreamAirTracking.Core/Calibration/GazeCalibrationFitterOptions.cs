namespace DreamAirTracking.Core.Calibration;

public sealed class GazeCalibrationFitterOptions
{
    public ISet<string> IgnoredStageIds { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "blink",
        "closed",
        "open"
    };

    public double MaxDeltaMs { get; set; } = 20;
    public double MinConfidence { get; set; } = 0.05;
    public double MinOpenness { get; set; } = 0.08;
    public double MaxStageOutlierDistancePixels { get; set; } = 18;
    public string ModelMode { get; set; } = "auto";
    public int MinQuadraticStages { get; set; } = 6;
    public double MinQuadraticResidualImprovement { get; set; } = 0.04;
    public double MinQuadraticResidualImprovementRatio { get; set; } = 0.15;
    public double MaxAutoQuadraticAverageResidual { get; set; } = 0.22;
    public double MaxAutoQuadraticStageResidual { get; set; } = 0.50;
    public double MinCenterDeadzone { get; set; } = 0.03;
    public double MaxCenterDeadzone { get; set; } = 0.18;
    public double CenterDeadzoneScale { get; set; } = 2.5;
    public bool AnchorCenterStage { get; set; } = true;
    public double CenterStageWeightMultiplier { get; set; } = 4.0;
}
