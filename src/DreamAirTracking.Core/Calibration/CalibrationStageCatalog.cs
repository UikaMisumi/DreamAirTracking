namespace DreamAirTracking.Core.Calibration;

public static class CalibrationStageCatalog
{
    public static IReadOnlyList<CalibrationStage> CreateFivePointGazeStages() =>
        CreateDefaultGazeStages().Take(5).ToArray();

    public static IReadOnlyList<CalibrationStage> CreateNinePointGazeStages() =>
        CreateDefaultGazeStages().Take(9).ToArray();

    public static IReadOnlyList<CalibrationStage> CreateDefaultGazeStages() =>
        new[]
        {
            Stage(1, "center", 0, 0, "Center"),
            Stage(2, "up", 0, 1, "Up"),
            Stage(3, "down", 0, -1, "Down"),
            Stage(4, "left", -1, 0, "Left"),
            Stage(5, "right", 1, 0, "Right"),
            Stage(6, "left_up", -1, 1, "Left up"),
            Stage(7, "right_up", 1, 1, "Right up"),
            Stage(8, "left_down", -1, -1, "Left down"),
            Stage(9, "right_down", 1, -1, "Right down"),
            Stage(10, "center_confirm", 0, 0, "Center check"),
            Stage(11, "closed", 0, 0, "Closed"),
            Stage(12, "open", 0, 0, "Open")
        };

    private static CalibrationStage Stage(int order, string stageId, double x, double y, string displayName) =>
        new()
        {
            Order = order,
            StageId = stageId,
            Target = new CalibrationTarget(x, y),
            DisplayName = displayName
        };
}
