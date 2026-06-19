using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class CalibrationSessionStoreTests
{
    [Fact]
    public async Task SavesAndLoadsSessionDataset()
    {
        var directory = Path.Combine(Path.GetTempPath(), $"dream-air-calibration-{Guid.NewGuid():N}");
        try
        {
            var session = new CalibrationSession
            {
                SessionId = "session-a",
                ProfileInput = "tracking_profile_final.json",
                Operator = new CalibrationOperatorInfo { DisplayName = "tester" },
                Stages =
                {
                    new CalibrationStage { Order = 1, StageId = "center", Target = new CalibrationTarget(0, 0) }
                }
            };
            var labels = new[]
            {
                new CalibrationLabel
                {
                    StageId = "center",
                    Target = new CalibrationTarget(0, 0),
                    FrameStart = 1,
                    FrameEnd = 2,
                    Accepted = true,
                    BadFrames = { 2 }
                }
            };
            var pairs = new[]
            {
                new CalibrationFramePair(1, "center", "left,1.jpg", "right\"1.jpg", 1.5, true, 100, 90, 0.8, 1, true, 102, 91, 0.7, 1)
            };
            var metrics = new CalibrationMetrics
            {
                ValidPairs = 1,
                Quality = "good",
                StageResiduals =
                {
                    ["center"] = new CalibrationResidual { X = 0.01, Y = 0.02, Distance = 0.03 }
                }
            };

            await CalibrationSessionStore.SaveSessionAsync(session, directory);
            await CalibrationSessionStore.SaveLabelsAsync(labels, directory);
            await CalibrationSessionStore.SavePairsAsync(pairs, directory);
            await CalibrationSessionStore.SaveMetricsAsync(metrics, directory);

            var loadedSession = await CalibrationSessionStore.LoadSessionAsync(directory);
            var loadedLabels = await CalibrationSessionStore.LoadLabelsAsync(directory);
            var loadedPairs = await CalibrationSessionStore.LoadPairsAsync(directory);
            var loadedMetrics = await CalibrationSessionStore.LoadMetricsAsync(directory);

            Assert.Equal("session-a", loadedSession.SessionId);
            Assert.Equal("tester", loadedSession.Operator.DisplayName);
            Assert.Single(loadedLabels);
            Assert.Equal(2, loadedLabels[0].BadFrames[0]);
            Assert.Single(loadedPairs);
            Assert.Equal("left,1.jpg", loadedPairs[0].LeftFile);
            Assert.Equal("right\"1.jpg", loadedPairs[0].RightFile);
            Assert.Equal("good", loadedMetrics.Quality);
            Assert.Equal(0.03, loadedMetrics.StageResiduals["center"].Distance, 3);
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
    }
}
