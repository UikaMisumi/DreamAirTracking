using System.IO.Compression;
using DreamAirTracking.Core.Calibration;

namespace DreamAirTracking.Tests;

public sealed class CalibrationCapturePackageExporterTests
{
    [Fact]
    public async Task ExportWritesTrainingPackageWithoutAbsolutePaths()
    {
        var root = Path.Combine(Path.GetTempPath(), "dreamair-package-" + Guid.NewGuid().ToString("N"));
        var sessionDirectory = Path.Combine(root, "calibration_data", "session_a");
        Directory.CreateDirectory(Path.Combine(sessionDirectory, "frames"));
        try
        {
            var session = new CalibrationSession
            {
                SessionId = "session_a",
                CreatedAt = DateTimeOffset.Parse("2026-06-19T10:00:00Z"),
                ProfileInput = Path.Combine(root, "secret", "tracking_profile.json"),
                Operator = new CalibrationOperatorInfo { DisplayName = "local user" },
                Stages = CalibrationStageCatalog.CreateNinePointGazeStages().ToList()
            };
            await CalibrationSessionStore.SaveSessionAsync(session, sessionDirectory);
            await CalibrationSessionStore.SaveLabelsAsync(new[]
            {
                new CalibrationLabel
                {
                    StageId = "center",
                    Target = CalibrationTarget.Center,
                    FrameStart = 1,
                    FrameEnd = 1,
                    Accepted = true
                }
            }, sessionDirectory);
            await CalibrationSessionStore.SavePairsAsync(new[]
            {
                new CalibrationFramePair(
                    1,
                    "center",
                    "frames/center_000001_left.jpg",
                    "frames/center_000001_right.jpg",
                    4,
                    true,
                    0.5,
                    0.5,
                    0.2,
                    1.0,
                    true,
                    0.5,
                    0.5,
                    0.2,
                    1.0)
            }, sessionDirectory);
            await File.WriteAllBytesAsync(Path.Combine(sessionDirectory, "frames", "center_000001_left.jpg"), new byte[] { 1, 2, 3 });
            await File.WriteAllBytesAsync(Path.Combine(sessionDirectory, "frames", "center_000001_right.jpg"), new byte[] { 4, 5, 6 });

            var result = await new CalibrationCapturePackageExporter().ExportAsync(new CalibrationCapturePackageExportOptions
            {
                SessionDirectory = sessionDirectory,
                OutputDirectory = root,
                SubjectId = "subject_test",
                WearId = "wear_001",
                RuntimeModelId = "dreamair-test"
            });

            Assert.True(File.Exists(result.OutputZipPath));
            using var archive = ZipFile.OpenRead(result.OutputZipPath);
            Assert.NotNull(archive.GetEntry("manifest.json"));
            Assert.NotNull(archive.GetEntry("device.json"));
            Assert.NotNull(archive.GetEntry("capture_protocol.json"));
            Assert.NotNull(archive.GetEntry("session.json"));
            Assert.NotNull(archive.GetEntry("labels.jsonl"));
            Assert.NotNull(archive.GetEntry("pairs.csv"));
            Assert.NotNull(archive.GetEntry("calibration_report.md"));
            Assert.NotNull(archive.GetEntry("runtime/runtime_summary.json"));
            Assert.NotNull(archive.GetEntry("frames/center_000001_left.jpg"));
            Assert.NotNull(archive.GetEntry("frames/center_000001_right.jpg"));

            foreach (var entry in archive.Entries.Where(entry => !entry.FullName.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)))
            {
                await using var stream = entry.Open();
                using var reader = new StreamReader(stream);
                var text = await reader.ReadToEndAsync();
                Assert.DoesNotContain(root, text, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain("local user", text, StringComparison.OrdinalIgnoreCase);
            }
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }
}

public sealed class ModelRegistryTests
{
    [Fact]
    public void RegistryResolvesRelativeModelPackagePaths()
    {
        var root = Path.Combine(Path.GetTempPath(), "dreamair-registry-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(root, "models"));
        try
        {
            File.WriteAllText(
                Path.Combine(root, "models", "model_registry.json"),
                """
                {
                  "schema": "dream_air_tracking.model_registry.v1",
                  "models": [
                    {
                      "id": "dreamair-main",
                      "displayName": "Dream Air Main",
                      "deviceFamily": "Dream Air",
                      "role": "main",
                      "runtime": "predict_live_multitask",
                      "onnx": "models/dreamair-main/model.onnx",
                      "metadata": "models/dreamair-main/metadata.json",
                      "outputs": ["gaze_xy", "openness_lr", "pupil_lr", "confidence"],
                      "default": true
                    }
                  ]
                }
                """);

            var registry = DreamAirTracking.Core.Models.ModelRegistry.TryLoadFromRepo(root);
            var model = registry?.FindDefaultMain();

            Assert.NotNull(model);
            Assert.Equal("dreamair-main", model!.Id);
            Assert.Equal(Path.Combine(root, "models", "dreamair-main", "model.onnx"), model.ResolvedOnnx);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }
}
