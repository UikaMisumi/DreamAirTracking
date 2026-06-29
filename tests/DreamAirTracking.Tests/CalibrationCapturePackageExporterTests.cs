using System.IO.Compression;
using System.Text.Json;
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
                Device = new CalibrationDeviceInfo { Source = "BrokenEye HTTP", Host = "192.168.50.181", Port = 5555 },
                ProfileInput = Path.Combine(root, "secret", "tracking_profile.json"),
                Operator = new CalibrationOperatorInfo { DisplayName = "local user" },
                Stages = CalibrationStageCatalog.CreateNinePointGazeStages().ToList()
            };
            session.Stages.Add(new CalibrationStage
            {
                Order = 10,
                StageId = "closed",
                Target = CalibrationTarget.Center
            });
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
                },
                new CalibrationLabel
                {
                    StageId = "closed",
                    Target = CalibrationTarget.Center,
                    FrameStart = 2,
                    FrameEnd = 2,
                    Accepted = true
                },
                new CalibrationLabel
                {
                    StageId = "right_down",
                    Target = new CalibrationTarget(1, -1),
                    FrameStart = 3,
                    FrameEnd = 3,
                    Accepted = false,
                    OperatorVerdict = "bad"
                }
            }, sessionDirectory);
            await CalibrationSessionStore.SavePairsAsync(new[]
            {
                new CalibrationFramePair(
                    1,
                    "center",
                    Path.Combine(sessionDirectory, "frames", "center_000001_left.jpg"),
                    Path.Combine(sessionDirectory, "frames", "center_000001_right.jpg"),
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
                    1.0),
                new CalibrationFramePair(
                    2,
                    "closed",
                    Path.Combine(sessionDirectory, "frames", "closed_000002_left.jpg"),
                    Path.Combine(sessionDirectory, "frames", "closed_000002_right.jpg"),
                    4,
                    true,
                    0.5,
                    0.5,
                    0.9,
                    0.02,
                    true,
                    0.5,
                    0.5,
                    0.9,
                    0.03),
                new CalibrationFramePair(
                    3,
                    "right_down",
                    Path.Combine(sessionDirectory, "frames", "right_down_000003_left.jpg"),
                    Path.Combine(sessionDirectory, "frames", "right_down_000003_right.jpg"),
                    4,
                    true,
                    0.9,
                    0.1,
                    0.2,
                    1.0,
                    true,
                    0.9,
                    0.1,
                    0.2,
                    1.0)
            }, sessionDirectory);
            await File.WriteAllBytesAsync(Path.Combine(sessionDirectory, "frames", "center_000001_left.jpg"), new byte[] { 1, 2, 3 });
            await File.WriteAllBytesAsync(Path.Combine(sessionDirectory, "frames", "center_000001_right.jpg"), new byte[] { 4, 5, 6 });
            await File.WriteAllBytesAsync(Path.Combine(sessionDirectory, "frames", "closed_000002_left.jpg"), new byte[] { 7, 8, 9 });
            await File.WriteAllBytesAsync(Path.Combine(sessionDirectory, "frames", "closed_000002_right.jpg"), new byte[] { 10, 11, 12 });
            await File.WriteAllBytesAsync(Path.Combine(sessionDirectory, "frames", "right_down_000003_left.jpg"), new byte[] { 13, 14, 15 });
            await File.WriteAllBytesAsync(Path.Combine(sessionDirectory, "frames", "right_down_000003_right.jpg"), new byte[] { 16, 17, 18 });

            var result = await new CalibrationCapturePackageExporter().ExportAsync(new CalibrationCapturePackageExportOptions
            {
                SessionDirectory = sessionDirectory,
                OutputDirectory = root,
                SubjectId = "subject_test",
                WearId = "wear_001",
                RuntimeModelId = "dreamair-test"
            });

            Assert.True(File.Exists(result.OutputZipPath));
            Assert.StartsWith("subject_", result.SubjectId);
            Assert.StartsWith("wear_", result.WearId);
            using var archive = ZipFile.OpenRead(result.OutputZipPath);
            Assert.NotNull(archive.GetEntry("manifest.json"));
            Assert.NotNull(archive.GetEntry("device.json"));
            Assert.NotNull(archive.GetEntry("capture_protocol.json"));
            Assert.NotNull(archive.GetEntry("session.json"));
            Assert.NotNull(archive.GetEntry("labels.jsonl"));
            Assert.NotNull(archive.GetEntry("pairs.csv"));
            Assert.NotNull(archive.GetEntry("training_manifest_v3.csv"));
            Assert.NotNull(archive.GetEntry("calibration_report.md"));
            Assert.NotNull(archive.GetEntry("runtime/runtime_summary.json"));
            Assert.NotNull(archive.GetEntry("frames/center_000001_left.jpg"));
            Assert.NotNull(archive.GetEntry("frames/center_000001_right.jpg"));
            Assert.NotNull(archive.GetEntry("frames/closed_000002_left.jpg"));
            Assert.NotNull(archive.GetEntry("frames/right_down_000003_left.jpg"));

            foreach (var entry in archive.Entries.Where(entry => !entry.FullName.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)))
            {
                await using var stream = entry.Open();
                using var reader = new StreamReader(stream);
                var text = await reader.ReadToEndAsync();
                Assert.DoesNotContain(root, text, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain("local user", text, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain("192.168.50.181", text, StringComparison.OrdinalIgnoreCase);
            }

            var deviceJson = await ReadEntryAsync(archive, "device.json");
            using (var document = JsonDocument.Parse(deviceJson))
            {
                Assert.False(document.RootElement.TryGetProperty("host", out _));
                Assert.False(document.RootElement.TryGetProperty("port", out _));
                Assert.True(document.RootElement.TryGetProperty("sourceFields", out _));
                Assert.True(document.RootElement.TryGetProperty("fingerprint", out _));
            }

            var pairsCsv = await ReadEntryAsync(archive, "pairs.csv");
            Assert.DoesNotContain(sessionDirectory, pairsCsv, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("frames/center_000001_left.jpg", pairsCsv);

            var manifest = await ReadEntryAsync(archive, "training_manifest_v3.csv");
            Assert.DoesNotContain(sessionDirectory, manifest, StringComparison.OrdinalIgnoreCase);
            var rows = manifest.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries);
            var header = rows[0].Split(',');
            var parsed = rows.Skip(1).Select(row => ToRow(header, row)).ToArray();
            var center = parsed.Single(row => row["stage"] == "center");
            var closed = parsed.Single(row => row["stage"] == "closed");
            var rejected = parsed.Single(row => row["stage"] == "right_down");

            Assert.Equal("1", center["gaze_weight"]);
            Assert.Equal("0", center["openness_valid_left"]);
            Assert.Equal("1", center["sample_weight"]);
            Assert.Equal("0", closed["gaze_weight"]);
            Assert.Equal("1", closed["openness_valid_left"]);
            Assert.Equal("0", closed["openness_target_left"]);
            Assert.Equal("0", closed["sample_weight"]);
            Assert.Equal("0", rejected["gaze_weight"]);
            Assert.Equal("0", rejected["openness_valid_left"]);
            Assert.Equal("0", rejected["sample_weight"]);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    private static async Task<string> ReadEntryAsync(ZipArchive archive, string name)
    {
        var entry = archive.GetEntry(name) ?? throw new InvalidOperationException($"Missing {name}");
        await using var stream = entry.Open();
        using var reader = new StreamReader(stream);
        return await reader.ReadToEndAsync();
    }

    private static Dictionary<string, string> ToRow(string[] header, string row)
    {
        var values = row.Split(',');
        return header.Zip(values, (name, value) => (name, value))
            .ToDictionary(item => item.name, item => item.value);
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
