using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Threading.Channels;
using DreamAirTracking.Core;
using DreamAirTracking.Core.Input;
using DreamAirTracking.Core.Runtime;

namespace DreamAirTracking.App.Services;

/// <summary>
/// In-app eyelid capture with time-RAMP labels (S3). Replaces the terminal-based staged
/// protocol: the schedule interleaves anchor holds with slow-close / slow-open ramps where the
/// per-frame openness label is the TIME progress through the ramp — a dense, continuous [0,1]
/// supervision signal that does not depend on any image measurement (the classical aperture
/// score is unusable on these side-view cameras, and stage constants trained the model into a
/// 4-level quantizer). One session produces both training labels and the v2 runtime calibration
/// (open_p95 / closed_p05 / half_p50) by running the installed main model on every pair.
/// </summary>
public sealed class EyelidRampCaptureService
{
    public static EyelidRampCaptureService Instance { get; } = new();

    private EyelidRampCaptureService() { }

    public bool IsRunning { get; private set; }

    /// <summary>One scheduled stage. Openness targets are functions of stage progress (0..1).</summary>
    public sealed record RampStage(
        string Id,
        string Prompt,
        double Seconds,
        Func<double, double> OpennessLeft,
        Func<double, double> OpennessRight,
        bool OpennessValidLeft = true,
        bool OpennessValidRight = true,
        double SettleSeconds = 0.8,      // hold stages: initial window marked invalid
        double RampTrimFraction = 0.0,   // ramp stages: head/tail fraction marked invalid
        double WideTarget = 0.0,
        bool WideValid = false,
        double SquintTarget = 0.0,
        bool SquintValid = false);

    public static IReadOnlyList<RampStage> DefaultSchedule { get; } = new[]
    {
        new RampStage("open_relaxed", "Relax, eyes open, look forward", 3.0, _ => 1.0, _ => 1.0,
            WideTarget: 0.0, WideValid: true, SquintTarget: 0.0, SquintValid: true),
        new RampStage("open_wide", "Open eyes WIDE", 2.0, _ => 1.0, _ => 1.0,
            WideTarget: 1.0, WideValid: true, SquintTarget: 0.0, SquintValid: true),
        new RampStage("slow_close_ramp", "SLOWLY close your eyes, steady pace", 5.0,
            p => 1.0 - p, p => 1.0 - p, SettleSeconds: 0.0, RampTrimFraction: 0.10),
        new RampStage("closed", "Keep eyes CLOSED", 2.0, _ => 0.0, _ => 0.0),
        new RampStage("slow_open_ramp", "SLOWLY open your eyes, steady pace", 5.0,
            p => p, p => p, SettleSeconds: 0.0, RampTrimFraction: 0.10),
        // squint supervises the squint head ONLY — its openness constant (0.7) polluted the
        // openness labels into a quantizer attractor, so openness is NOT supervised here.
        new RampStage("squint", "SQUINT (narrow your eyes)", 2.0, _ => 0.7, _ => 0.7,
            OpennessValidLeft: false, OpennessValidRight: false,
            SquintTarget: 1.0, SquintValid: true),
        new RampStage("wink_left", "Close LEFT eye only", 2.0, _ => 0.0, _ => 1.0),
        new RampStage("wink_right", "Close RIGHT eye only", 2.0, _ => 1.0, _ => 0.0),
        new RampStage("open_confirm", "Eyes open again", 2.0, _ => 1.0, _ => 1.0,
            WideTarget: 0.0, WideValid: true, SquintTarget: 0.0, SquintValid: true),
    };

    public sealed record RampProgress(
        string StageId,
        string Prompt,
        double StageRemainingSeconds,
        double TargetOpenness,      // for the UI bar (mean of both eyes' targets)
        double TotalFraction,
        int PairsCaptured);

    public sealed record RampResult(
        bool Success,
        string Message,
        string? SessionDirectory,
        int PairCount,
        string? CalibrationPath);

    public async Task<RampResult> RunAsync(
        string host,
        int port,
        string mainOnnxPath,
        int imageSize,
        string sessionDirectory,
        IProgress<RampProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (IsRunning)
        {
            return new RampResult(false, "An eyelid capture is already running.", null, 0, null);
        }

        IsRunning = true;
        try
        {
            return await RunCoreAsync(host, port, mainOnnxPath, imageSize, sessionDirectory, progress, cancellationToken);
        }
        finally
        {
            IsRunning = false;
        }
    }

    private static async Task<RampResult> RunCoreAsync(
        string host,
        int port,
        string mainOnnxPath,
        int imageSize,
        string sessionDirectory,
        IProgress<RampProgress>? progress,
        CancellationToken cancellationToken)
    {
        var schedule = DefaultSchedule;
        double totalSeconds = schedule.Sum(s => s.Seconds);

        Directory.CreateDirectory(sessionDirectory);
        var framesDirectory = Path.Combine(sessionDirectory, "frames");
        Directory.CreateDirectory(framesDirectory);

        using var model = new MultitaskEyeModel(mainOnnxPath, imageSize);

        var channel = Channel.CreateBounded<CapturedJpeg>(new BoundedChannelOptions(32)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.DropOldest
        });

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(TimeSpan.FromSeconds(totalSeconds + 15));
        var token = cts.Token;
        var leftTask = Task.Run(() => PumpSideAsync(host, port, "left", channel.Writer, token), token);
        var rightTask = Task.Run(() => PumpSideAsync(host, port, "right", channel.Writer, token), token);

        var rows = new List<CapturedRow>();
        var leftQueue = new List<CapturedJpeg>();
        var rightQueue = new List<CapturedJpeg>();
        const double maxDeltaMs = 30.0;

        // wait for the first frame before the clock starts, so stage 1 isn't eaten by stream spin-up
        try
        {
            var first = await channel.Reader.ReadAsync(token);
            (first.Side == "left" ? leftQueue : rightQueue).Add(first);
        }
        catch (OperationCanceledException)
        {
            return new RampResult(false, "BrokenEye stream did not produce frames on 127.0.0.1:" + port + ".", null, 0, null);
        }

        var clock = Stopwatch.StartNew();
        int stageIndex = -1;
        double stageStart = 0;

        try
        {
            while (true)
            {
                double elapsed = clock.Elapsed.TotalSeconds;
                if (elapsed >= totalSeconds)
                {
                    break;
                }

                // resolve current stage
                double acc = 0;
                int idx = 0;
                for (; idx < schedule.Count; idx++)
                {
                    if (elapsed < acc + schedule[idx].Seconds) break;
                    acc += schedule[idx].Seconds;
                }
                if (idx >= schedule.Count) break;
                var stage = schedule[idx];
                if (idx != stageIndex)
                {
                    stageIndex = idx;
                    stageStart = acc;
                    _ = Task.Run(() => StageBeep(stage.Id));
                }
                double stageElapsed = elapsed - stageStart;
                double p = Math.Clamp(stageElapsed / Math.Max(0.001, stage.Seconds), 0.0, 1.0);

                progress?.Report(new RampProgress(
                    stage.Id, stage.Prompt, Math.Max(0, stage.Seconds - stageElapsed),
                    (stage.OpennessLeft(p) + stage.OpennessRight(p)) * 0.5,
                    elapsed / totalSeconds, rows.Count));

                CapturedJpeg packet;
                try
                {
                    packet = await channel.Reader.ReadAsync(token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                (packet.Side == "left" ? leftQueue : rightQueue).Add(packet);
                Trim(leftQueue); Trim(rightQueue);
                var pair = TryTakeBestPair(leftQueue, rightQueue, maxDeltaMs);
                if (pair is null) continue;

                // validity: settle window on holds, head/tail trim on ramps
                bool inSettle = stage.SettleSeconds > 0 && stageElapsed < stage.SettleSeconds;
                bool inTrim = stage.RampTrimFraction > 0 &&
                              (p < stage.RampTrimFraction || p > 1.0 - stage.RampTrimFraction);
                bool timeValid = !inSettle && !inTrim;

                var seq = rows.Count + 1;
                string leftFile = $"frames/{stage.Id}_{seq:000000}_left.jpg";
                string rightFile = $"frames/{stage.Id}_{seq:000000}_right.jpg";
                await File.WriteAllBytesAsync(Path.Combine(framesDirectory, Path.GetFileName(leftFile)), pair.Value.Left.Jpeg, token);
                await File.WriteAllBytesAsync(Path.Combine(framesDirectory, Path.GetFileName(rightFile)), pair.Value.Right.Jpeg, token);

                var outputs = model.RunMain(
                    EyeTensorPreprocessor.Preprocess(pair.Value.Left.Jpeg, imageSize),
                    EyeTensorPreprocessor.Preprocess(pair.Value.Right.Jpeg, imageSize));

                rows.Add(new CapturedRow(
                    seq, stage.Id, Math.Round(elapsed, 3), leftFile, rightFile,
                    Math.Abs((pair.Value.Left.ReceivedAt - pair.Value.Right.ReceivedAt).TotalMilliseconds),
                    stage.OpennessLeft(p), stage.OpennessRight(p),
                    timeValid && stage.OpennessValidLeft, timeValid && stage.OpennessValidRight,
                    stage.WideTarget, timeValid && stage.WideValid,
                    stage.SquintTarget, timeValid && stage.SquintValid,
                    outputs.Openness.Left, outputs.Openness.Right));
            }
        }
        finally
        {
            cts.Cancel();
            await Task.WhenAny(Task.WhenAll(leftTask, rightTask), Task.Delay(500, CancellationToken.None));
        }

        if (rows.Count < 60)
        {
            return new RampResult(false,
                $"Capture produced only {rows.Count} pairs — is BrokenEye streaming? Nothing was saved to the registry.",
                sessionDirectory, rows.Count, null);
        }

        WriteLabelsCsv(sessionDirectory, rows);
        var calibrationPath = WriteCalibration(sessionDirectory, rows);
        return new RampResult(true,
            $"Captured {rows.Count} pairs. Calibration written (open/closed/half anchors); ramp labels ready for training.",
            sessionDirectory, rows.Count, calibrationPath);
    }

    private static void WriteLabelsCsv(string sessionDirectory, List<CapturedRow> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("sequence,stage,elapsed_s,left_file,right_file,delta_ms," +
                      "openness_target_left,openness_target_right,openness_valid_left,openness_valid_right," +
                      "wide_target,wide_valid,squint_target,squint_valid,model_openness_left,model_openness_right");
        foreach (var r in rows)
        {
            sb.AppendLine(string.Join(",",
                r.Sequence, r.Stage, F(r.ElapsedS), r.LeftFile, r.RightFile, F(r.DeltaMs),
                F(r.OpennessTargetLeft), F(r.OpennessTargetRight),
                r.OpennessValidLeft ? 1 : 0, r.OpennessValidRight ? 1 : 0,
                F(r.WideTarget), r.WideValid ? 1 : 0, F(r.SquintTarget), r.SquintValid ? 1 : 0,
                F(r.ModelOpennessLeft), F(r.ModelOpennessRight)));
        }
        File.WriteAllText(Path.Combine(sessionDirectory, "eyelid_ramp_labels.csv"), sb.ToString());
    }

    // v2 calibration: open/closed anchors from the hold stages, half from ramp frames near mid.
    private static string? WriteCalibration(string sessionDirectory, List<CapturedRow> rows)
    {
        static double Percentile(List<double> values, double pct)
        {
            var sorted = values.OrderBy(v => v).ToList();
            double rank = (pct / 100.0) * (sorted.Count - 1);
            int lo = (int)Math.Floor(rank);
            int hi = (int)Math.Ceiling(rank);
            return lo == hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
        }

        string? Side(string side, out string payload)
        {
            payload = string.Empty;
            bool left = side == "left";
            double Model(CapturedRow r) => left ? r.ModelOpennessLeft : r.ModelOpennessRight;
            double Target(CapturedRow r) => left ? r.OpennessTargetLeft : r.OpennessTargetRight;
            bool Valid(CapturedRow r) => left ? r.OpennessValidLeft : r.OpennessValidRight;

            var open = rows.Where(r => r.Stage is "open_relaxed" or "open_confirm" && Valid(r)).Select(Model).ToList();
            var closed = rows.Where(r => r.Stage == "closed" && Valid(r)).Select(Model).ToList();
            var half = rows.Where(r => r.Stage is "slow_close_ramp" or "slow_open_ramp" && Valid(r)
                                       && Target(r) >= 0.40 && Target(r) <= 0.60).Select(Model).ToList();
            if (open.Count < 10 || closed.Count < 10) return null;

            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append($"\"open_p95\": {F(Percentile(open, 95))}, \"closed_p05\": {F(Percentile(closed, 5))}, ");
            sb.Append($"\"model_open_samples\": {open.Count}, \"model_closed_samples\": {closed.Count}");
            if (half.Count >= 10)
            {
                sb.Append($", \"half_p50\": {F(Percentile(half, 50))}, \"model_half_samples\": {half.Count}");
            }
            sb.Append("}");
            payload = sb.ToString();
            return payload;
        }

        if (Side("left", out var leftPayload) is null || Side("right", out var rightPayload) is null)
        {
            return null;
        }

        var json =
            "{\n" +
            "  \"schema\": \"dreamair.openness_calibration.v2\",\n" +
            $"  \"created_at\": \"{DateTimeOffset.UtcNow:yyyy-MM-ddTHH:mm:ssZ}\",\n" +
            "  \"source\": \"in_app_ramp_capture\",\n" +
            $"  \"left\": {leftPayload},\n" +
            $"  \"right\": {rightPayload}\n" +
            "}\n";
        var path = Path.Combine(sessionDirectory, "openness_calibration.json");
        File.WriteAllText(path, json);
        return path;
    }

    private static string F(double v) => v.ToString("0.######", CultureInfo.InvariantCulture);

    private static void StageBeep(string stageId)
    {
        try
        {
            // distinct tones so the schedule can be followed with eyes closed
            uint freq = stageId switch
            {
                "closed" => 880,
                "slow_close_ramp" => 660,
                "slow_open_ramp" => 990,
                _ => 1320,
            };
            NativeBeep(freq, 180);
        }
        catch
        {
        }
    }

    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    private static extern bool NativeBeep(uint frequency, uint duration);

    private static async Task PumpSideAsync(string host, int port, string side, ChannelWriter<CapturedJpeg> writer, CancellationToken token)
    {
        try
        {
            using var httpClient = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
            var client = new MjpegStreamClient(httpClient);
            var uri = new Uri($"http://{host}:{port}/eye/{side}");
            await foreach (var jpeg in client.ReadJpegFramesAsync(uri, token))
            {
                await writer.WriteAsync(new CapturedJpeg(side, jpeg, DateTimeOffset.UtcNow), token);
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch when (!token.IsCancellationRequested)
        {
        }
    }

    private static (CapturedJpeg Left, CapturedJpeg Right)? TryTakeBestPair(List<CapturedJpeg> leftQueue, List<CapturedJpeg> rightQueue, double maxDeltaMs)
    {
        if (leftQueue.Count == 0 || rightQueue.Count == 0) return null;
        int bestL = -1, bestR = -1;
        double best = double.MaxValue;
        for (int li = 0; li < leftQueue.Count; li++)
        {
            for (int ri = 0; ri < rightQueue.Count; ri++)
            {
                double d = Math.Abs((leftQueue[li].ReceivedAt - rightQueue[ri].ReceivedAt).TotalMilliseconds);
                if (d < best) { best = d; bestL = li; bestR = ri; }
            }
        }
        if (best > maxDeltaMs)
        {
            if (leftQueue.Count == 0) rightQueue.RemoveAt(0);
            else if (rightQueue.Count == 0) leftQueue.RemoveAt(0);
            else if (leftQueue[0].ReceivedAt <= rightQueue[0].ReceivedAt) leftQueue.RemoveAt(0);
            else rightQueue.RemoveAt(0);
            return null;
        }
        var left = leftQueue[bestL];
        var right = rightQueue[bestR];
        leftQueue.RemoveAt(bestL);
        rightQueue.RemoveAt(bestR);
        leftQueue.RemoveAll(f => f.ReceivedAt < left.ReceivedAt.AddMilliseconds(-maxDeltaMs * 2));
        rightQueue.RemoveAll(f => f.ReceivedAt < right.ReceivedAt.AddMilliseconds(-maxDeltaMs * 2));
        return (left, right);
    }

    private static void Trim(List<CapturedJpeg> queue)
    {
        while (queue.Count > 10) queue.RemoveAt(0);
    }

    private sealed record CapturedJpeg(string Side, byte[] Jpeg, DateTimeOffset ReceivedAt);

    private sealed record CapturedRow(
        int Sequence, string Stage, double ElapsedS, string LeftFile, string RightFile, double DeltaMs,
        double OpennessTargetLeft, double OpennessTargetRight, bool OpennessValidLeft, bool OpennessValidRight,
        double WideTarget, bool WideValid, double SquintTarget, bool SquintValid,
        double ModelOpennessLeft, double ModelOpennessRight);
}
