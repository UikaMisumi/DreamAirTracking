using System.Diagnostics;
using System.Threading.Channels;
using DreamAirTracking.Core;
using DreamAirTracking.Core.Calibration;
using DreamAirTracking.Core.Input;
using DreamAirTracking.Core.Profiles;
using DreamAirTracking.Core.Tracking;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DreamAirTracking.App.Services;

public sealed class BrokenEyeCalibrationCaptureService
{
    public async Task<StageCaptureResult> CaptureStageAsync(
        CalibrationStage stage,
        TrackingProfile profile,
        string sessionDirectory,
        long firstSequence,
        StageCaptureOptions options,
        IProgress<StageCaptureProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(sessionDirectory);
        var framesDirectory = Path.Combine(sessionDirectory, "frames");
        Directory.CreateDirectory(framesDirectory);

        var channel = Channel.CreateBounded<CapturedJpeg>(new BoundedChannelOptions(32)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.DropOldest
        });

        using var captureCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        captureCts.CancelAfter(options.CaptureDuration + options.StreamGracePeriod);
        var token = captureCts.Token;
        var leftTask = Task.Run(() => PumpHttpSideAsync(options.Host, options.Port, EyeSide.Left, channel.Writer, token), token);
        var rightTask = Task.Run(() => PumpHttpSideAsync(options.Host, options.Port, EyeSide.Right, channel.Writer, token), token);

        var tracker = new EyeTracker(profile);
        var opennessDetector = new EyeOpennessDetector();
        var leftQueue = new List<CapturedJpeg>();
        var rightQueue = new List<CapturedJpeg>();
        var pairs = new List<CalibrationFramePair>();
        var stopwatch = Stopwatch.StartNew();
        var startedCollecting = false;

        try
        {
            while (pairs.Count < options.MaxPairs)
            {
                if (startedCollecting && stopwatch.Elapsed >= options.CaptureDuration)
                {
                    break;
                }

                CapturedJpeg packet;
                try
                {
                    packet = await channel.Reader.ReadAsync(token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                if (!startedCollecting)
                {
                    startedCollecting = true;
                    stopwatch.Restart();
                }

                var queue = packet.Side == EyeSide.Left ? leftQueue : rightQueue;
                queue.Add(packet);
                TrimQueue(queue, 10);

                var pair = TryTakeBestPair(leftQueue, rightQueue, options.MaxDeltaMs);
                if (pair is null)
                {
                    continue;
                }

                var sequence = firstSequence + pairs.Count;
                var captured = await BuildFramePairAsync(
                    stage,
                    profile,
                    tracker,
                    opennessDetector,
                    framesDirectory,
                    pair.Value.Left,
                    pair.Value.Right,
                    sequence,
                    token);

                pairs.Add(captured);
                progress?.Report(new StageCaptureProgress(
                    pairs.Count,
                    options.MaxPairs,
                    Math.Clamp(stopwatch.Elapsed.TotalMilliseconds / options.CaptureDuration.TotalMilliseconds, 0, 1)));
            }
        }
        finally
        {
            captureCts.Cancel();
            await Task.WhenAny(Task.WhenAll(leftTask, rightTask), Task.Delay(500, CancellationToken.None));
        }

        return new StageCaptureResult(stage.StageId, pairs, BuildQuickCheck(stage.StageId, pairs, options));
    }

    private static async Task<CalibrationFramePair> BuildFramePairAsync(
        CalibrationStage stage,
        TrackingProfile profile,
        EyeTracker tracker,
        EyeOpennessDetector opennessDetector,
        string framesDirectory,
        CapturedJpeg left,
        CapturedJpeg right,
        long sequence,
        CancellationToken cancellationToken)
    {
        var leftDetection = tracker.Process(left.Frame);
        var rightDetection = tracker.Process(right.Frame);
        var leftOpenness = opennessDetector.Detect(left.Frame, new EyeOpennessDetectorOptions { Roi = profile.Left.Roi });
        var rightOpenness = opennessDetector.Detect(right.Frame, new EyeOpennessDetectorOptions { Roi = profile.Right.Roi });
        var deltaMs = Math.Abs((left.ReceivedAt - right.ReceivedAt).TotalMilliseconds);

        var leftFile = Path.Combine("frames", $"{stage.StageId}_{sequence:000000}_left.jpg");
        var rightFile = Path.Combine("frames", $"{stage.StageId}_{sequence:000000}_right.jpg");
        await File.WriteAllBytesAsync(Path.Combine(framesDirectory, Path.GetFileName(leftFile)), left.Jpeg, cancellationToken);
        await File.WriteAllBytesAsync(Path.Combine(framesDirectory, Path.GetFileName(rightFile)), right.Jpeg, cancellationToken);

        return new CalibrationFramePair(
            sequence,
            stage.StageId,
            leftFile,
            rightFile,
            deltaMs,
            leftDetection.Found,
            leftDetection.CenterX,
            leftDetection.CenterY,
            leftDetection.Confidence,
            leftOpenness.Found ? leftOpenness.Openness : 1,
            rightDetection.Found,
            rightDetection.CenterX,
            rightDetection.CenterY,
            rightDetection.Confidence,
            rightOpenness.Found ? rightOpenness.Openness : 1);
    }

    private static StageQuickCheck BuildQuickCheck(string stageId, IReadOnlyList<CalibrationFramePair> pairs, StageCaptureOptions options)
    {
        var syncLatePairs = pairs.Count(pair => pair.DeltaMs > options.MaxDeltaMs);
        var notFoundPairs = pairs.Count(pair => !pair.LeftFound || !pair.RightFound);
        var lowConfidencePairs = pairs.Count(pair =>
            pair.LeftConfidence < options.MinConfidence ||
            pair.RightConfidence < options.MinConfidence);
        var lowOpennessPairs = pairs.Count(pair => pair.LeftOpenness < options.MinOpenness || pair.RightOpenness < options.MinOpenness);
        var validPairs = pairs.Count(pair =>
            pair.DeltaMs <= options.MaxDeltaMs &&
            pair.LeftFound &&
            pair.RightFound &&
            pair.LeftConfidence >= options.MinConfidence &&
            pair.RightConfidence >= options.MinConfidence &&
            pair.LeftOpenness >= options.MinOpenness &&
            pair.RightOpenness >= options.MinOpenness);
        var averageDeltaMs = pairs.Count == 0 ? 0 : pairs.Average(pair => pair.DeltaMs);
        var averageConfidence = pairs.Count == 0
            ? 0
            : pairs.Average(pair => (pair.LeftConfidence + pair.RightConfidence) / 2);

        var quality = validPairs >= Math.Min(20, options.MaxPairs * 0.7) && averageDeltaMs <= options.MaxDeltaMs
            ? "good"
            : validPairs >= Math.Min(10, options.MaxPairs * 0.4)
                ? "fair"
                : "poor";
        var recommendation = CalibrationStageReviewAdvisor.Recommend(new CalibrationStageReviewInput(
            stageId,
            pairs.Count,
            validPairs,
            lowOpennessPairs,
            lowConfidencePairs,
            notFoundPairs,
            syncLatePairs,
            quality));

        return new StageQuickCheck(
            pairs.Count,
            validPairs,
            averageDeltaMs,
            averageConfidence,
            lowOpennessPairs,
            lowConfidencePairs,
            notFoundPairs,
            syncLatePairs,
            quality,
            recommendation.SuggestedAction,
            recommendation.Reason,
            recommendation.ImprovementPlan);
    }

    private static async Task PumpHttpSideAsync(string host, int port, EyeSide side, ChannelWriter<CapturedJpeg> writer, CancellationToken cancellationToken)
    {
        try
        {
            var sideName = side == EyeSide.Left ? "left" : "right";
            var uri = new Uri($"http://{host}:{port}/eye/{sideName}");
            using var httpClient = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
            var client = new MjpegStreamClient(httpClient);
            var index = 0;

            await foreach (var jpeg in client.ReadJpegFramesAsync(uri, cancellationToken))
            {
                index++;
                var frame = await DecodeJpegFrameAsync(jpeg, side);
                await writer.WriteAsync(new CapturedJpeg(side, index, jpeg, frame, DateTimeOffset.UtcNow), cancellationToken);
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception) when (!cancellationToken.IsCancellationRequested)
        {
            return;
        }
    }

    private static async Task<EyeFrame> DecodeJpegFrameAsync(byte[] bytes, EyeSide side)
    {
        using var image = Image.Load<L8>(bytes);
        var pixels = new byte[image.Width * image.Height];
        image.CopyPixelDataTo(pixels);
        await Task.CompletedTask;
        return new EyeFrame(side, image.Width, image.Height, 8, pixels, DateTimeOffset.UtcNow);
    }

    private static (CapturedJpeg Left, CapturedJpeg Right)? TryTakeBestPair(List<CapturedJpeg> leftQueue, List<CapturedJpeg> rightQueue, double maxDeltaMs)
    {
        if (leftQueue.Count == 0 || rightQueue.Count == 0)
        {
            return null;
        }

        var bestLeft = -1;
        var bestRight = -1;
        var bestDelta = double.MaxValue;

        for (var li = 0; li < leftQueue.Count; li++)
        {
            for (var ri = 0; ri < rightQueue.Count; ri++)
            {
                var delta = Math.Abs((leftQueue[li].ReceivedAt - rightQueue[ri].ReceivedAt).TotalMilliseconds);
                if (delta < bestDelta)
                {
                    bestDelta = delta;
                    bestLeft = li;
                    bestRight = ri;
                }
            }
        }

        if (bestDelta > maxDeltaMs)
        {
            DropOldestFrame(leftQueue, rightQueue);
            return null;
        }

        var left = leftQueue[bestLeft];
        var right = rightQueue[bestRight];
        leftQueue.RemoveAt(bestLeft);
        rightQueue.RemoveAt(bestRight);
        DropOlderThan(leftQueue, left.ReceivedAt.AddMilliseconds(-maxDeltaMs * 2));
        DropOlderThan(rightQueue, right.ReceivedAt.AddMilliseconds(-maxDeltaMs * 2));
        return (left, right);
    }

    private static void DropOldestFrame(List<CapturedJpeg> leftQueue, List<CapturedJpeg> rightQueue)
    {
        if (leftQueue.Count == 0)
        {
            rightQueue.RemoveAt(0);
            return;
        }

        if (rightQueue.Count == 0)
        {
            leftQueue.RemoveAt(0);
            return;
        }

        if (leftQueue[0].ReceivedAt <= rightQueue[0].ReceivedAt)
        {
            leftQueue.RemoveAt(0);
        }
        else
        {
            rightQueue.RemoveAt(0);
        }
    }

    private static void TrimQueue(List<CapturedJpeg> queue, int maxCount)
    {
        while (queue.Count > maxCount)
        {
            queue.RemoveAt(0);
        }
    }

    private static void DropOlderThan(List<CapturedJpeg> queue, DateTimeOffset cutoff)
    {
        queue.RemoveAll(frame => frame.ReceivedAt < cutoff);
    }

    private sealed record CapturedJpeg(EyeSide Side, int Index, byte[] Jpeg, EyeFrame Frame, DateTimeOffset ReceivedAt);
}

public sealed record StageCaptureOptions(
    string Host,
    int Port,
    TimeSpan CaptureDuration,
    TimeSpan StreamGracePeriod,
    int MaxPairs,
    double MaxDeltaMs,
    double MinConfidence,
    double MinOpenness);

public sealed record StageCaptureProgress(
    int CapturedPairs,
    int TargetPairs,
    double ElapsedFraction);

public sealed record StageCaptureResult(
    string StageId,
    IReadOnlyList<CalibrationFramePair> Pairs,
    StageQuickCheck QuickCheck);

public sealed record StageQuickCheck(
    int PairCount,
    int ValidPairCount,
    double AverageDeltaMs,
    double AverageConfidence,
    int LowOpennessPairs,
    int LowConfidencePairs,
    int NotFoundPairs,
    int SyncLatePairs,
    string Quality,
    string SuggestedAction,
    string ReviewReason,
    string ImprovementPlan);
