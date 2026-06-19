using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using System.Threading.Channels;
using DreamAirTracking.Core;
using DreamAirTracking.Core.Bridge;
using DreamAirTracking.Core.Input;
using DreamAirTracking.Core.Profiles;
using DreamAirTracking.Core.Tracking;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

var exitCode = await BridgeMain.RunAsync(args);
return exitCode;

internal static class BridgeMain
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintHelp();
            return 0;
        }

        if (args[0] != "run")
        {
            Console.Error.WriteLine($"Unknown command: {args[0]}");
            PrintHelp();
            return 1;
        }

        var host = GetOption(args, "--host") ?? "127.0.0.1";
        var port = int.Parse(GetOption(args, "--port") ?? "5555", CultureInfo.InvariantCulture);
        var profilePath = GetOption(args, "--profile") ?? "tracking_profile_final.json";
        var udpHost = GetOption(args, "--udp-host") ?? "127.0.0.1";
        var udpPort = int.Parse(GetOption(args, "--udp-port") ?? "9400", CultureInfo.InvariantCulture);
        var monitorUdpHost = GetOption(args, "--monitor-udp-host") ?? "127.0.0.1";
        var monitorUdpPort = int.Parse(GetOption(args, "--monitor-udp-port") ?? "9401", CultureInfo.InvariantCulture);
        var maxDeltaMs = double.Parse(GetOption(args, "--max-delta-ms") ?? "20", CultureInfo.InvariantCulture);
        var printEvery = int.Parse(GetOption(args, "--print-every") ?? "30", CultureInfo.InvariantCulture);
        var pupilCalibrationPath = GetOption(args, "--pupil-calibration");
        var quickGazeCalibrationPath = GetOption(args, "--quick-gaze-calibration");
        var quickGazeCalibrationSession = GetOption(args, "--quick-gaze-calibration-session");
        var enableEyeTracking = GetBoolOption(args, "--enable-eye-tracking", true);
        var enablePupilAssist = GetBoolOption(args, "--enable-pupil-assist", false);
        var enablePupilDiameter = GetBoolOption(args, "--enable-pupil-diameter", pupilCalibrationPath is not null);
        if (enablePupilDiameter && pupilCalibrationPath is null)
        {
            Console.Error.WriteLine("Pupil diameter output requested, but --pupil-calibration was not provided. Pupil diameter output is disabled.");
            enablePupilDiameter = false;
        }

        using var singleInstance = new BridgeSingleInstance(udpPort);
        if (!singleInstance.Acquired)
        {
            Console.Error.WriteLine($"Another DreamAirTracking Bridge is already running for UDP port {udpPort}.");
            return 2;
        }

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };

        var profile = await TrackingProfileStore.LoadOrDefaultAsync(profilePath);
        var pupilCalibration = enablePupilDiameter && pupilCalibrationPath is not null
            ? await PupilDiameterCalibrationStore.LoadAsync(pupilCalibrationPath, cts.Token)
            : null;
        QuickGazeCalibrationLayer? quickGazeCalibration = null;
        if (quickGazeCalibrationPath is not null)
        {
            try
            {
                quickGazeCalibration = await QuickGazeCalibrationLayer.LoadAsync(quickGazeCalibrationPath, quickGazeCalibrationSession, cts.Token);
            }
            catch (Exception ex) when (ex is InvalidDataException or JsonException)
            {
                Console.Error.WriteLine($"Quick gaze calibration could not be loaded: {ex.Message}");
                return 1;
            }
        }
        var pupilDiameterNormalizer = pupilCalibration is not null ? new PupilDiameterNormalizer(pupilCalibration) : null;
        var enableRawPupilDiameter = enablePupilDiameter && pupilDiameterNormalizer is not null;
        var channel = Channel.CreateBounded<CapturedJpeg>(new BoundedChannelOptions(32)
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.DropOldest
        });

        var leftTask = Task.Run(() => PumpHttpSideAsync(host, port, EyeSide.Left, channel.Writer, cts.Token));
        var rightTask = Task.Run(() => PumpHttpSideAsync(host, port, EyeSide.Right, channel.Writer, cts.Token));
        using var udp = new UdpClient();
        var endpoint = new IPEndPoint(IPAddress.Parse(udpHost), udpPort);
        using var monitorUdp = monitorUdpPort > 0 ? new UdpClient() : null;
        var monitorEndpoint = monitorUdpPort > 0
            ? new IPEndPoint(IPAddress.Parse(monitorUdpHost), monitorUdpPort)
            : null;
        var tracker = new EyeTracker(profile);
        var mapper = new PupilOffsetMapper();
        var pupilDiameterEstimator = new PupilDiameterEstimator();
        var opennessDetector = new EyeOpennessDetector();
        var opennessFilter = new BridgeOpennessFilter();
        var gazeFilter = new BridgeGazeFilter();
        var binocularGazeFilter = new BridgeBinocularGazeFilter();
        var closureGazeStabilizer = new BridgeClosureGazeStabilizer();
        var outputMapper = new BridgeOutputMapper(profile.Output);
        var leftQueue = new List<CapturedJpeg>();
        var rightQueue = new List<CapturedJpeg>();
        long sequence = 0;

        Console.WriteLine($"DreamAirTracking Bridge");
        Console.WriteLine($"BrokenEye HTTP: http://{host}:{port}");
        Console.WriteLine($"Profile: {Path.GetFullPath(profilePath)}");
        Console.WriteLine($"Eye tracking output: {(enableEyeTracking ? "enabled" : "disabled")}");
        Console.WriteLine($"Pupil assist output: {(enablePupilAssist ? "enabled" : "disabled")}");
        Console.WriteLine($"Raw pupil diameter output: {(enableRawPupilDiameter ? "enabled" : "disabled")}");
        if (enablePupilDiameter && pupilCalibrationPath is not null)
        {
            Console.WriteLine($"Pupil diameter calibration: {Path.GetFullPath(pupilCalibrationPath)}");
        }
        if (quickGazeCalibrationPath is not null)
        {
            Console.WriteLine($"Quick gaze calibration: {Path.GetFullPath(quickGazeCalibrationPath)} session={quickGazeCalibration?.Session ?? "n/a"}");
        }

        Console.WriteLine($"UDP output: {udpHost}:{udpPort}");
        if (monitorEndpoint is not null)
        {
            Console.WriteLine($"GUI monitor UDP: {monitorUdpHost}:{monitorUdpPort}");
        }

        Console.WriteLine("Press Ctrl+C to stop.");

        try
        {
            while (!cts.Token.IsCancellationRequested)
            {
                var packet = await channel.Reader.ReadAsync(cts.Token);
                var queue = packet.Side == EyeSide.Left ? leftQueue : rightQueue;
                queue.Add(packet);
                TrimQueue(queue, 10);

                var pair = TryTakeBestPair(leftQueue, rightQueue, maxDeltaMs);
                if (pair is null)
                {
                    continue;
                }

                var (left, right) = pair.Value;
                var leftDetection = tracker.Process(left.Frame);
                var rightDetection = tracker.Process(right.Frame);
                var leftMapped = mapper.Map(EyeSide.Left, leftDetection, profile);
                var rightMapped = mapper.Map(EyeSide.Right, rightDetection, profile);
                var leftOpenness = opennessDetector.Detect(left.Frame, new EyeOpennessDetectorOptions { Roi = profile.Left.Roi });
                var rightOpenness = opennessDetector.Detect(right.Frame, new EyeOpennessDetectorOptions { Roi = profile.Right.Roi });
                var filteredOpenness = opennessFilter.Update(leftOpenness, rightOpenness);
                var (leftPupilDiameter, rightPupilDiameter) = enablePupilAssist
                    ? PupilAssistEstimator.EstimateLinked(
                        filteredOpenness.Left,
                        leftMapped.Confidence,
                        filteredOpenness.Right,
                        rightMapped.Confidence,
                        leftOpenness.ApertureHeight,
                        rightOpenness.ApertureHeight)
                    : (
                        GetRawPupilDiameterState(EyeSide.Left, pupilDiameterNormalizer, pupilDiameterEstimator, left.Frame, leftDetection, leftOpenness.Openness),
                        GetRawPupilDiameterState(EyeSide.Right, pupilDiameterNormalizer, pupilDiameterEstimator, right.Frame, rightDetection, rightOpenness.Openness));
                var monocularGaze = gazeFilter.Update(leftMapped, rightMapped);
                var fusedGaze = binocularGazeFilter.Update(monocularGaze, leftMapped, rightMapped);
                var stabilizedGaze = closureGazeStabilizer.Update(fusedGaze, filteredOpenness);
                var finalGaze = outputMapper.Update(stabilizedGaze);
                if (quickGazeCalibration is not null)
                {
                    finalGaze = quickGazeCalibration.Apply(finalGaze);
                }
                var deltaMs = Math.Abs((left.ReceivedAt - right.ReceivedAt).TotalMilliseconds);

                var state = new BridgeTrackingState(
                    Interlocked.Increment(ref sequence),
                    DateTimeOffset.UtcNow,
                    deltaMs,
                    ToBridgeEyeState(leftDetection, leftMapped, finalGaze.LeftX, finalGaze.LeftY, monocularGaze.LeftX, monocularGaze.LeftY, leftOpenness, filteredOpenness.Left, leftPupilDiameter),
                    ToBridgeEyeState(rightDetection, rightMapped, finalGaze.RightX, finalGaze.RightY, monocularGaze.RightX, monocularGaze.RightY, rightOpenness, filteredOpenness.Right, rightPupilDiameter))
                {
                    EyeTrackingEnabled = enableEyeTracking,
                    PupilDiameterEnabled = enablePupilAssist || enableRawPupilDiameter,
                    PupilDiameterMode = enablePupilAssist ? "assist" : enableRawPupilDiameter ? "raw" : "off"
                };

                var payload = JsonSerializer.SerializeToUtf8Bytes(state, JsonOptions);
                await udp.SendAsync(payload, payload.Length, endpoint);
                if (monitorUdp is not null && monitorEndpoint is not null)
                {
                    await monitorUdp.SendAsync(payload, payload.Length, monitorEndpoint);
                }

                if (state.Sequence % printEvery == 0)
                {
                    Console.WriteLine($"{state.Sequence:000000} dt={deltaMs:0.0} L=({state.Left.Found},{state.Left.NormalizedX:0.00},{state.Left.NormalizedY:0.00},mono={state.Left.MonocularNormalizedX:0.00},{state.Left.MonocularNormalizedY:0.00},raw={state.Left.RawNormalizedX:0.00},{state.Left.RawNormalizedY:0.00},c={state.Left.Confidence:0.00},o={state.Left.Openness:0.00},po={state.Left.PupilOpenness:0.00},pd=({state.Left.PupilDiameterQuality},{state.Left.PupilDiameterNormalized:0.00},{state.Left.PupilDiameterPx:0.0}px),ao={state.Left.ApertureOpenness:0.00},ah={state.Left.ApertureHeight}) R=({state.Right.Found},{state.Right.NormalizedX:0.00},{state.Right.NormalizedY:0.00},mono={state.Right.MonocularNormalizedX:0.00},{state.Right.MonocularNormalizedY:0.00},raw={state.Right.RawNormalizedX:0.00},{state.Right.RawNormalizedY:0.00},c={state.Right.Confidence:0.00},o={state.Right.Openness:0.00},po={state.Right.PupilOpenness:0.00},pd=({state.Right.PupilDiameterQuality},{state.Right.PupilDiameterNormalized:0.00},{state.Right.PupilDiameterPx:0.0}px),ao={state.Right.ApertureOpenness:0.00},ah={state.Right.ApertureHeight})");
                }
            }
        }
        catch (ChannelClosedException ex) when (!cts.Token.IsCancellationRequested)
        {
            Console.Error.WriteLine($"BrokenEye stream stopped: {ex.InnerException?.Message ?? ex.Message}");
            Console.Error.WriteLine($"Check that BrokenEye is running and serving /eye/left and /eye/right on http://{host}:{port}.");
            return 2;
        }

        await Task.WhenAny(Task.WhenAll(leftTask, rightTask), Task.Delay(1000));
        return 0;
    }

    private static PupilDiameterTrackingState GetRawPupilDiameterState(
        EyeSide side,
        PupilDiameterNormalizer? normalizer,
        PupilDiameterEstimator estimator,
        EyeFrame frame,
        PupilDetection detection,
        double apertureOpenness)
    {
        return normalizer is not null
            ? normalizer.Normalize(side, estimator.Estimate(frame, detection), apertureOpenness)
            : PupilDiameterTrackingState.NotFound("pupil diameter calibration is not loaded");
    }

    private static BridgeEyeState ToBridgeEyeState(
        PupilDetection detection,
        NormalizedEyePoint mapped,
        double filteredX,
        double filteredY,
        double monocularX,
        double monocularY,
        EyeOpennessDetection openness,
        double filteredOpenness,
        PupilDiameterTrackingState pupilDiameter)
    {
        var pupilOpenness = detection.Found && detection.Confidence > 0.055 ? 1.0 : 0.0;
        return new BridgeEyeState(
            mapped.Found,
            mapped.Confidence,
            detection.Found ? detection.CenterX : 0,
            detection.Found ? detection.CenterY : 0,
            filteredX,
            filteredY,
            filteredOpenness)
        {
            PupilOpenness = pupilOpenness,
            ApertureOpenness = openness.Openness,
            AperturePeakDarkFraction = openness.PeakDarkFraction,
            ApertureHeight = openness.ApertureHeight,
            ApertureReason = openness.Reason,
            RawNormalizedX = mapped.X,
            RawNormalizedY = mapped.Y,
            MonocularNormalizedX = monocularX,
            MonocularNormalizedY = monocularY,
            PupilDiameterFound = pupilDiameter.Found,
            PupilDiameterQuality = pupilDiameter.Quality,
            PupilDiameterConfidence = pupilDiameter.Confidence,
            PupilDiameterPx = pupilDiameter.DiameterPx,
            PupilDiameterNormalized = pupilDiameter.Normalized,
            PupilDiameterAxisRatio = pupilDiameter.AxisRatio,
            PupilDiameterReason = pupilDiameter.Reason
        };
    }

    private static async Task PumpHttpSideAsync(string host, int port, EyeSide side, ChannelWriter<CapturedJpeg> writer, CancellationToken cancellationToken)
    {
        try
        {
            var sideName = side == EyeSide.Left ? "left" : "right";
            var uri = new Uri($"http://{host}:{port}/eye/{sideName}");
            var client = new MjpegStreamClient();
            var index = 0;

            await foreach (var jpeg in client.ReadJpegFramesAsync(uri, cancellationToken))
            {
                index++;
                var frame = await DecodeJpegFrameAsync(jpeg, side);
                await writer.WriteAsync(new CapturedJpeg(side, index, frame, DateTimeOffset.UtcNow), cancellationToken);
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            writer.TryComplete(ex);
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

    private static void TrimQueue<T>(List<T> queue, int maxCount)
    {
        while (queue.Count > maxCount)
        {
            queue.RemoveAt(0);
        }
    }

    private static string? GetOption(string[] args, string name)
    {
        for (var i = 0; i < args.Length; i++)
        {
            if (args[i] == name && i + 1 < args.Length)
            {
                return args[i + 1];
            }
        }

        return null;
    }

    private static bool GetBoolOption(string[] args, string name, bool fallback)
    {
        var raw = GetOption(args, name);
        if (raw is null)
        {
            return fallback;
        }

        return raw.Trim().ToLowerInvariant() switch
        {
            "1" or "true" or "yes" or "on" => true,
            "0" or "false" or "no" or "off" => false,
            _ => fallback
        };
    }

    private static void PrintHelp()
    {
        Console.WriteLine("""
        DreamAirTracking Bridge

        Commands:
          run [--profile tracking_profile_final.json] [--host 127.0.0.1] [--port 5555] [--udp-host 127.0.0.1] [--udp-port 9400] [--monitor-udp-host 127.0.0.1] [--monitor-udp-port 9401] [--enable-eye-tracking true|false] [--enable-pupil-assist true|false] [--enable-pupil-diameter true|false] [--pupil-calibration pupil_diameter_calibration.json] [--quick-gaze-calibration quick_calibration_layer.json] [--quick-gaze-calibration-session session_id]
        """);
    }

    private sealed class BridgeSingleInstance : IDisposable
    {
        private readonly Mutex _mutex;

        public BridgeSingleInstance(int udpPort)
        {
            _mutex = new Mutex(false, $"DreamAirTracking.Bridge.Udp.{udpPort}");
            try
            {
                Acquired = _mutex.WaitOne(0);
            }
            catch (AbandonedMutexException)
            {
                Acquired = true;
            }
        }

        public bool Acquired { get; }

        public void Dispose()
        {
            if (Acquired)
            {
                try
                {
                    _mutex.ReleaseMutex();
                }
                catch (ApplicationException)
                {
                    // Mutex ownership is thread-affine; async continuations can dispose on a different thread.
                    // Process exit will release the named mutex handle.
                }
            }

            _mutex.Dispose();
        }
    }

    private sealed record CapturedJpeg(EyeSide Side, int Index, EyeFrame Frame, DateTimeOffset ReceivedAt);
}
