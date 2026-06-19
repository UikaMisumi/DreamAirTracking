using System.Diagnostics;
using System.Net;
using System.Net.Http;
using System.Net.NetworkInformation;
using System.Text.Json;
using DreamAirTracking.Core.Bridge;

namespace DreamAirTracking.App.Services;

public sealed class PipelineDiagnosticsService : IDisposable
{
    private const int BrokenEyePort = 5555;
    private const int VrcftUdpPort = 9400;
    private static readonly TimeSpan ProbeTimeout = TimeSpan.FromMilliseconds(650);
    private static readonly TimeSpan BridgeFreshness = TimeSpan.FromSeconds(2.5);

    private readonly HttpClient _http = new();
    private int? _oscQueryPort;

    public async Task<PipelineDiagnosticsSnapshot> CaptureAsync(
        BridgeTrackingState? bridgeState,
        DateTimeOffset? bridgeStateAt,
        CancellationToken cancellationToken)
    {
        var brokenEyeLeftTask = ProbeBrokenEyeStreamAsync("left", cancellationToken);
        var brokenEyeRightTask = ProbeBrokenEyeStreamAsync("right", cancellationToken);
        var avatarTask = ProbeVrchatAvatarAsync(cancellationToken);

        var capturedAt = DateTimeOffset.Now;
        var bridgeMonitorLive = bridgeState is not null
            && bridgeStateAt is not null
            && capturedAt - bridgeStateAt.Value <= BridgeFreshness;

        var vrcftProcessRunning = IsProcessRunning("VRCFaceTracking");
        var vrcftModuleProcessRunning = IsProcessRunning("VRCFaceTracking.ModuleProcess");
        var vrcftUdp9400Listening = IsUdpPortListening(VrcftUdpPort);
        var wearTemplate = BridgeProcessService.Instance.LatestWearTemplateProbe
            ?? WearTemplateProbeResult.NotRun("runtime not probed");

        var brokenEyeLeftLive = await brokenEyeLeftTask;
        var brokenEyeRightLive = await brokenEyeRightTask;
        var avatar = await avatarTask;

        return new PipelineDiagnosticsSnapshot(
            capturedAt,
            brokenEyeLeftLive,
            brokenEyeRightLive,
            bridgeMonitorLive,
            bridgeState?.PupilDiameterEnabled == true,
            bridgeState?.PupilDiameterMode ?? "off",
            wearTemplate.Status,
            wearTemplate.Action,
            wearTemplate.TemplateId,
            wearTemplate.Distance,
            vrcftProcessRunning,
            vrcftModuleProcessRunning,
            vrcftUdp9400Listening,
            avatar.IsLive,
            avatar.Port,
            avatar.AvatarId,
            avatar.FtPupilDilationPresent,
            avatar.FtPupilDilation,
            avatar.FtPupilDiameterLeft,
            avatar.FtPupilDiameterRight,
            avatar.PupilParameterSummary,
            avatar.PupilGateAliasSummary,
            avatar.EyeDilationEnable,
            avatar.Vf68EyeDilationEnable);
    }

    public void Dispose()
    {
        _http.Dispose();
    }

    private async Task<bool> ProbeBrokenEyeStreamAsync(string eye, CancellationToken cancellationToken)
    {
        try
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(ProbeTimeout);
            using var request = new HttpRequestMessage(HttpMethod.Get, $"http://127.0.0.1:{BrokenEyePort}/eye/{eye}");
            using var response = await _http.SendAsync(
                request,
                HttpCompletionOption.ResponseHeadersRead,
                timeout.Token);
            var mediaType = response.Content.Headers.ContentType?.MediaType ?? string.Empty;
            return response.IsSuccessStatusCode
                && mediaType.Contains("multipart", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private async Task<VrchatAvatarProbe> ProbeVrchatAvatarAsync(CancellationToken cancellationToken)
    {
        if (_oscQueryPort is not null)
        {
            var cached = await TryProbeAvatarPortAsync(_oscQueryPort.Value, cancellationToken);
            if (cached.IsLive)
            {
                return cached;
            }

            _oscQueryPort = null;
        }

        var probes = GetLikelyHttpPorts()
            .Select(port => TryProbeAvatarPortAsync(port, cancellationToken))
            .ToArray();
        if (probes.Length == 0)
        {
            return VrchatAvatarProbe.Offline;
        }

        var results = await Task.WhenAll(probes);
        var live = results.FirstOrDefault(result => result.IsLive);
        if (live is not null && live.IsLive)
        {
            _oscQueryPort = live.Port;
            return live;
        }

        return VrchatAvatarProbe.Offline;
    }

    private async Task<VrchatAvatarProbe> TryProbeAvatarPortAsync(int port, CancellationToken cancellationToken)
    {
        try
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(ProbeTimeout);
            var json = await _http.GetStringAsync($"http://127.0.0.1:{port}/avatar", timeout.Token);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (!TryGetString(root, "FULL_PATH", out var fullPath)
                || !fullPath.Equals("/avatar", StringComparison.OrdinalIgnoreCase))
            {
                return VrchatAvatarProbe.Offline;
            }

            TryGetString(root, "NAME", out var avatarId);
            var leaves = CollectOscLeaves(root).ToArray();
            var pupilDilation = FindFloatParameter(leaves, "/v2/PupilDilation");
            var pupilDiameterLeft = FindFloatParameter(leaves, "/v2/PupilDiameterLeft");
            var pupilDiameterRight = FindFloatParameter(leaves, "/v2/PupilDiameterRight");
            var pupilParameterSummary = SummarizePupilParameters(leaves, pupilDilation.Value);
            var gateAliases = leaves
                .Where(leaf => leaf.FullPath.Contains("EyeDilationEnable", StringComparison.OrdinalIgnoreCase))
                .Select(leaf => $"{GetLeafName(leaf.FullPath)}={FormatOscValue(leaf.Node)}")
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            return new VrchatAvatarProbe(
                true,
                port,
                avatarId,
                pupilDilation.Found,
                pupilDilation.Value,
                pupilDiameterLeft.Value,
                pupilDiameterRight.Value,
                pupilParameterSummary,
                gateAliases.Length == 0 ? "not found" : string.Join(", ", gateAliases),
                TryGetOscNode(root, new[] { "parameters", "EyeDilationEnable" }, out var eyeGateNode)
                    && TryGetOscBool(eyeGateNode, out var eyeGate)
                        ? eyeGate
                        : null,
                TryGetOscNode(root, new[] { "parameters", "VF68_EyeDilationEnable" }, out var vf68GateNode)
                    && TryGetOscBool(vf68GateNode, out var vf68Gate)
                        ? vf68Gate
                        : null);
        }
        catch
        {
            return VrchatAvatarProbe.Offline;
        }
    }

    private static IEnumerable<int> GetLikelyHttpPorts()
    {
        var activePorts = IPGlobalProperties.GetIPGlobalProperties()
            .GetActiveTcpListeners()
            .Where(endpoint => IsLoopbackOrAny(endpoint.Address))
            .Select(endpoint => endpoint.Port)
            .Where(port => port >= 49152 && port != BrokenEyePort)
            .Distinct()
            .OrderByDescending(port => port >= 60000)
            .ThenByDescending(port => port)
            .Take(24);

        foreach (var port in new[] { 61447 }.Concat(activePorts).Distinct())
        {
            yield return port;
        }
    }

    private static bool IsProcessRunning(string processName)
    {
        foreach (var process in Process.GetProcessesByName(processName))
        {
            using (process)
            {
                try
                {
                    if (!process.HasExited)
                    {
                        return true;
                    }
                }
                catch
                {
                }
            }
        }

        return false;
    }

    private static bool IsUdpPortListening(int port)
    {
        try
        {
            return IPGlobalProperties.GetIPGlobalProperties()
                .GetActiveUdpListeners()
                .Any(endpoint => endpoint.Port == port && IsLoopbackOrAny(endpoint.Address));
        }
        catch
        {
            return false;
        }
    }

    private static bool IsLoopbackOrAny(IPAddress address)
        => IPAddress.IsLoopback(address)
           || address.Equals(IPAddress.Any)
           || address.Equals(IPAddress.IPv6Any);

    private static bool TryGetOscNode(JsonElement root, IReadOnlyList<string> path, out JsonElement node)
    {
        node = root;
        foreach (var segment in path)
        {
            if (!node.TryGetProperty("CONTENTS", out var contents)
                || contents.ValueKind != JsonValueKind.Object
                || !contents.TryGetProperty(segment, out node))
            {
                node = default;
                return false;
            }
        }

        return true;
    }

    private static bool TryGetOscDouble(JsonElement node, out double value)
    {
        value = 0;
        return TryGetValueElement(node, out var valueElement)
            && valueElement.ValueKind == JsonValueKind.Number
            && valueElement.TryGetDouble(out value);
    }

    private static bool TryGetOscBool(JsonElement node, out bool value)
    {
        value = false;
        return TryGetValueElement(node, out var valueElement)
            && (valueElement.ValueKind switch
            {
                JsonValueKind.True => SetBool(true, out value),
                JsonValueKind.False => SetBool(false, out value),
                _ => false
            });
    }

    private static bool TryGetValueElement(JsonElement node, out JsonElement valueElement)
    {
        valueElement = default;
        if (!node.TryGetProperty("VALUE", out var values)
            || values.ValueKind != JsonValueKind.Array
            || values.GetArrayLength() == 0)
        {
            return false;
        }

        valueElement = values[0];
        return true;
    }

    private static bool TryGetString(JsonElement node, string property, out string value)
    {
        value = string.Empty;
        if (!node.TryGetProperty(property, out var element) || element.ValueKind != JsonValueKind.String)
        {
            return false;
        }

        value = element.GetString() ?? string.Empty;
        return true;
    }

    private static bool SetBool(bool input, out bool output)
    {
        output = input;
        return true;
    }

    private static IEnumerable<OscLeaf> CollectOscLeaves(JsonElement node)
    {
        if (TryGetString(node, "FULL_PATH", out var fullPath) && node.TryGetProperty("VALUE", out _))
        {
            yield return new OscLeaf(fullPath, node);
        }

        if (!node.TryGetProperty("CONTENTS", out var contents) || contents.ValueKind != JsonValueKind.Object)
        {
            yield break;
        }

        foreach (var child in contents.EnumerateObject())
        {
            foreach (var leaf in CollectOscLeaves(child.Value))
            {
                yield return leaf;
            }
        }
    }

    private static (bool Found, double? Value) FindFloatParameter(IReadOnlyList<OscLeaf> leaves, string suffix)
    {
        foreach (var leaf in leaves)
        {
            if (leaf.FullPath.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)
                && TryGetOscDouble(leaf.Node, out var value))
            {
                return (true, value);
            }
        }

        return (false, null);
    }

    private static string SummarizePupilParameters(IReadOnlyList<OscLeaf> leaves, double? pupilDilation)
    {
        var pupilLeaves = leaves
            .Where(leaf => leaf.FullPath.Contains("/v2/Pupil", StringComparison.OrdinalIgnoreCase))
            .Select(leaf => GetLeafName(leaf.FullPath))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(8)
            .ToArray();

        if (pupilLeaves.Length == 0)
        {
            return "not found";
        }

        var value = pupilDilation is null ? "n/a" : pupilDilation.Value.ToString("0.000");
        return $"PupilDilation={value}; params={string.Join(", ", pupilLeaves)}";
    }

    private static string FormatOscValue(JsonElement node)
    {
        if (!TryGetValueElement(node, out var value))
        {
            return "n/a";
        }

        return value.ValueKind switch
        {
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            JsonValueKind.Number when value.TryGetDouble(out var number) => number.ToString("0.###"),
            JsonValueKind.String => value.GetString() ?? string.Empty,
            _ => value.ValueKind.ToString()
        };
    }

    private static string GetLeafName(string fullPath)
    {
        var index = fullPath.LastIndexOf('/');
        return index >= 0 && index < fullPath.Length - 1
            ? fullPath[(index + 1)..]
            : fullPath;
    }

    private sealed record VrchatAvatarProbe(
        bool IsLive,
        int? Port,
        string? AvatarId,
        bool FtPupilDilationPresent,
        double? FtPupilDilation,
        double? FtPupilDiameterLeft,
        double? FtPupilDiameterRight,
        string PupilParameterSummary,
        string PupilGateAliasSummary,
        bool? EyeDilationEnable,
        bool? Vf68EyeDilationEnable)
    {
        public static VrchatAvatarProbe Offline { get; } = new(
            false,
            null,
            null,
            false,
            null,
            null,
            null,
            "not found",
            "not found",
            null,
            null);
    }

    private sealed record OscLeaf(string FullPath, JsonElement Node);
}
