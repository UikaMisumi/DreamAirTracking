using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using DreamAirTracking.Core.Bridge;

namespace DreamAirTracking.App.Services;

public sealed class BridgeMonitorService
{
    public static BridgeMonitorService Instance { get; } = new();

    private const int MonitorUdpPort = 9401;
    private static readonly JsonSerializerOptions JsonOptions = AppJsonOptions.Web();

    private readonly object _gate = new();
    private CancellationTokenSource? _cts;
    private UdpClient? _udp;

    private BridgeMonitorService()
    {
    }

    public event EventHandler<BridgeTrackingState>? StateReceived;

    public BridgeTrackingState? LatestState { get; private set; }

    public DateTimeOffset? LatestStateAt { get; private set; }

    public string? ErrorMessage { get; private set; }

    public void Start()
    {
        lock (_gate)
        {
            if (_cts is not null)
            {
                return;
            }

            _cts = new CancellationTokenSource();
            try
            {
                _udp = new UdpClient(new IPEndPoint(IPAddress.Loopback, MonitorUdpPort));
                ErrorMessage = null;
            }
            catch (SocketException ex)
            {
                ErrorMessage = $"Could not listen on 127.0.0.1:{MonitorUdpPort}: {ex.Message}";
                _cts.Dispose();
                _cts = null;
                return;
            }

            var udp = _udp;
            var token = _cts.Token;
            _ = Task.Run(() => MonitorLoopAsync(udp, token), token);
        }
    }

    private async Task MonitorLoopAsync(UdpClient udp, CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            UdpReceiveResult result;
            try
            {
                result = await udp.ReceiveAsync(token);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch (ObjectDisposedException)
            {
                return;
            }
            catch (SocketException) when (token.IsCancellationRequested)
            {
                return;
            }
            catch (SocketException ex)
            {
                ErrorMessage = ex.Message;
                continue;
            }

            BridgeTrackingState? state;
            try
            {
                state = JsonSerializer.Deserialize<BridgeTrackingState>(result.Buffer, JsonOptions);
            }
            catch (JsonException)
            {
                continue;
            }

            if (state is null)
            {
                continue;
            }

            LatestState = state;
            LatestStateAt = DateTimeOffset.Now;
            StateReceived?.Invoke(this, state);
        }
    }
}
