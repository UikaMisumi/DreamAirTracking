using DreamAirTracking.Core.Runtime;

namespace DreamAirTracking.App.Services;

/// <summary>
/// Hosts the in-process native <see cref="NeuralEyeRuntime"/> on a background task (E11).
/// Isolates the native engine lifecycle so BridgeProcessService can branch to it without a
/// child process. The runtime sends UDP 9400/9401 itself, so the monitor + VRCFT paths are
/// unchanged.
/// </summary>
public sealed class NativeRuntimeHost
{
    private CancellationTokenSource? _cts;
    private Task? _task;

    public bool IsRunning => _task is { IsCompleted: false };

    /// <summary>Start the runtime. Returns false (and calls onError) if ONNX init fails synchronously.</summary>
    public bool Start(NeuralRuntimeConfig config, Action? onStopped, Action<Exception>? onError)
    {
        Stop();
        NeuralEyeRuntime runtime;
        try
        {
            runtime = new NeuralEyeRuntime(config); // loads ONNX sessions — may throw
        }
        catch (Exception ex)
        {
            onError?.Invoke(ex);
            return false;
        }

        var cts = new CancellationTokenSource();
        _cts = cts;
        _task = Task.Run(async () =>
        {
            try
            {
                await runtime.RunAsync(null, cts.Token);
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                onError?.Invoke(ex);
            }
            finally
            {
                runtime.Dispose();
                onStopped?.Invoke();
            }
        });
        return true;
    }

    public void Stop()
    {
        try { _cts?.Cancel(); } catch { /* already disposed */ }
        _cts = null;
        _task = null;
    }
}
