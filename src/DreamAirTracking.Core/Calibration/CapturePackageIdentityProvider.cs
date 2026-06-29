using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;

namespace DreamAirTracking.Core.Calibration;

public sealed record CapturePackageIdentity(
    string SubjectId,
    string WearId,
    CaptureDeviceFingerprint Fingerprint);

public sealed record CaptureDeviceFingerprint(
    string Schema,
    string SubjectHash,
    string WearHash,
    string MachineNameHash,
    string UserNameHash,
    string HostNameHash,
    IReadOnlyList<string> Ipv4AddressHashes,
    string SessionHash,
    IReadOnlyList<string> SourceFields);

public static class CapturePackageIdentityProvider
{
    private const string Schema = "dream_air_tracking.capture_identity.v1";
    private const string Namespace = "dream_air_tracking.capture_package";

    public static CapturePackageIdentity Create(string sessionId)
    {
        var normalizedSession = Normalize(sessionId);
        var machineName = Normalize(Environment.MachineName);
        var userName = Normalize(Environment.UserName);
        var hostName = Normalize(SafeHostName());
        var ipv4 = GetLocalIpv4Addresses();

        var machineHash = ShortHash($"machine|{machineName}", 16);
        var userHash = ShortHash($"user|{userName}", 16);
        var hostHash = ShortHash($"host|{hostName}", 16);
        var ipHashes = ipv4.Select(address => ShortHash($"ipv4|{address}", 16)).ToArray();
        var sessionHash = ShortHash($"session|{normalizedSession}", 16);
        var subjectHash = ShortHash(
            $"subject|{machineName}|{userName}|{hostName}|{string.Join("|", ipv4)}",
            12);
        var wearHash = ShortHash($"wear|{subjectHash}|{normalizedSession}", 12);
        var safeSession = SafeToken(string.IsNullOrWhiteSpace(normalizedSession)
            ? DateTimeOffset.Now.ToString("yyyyMMdd_HHmmss")
            : normalizedSession);

        return new CapturePackageIdentity(
            $"subject_{subjectHash}",
            $"wear_{safeSession}_{wearHash}",
            new CaptureDeviceFingerprint(
                Schema,
                subjectHash,
                wearHash,
                machineHash,
                userHash,
                hostHash,
                ipHashes,
                sessionHash,
                new[] { "machine_name", "windows_user", "host_name", "local_ipv4", "session_id" }));
    }

    private static IReadOnlyList<string> GetLocalIpv4Addresses()
    {
        try
        {
            return NetworkInterface.GetAllNetworkInterfaces()
                .Where(adapter =>
                    adapter.OperationalStatus == OperationalStatus.Up &&
                    adapter.NetworkInterfaceType != NetworkInterfaceType.Loopback)
                .SelectMany(adapter => adapter.GetIPProperties().UnicastAddresses)
                .Where(address =>
                    address.Address.AddressFamily == AddressFamily.InterNetwork &&
                    !IPAddress.IsLoopback(address.Address))
                .Select(address => address.Address.ToString())
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(address => address, StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }
        catch
        {
            return Array.Empty<string>();
        }
    }

    private static string SafeHostName()
    {
        try
        {
            return Dns.GetHostName();
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string ShortHash(string value, int length)
    {
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes($"{Namespace}|{value}"));
        var text = Convert.ToHexString(bytes).ToLowerInvariant();
        return text[..Math.Min(length, text.Length)];
    }

    private static string Normalize(string? value) =>
        (value ?? string.Empty).Trim().ToLowerInvariant();

    private static string SafeToken(string value)
    {
        var builder = new StringBuilder(value.Length);
        foreach (var ch in value)
        {
            builder.Append(char.IsLetterOrDigit(ch) || ch is '_' or '-' ? ch : '_');
        }

        var token = builder.ToString().Trim('_');
        return string.IsNullOrWhiteSpace(token) ? DateTimeOffset.Now.ToString("yyyyMMdd_HHmmss") : token;
    }
}
