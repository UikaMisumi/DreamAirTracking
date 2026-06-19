#!/usr/bin/env python3
"""Serve manifest eye images as fake BrokenEye MJPEG streams for runtime smoke tests."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def read_manifest_pairs(path: Path, limit: int) -> tuple[list[bytes], list[bytes]]:
    left: list[bytes] = []
    right: list[bytes] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            left_path = Path(row["left_file"])
            right_path = Path(row["right_file"])
            if not left_path.exists() or not right_path.exists():
                continue
            left.append(left_path.read_bytes())
            right.append(right_path.read_bytes())
            if len(left) >= limit:
                break
    if not left or not right:
        raise SystemExit(f"No usable image pairs found in {path}")
    return left, right


class FakeBrokenEyeHandler(BaseHTTPRequestHandler):
    server_version = "FakeBrokenEyeMJPEG/1.0"

    def do_GET(self) -> None:
        if self.path in {"/", ""}:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Fake BrokenEye MJPEG server. Use /eye/left or /eye/right.\n")
            return
        if self.path not in {"/eye/left", "/eye/right"}:
            self.send_response(404)
            self.end_headers()
            return
        side = "left" if self.path.endswith("/left") else "right"
        frames = self.server.left_frames if side == "left" else self.server.right_frames  # type: ignore[attr-defined]
        fps = float(self.server.fps)  # type: ignore[attr-defined]
        sleep_seconds = 1.0 / max(fps, 0.1)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for frame in itertools.cycle(frames):
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(sleep_seconds)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(format, *args)


class FakeBrokenEyeServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], left_frames: list[bytes], right_frames: list[bytes], fps: float, verbose: bool) -> None:
        super().__init__(server_address, handler_class)
        self.left_frames = left_frames
        self.right_frames = right_frames
        self.fps = fps
        self.verbose = verbose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runs/eye_manifest_v4_round6_clean_20260618/manifest_v4_clean.csv"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5599)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    left, right = read_manifest_pairs(args.manifest.resolve(), args.limit)
    server = FakeBrokenEyeServer((args.host, args.port), FakeBrokenEyeHandler, left, right, args.fps, args.verbose)
    print(f"Fake BrokenEye MJPEG: http://{args.host}:{args.port}/eye/left and /eye/right")
    print(f"Frames loaded: left={len(left)} right={len(right)} fps={args.fps}")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping fake server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
