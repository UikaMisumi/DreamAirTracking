#!/usr/bin/env python3
"""Golden-recording regression gate (E10).

Replays frozen BrokenEye recordings through the full Python runtime and compares
per-stage gaze / openness metrics against a frozen baseline. A new model package
must not regress beyond tolerance on the golden set before it may ship.

Frames stay OUT of the repo (biometric data): tests/golden/recordings.json points
to LOCAL manifest paths. Baselines are metrics-only (safe to commit) and live in
tests/golden/expected_metrics/<name>.json.

  python run_golden_regression.py                    # gate config defaults vs baselines
  python run_golden_regression.py --update-baseline  # (re)freeze baselines from current model
  python run_golden_regression.py --model X.onnx --metadata X.json   # gate a candidate model
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_live_multitask_csv import summarize_csv  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
DEFAULT_CONFIG = GOLDEN_DIR / "recordings.json"
BASELINE_DIR = GOLDEN_DIR / "expected_metrics"


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_port(port: int, proc: subprocess.Popen, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(port):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.3)
    return False


def compact_metrics(summary: dict) -> dict:
    stages = {}
    for stage, data in (summary.get("by_stage") or {}).items():
        stages[stage] = {
            "count": data.get("count"),
            "smooth_median": (data.get("smooth") or {}).get("median"),
            "openness_min_median": (data.get("openness_min") or {}).get("median"),
        }
    fields = summary.get("fields") or {}
    return {
        "row_count": summary.get("row_count"),
        "fps": round(summary.get("fps") or 0.0, 2),
        "left_openness_median": (fields.get("left_openness") or {}).get("median"),
        "right_openness_median": (fields.get("right_openness") or {}).get("median"),
        "by_stage": stages,
    }


def run_recording(rec: dict, model: str, metadata: str, out_dir: Path, port: int) -> dict:
    manifest = Path(rec["manifest"])
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    server = subprocess.Popen(
        [
            sys.executable, str(SCRIPT_DIR / "fake_brokeneye_mjpeg_server.py"),
            "--manifest", str(manifest), "--port", str(port),
            "--fps", str(rec.get("fps", 120)), "--limit", str(rec.get("limit", 300)),
        ],
        cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        if not wait_port(port, server):
            out = server.stdout.read() if server.stdout else ""
            raise SystemExit(f"fake server failed for {rec['name']}: {out[:400]}")
        time.sleep(0.4)
        cmd = [
            sys.executable, str(SCRIPT_DIR / "predict_live_multitask.py"),
            "--host", "127.0.0.1", "--port", str(port),
            "--onnx", str(model), "--metadata", str(metadata),
            "--output-dir", str(out_dir), "--duration-seconds", str(rec.get("duration_seconds", 6)),
            "--print-every", "100000",
        ] + list(rec.get("runtime_flags", []))
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"runtime failed for {rec['name']}: {result.stderr[-400:]}")
        csvs = sorted(out_dir.glob("live_multitask_*.csv"))
        if not csvs:
            raise SystemExit(f"no CSV produced for {rec['name']}")
        return compact_metrics(summarize_csv(csvs[-1]))
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


def compare(baseline: dict, current: dict, gaze_tol: float, open_tol: float) -> list[str]:
    regressions: list[str] = []
    for stage, base in baseline.get("by_stage", {}).items():
        cur = current.get("by_stage", {}).get(stage)
        if not cur:
            regressions.append(f"{stage}: missing in candidate run")
            continue
        bm, cm = base.get("smooth_median"), cur.get("smooth_median")
        if bm and cm:
            for i, axis in enumerate("xy"):
                if bm[i] is not None and cm[i] is not None and abs(cm[i] - bm[i]) > gaze_tol:
                    regressions.append(f"{stage}: gaze {axis} drift {cm[i] - bm[i]:+.3f} > {gaze_tol}")
        bo, co = base.get("openness_min_median"), cur.get("openness_min_median")
        if bo is not None and co is not None and abs(co - bo) > open_tol:
            regressions.append(f"{stage}: openness_min drift {co - bo:+.3f} > {open_tol}")
    return regressions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--model", type=Path, help="Override model ONNX (default: config-level or per-recording).")
    ap.add_argument("--metadata", type=Path)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--port", type=int, default=5601)
    ap.add_argument("--gaze-tol", type=float, default=0.06)
    ap.add_argument("--open-tol", type=float, default=0.10)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/golden_regression")
    args = ap.parse_args()

    if not args.config.exists():
        raise SystemExit(f"No golden config at {args.config}. See tests/golden/README.md.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_ok = True
    results = []
    for rec in config.get("recordings", []):
        model = args.model or rec.get("model") or config.get("model")
        metadata = args.metadata or rec.get("metadata") or config.get("metadata")
        if not model or not metadata:
            raise SystemExit(f"recording {rec['name']} has no model/metadata (set in config or pass --model).")
        out_dir = args.output_dir / rec["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        current = run_recording(rec, model, metadata, out_dir, args.port)
        baseline_path = BASELINE_DIR / f"{rec['name']}.json"
        if args.update_baseline:
            baseline_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
            results.append({"name": rec["name"], "action": "baseline_written", "row_count": current["row_count"]})
            continue
        if not baseline_path.exists():
            raise SystemExit(f"No baseline for {rec['name']}; run with --update-baseline first.")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        regressions = compare(baseline, current, args.gaze_tol, args.open_tol)
        overall_ok = overall_ok and not regressions
        results.append({
            "name": rec["name"],
            "passed": not regressions,
            "regressions": regressions,
            "row_count": current["row_count"],
        })

    print(json.dumps({"update_baseline": args.update_baseline, "passed": overall_ok, "results": results},
                     indent=2, ensure_ascii=False))
    return 0 if (overall_ok or args.update_baseline) else 2


if __name__ == "__main__":
    raise SystemExit(main())
