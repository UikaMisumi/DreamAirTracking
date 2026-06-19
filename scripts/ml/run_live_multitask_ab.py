#!/usr/bin/env python3
"""Run live multitask A/B validation against BrokenEye streams."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_live_multitask_csv import summarize_csv  # noqa: E402
from check_live_multitask_ab import check_report  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parent.parent


VARIANTS = {
    "dual": {
        "label": "dual_round2_main_round8_expression",
        "onnx": REPO_ROOT / "runs/eye_multitask_siamese_per_eye_openness_tune_round2_20260617/eye_multitask.onnx",
        "metadata": REPO_ROOT / "runs/eye_multitask_siamese_per_eye_openness_tune_round2_20260617/eye_multitask.metadata.json",
        "expression_onnx": REPO_ROOT / "runs/eye_multitask_siamese_v4_round8_headmask_freezebn_20260618/eye_multitask.onnx",
        "expression_metadata": REPO_ROOT / "runs/eye_multitask_siamese_v4_round8_headmask_freezebn_20260618/eye_multitask.metadata.json",
        "expression_every_n_frames": "3",
        "expression_ema_alpha": "0.65",
    },
    "split": {
        "label": "split_round2_geometry_round8_expression",
        "onnx": REPO_ROOT / "runs/eye_multitask_split_round2_geometry_round8_expression_20260618/eye_multitask.onnx",
        "metadata": REPO_ROOT / "runs/eye_multitask_split_round2_geometry_round8_expression_20260618/eye_multitask.metadata.json",
    },
}


def brokeneye_ready(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_brokeneye(host: str, port: int, wait_seconds: float) -> bool:
    deadline = time.time() + max(0.0, wait_seconds)
    while time.time() <= deadline:
        if brokeneye_ready(host, port):
            return True
        time.sleep(1.0)
    return brokeneye_ready(host, port)


def latest_csv(directory: Path) -> Path | None:
    files = sorted(directory.glob("live_multitask_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def default_check_args() -> SimpleNamespace:
    return SimpleNamespace(
        min_rows=20,
        min_stage_rows=5,
        min_fps=12.0,
        max_delta_p95_ms=30.0,
        min_pair_confidence_median=0.50,
        min_eye_confidence_median=0.50,
        min_openness_stage_median=0.05,
        max_ab_smooth_delta_warning=0.35,
    )


def write_report_and_check(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    report_path = output_dir / "live_ab_report.json"
    check_path = output_dir / "live_ab_check.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    check = check_report(report, default_check_args())
    check_path.write_text(json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path, check_path, check


def compact_run(run: dict[str, Any], check: dict[str, Any] | None) -> dict[str, Any]:
    summary = run.get("summary") or {}
    fields = summary.get("fields") or {}
    return {
        "variant": run.get("variant"),
        "label": run.get("label"),
        "returncode": run.get("returncode"),
        "passed": check.get("passed") if check else None,
        "warning_count": check.get("warning_count") if check else None,
        "row_count": summary.get("row_count"),
        "fps": summary.get("fps"),
        "delta_ms_p95": (fields.get("delta_ms") or {}).get("p95"),
        "csv": run.get("csv"),
        "run_dir": run.get("run_dir"),
    }


def compact_report(report: dict[str, Any], check: dict[str, Any], report_path: Path, check_path: Path) -> dict[str, Any]:
    checks_by_label = {run.get("label"): run for run in check.get("runs", [])}
    return {
        "state": report.get("state"),
        "host": report.get("host"),
        "port": report.get("port"),
        "direction_preset": report.get("direction_preset"),
        "check_passed": check.get("passed"),
        "report": str(report_path.resolve()),
        "check": str(check_path.resolve()),
        "runs": [compact_run(run, checks_by_label.get(run.get("label"))) for run in report.get("runs", [])],
        "comparisons": check.get("comparisons", []),
        "message": report.get("message"),
    }


def ensure_variant_files(variant: dict[str, Any]) -> None:
    for key in ("onnx", "metadata", "expression_onnx", "expression_metadata"):
        path = variant.get(key)
        if path is not None and not Path(path).exists():
            raise SystemExit(f"Missing {key}: {path}")


def run_variant(args: argparse.Namespace, name: str, variant: dict[str, Any]) -> dict[str, Any]:
    ensure_variant_files(variant)
    run_dir = args.output_dir / variant["label"]
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "predict_live_multitask.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--output-dir",
        str(run_dir),
        "--onnx",
        str(variant["onnx"]),
        "--metadata",
        str(variant["metadata"]),
        "--print-every",
        str(args.print_every),
        "--snapshot-every",
        "0",
    ]
    if args.direction_preset:
        command.extend(
            [
                "--direction-preset",
                args.direction_preset,
                "--settle-seconds",
                str(args.settle_seconds),
                "--stage-seconds",
                str(args.stage_seconds),
            ]
        )
    else:
        command.extend(["--duration-seconds", str(args.duration_seconds)])
    if args.udp_port > 0:
        command.extend(["--udp-port", str(args.udp_port), "--udp-host", args.udp_host])
    if args.monitor_udp_port > 0:
        command.extend(["--monitor-udp-port", str(args.monitor_udp_port), "--monitor-udp-host", args.monitor_udp_host])
    if variant.get("expression_onnx") is not None:
        command.extend(
            [
                "--expression-onnx",
                str(variant["expression_onnx"]),
                "--expression-metadata",
                str(variant["expression_metadata"]),
                "--expression-every-n-frames",
                str(variant.get("expression_every_n_frames", "3")),
                "--expression-ema-alpha",
                str(variant.get("expression_ema_alpha", "0.65")),
            ]
        )

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, text=True, check=False)
    csv_path = latest_csv(run_dir)
    summary = summarize_csv(csv_path) if csv_path is not None else None
    return {
        "variant": name,
        "label": variant["label"],
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
        "command": command,
        "run_dir": str(run_dir.resolve()),
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "csv": str(csv_path.resolve()) if csv_path is not None else None,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--wait-seconds", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/live_multitask_ab")
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="Variant to run; repeatable. Default: dual and split.")
    parser.add_argument("--direction-preset", choices=("five", "nine"))
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--stage-seconds", type=float, default=3.0)
    parser.add_argument("--duration-seconds", type=float, default=20.0)
    parser.add_argument("--udp-port", type=int, default=0)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--monitor-udp-port", type=int, default=0)
    parser.add_argument("--monitor-udp-host", default="127.0.0.1")
    parser.add_argument("--print-every", type=int, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not wait_for_brokeneye(args.host, args.port, args.wait_seconds):
        report = {
            "state": "blocked_waiting_for_brokeneye",
            "host": args.host,
            "port": args.port,
            "wait_seconds": args.wait_seconds,
            "message": "BrokenEye TCP port is not accepting connections.",
        }
        report_path, check_path, check = write_report_and_check(args.output_dir, report)
        print(json.dumps(compact_report(report, check, report_path, check_path), indent=2, ensure_ascii=False))
        return 2

    selected = args.variant or ["dual", "split"]
    runs = [run_variant(args, name, VARIANTS[name]) for name in selected]
    report = {
        "state": "complete" if all(run["returncode"] == 0 for run in runs) else "error",
        "host": args.host,
        "port": args.port,
        "direction_preset": args.direction_preset,
        "runs": runs,
    }
    report_path, check_path, check = write_report_and_check(args.output_dir, report)
    print(json.dumps(compact_report(report, check, report_path, check_path), indent=2, ensure_ascii=False))
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
