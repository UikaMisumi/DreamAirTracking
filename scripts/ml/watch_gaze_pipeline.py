#!/usr/bin/env python3
"""Watch calibration_data and rerun the gaze auto pipeline when new sessions appear."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CALIBRATION_ROOT = Path.home() / "Documents" / "DreamAirTracking" / "calibration_data"


def snapshot_sessions(root: Path) -> dict[str, float]:
    if not root.exists():
        return {}
    result: dict[str, float] = {}
    for pairs in root.rglob("pairs.csv"):
        session = pairs.parent.name
        files = [pairs]
        metrics = pairs.parent / "metrics.json"
        labels = pairs.parent / "labels.jsonl"
        if metrics.exists():
            files.append(metrics)
        if labels.exists():
            files.append(labels)
        result[str(pairs.parent.resolve())] = max(path.stat().st_mtime for path in files)
    return result


def snapshot_roots(roots: list[Path]) -> dict[str, float]:
    result: dict[str, float] = {}
    for root in roots:
        result.update(snapshot_sessions(root))
    return result


def make_attempt_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_root / f"attempt_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_pipeline(args: argparse.Namespace, attempt_dir: Path) -> int:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "auto_gaze_pipeline.py"),
        "--output-dir",
        str(attempt_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--model",
        args.model,
        "--image-session-quality",
        args.image_session_quality,
        "--require-metrics-quality",
        args.require_metrics_quality,
        "--min-stage-samples",
        str(args.min_stage_samples),
        "--min-openness",
        str(args.min_openness),
    ]
    for root in args.root:
        command.extend(["--root", str(root)])
    if args.cpu:
        command.append("--cpu")

    with (attempt_dir / "command.json").open("w", encoding="utf-8") as handle:
        json.dump(command, handle, indent=2, ensure_ascii=False)

    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=str(WORKSPACE_ROOT))
    return completed.returncode


def run_audit(args: argparse.Namespace, attempt_dir: Path) -> Path | None:
    audit_path = attempt_dir / "calibration_session_audit.md"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "audit_gaze_sessions.py"),
        "--root",
        str(args.audit_root),
        "--latest",
        "0",
        "--markdown",
        str(audit_path),
    ]
    with (attempt_dir / "audit_command.json").open("w", encoding="utf-8") as handle:
        json.dump(command, handle, indent=2, ensure_ascii=False)
    with (attempt_dir / "audit_stdout.txt").open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(command, cwd=str(WORKSPACE_ROOT), stdout=stdout, stderr=subprocess.STDOUT)
    if completed.returncode != 0 or not audit_path.exists():
        return None
    latest = args.output_root / "latest_calibration_session_audit.md"
    shutil.copyfile(audit_path, latest)
    return audit_path


def first_existing(*paths: Path) -> str | None:
    for path in paths:
        if path.exists():
            return str(path.resolve())
    return None


def write_status(output_root: Path, payload: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "watch_status.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def wait_for_settled_snapshot(
    roots: list[Path],
    settle_seconds: float,
    output_root: Path,
    attempt_count: int,
) -> dict[str, float]:
    if settle_seconds <= 0:
        return snapshot_roots(roots)

    write_status(
        output_root,
        {
            "state": "settling",
            "return_code": 2,
            "attempt_count": attempt_count,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "reason": f"waiting {settle_seconds:0.1f}s for calibration files to stop changing before review",
        },
    )
    stable_since = time.monotonic()
    previous = snapshot_roots(roots)
    while True:
        time.sleep(min(1.0, max(0.1, settle_seconds / 2)))
        current = snapshot_roots(roots)
        if current != previous:
            previous = current
            stable_since = time.monotonic()
            continue
        if time.monotonic() - stable_since >= settle_seconds:
            return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, default=[WORKSPACE_ROOT, DEFAULT_CALIBRATION_ROOT])
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "runs" / "gaze_watch")
    parser.add_argument("--interval-seconds", type=float, default=20.0)
    parser.add_argument("--max-wait-seconds", type=float, default=0.0, help="0 means wait forever.")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help="wait for calibration files to remain unchanged before reviewing a new session",
    )
    parser.add_argument("--run-immediately", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", choices=("cnn", "hybrid"), default="cnn")
    parser.add_argument("--image-session-quality", choices=("none", "fair", "good"), default="none")
    parser.add_argument("--require-metrics-quality", choices=("fair", "good", "none"), default="fair")
    parser.add_argument("--min-stage-samples", type=int, default=20)
    parser.add_argument("--min-openness", type=float, default=0.08)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    args.root = [root.resolve() for root in args.root]
    args.audit_root = args.audit_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous = snapshot_roots(args.root)
    attempt_count = 0
    last_status: dict[str, object] | None = None

    while True:
        current = snapshot_roots(args.root)
        changed = current != previous
        immediate_first_run = args.run_immediately and attempt_count == 0
        if changed and not immediate_first_run:
            current = wait_for_settled_snapshot(args.root, args.settle_seconds, args.output_root, attempt_count)
            changed = current != previous
        should_run = immediate_first_run or changed
        if should_run:
            attempt_count += 1
            attempt_dir = make_attempt_dir(args.output_root)
            code = run_pipeline(args, attempt_dir)
            audit_path = run_audit(args, attempt_dir)
            status = {
                "state": "complete" if code == 0 else "waiting_for_data" if code == 2 else "failed",
                "return_code": code,
                "attempt_dir": str(attempt_dir.resolve()),
                "recording_request": first_existing(attempt_dir / "recording_request.md"),
                "session_discovery": first_existing(attempt_dir / "session_discovery.json"),
                "audit_report": str(audit_path.resolve()) if audit_path is not None else None,
                "latest_audit_report": first_existing(args.output_root / "latest_calibration_session_audit.md"),
                "attempt_count": attempt_count,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_status(args.output_root, status)
            last_status = status
            if code == 0:
                print(f"Pipeline succeeded: {attempt_dir}")
                return 0
            if code not in (0, 2):
                print(f"Pipeline failed: {attempt_dir}")
                return code
            previous = current

        elapsed = time.monotonic() - started
        if args.max_wait_seconds > 0 and elapsed >= args.max_wait_seconds:
            timeout_status = dict(last_status or {})
            timeout_status.update(
                {
                    "state": "waiting_for_data",
                    "return_code": 2,
                    "attempt_count": attempt_count,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "reason": "max wait elapsed before enough usable calibration data appeared",
                    "latest_audit_report": first_existing(args.output_root / "latest_calibration_session_audit.md"),
                }
            )
            write_status(
                args.output_root,
                timeout_status,
            )
            print("Still waiting for enough usable calibration data.")
            return 2

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
