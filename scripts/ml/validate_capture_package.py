#!/usr/bin/env python3
"""Validate a DreamAirTracking capture package against the data-audit checklist (E10).

Turns the "Quality gate" checklist in docs/MODEL_ADAPTATION_DATA_AUDIT_zh.md into
code so bad packages can be rejected automatically on import.

Checks (violation = fail; softer issues = warning):
  - required members present (manifest/device/capture_protocol/session/labels/pairs/metrics + frames/)
  - manifest schema + deviceFamily == "Dream Air" + subjectId/wearId/captureProtocol present
  - pairs.csv: delta_ms p95 <= --max-delta-ms (warn), stage labels present (fail if none)
  - blink/closed pollution ratio (warn if it dominates a gaze protocol)
  - if a training_manifest_v3.csv is bundled, run the head-mask gate on it

Accepts a .zip package or an extracted directory.
  python validate_capture_package.py --package DreamAirTrackingCapture_XXXX.zip [--json]
Exit 0 = pass, 2 = fail.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

REQUIRED_MEMBERS = [
    "manifest.json", "device.json", "capture_protocol.json",
    "session.json", "labels.jsonl", "pairs.csv", "metrics.json",
]
BLINK_CLOSED_STAGES = {
    "closed", "blink", "both_closed", "left_closed_right_open", "left_open_right_closed",
}


class PackageSource:
    """Uniform reader over a .zip package or an extracted directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.suffix.lower() == ".zip" else None
        if self.zip is not None:
            self.names = list(self.zip.namelist())
        else:
            self.names = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()]

    def _resolve(self, name: str) -> str | None:
        if name in self.names:
            return name
        return next((n for n in self.names if n.endswith("/" + name)), None)

    def has(self, name: str) -> bool:
        return self._resolve(name) is not None

    def read_text(self, name: str) -> str | None:
        real = self._resolve(name)
        if real is None:
            return None
        if self.zip is not None:
            return self.zip.read(real).decode("utf-8-sig")
        return (self.path / real).read_text(encoding="utf-8-sig")

    def frame_count(self) -> int:
        return sum(1 for n in self.names if n.startswith("frames/") and n.lower().endswith((".jpg", ".jpeg", ".png")))

    def close(self) -> None:
        if self.zip is not None:
            self.zip.close()


def _floats(rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed):
            out.append(parsed)
    return out


def validate(path: Path, max_delta_ms: float = 30.0) -> dict:
    src = PackageSource(path)
    violations: list[str] = []
    warnings: list[str] = []
    info: dict[str, object] = {}
    try:
        missing = [m for m in REQUIRED_MEMBERS if not src.has(m)]
        if missing:
            violations.append(f"missing required files: {missing}")
        info["frame_files"] = src.frame_count()
        if info["frame_files"] == 0:
            violations.append("no frames/*.jpg in package")

        manifest_txt = src.read_text("manifest.json")
        if manifest_txt:
            manifest = json.loads(manifest_txt)
            info["schema"] = manifest.get("schema")
            if not str(manifest.get("schema", "")).startswith("dream_air_tracking.capture_package"):
                warnings.append(f"unexpected manifest schema: {manifest.get('schema')}")
            if manifest.get("deviceFamily") != "Dream Air":
                violations.append(f"deviceFamily != Dream Air: {manifest.get('deviceFamily')}")
            for key in ("subjectId", "wearId", "captureProtocol"):
                if not manifest.get(key):
                    violations.append(f"manifest missing {key}")
            info["subjectId"] = manifest.get("subjectId")
            info["wearId"] = manifest.get("wearId")
            info["captureProtocol"] = manifest.get("captureProtocol")
            info["pairCount"] = manifest.get("pairCount")

        pairs_txt = src.read_text("pairs.csv")
        if pairs_txt:
            rows = list(csv.DictReader(io.StringIO(pairs_txt)))
            info["pairs_rows"] = len(rows)
            deltas = sorted(_floats(rows, "delta_ms"))
            if deltas:
                p95 = deltas[min(len(deltas) - 1, int(0.95 * (len(deltas) - 1)))]
                info["delta_ms_p95"] = p95
                if p95 > max_delta_ms:
                    warnings.append(f"delta_ms p95 {p95:.1f} > {max_delta_ms}")
            stages = Counter((row.get("stage", "") or "-") for row in rows)
            info["stage_counts"] = dict(stages.most_common(20))
            if rows:
                blink = sum(count for stage, count in stages.items() if stage in BLINK_CLOSED_STAGES)
                ratio = blink / len(rows)
                info["blink_closed_ratio"] = round(ratio, 3)
                if "gaze" in str(info.get("captureProtocol", "")).lower() and ratio > 0.5:
                    warnings.append(f"blink/closed frames dominate a gaze protocol: {ratio:.0%}")
            if not stages or set(stages) <= {"-", ""}:
                violations.append("pairs.csv has no stage labels")

        training_manifest = next((n for n in src.names if n.endswith("training_manifest_v3.csv")), None)
        if training_manifest:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_csv = Path(tmp) / "training_manifest.csv"
                tmp_csv.write_text(src.read_text(training_manifest) or "", encoding="utf-8")
                from manifest_head_mask_gate import gate_manifest

                # Inside a package check, gaze/head co-supervision is a warning (it is a
                # training-time decision); missing columns / split leakage still fail.
                gate = gate_manifest(tmp_csv, allow_gaze_head_cosupervision=True)
                info["head_mask_gate"] = {
                    "passed": gate["passed"],
                    "violations": gate["violations"],
                    "warnings": gate["warnings"],
                }
                violations.extend(f"training_manifest: {v}" for v in gate["violations"])
                warnings.extend(f"training_manifest: {w}" for w in gate["warnings"])
    finally:
        src.close()

    return {
        "package": str(path.resolve()),
        "info": info,
        "warnings": warnings,
        "violations": violations,
        "passed": not violations,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, type=Path, help="Capture package .zip or extracted directory.")
    ap.add_argument("--max-delta-ms", type=float, default=30.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if not args.package.exists():
        raise SystemExit(f"not found: {args.package}")
    report = validate(args.package, args.max_delta_ms)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"[capture-validate] {'PASSED' if report['passed'] else 'FAILED'}: {report['package']}")
        for violation in report["violations"]:
            print(f"  VIOLATION: {violation}")
        for warning in report["warnings"]:
            print(f"  warning: {warning}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
