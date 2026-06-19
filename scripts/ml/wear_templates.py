#!/usr/bin/env python3
"""Wear-template signature and matching utilities for DreamAirTracking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


SCHEMA = "dream_air_tracking.wear_templates.v1"
DEFAULT_TEMPLATE_PATH = Path("runs/wear_templates/wear_templates.json")
DEFAULT_LEFT_PUPIL_CENTER = [0.81, 0.54]
DEFAULT_RIGHT_PUPIL_CENTER = [0.20, 0.54]

WEIGHTS = {
    "pupil_center_distance": 0.45,
    "crop_shift_distance": 0.30,
    "raw_gaze_center_distance": 0.05,
    "openness_open_distance": 0.10,
    "confidence_asymmetry_distance": 0.10,
}


@dataclass(frozen=True)
class ProbeRequirements:
    required_pairs: int = 60
    average_openness: float = 0.55
    median_confidence: float = 0.12
    pupil_center_std: float = 0.035


def empty_store() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created_at": _now(),
        "updated_at": _now(),
        "max_templates": 12,
        "weights": WEIGHTS,
        "thresholds": {
            "good_match": 0.18,
            "weak_match": 0.32,
            "merge": 0.12,
        },
        "probe_requirements": {
            "required_pairs": 60,
            "average_openness": 0.55,
            "median_confidence": 0.12,
            "pupil_center_std": 0.035,
        },
        "templates": [],
    }


def default_template(template_id: str | None = None, sample_count: int = 0) -> dict[str, Any]:
    timestamp = _now()
    return {
        "id": template_id or time.strftime("wear_%Y%m%d_%H%M%S"),
        "created_at": timestamp,
        "updated_at": timestamp,
        "sample_count": sample_count,
        "left_pupil_center_median": list(DEFAULT_LEFT_PUPIL_CENTER),
        "right_pupil_center_median": list(DEFAULT_RIGHT_PUPIL_CENTER),
        "left_crop_shift_median": [0.0, 0.0],
        "right_crop_shift_median": [0.0, 0.0],
        "open_openness_median": [1.0, 1.0],
        "confidence_median": [1.0, 1.0],
        "raw_gaze_center_median": [0.0, 0.0],
        "center_offset": [0.0, 0.0],
        "gain": [1.0, 1.0],
        "runtime_calibration": None,
        "quick_calibration_layer": None,
        "openness_calibration": None,
        "notes": "",
    }


def load_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_store()
    payload = json.load(path.open("r", encoding="utf-8-sig"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported wear template schema: {payload.get('schema')}")
    payload["weights"] = WEIGHTS
    payload["probe_requirements"] = empty_store()["probe_requirements"]
    payload.setdefault("templates", [])
    return payload


def save_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store["updated_at"] = _now()
    path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def signature_from_csv(path: Path, requirements: ProbeRequirements = ProbeRequirements()) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    return signature_from_rows(rows, requirements, source=str(path.resolve()))


def signature_from_rows(
    rows: list[dict[str, Any]],
    requirements: ProbeRequirements = ProbeRequirements(),
    source: str | None = None,
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if _row_bool(row, "left_norm_found")
        and _row_bool(row, "right_norm_found")
        and _row_float(row, "left_openness", "0") >= 0
        and _row_float(row, "right_openness", "0") >= 0
    ]
    sample_count = len(usable)
    left_openness = [_row_float(row, "left_openness", "1") for row in usable]
    right_openness = [_row_float(row, "right_openness", "1") for row in usable]
    left_conf = [_first_float(row, ("left_norm_confidence", "left_confidence", "confidence"), 1.0) for row in usable]
    right_conf = [_first_float(row, ("right_norm_confidence", "right_confidence", "confidence"), 1.0) for row in usable]
    raw_x = [_row_float(row, "raw_x", "0") for row in usable]
    raw_y = [_row_float(row, "raw_y", "0") for row in usable]
    left_pupil_x = [_row_float(row, "left_norm_pupil_x", str(DEFAULT_LEFT_PUPIL_CENTER[0])) for row in usable]
    left_pupil_y = [_row_float(row, "left_norm_pupil_y", str(DEFAULT_LEFT_PUPIL_CENTER[1])) for row in usable]
    right_pupil_x = [_row_float(row, "right_norm_pupil_x", str(DEFAULT_RIGHT_PUPIL_CENTER[0])) for row in usable]
    right_pupil_y = [_row_float(row, "right_norm_pupil_y", str(DEFAULT_RIGHT_PUPIL_CENTER[1])) for row in usable]
    average_open = _mean([(l + r) * 0.5 for l, r in zip(left_openness, right_openness)])
    pair_conf = [(l + r) * 0.5 for l, r in zip(left_conf, right_conf)]
    raw_std = math.sqrt((_variance(raw_x) + _variance(raw_y)) * 0.5)
    pupil_std = math.sqrt(
        (
            _variance(left_pupil_x)
            + _variance(left_pupil_y)
            + _variance(right_pupil_x)
            + _variance(right_pupil_y)
        )
        * 0.25
    )
    quality = {
        "sample_count": sample_count,
        "average_openness": average_open,
        "median_confidence": _median(pair_conf, 0.0),
        "raw_gaze_std": raw_std,
        "pupil_center_std": pupil_std,
        "requirements_met": sample_count >= requirements.required_pairs
        and average_open >= requirements.average_openness
        and _median(pair_conf, 0.0) >= requirements.median_confidence
        and pupil_std <= requirements.pupil_center_std,
    }
    if not usable:
        return {
            "source": source,
            "quality": quality,
            "signature": default_template("wear_signature_empty", 0),
            "reason": "no usable rows",
        }

    signature = default_template("wear_signature", sample_count)
    signature.update(
        {
            "left_pupil_center_median": [
                _median(left_pupil_x, DEFAULT_LEFT_PUPIL_CENTER[0]),
                _median(left_pupil_y, DEFAULT_LEFT_PUPIL_CENTER[1]),
            ],
            "right_pupil_center_median": [
                _median(right_pupil_x, DEFAULT_RIGHT_PUPIL_CENTER[0]),
                _median(right_pupil_y, DEFAULT_RIGHT_PUPIL_CENTER[1]),
            ],
            "left_crop_shift_median": [
                _median([_row_float(row, "left_norm_shift_x", "0") for row in usable], 0.0),
                _median([_row_float(row, "left_norm_shift_y", "0") for row in usable], 0.0),
            ],
            "right_crop_shift_median": [
                _median([_row_float(row, "right_norm_shift_x", "0") for row in usable], 0.0),
                _median([_row_float(row, "right_norm_shift_y", "0") for row in usable], 0.0),
            ],
            "open_openness_median": [
                _median(left_openness, 1.0),
                _median(right_openness, 1.0),
            ],
            "confidence_median": [
                _median(left_conf, 1.0),
                _median(right_conf, 1.0),
            ],
            "raw_gaze_center_median": [
                _median(raw_x, 0.0),
                _median(raw_y, 0.0),
            ],
        }
    )
    return {"source": source, "quality": quality, "signature": signature, "reason": None}


def match_template(signature: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
    templates = [item for item in store.get("templates", []) if isinstance(item, dict)]
    if not templates:
        return {"status": "new", "action": "run_5_point_calibration", "template_id": None, "distance": None, "components": {}}

    scored = [(template, template_distance(signature, template)) for template in templates]
    scored.sort(key=lambda item: item[1]["distance"])
    template, distance = scored[0]
    thresholds = store.get("thresholds", {})
    good = float(thresholds.get("good_match", 0.18))
    weak = float(thresholds.get("weak_match", 0.32))
    value = float(distance["distance"])
    if value <= good:
        status = "good"
        action = "load_template"
    elif value <= weak:
        status = "weak"
        action = "suggest_5_point_calibration"
    else:
        status = "new"
        action = "run_5_point_calibration"
    return {
        "status": status,
        "action": action,
        "template_id": template.get("id"),
        "distance": value,
        "components": distance["components"],
        "runtime_calibration": template.get("runtime_calibration"),
        "has_runtime_calibration": _has_runtime_calibration(template),
    }


def template_distance(signature: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    components = {
        "pupil_center_distance": (
            _distance(signature.get("left_pupil_center_median"), template.get("left_pupil_center_median"))
            + _distance(signature.get("right_pupil_center_median"), template.get("right_pupil_center_median"))
        )
        * 0.5,
        "crop_shift_distance": (
            _distance(signature.get("left_crop_shift_median"), template.get("left_crop_shift_median"))
            + _distance(signature.get("right_crop_shift_median"), template.get("right_crop_shift_median"))
        )
        * 0.5,
        "raw_gaze_center_distance": _distance(signature.get("raw_gaze_center_median"), template.get("raw_gaze_center_median")),
        "openness_open_distance": _distance(signature.get("open_openness_median"), template.get("open_openness_median")),
        "confidence_asymmetry_distance": abs(
            _asymmetry(signature.get("confidence_median")) - _asymmetry(template.get("confidence_median"))
        ),
    }
    weighted = sum(components[key] * WEIGHTS[key] for key in WEIGHTS)
    return {"distance": weighted, "components": components}


def merge_template(template: dict[str, Any], signature: dict[str, Any], alpha: float = 0.08) -> dict[str, Any]:
    merged = dict(template)
    for key in (
        "left_pupil_center_median",
        "right_pupil_center_median",
        "left_crop_shift_median",
        "right_crop_shift_median",
        "open_openness_median",
        "confidence_median",
        "raw_gaze_center_median",
    ):
        merged[key] = _ewma_vector(template.get(key), signature.get(key), alpha)
    merged["sample_count"] = int(template.get("sample_count", 0)) + int(signature.get("sample_count", 0))
    merged["updated_at"] = _now()
    return merged


def command_init(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists() and not args.force:
        print(f"Already exists: {path.resolve()}")
        return 0
    store = empty_store()
    save_store(path, store)
    print(f"Wrote {path.resolve()}")
    return 0


def command_match(args: argparse.Namespace) -> int:
    store = load_store(args.templates)
    payload = signature_from_csv(args.csv)
    result = match_template(payload["signature"], store)
    output = {"quality": payload["quality"], "match": result, "signature": payload["signature"]}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def command_add(args: argparse.Namespace) -> int:
    store = load_store(args.templates)
    payload = signature_from_csv(args.csv)
    signature = payload["signature"]
    signature["id"] = args.id or time.strftime("wear_%Y%m%d_%H%M%S")
    signature["notes"] = args.notes or ""
    store.setdefault("templates", []).append(signature)
    max_templates = int(store.get("max_templates", 12))
    store["templates"] = store["templates"][-max_templates:]
    save_store(args.templates, store)
    print(json.dumps({"quality": payload["quality"], "added": signature["id"], "templates": len(store["templates"])}, indent=2))
    return 0


def command_upsert(args: argparse.Namespace) -> int:
    store = load_store(args.templates)
    payload = signature_from_csv(args.csv)
    signature = payload["signature"]
    quality = payload["quality"]
    if not quality.get("requirements_met", False) and not args.force:
        print(json.dumps({"quality": quality, "updated": False, "reason": "requirements not met"}, indent=2, ensure_ascii=False))
        return 2

    match = match_template(signature, store)
    thresholds = store.get("thresholds", {})
    merge_threshold = float(thresholds.get("merge", 0.12))
    templates = [item for item in store.get("templates", []) if isinstance(item, dict)]
    if match["template_id"] is not None and match["distance"] is not None and float(match["distance"]) <= merge_threshold:
        for index, template in enumerate(templates):
            if template.get("id") == match["template_id"]:
                templates[index] = merge_template(template, signature, args.alpha)
                store["templates"] = templates
                save_store(args.templates, store)
                print(json.dumps({"quality": quality, "match": match, "updated": True, "merged": match["template_id"]}, indent=2, ensure_ascii=False))
                return 0

    signature["id"] = args.id or time.strftime("wear_%Y%m%d_%H%M%S")
    signature["notes"] = args.notes or ""
    templates.append(signature)
    max_templates = int(store.get("max_templates", 12))
    store["templates"] = templates[-max_templates:]
    save_store(args.templates, store)
    print(json.dumps({"quality": quality, "match": match, "updated": True, "added": signature["id"], "templates": len(store["templates"])}, indent=2, ensure_ascii=False))
    return 0


def command_bind_calibration(args: argparse.Namespace) -> int:
    store = load_store(args.templates)
    template_id = args.template_id
    calibration_path: Path = args.calibration
    calibration = _read_runtime_calibration(calibration_path)
    templates = [item for item in store.get("templates", []) if isinstance(item, dict)]
    for template in templates:
        if template.get("id") != template_id:
            continue
        runtime_calibration = {
            "schema": "dream_air_tracking.runtime_template_calibration.v1",
            "updated_at": _now(),
            "path": str(calibration_path.resolve()),
            "center_offset": calibration["center_offset"],
            "gain": calibration["gain"],
            "sample_count": calibration["sample_count"],
            "reason": calibration.get("reason", "ok"),
        }
        template["center_offset"] = list(runtime_calibration["center_offset"])
        template["gain"] = list(runtime_calibration["gain"])
        template["runtime_calibration"] = runtime_calibration
        template["updated_at"] = runtime_calibration["updated_at"]
        save_store(args.templates, store)
        print(json.dumps({"updated": True, "template_id": template_id, "runtime_calibration": runtime_calibration}, indent=2, ensure_ascii=False))
        return 0

    print(json.dumps({"updated": False, "template_id": template_id, "reason": "template not found"}, indent=2, ensure_ascii=False))
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create an empty wear template store.")
    init.add_argument("--path", type=Path, default=DEFAULT_TEMPLATE_PATH)
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=command_init)

    match = sub.add_parser("match", help="Estimate a signature from predict_live CSV and match templates.")
    match.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_PATH)
    match.add_argument("--csv", type=Path, required=True)
    match.set_defaults(func=command_match)

    add = sub.add_parser("add", help="Add a template from predict_live CSV.")
    add.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_PATH)
    add.add_argument("--csv", type=Path, required=True)
    add.add_argument("--id")
    add.add_argument("--notes")
    add.set_defaults(func=command_add)

    upsert = sub.add_parser("upsert", help="Merge into a matching template or add a new template from predict_live CSV.")
    upsert.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_PATH)
    upsert.add_argument("--csv", type=Path, required=True)
    upsert.add_argument("--id")
    upsert.add_argument("--notes")
    upsert.add_argument("--alpha", type=float, default=0.08)
    upsert.add_argument("--force", action="store_true")
    upsert.set_defaults(func=command_upsert)

    bind = sub.add_parser("bind-calibration", help="Attach a runtime gaze calibration JSON to an existing wear template.")
    bind.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_PATH)
    bind.add_argument("--template-id", required=True)
    bind.add_argument("--calibration", type=Path, required=True)
    bind.set_defaults(func=command_bind_calibration)
    return parser


def _row_float(row: dict[str, Any], key: str, default: str) -> float:
    return _to_float(row.get(key, default), float(default))


def _row_bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def _first_float(row: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return _to_float(row[key], default)
    return default


def _to_float(value: Any, default: float) -> float:
    try:
        output = float(value)
        return output if math.isfinite(output) else default
    except (TypeError, ValueError):
        return default


def _median(values: list[float], default: float) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(median(clean)) if clean else default


def _mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _variance(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return 0.0
    mean = sum(clean) / len(clean)
    return float(sum((value - mean) ** 2 for value in clean) / len(clean))


def _distance(left: Any, right: Any) -> float:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return 1.0
    deltas = [(_to_float(a, 0.0) - _to_float(b, 0.0)) ** 2 for a, b in zip(left, right)]
    return float(math.sqrt(sum(deltas)))


def _asymmetry(values: Any) -> float:
    if not isinstance(values, list) or len(values) < 2:
        return 0.0
    return abs(_to_float(values[0], 0.0) - _to_float(values[1], 0.0))


def _has_runtime_calibration(template: dict[str, Any]) -> bool:
    calibration = template.get("runtime_calibration")
    if not isinstance(calibration, dict):
        return False
    center_offset = calibration.get("center_offset")
    gain = calibration.get("gain")
    return (
        isinstance(center_offset, list)
        and isinstance(gain, list)
        and len(center_offset) >= 2
        and len(gain) >= 2
        and all(math.isfinite(_to_float(value, float("nan"))) for value in center_offset[:2] + gain[:2])
    )


def _read_runtime_calibration(path: Path) -> dict[str, Any]:
    payload = json.load(path.open("r", encoding="utf-8-sig"))
    succeeded = payload.get("succeeded", payload.get("Succeeded", True))
    if str(succeeded).lower() in ("false", "0", "no"):
        raise ValueError(f"Runtime calibration did not succeed: {path}")
    center_offset = [
        _first_payload_float(payload, ("centerOffsetX", "center_offset_x", "CenterOffsetX"), 0.0),
        _first_payload_float(payload, ("centerOffsetY", "center_offset_y", "CenterOffsetY"), 0.0),
    ]
    gain = [
        _first_payload_float(payload, ("xGain", "x_gain", "XGain"), 1.0),
        _first_payload_float(payload, ("yGain", "y_gain", "YGain"), 1.0),
    ]
    sample_count = int(_first_payload_float(payload, ("sampleCount", "sample_count", "SampleCount"), 0.0))
    return {
        "center_offset": center_offset,
        "gain": gain,
        "sample_count": sample_count,
        "reason": str(payload.get("reason", payload.get("Reason", "ok"))),
    }


def _first_payload_float(payload: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in payload:
            return _to_float(payload[key], default)
    return default


def _ewma_vector(old: Any, new: Any, alpha: float) -> list[float]:
    if not isinstance(old, list) or not isinstance(new, list) or len(old) != len(new):
        return list(new) if isinstance(new, list) else []
    return [
        _to_float(a, 0.0) * (1.0 - alpha) + _to_float(b, 0.0) * alpha
        for a, b in zip(old, new)
    ]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
