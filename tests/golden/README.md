# Golden Recording Regression (E10)

Replay frozen BrokenEye recordings through the full Python runtime and gate new
model packages against a frozen baseline, so shipping a new model cannot silently
regress gaze / openness on known-hard cases.

## Privacy

Eye frames are **biometric data and are NOT committed**. `recordings.json`
(machine-local, gitignored) points to LOCAL manifest paths that reference frames
on your disk. Only `expected_metrics/*.json` (metrics, no images) are committed.

## Files

- `recordings.json` — machine-local config (copy from `recordings.example.json`, gitignored).
- `recordings.example.json` — template.
- `expected_metrics/<name>.json` — frozen baseline metrics per recording (committed).

## Usage

```powershell
# 1) Freeze baselines from the current shipping model:
python scripts/ml/run_golden_regression.py --update-baseline

# 2) Gate a candidate model before shipping (exit 0 = ok, exit 2 = regression):
python scripts/ml/run_golden_regression.py --model <cand.onnx> --metadata <cand.json>
```

A recording manifest is any CSV with `left_file`, `right_file` (and optional
`stage`) columns — e.g. the `openness_samples_*.csv` produced by
`live_openness_calibration.py --save-training-images`, or an existing training
manifest.

Tolerances: `--gaze-tol` (default 0.06 — per-stage smooth-median drift) and
`--open-tol` (default 0.10 — per-stage min-openness median drift). Any per-stage
metric drifting beyond tolerance is reported as a regression.
