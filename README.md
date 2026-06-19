# DreamAirTracking

DreamAirTracking 是一个面向 **Dream Air 单一头显链路** 的本地眼动追踪工具。它把 BrokenEye 的左右眼原始图像接入本地 ONNX 模型，再通过 VRCFaceTracking 模块输出 VRChat 可用的 eye tracking 数据。

当前项目重点不是做多头显泛化，而是把 Dream Air 在不同用户、不同佩戴、不同眼型下的 gaze、openness、pupil 表现调到稳定可用。

```text
Dream Air + BrokenEye
  -> DreamAirTracking App
  -> Dream Air model package
  -> VRCFT eye slot
  -> VRChat
```

SRanipal 或其它 face module 可以继续负责面部/表情。DreamAirTracking 默认只处理 eye slot，避免和其它 face tracking 模块抢 expression 输出。

## Supported Features

- Dream Air + BrokenEye 左右眼 MJPEG 输入：`http://127.0.0.1:5555/eye/left` 和 `/eye/right`
- WinUI 本地 app：启动 runtime、选择模型包、显示 pipeline diagnostics
- 模型包加载：从 `%LOCALAPPDATA%\DreamAirTracking\models\model_registry.json` 读取已安装模型
- Gaze 校准：5 点/9 点校准，支持佩戴偏移和 quick runtime correction
- Eyelid / openness：眼皮开合状态、眨眼曲线和 runtime 后处理
- Pupil：pupil runtime 状态和弱标签/质量 gate
- VRCFT 输出：通过 DreamAirTracking VRCFT module 向 VRChat 输出眼动数据
- 本地采集包：校准 session 可导出为训练数据包，用户手动分享，不由 app 自动上传
- 训练与评估脚本：包含 manifest 审计、multitask 训练、ONNX 导出、edge gaze 验收等工具

## Current Limitations

- 只支持 Dream Air 这一类头显链路。
- 需要 BrokenEye 正常提供 `/eye/left` 和 `/eye/right`；否则 runtime 会停在 `waiting_for_brokeneye`。
- 需要 VRCFaceTracking 和 DreamAirTracking module 监听 UDP `9400`，才能完整输出到 VRChat。
- VRCFaceTracking module 单独发布在：
  [UikaMisumi/DreamAirTracking.VrcftModule](https://github.com/UikaMisumi/DreamAirTracking.VrcftModule)。
- 模型权重不直接提交到 GitHub。公开模型包放在 Hugging Face：
  [Sumirui/dreamairtracking-dreamair-main-current](https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current)。
- 当前 public runtime 默认不负责 SRanipal face/expression slot。

## How To Use

### 1. Build

```powershell
dotnet build .\DreamAirTracking.sln -c Release
dotnet test .\DreamAirTracking.sln -c Release
```

Python runtime:

```powershell
pip install -r requirements-runtime.txt
```

Training tools:

```powershell
pip install -r requirements-training.txt
```

### 2. Install A Model Package

Download the current public Dream Air model package from:

[https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current](https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current)

Copy the Hugging Face repository contents into this local folder:

```text
%LOCALAPPDATA%\DreamAirTracking\models\
```

After install, the local files must look like this:

```text
%LOCALAPPDATA%\DreamAirTracking\models\model_registry.json
%LOCALAPPDATA%\DreamAirTracking\models\sha256.txt

%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\model.onnx
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\metadata.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\runtime_defaults.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\acceptance.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\model_card.md

%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\model.onnx
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\metadata.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\runtime_defaults.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\acceptance.json
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-expression-current\model_card.md
```

Do not put the files inside an extra nested folder such as:

```text
%LOCALAPPDATA%\DreamAirTracking\models\dreamairtracking-dreamair-main-current-main\...
```

The app reads `model_registry.json` from `%LOCALAPPDATA%\DreamAirTracking\models\model_registry.json`, and that registry points to the two folders above.

PowerShell download/install example:

```powershell
$modelRoot = Join-Path $env:LOCALAPPDATA "DreamAirTracking\models"
$zip = Join-Path $env:TEMP "dreamairtracking-dreamair-main-current.zip"
$extract = Join-Path $env:TEMP "dreamairtracking-dreamair-main-current"

New-Item -ItemType Directory -Force $modelRoot | Out-Null
Remove-Item -Recurse -Force $extract -ErrorAction SilentlyContinue

Invoke-WebRequest `
  -Uri "https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/archive/main.zip" `
  -OutFile $zip

Expand-Archive -Force $zip $extract
$downloadedRoot = Get-ChildItem $extract -Directory | Select-Object -First 1
Copy-Item -Recurse -Force (Join-Path $downloadedRoot.FullName "*") $modelRoot
```

Quick check:

```powershell
Test-Path "$env:LOCALAPPDATA\DreamAirTracking\models\model_registry.json"
Test-Path "$env:LOCALAPPDATA\DreamAirTracking\models\dreamair-main-current\model.onnx"
Test-Path "$env:LOCALAPPDATA\DreamAirTracking\models\dreamair-expression-current\model.onnx"
```

All three commands should print `True`.

The shipped `model_registry.json` already contains the needed entries. For reference, it points to:

```json
{
  "schema": "dream_air_tracking.model_registry.v1",
  "models": [
    {
      "id": "dreamair-main-current",
      "displayName": "Dream Air Main",
      "deviceFamily": "Dream Air",
      "role": "main",
      "architecture": "Siamese MobileNetV3-small",
      "runtime": "predict_live_multitask",
      "onnx": "models/dreamair-main-current/model.onnx",
      "metadata": "models/dreamair-main-current/metadata.json",
      "runtimeDefaults": "models/dreamair-main-current/runtime_defaults.json",
      "acceptance": "models/dreamair-main-current/acceptance.json",
      "outputs": ["gaze_xy", "openness_lr", "pupil_lr", "confidence"],
      "default": true
    },
    {
      "id": "dreamair-expression-current",
      "displayName": "Dream Air Expression Auxiliary",
      "deviceFamily": "Dream Air",
      "role": "expression",
      "architecture": "Siamese MobileNetV3-small",
      "runtime": "predict_live_multitask",
      "onnx": "models/dreamair-expression-current/model.onnx",
      "metadata": "models/dreamair-expression-current/metadata.json",
      "runtimeDefaults": "models/dreamair-expression-current/runtime_defaults.json",
      "acceptance": "models/dreamair-expression-current/acceptance.json",
      "outputs": ["wide_lr", "squint_lr"],
      "default": true
    }
  ]
}
```

The app shows only real imported model packages. If no package exists, the model list stays empty and eye tracking start is disabled.

Schema files:

- `models/model_registry.schema.json`
- `models/model_registry.example.json`

The matching held-out test package is kept private because it contains raw eye-frame images. Maintainers keep it separately at `Sumirui/dreamairtracking-dreamair-main-current-testset`.

### 3. Install The VRCFT Module

Download the latest module release:

[https://github.com/UikaMisumi/DreamAirTracking.VrcftModule/releases/latest](https://github.com/UikaMisumi/DreamAirTracking.VrcftModule/releases/latest)

Extract `DreamAirTracking.VrcftModule-v0.1.0.zip` into:

```text
%APPDATA%\VRCFaceTracking\CustomLibs\b24a50f2-36bd-4a56-88f0-daa2a3727b5d\
```

After install, the folder should contain:

```text
DreamAirTracking.VrcftModule.dll
DreamAirTracking.VrcftModule.pdb
module.json
```

Restart VRCFaceTracking and enable the `DreamAirTracking` module. Keep SRanipal or another module enabled for face/lip tracking if you need it; this module is for the eye slot.

### 4. Start Runtime

1. Start BrokenEye and confirm both streams are live:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5555/eye/left
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5555/eye/right
```

2. Start VRCFaceTracking with the DreamAirTracking module enabled.
3. Open DreamAirTracking App.
4. Select the installed model package.
5. Start eye tracking.

Runtime status files:

```text
%LOCALAPPDATA%\DreamAirTracking\runtime_launch_status.json
%LOCALAPPDATA%\DreamAirTracking\bridge_launch_options.json
%LOCALAPPDATA%\DreamAirTracking\app_crash.log
```

Expected ports:

```text
BrokenEye HTTP input: 127.0.0.1:5555
VRCFT eye output:     127.0.0.1:9400
App monitor UDP:      127.0.0.1:9401
```

## Calibration

The app has three calibration modules:

| Module | Purpose | Training Use |
|---|---|---|
| Gaze | 5-point/9-point gaze correction | gaze hardcases and edge reach |
| Eyelid | openness and blink curve | openness labels |
| Pupil | pupil status and quality gate | weak pupil labels only when valid |

Gaze calibration can improve local fit without retraining. For better future models, accepted calibration sessions can be exported as capture packages and manually shared.

## Help Us Build Better Models

The app is designed so normal calibration can also become useful personal training data. You do not need to understand manifests or CSV files. Pick the calibration page that matches the problem, finish the prompts, then export the capture package.

### Which Calibration Should I Run?

| If you see this problem | Open this tab | Press this button | What it collects |
|---|---|---|---|
| Eyes look in the wrong direction | `Calibration -> Gaze` | `Start gaze calibration` | Personal gaze samples for center/up/down/left/right, or 9 points if advanced mode is enabled |
| One corner is weak, such as down-right | `Calibration -> Gaze` | `Start gaze calibration` with 9-point mode | Edge gaze samples, especially `left_up`, `right_up`, `left_down`, `right_down` |
| Blink does not close, or eyes look squeezed | `Calibration -> Eyelid` | `Start eyelid calibration` | Personal open / half-open / closed / blink eyelid shape |
| Pupil size or pupil response looks unstable | `Calibration -> Pupil` | `Start pupil calibration` | Pupil runtime samples and quality gates; useful only when BrokenEye confidence/openness are good |

In short:

```text
Gaze Cali   = where your eyes are looking
Eyelid Cali = how open/closed your eyelids look
Pupil Cali  = pupil size/quality behavior
```

### Recommended Beginner Collection

If you want to help improve the public model, do this:

1. Wear the Dream Air normally.
2. Start BrokenEye and make sure both eye streams are live.
3. Open DreamAirTracking.
4. Go to `Calibration -> Gaze`.
5. Enable 9-point mode if available.
6. Press `Start gaze calibration`.
7. Follow the dots until the app says calibration is complete.
8. Without moving the headset, go to `Calibration -> Eyelid`.
9. Press `Start eyelid calibration`.
10. Follow the open / half-open / closed / blink prompts.
11. If pupil behavior is the problem, also go to `Calibration -> Pupil` and press `Start pupil calibration`.
12. Export the capture package.

For a better personal package, repeat the same steps after taking the headset off and putting it back on. This creates a new `wearId`, which helps the model learn real headset fit changes.

### How To Export

The intended user flow is:

```text
Calibration complete
  -> Export training package
  -> DreamAirTrackingCapture_YYYYMMDD_HHMMSS.zip
  -> Upload the zip yourself
  -> Share the link in a GitHub issue
```

The app does not upload anything automatically. The zip should contain:

```text
session.json
device.json
capture_protocol.json
labels.jsonl
pairs.csv
metrics.json
calibration_report.md
frames/
```

After a successful gaze calibration, `Export training package` appears on the Calibration page. Press it once, then upload the generated zip yourself.

### What Counts As Useful Data?

Useful data:

- Full 9-point gaze calibration, especially `right_down`, `left_down`, `right_up`, `left_up`
- Multiple re-wear sessions from the same person
- Open / half-open / closed / blink eyelid captures
- Short failure replay after you notice a bad direction in VRChat
- Notes about glasses, headset fit, lighting, and whether BrokenEye streams looked stable

Not useful by itself:

- Random screenshots
- Only `runtime_launch_status.json`
- Only CSV row counts
- A recording where BrokenEye `/eye/left` or `/eye/right` was not live
- A calibration you know was bad but did not mark or mention

### What To Send

Upload the exported zip to Google Drive, OneDrive, or another file host. Then open a GitHub issue with:

```text
Problem:
  Example: right_down is weak / blink does not close / pupil unstable

What I collected:
  Gaze 9-point: yes/no
  Eyelid: yes/no
  Pupil: yes/no
  Re-wear count: 1 / 2 / 3+

Runtime:
  Model package id:
  BrokenEye /eye/left live: yes/no
  BrokenEye /eye/right live: yes/no
  VRCFT module running: yes/no

Notes:
  Glasses:
  Lighting:
  Headset fit:
  Which direction or expression failed:

Capture package link:
```

Detailed developer requirements are in:

- `datasets/session_package.schema.md`
- `docs/MODEL_ADAPTATION_DATA_AUDIT_zh.md`
- `docs/SINGLE_HEADSET_RELEASE_AND_TRAINING_PLAN_zh.md`

## Contact

- Use GitHub Issues for bugs, model requests, and capture package coordination.
- Maintainer: `UikaMisumi` / `Sumirui`
- For private capture links, contact through the maintainer's GitHub profile or discuss a temporary private channel in an issue.

When reporting a runtime problem, include:

```text
%LOCALAPPDATA%\DreamAirTracking\runtime_launch_status.json
%LOCALAPPDATA%\DreamAirTracking\bridge_launch_options.json
BrokenEye /eye/left and /eye/right status
VRCFaceTracking module status
DreamAirTracking model package id
```

## Repository Layout

```text
src/
  DreamAirTracking.Core        Core capture, calibration, filtering, profile logic
  DreamAirTracking.App         WinUI app and runtime launcher
  DreamAirTracking.Bridge      Legacy/debug C# bridge
  DreamAirTracking.Cli         CLI tools
  DreamAirTracking.VrcftModule Optional VRCFT module source
tests/
scripts/
  ml/                          Training, eval, ONNX export, runtime scripts
docs/
models/
datasets/
```

The default solution currently excludes `DreamAirTracking.VrcftModule` because it depends on the VRCFaceTracking SDK/source, which is not vendored into this clean repository.

## Development Notes

- Keep app code and model artifacts decoupled.
- Do not commit `runs/`, real eye images, runtime CSV, local calibration data, or model weights.
- Use model registry packages instead of hard-coded training run names.
- Do not train all heads on every row. Gaze, eyelid, pupil, wide, and squint data must use head-specific masks.
- Validation must include edge reach and held-out wear/subject splits; `val_loss` alone is not enough.

## License

License has not been finalized yet. Add a license before public release.
