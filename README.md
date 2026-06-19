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
- 模型权重不直接提交到 GitHub。请通过模型包/Release 分发。
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

Create a local model package:

```text
%LOCALAPPDATA%\DreamAirTracking\models\dreamair-main-current\
  model.onnx
  metadata.json
  runtime_defaults.json
  acceptance.json
  model_card.md
  sha256.txt
```

Then create:

```text
%LOCALAPPDATA%\DreamAirTracking\models\model_registry.json
```

Example:

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
      "onnx": "C:\\Users\\You\\AppData\\Local\\DreamAirTracking\\models\\dreamair-main-current\\model.onnx",
      "metadata": "C:\\Users\\You\\AppData\\Local\\DreamAirTracking\\models\\dreamair-main-current\\metadata.json",
      "outputs": ["gaze_xy", "openness_lr", "pupil_lr", "confidence"],
      "default": true
    }
  ]
}
```

The app shows only real imported model packages. If no package exists, the model list stays empty and eye tracking start is disabled.

Schema files:

- `models/model_registry.schema.json`
- `models/model_registry.example.json`

### 3. Start Runtime

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

If your Dream Air tracking is weak in a specific direction or eye state, useful data is:

- Full 9-point gaze calibration, especially `right_down`, `left_down`, `right_up`, `left_up`
- Multiple re-wear sessions from the same person
- Open / half-open / closed / blink eyelid captures
- Short failure replay after you notice a bad direction in VRChat
- Notes about glasses, headset fit, lighting, and whether BrokenEye streams looked stable

Please do **not** send random loose screenshots or only CSV row counts. A useful package must include the left/right eye frames, labels, session metadata, stage names, accepted/rejected state, and quality metrics.

Detailed requirements are in:

- `datasets/session_package.schema.md`
- `docs/MODEL_ADAPTATION_DATA_AUDIT_zh.md`
- `docs/SINGLE_HEADSET_RELEASE_AND_TRAINING_PLAN_zh.md`

Users can upload capture packages to Google Drive / OneDrive / another file host and share the link manually. The app does not upload data automatically.

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
