# DreamAirTracking 单一头显发布与训练路线

本文取代“跨设备/多头显泛化”的表述。当前项目只服务一个硬件目标：Dream Air 这一类固定头显和固定 BrokenEye/VRCFT 链路。我们要解决的是同一型号头显下的不同佩戴、不同人、不同眼型，而不是同时兼容多种头显。

## 当前目标

```text
Dream Air + BrokenEye
  -> DreamAirTracking App
  -> ONNX eye model
  -> VRCFT eye slot
  -> VRChat
```

SRanipal 或其它 face module 继续负责 face/expression slot。DreamAirTracking 默认只占 VRCFT eye slot，避免和 SR 抢 expression slot。

## Clean Repo 边界

这个 clean repo 应该发布源码、测试、脚本、schema 和路线文档，不发布本地训练产物。

应发布：

```text
src/
tests/
scripts/
docs/
models/model_registry.schema.json
datasets/session_package.schema.md
README.md
requirements-runtime.txt
requirements-training.txt
```

不应发布：

```text
runs/
真实眼图采集 session
runtime CSV
*.onnx
*.pt
.external/
VRCFaceTracking 本机源码副本
个人绝对路径配置
```

模型通过 GitHub Release、Git LFS、Hugging Face 或手动下载包分发。repo 内只放 registry/schema/model card。

## App 和模型怎么解耦

当前 app 仍有历史硬编码 run 目录。发布前建议改成：

1. app 读取 `models/model_registry.json`。
2. 现有模型选择按钮显示 registry 中的 model package。
3. model package 提供 ONNX、metadata、runtime defaults、acceptance report。
4. app 只负责启动 runtime，不关心模型来自哪个训练 run。

模型包结构：

```text
models/dreamair-round9-gaze/
  model.onnx
  metadata.json
  runtime_defaults.json
  acceptance.json
  model_card.md
  sha256.txt
```

Runtime contract 保持稳定：

```text
input: left/right grayscale eye image pair
output: gaze_xy, openness_lr, pupil_lr, confidence
optional output: wide_lr, squint_lr
```

即使模型内部升级，app 只看 metadata 和 output names。

## App 采集必须服务训练

app 的 5 点/9 点校准不是单纯用户校准 UI，它应该产出可训练数据。

app 侧校准算法、runtime adapter、openness 后处理和训练包导出边界见：

```text
docs/APP_CALIBRATION_RUNTIME_DESIGN_zh.md
```

当前已有基础：

- `session.json`
- `labels.jsonl`
- `pairs.csv`
- `metrics.json`
- 左右眼帧
- stage id
- raw detector 坐标
- confidence
- openness
- calibration report

下一步应新增一键导出训练包：

```text
DreamAirTrackingCapture_YYYYMMDD_HHMMSS.zip
```

用户不需要 app 上传。app 只生成 zip 和隐私提示，用户自己传 Google Drive / OneDrive / 私信链接。

## 单头显数据采集策略

因为只服务 Dream Air，数据重点不是 device diversity，而是：

- 不同用户
- 不同佩戴位置
- 同一用户反复摘戴
- 眼皮/睫毛遮挡
- right/down/right_down 边缘失败
- blink/open/half/closed 的真实眼皮曲线

建议协议：

| 协议 | 内容 | 用途 |
|---|---|---|
| Quick 5-point | center/up/down/left/right | 本地 runtime 校准，不一定进入全局训练 |
| Full 9-point | 5 点 + 四个 diagonal | gaze 主训练和 hardcase |
| Eyelid | open/half/closed/blink/squint | openness 训练 |
| Redon | 同一人摘戴后重复 9 点 | 佩戴鲁棒性 |
| Failure replay | 用户觉得差的方向短采样 | hardcase fine-tune |

每个采集包必须写明 `subjectId`, `wearId`, `deviceFamily=Dream Air`, `capture_protocol`。

## 数据变多后是否需要换模型

短答案：先不急着换。单一头显下，当前 Siamese MobileNetV3-small 路线仍然合理。数据量增加后的第一收益应该来自更干净的 split、mask、hardcase 和校准，而不是马上换大模型。

建议决策：

| 数据规模 | 建议模型策略 |
|---|---|
| 当前单人/少量 wear | 保持 Round9 类模型，app calibration + 小步 hardcase fine-tune |
| 5-10 人，30-50 wear | 仍用 Siamese MobileNetV3-small，重新训练/微调，重点做 subject/wear held-out |
| 10-30 人，100+ wear | 可尝试 Siamese MobileNetV3-small + metadata/wear normalization；不要直接扩大到很重模型 |
| 30+ 人后边缘 reach 仍卡住 | 再评估 MobileNetV3-large、EfficientNet-Lite0、ConvNeXt-Tiny 或 160/192 输入 |

换模型的触发条件：

- subject-held-out 的 right/down/right_down reach 长期不达标；
- openness 在多人上需要过强后处理才像闭眼；
- pupil/眼皮遮挡下 confidence 无法区分坏样本；
- 小模型在训练集也欠拟合，而不是只验证集差。

不建议因为 CSV 行数变多就换模型。之前的问题就是行数变多但真实信息量没变多。

## 数据变多后是否需要重新训练

需要，但不要每来一个用户就全量重训。

推荐节奏：

1. 用户本地只做 app 校准：不重训。
2. 收到少量 failure replay：生成 hardcase manifest，从当前模型低 LR 微调。
3. 收到一批新用户：重新构建 head-specific manifest，做一次正式 retrain/fine-tune。
4. 达到多人 subject-held-out 阶段：每个候选模型必须跑完整 acceptance。

训练时从当前 best checkpoint 初始化，而不是从零开始。当前路线应保留 teacher/base：

```text
base/teacher: 当前 best gaze model
new data: head-specific manifest
training: low LR + freeze BN + teacher gaze distillation
selection: edge reach + center radius + openness gate
```

## Fine-tune 规则

### Gaze hardcase

例如 right/down/right_down 失败：

```text
gaze_weight > 0
openness_valid = 0
wide_valid = 0
squint_valid = 0
pupil_valid = 0
sample_weight 按 hardcase 加权
```

不要让这些行污染 openness/expression/pupil。

### Eyelid/openness

眼皮数据：

```text
gaze_weight = 0
openness_valid_left/right = 1
wide/squint 视协议决定
pupil_valid 只有质量足够时打开
```

训练后要检查：

- open 是否稳定接近 1；
- gentle blink 是否能掉；
- intentional closed 是否能接近 0；
- 中段是否平滑；
- gaze edge reach 是否没有退化。

### Pupil

pupil 目前主要是 weak label。可以辅助，但不应主导共享 encoder。质量低、眼皮闭合、边缘遮挡时应关闭 pupil loss。

### Expression

当前发布路线默认 DreamAir 不占 VRCFT expression slot，让 SR 负责 face/expression。wide/squint 可以保留在模型和 CSV 里做研究，但 public runtime 默认不发到 VRCFT expression。

## 推荐训练流水线

```text
capture package
  -> validate_capture_package.py
  -> build_eye_manifest_v4/v5
  -> audit_eye_manifest_series.py
  -> train_eye_multitask.py
  -> eval_eye_multitask.py
  -> analyze_multitask_outputs.py
  -> check_eye_model_acceptance.py
  -> export_eye_multitask_onnx.py
  -> model package
```

必须保留的 split：

```text
train: 训练用户和 wear
val_wear: 已见用户的新佩戴
val_subject: 未见用户
test_subject: 完全保留到最后的未见用户
```

如果数据还少，至少先做 `wear-held-out`，不要随机行切分。

## 验收指标

不能只看 `val_gaze_l2`。

必须看：

```text
center_median_radius <= 0.055
left/right/up/down reach
right_down diagonal reach
open openness median
closed openness median
blink drop latency
pupil center error
confidence 与坏帧相关性
runtime live CSV stability
```

模型选择优先级：

1. gaze edge reach 不退化；
2. center 不漂；
3. openness 曲线有效；
4. pupil 稳；
5. runtime 不卡顿。

## 近期工程 TODO

P0:

- clean repo 不包含 `runs/`、个人采集和 `.external`。
- app 默认只占 VRCFT eye slot。
- 模型选择改成 registry/package。
- 写 capture package exporter。

P1:

- `validate_capture_package.py`。
- manifest builder 直接消费 app zip。
- subject/wear split 支持。
- model card 和 acceptance 自动生成。

P2:

- runtime/training requirements 分离。
- VRCFT module SDK 依赖文档化或 submodule 化。
- GitHub Release model package 模板。

## 对外表述

建议公开 README 只写：

```text
DreamAirTracking is an experimental local eye-tracking pipeline for Dream Air.
It provides app-side calibration, ONNX model runtime integration, VRCFT eye output,
and tools for collecting training packages for future same-headset model improvement.
```

不要写多头显通用。我们只承诺 Dream Air 这一条链路。
