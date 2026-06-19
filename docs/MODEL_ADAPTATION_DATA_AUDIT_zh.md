# DreamAirTracking 模型适配数据审计

本文回答一个发布前必须说清楚的问题：如果要为 Dream Air 训练或微调一个更好的模型，新增一个用户/佩戴/失败方向适配到底需要哪些数据。

结论：**不要只要 CSV 行数。需要按 head 隔离的数据包、明确 protocol、accepted labels、左右眼帧、质量指标和 held-out split。**

## 适配分级

| 目标 | 是否需要上传数据 | 是否需要重新训练 | 所需数据 |
|---|---:|---:|---|
| 本机佩戴偏移修正 | 否 | 否 | App 内 5 点/9 点校准 |
| 某个方向小幅修正 | 可选 | 通常不全量重训 | 9 点 hardcase capture package |
| 某人/某眼型适配 | 是 | 低 LR fine-tune | 多次 re-wear 9 点 + eyelid |
| 发布更好通用模型 | 是 | 是 | 多用户、多 wear、subject/wear held-out |
| 换模型架构 | 是 | 是 | 先证明小模型欠拟合，而不是数据/label 问题 |

## 一个有效 Capture Package 必须包含

结构见 `datasets/session_package.schema.md`。最低要求：

```text
DreamAirTrackingCapture_YYYYMMDD_HHMMSS/
  session.json
  device.json
  capture_protocol.json
  labels.jsonl
  pairs.csv
  metrics.json
  calibration_report.md
  frames/
    center_000001_left.jpg
    center_000001_right.jpg
```

必须有：

- `deviceFamily = Dream Air`
- `subjectId`：匿名即可，但同一人要稳定一致
- `wearId`：每次摘戴/重新佩戴必须不同
- `protocolId`：例如 `five_point_gaze`、`nine_point_gaze`、`eyelid_open_half_closed`
- `stage`：center/up/down/left/right/diagonal/open/half/closed/blink 等
- `accepted`：用户或自动质量 gate 接受的 stage 才能进正式训练
- 左右眼帧：不能只有推理输出 CSV
- 每对帧的 `delta_ms`、confidence、openness、found 状态

## 推荐采集协议

### 最小可用包

用于判断某个用户是否适合加入训练：

```text
1 个 wear
1 次 9-point gaze
1 次 open / half / closed / blink eyelid
短 notes
```

这只能做审计和小 hardcase，不足以代表一个人。

### 推荐适配包

用于给某个用户/眼型做可靠微调：

```text
3 个 wear
每个 wear 1 次 full 9-point gaze
每个 wear 1 次 eyelid open / half / closed / blink
至少 1 段 failure replay
```

重点覆盖：

- center
- left / right
- up / down
- left_up / right_up / left_down / right_down
- open / half / closed / blink
- 用户自己觉得差的方向，例如 right_down

### 发布级训练数据

用于推出更好的公开模型：

```text
5-10 人：可以做第一版多人 fine-tune
10-30 人：可以开始 subject-held-out 验证
30+ 人：再考虑是否需要更大模型
```

每个人至少：

```text
2-3 个 wear
每个 wear 一个 accepted 9-point gaze
至少一个 eyelid protocol
失败方向 replay，若有
```

## 质量门槛

一个 session 进入训练前必须审计：

| 项目 | 要求 |
|---|---|
| BrokenEye stream | `/eye/left` 和 `/eye/right` 都 live |
| 左右眼同步 | `delta_ms` 应稳定，严重超时帧剔除 |
| stage 覆盖 | 9 点必须包含 center、四方向、四 diagonal |
| accepted labels | 只使用 accepted stage |
| bad frames | 必须排除 |
| gaze edge | right/down/right_down 不能缺失 |
| eyelid | open/half/closed/blink 要分开 |
| pupil | 只在 openness 和 confidence gate 通过时使用 |

## Head-Specific Mask 规则

这是当前项目最重要的训练纪律。

### Gaze rows

```text
gaze_valid = 1
gaze_weight > 0
openness_valid_left/right = 0
wide_valid_left/right = 0
squint_valid_left/right = 0
pupil_valid_left/right = 0
```

用途：训练 gaze，不允许顺便训练 eyelid/pupil/expression。

### Eyelid rows

```text
gaze_weight = 0
openness_valid_left/right = 1
wide/squint 根据协议决定
pupil_valid 只有质量足够时打开
```

用途：训练 openness，不允许中心 gaze 被大量重复污染。

### Pupil rows

```text
pupil_valid_left/right = 1 仅在 confidence/openness gate 通过时
gaze_weight 默认 0，除非同一行也是明确 gaze protocol
```

用途：弱监督 pupil。pupil 标签不是人工精标，不能主导共享 encoder。

### Expression rows

公开 runtime 默认不占 SR/face expression slot。wide/squint 可以保留做研究，但训练时必须和 gaze/openness 分开 mask。

## Split 规则

禁止随机按行切分。必须按 session/wear/subject 切：

```text
train: 训练用户和 wear
val_wear: 已见用户的新佩戴
val_subject: 未见用户
test_subject: 完全保留到最后
```

如果数据还少，至少做 wear-held-out，不要把同一个 calibration session 的相邻帧同时放进 train 和 val。

## 训练流程

推荐流程：

```text
capture package
  -> validate package
  -> build manifest
  -> audit manifest unique pairs / head masks / stage coverage
  -> train_eye_multitask.py
  -> eval_eye_multitask.py
  -> analyze_multitask_outputs.py
  -> check_eye_model_acceptance.py
  -> export_eye_multitask_onnx.py
  -> model package
```

推荐训练策略：

- 从当前 best model 初始化，不从零开始；
- low learning rate；
- freeze BatchNorm；
- teacher gaze distillation 保护旧 gaze；
- checkpoint selection 优先 `gaze_l2_edge` 或 edge-aware combo；
- openness/pupil/expression 不得靠整体 loss 抢 checkpoint。

## 验收指标

不能只看 `val_loss` 或 `val_gaze_l2`。

必须看：

```text
center median radius
left/right/up/down reach
left_up/right_up/left_down/right_down reach
gaze_l2_edge
open openness median
half openness curve
closed openness median
blink drop latency
pupil MAE under valid mask
confidence/gate behavior
```

失败模式要分开记录：

- source stream 坏：BrokenEye 本身没给稳定图；
- local fit 坏：app 校准/佩戴偏移问题；
- model gaze 坏：edge reach 学不到；
- openness 坏：曲线或标签问题；
- multitask 污染：某个 head 的数据让 gaze 退化。

## 用户如何提交数据

1. 在 app 内完成校准/采集。
2. 导出 capture package zip。
3. 自己上传到 Google Drive / OneDrive / 其它网盘。
4. 在 GitHub Issue 里说明：

```text
Headset: Dream Air
Problem: e.g. right_down weak / blink not closing / pupil unstable
Model package id:
BrokenEye status:
VRCFT status:
Capture package link:
Notes: glasses, fit, lighting, whether re-worn
```

不要直接公开上传包含个人隐私的眼图，除非你明确接受公开发布。更推荐用私密链接发给维护者。

## 什么时候需要换模型

先不因为数据变多就换模型。当前 Siamese MobileNetV3-small 仍是合理基线。

考虑换模型的条件：

- subject-held-out 的 edge reach 长期失败；
- 小模型在训练集也明显欠拟合；
- openness 必须靠过强 runtime curve 才能闭眼；
- pupil/confidence 在遮挡和闭眼下无法分离坏样本；
- 已经确认不是 label/mask/split/checkpoint selection 问题。

否则优先做：

- 更多真实 accepted 9-point；
- right/down/right_down hardcase；
- 更严格 head masks；
- wear-held-out / subject-held-out validation；
- teacher/base 保护 gaze。
