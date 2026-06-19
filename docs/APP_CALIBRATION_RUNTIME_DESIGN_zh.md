# DreamAirTracking App 校准、Runtime 与数据采集设计

这份文档把旧整理文档里的 app 侧内容拆出来，作为 clean repo 的 app 工程路线。当前范围只服务 Dream Air 单一头显链路，不讨论多头显泛化。

## 结论

app 侧不是简单的启动器。它现在已经承担四件事：

1. 从 BrokenEye 读取左右眼图；
2. 做 5 点/9 点 gaze 校准、质量审核和 session 落盘；
3. 在 runtime 里做 gaze/openness/pupil 的轻量后处理；
4. 启动 Bridge，把结果送到 VRCFT eye slot。

下一步不要新增一堆采集按钮。应该复用现有 Calibration/Session 流程，把当前已经采集到的 session 规范化导出成训练包，让 app 的 5 点/9 点、眼皮、pupil 数据真的服务下一代模型。

## 当前 App 侧职责边界

```text
BrokenEye HTTP /eye/left, /eye/right
  -> App calibration capture
  -> session.json / labels.jsonl / pairs.csv / frames/
  -> local runtime calibration/profile
  -> Bridge runtime
  -> VRCFT eye output
```

app 侧应该做：

- 本地校准：offset、gain、quick affine、profile/input normalization；
- 数据采集：稳定保存左右眼帧、stage、目标点、检测质量、openness；
- 模型接入：通过 model registry/package 选择 ONNX；
- 输出治理：只占 VRCFT eye slot，默认不抢 SRanipal/其它 face module 的 expression slot。

app 侧不应该做：

- 在 repo 中硬编码某个 `runs/...` 训练目录；
- 把真实采集、runtime CSV、模型权重塞进 Git；
- 把 5 点/9 点校准数据随机混进所有 head 的 loss；
- 让 openness/wide/squint 这种 shape 输出默认影响 SR face/expression。

## 现有实现库存

| 功能 | 当前实现 | 作用 |
|---|---|---|
| 5 点/9 点 stage | `CalibrationStageCatalog` | 定义 center/up/down/left/right 和四个 diagonal |
| 同步采集 | `BrokenEyeCalibrationCaptureService` | 从 `/eye/left`、`/eye/right` 读取 MJPEG，按时间配对 |
| session 落盘 | `CalibrationSessionStore` | 保存 `session.json`、`labels.jsonl`、`pairs.csv`、`metrics.json` |
| 人工审核 | `CalibrationLabel` + `CalibrationStageReviewAdvisor` | accepted/bad/range/bad frames，记录 stage 质量 |
| gaze profile 拟合 | `GazeCalibrationFitter` | 从 accepted labels 拟合 affine/quadratic gaze map |
| raw 坐标归一 | `RawCoordinateNormalizationProfileBuilder` | 用当前佩戴中心/跨度修正输入坐标 |
| runtime 5 点校准 | `RuntimeGazeCalibrationFitter` | 从 center/left/right/up/down 估计 offset/gain |
| quick affine layer | `QuickGazeCalibrationLayer` | 对 Bridge 输出再套一层 `A*x+b` |
| openness 检测 | `EyeOpennessDetector` | 用暗像素行分布估计眼裂高度和开放度 |
| openness runtime filter | `BridgeOpennessFilter` | 自适应 open baseline、平滑、抑制单眼误掉 |
| 闭眼 gaze 稳定 | `BridgeClosureGazeStabilizer` | 闭眼/眨眼时短暂 hold gaze，避免乱跳 |
| 输出映射 | `BridgeOutputMapper` | 双眼融合后做 center lock、deadzone、gain、offset |

## 5 点/9 点校准含义

当前 stage 定义如下：

```text
center      -> ( 0,  0)
up          -> ( 0,  1)
down        -> ( 0, -1)
left        -> (-1,  0)
right       -> ( 1,  0)
left_up     -> (-1,  1)
right_up    -> ( 1,  1)
left_down   -> (-1, -1)
right_down  -> ( 1, -1)
```

5 点主要服务本地 runtime 校准：修 offset、少量 gain、佩戴偏移。

9 点更重要：它同时服务本地校准和训练数据。尤其 `right_down` 是我们已经观察到的失败方向，不能只靠 5 点推断。

`closed`、`open` 这类 stage 应该服务 openness/eyelid，不应该参与 gaze fitting 或 gaze loss。

## 采集算法

`BrokenEyeCalibrationCaptureService` 的流程是：

1. 建立 bounded channel；
2. 并行读取 `http://127.0.0.1:5555/eye/left` 和 `/eye/right`；
3. 保留左右短队列；
4. 用 `MaxDeltaMs` 找最接近的左右眼帧；
5. 对每对帧运行传统 pupil detector 和 openness detector；
6. 保存左右 JPEG 到 `frames/`；
7. 写入 `CalibrationFramePair`：

```text
sequence
stage
left/right file
delta_ms
left/right found
left/right raw_x/raw_y
left/right confidence
left/right openness
```

每个 stage 结束后会生成 quick check：

```text
pair_count
valid_pair_count
average_delta_ms
average_confidence
low_openness_pairs
low_confidence_pairs
not_found_pairs
sync_late_pairs
quality
suggested_action
review_reason
improvement_plan
```

这已经足够作为训练包的基础，不需要从零设计采集 app。

## 标签与审核

用户在 app 里接受/拒绝一个 stage 时，会写入 `labels.jsonl`。标签包含：

```text
stage_id
target
frame_start
frame_end
accepted
operator_verdict
bad_frames
notes
```

训练侧必须尊重这些标签：

- `accepted=false` 不进正式训练；
- `operator_verdict=bad` 不进正式训练；
- `bad_frames` 排除；
- `closed/open/blink` 不进 gaze；
- low confidence / low openness 行只能作为 debug 或对应 head 的 hardcase。

## Gaze Profile 拟合

`GazeCalibrationFitter` 使用 accepted labels 和 `pairs.csv` 来拟合每只眼的 gaze map。

过滤条件：

```text
delta_ms <= MaxDeltaMs
confidence >= MinConfidence
openness >= MinOpenness
raw_x/raw_y finite
stage 不在 blink/closed/open 忽略列表
```

拟合策略：

- 先按 stage 求 raw median；
- 默认至少 3 个 stage 才能 affine；
- 6 个以上 stage 可以尝试 quadratic；
- `auto` 模式只有在 quadratic 明显降低 residual 时才切换；
- center stage 会加权，避免中心漂；
- 输出 stage residual、center drift、quality。

这套适合生成或修正 `tracking_profile*.json`。但它不是万能模型训练。它修的是传统 detector/raw 坐标到输出空间的映射，不能凭空补出模型没有学会的眼图特征。

## Runtime 5 点校准

`RuntimeGazeCalibrationFitter` 是更轻的 runtime adapter。它只要求：

```text
center
left
right
up
down
```

它会计算每个 stage 的输出 median，然后检查：

- left/right 水平跨度不能太小；
- up/down 垂直跨度不能太小；
- left 必须真的在 center 左边；
- right 必须真的在 center 右边；
- up/down 也必须和 center 有明确方向 margin。

通过后生成：

```text
centerOffsetX/Y
xGain/yGain
stageMedians
sampleCount
```

当前默认更偏保守：offset 是主要收益，gain 是否应用需要受控。原因是模型 edge reach 差时，盲目放大 gain 会让中心和边缘一起变不稳。

## Quick Affine Layer

`QuickGazeCalibrationLayer` 是 Bridge 输出空间上的最后一层校正：

```text
output = clamp(A * bridge_output_xy + b, -1, 1)
```

它从 JSON 中读取：

```text
raw_source = bridge_output_xy
sessions[session_id].used = true
sessions[session_id].A
sessions[session_id].b
```

适用场景：

- 同一个模型在当前佩戴下整体偏移；
- right/down 有轻微比例不足；
- 用户只想快速修正当前 runtime。

不适用场景：

- 模型在 right_down 完全没有可分辨特征；
- openness 曲线错误；
- pupil weak label 失真；
- 训练集中 head loss 混淆导致主干退化。

## Raw 坐标归一

`RawCoordinateNormalizationProfileBuilder` 用当前 session 的 center 和边缘 stage median，估计当前佩戴相对基础 profile 的中心和尺度变化：

```text
CurrentCenterX/Y
TargetCenterX/Y
ScaleX/ScaleY
```

这适合 Dream Air 单一头显里的不同佩戴。它应该作为 app 侧 profile/adapter，而不是每个用户都重训模型。

## Openness 算法

当前 openness 分两层。

第一层是 `EyeOpennessDetector`：

1. 在 ROI 内估计暗像素阈值；
2. 按行统计 dark fraction；
3. 平滑行分布；
4. 找连续暗像素 aperture；
5. 根据 peak dark fraction 和 aperture height 得到 `openness`。

第二层是 `BridgeOpennessFilter`：

- 自适应 open baseline；
- 用 peak fraction 和 aperture height 混合；
- 闭合时用较快 fall alpha；
- 张开时用 rise alpha；
- 抑制“单眼突然掉、另一只眼仍完全 open”的误判。

还有 `BridgeClosureGazeStabilizer`：

- 检测到 closing/closed 时短暂 hold gaze；
- 重新睁眼时用小 alpha 释放；
- 目的不是让闭眼更夸张，而是避免闭眼瞬间 gaze 乱跳。

## Openness 近期调教原则

你在 VRC 里观察到的问题是：闭眼时 shape 有点挤压，中段不够丝滑，尾部又不够容易闭严。

这不应该优先靠重训解决。最小改进应该在 runtime 曲线做：

- 保持 open 区域更稳定，避免轻微眨动就改变 shape；
- 中段用更平滑的 S curve；
- closed 尾部保留足够下压力，让故意闭眼能闭到底；
- 不让 wide/squint 默认抢 expression slot；
- 眼皮输出和 gaze 稳定分开调，不要为了闭眼形状牺牲 gaze。

也就是说，app 侧应输出“眼皮 openness”，而不是把它过度解释成表情 shape。

## Pupil

pupil 当前有两类输出：

- pupil assist：由 openness/confidence 推出辅助参数；
- raw pupil diameter：传统 pupil diameter estimator + calibration profile。

pupil 在当前模型里表现不错，但训练侧仍要当 weak label 使用。app 导出的训练包里应该保留 pupil 相关字段，但 manifest 里必须有 `pupil_valid_left/right`，闭眼、遮挡、低 confidence 时关闭 pupil loss。

## 模型接入方式

当前应该从硬编码 run 路径改成 model package：

```text
models/<model_id>/
  model.onnx
  metadata.json
  runtime_defaults.json
  acceptance.json
  model_card.md
  sha256.txt
```

app 读取：

```text
models/model_registry.json
```

现有按钮继续保留，但按钮展示的是 registry 中的模型条目。这样 app 不关心 Round2/Round8/Round9 的历史目录，也不会把本机训练 workspace 绑定进发布版。

## App 数据如何服务训练

导出的训练包建议结构：

```text
DreamAirTrackingCapture_YYYYMMDD_HHMMSS/
  manifest.json
  session.json
  labels.jsonl
  pairs.csv
  metrics.json
  frames/
  runtime/
    bridge_options.json
    quick_gaze_calibration.json
    openness_calibration.json
    pupil_calibration.json
```

`manifest.json` 至少包含：

```text
schema
deviceFamily = Dream Air
subjectId
wearId
captureProtocol
appVersion
brokenEyeVersion
modelId
createdAt
privacyNote
```

训练脚本消费 app package 时必须生成 head-specific mask：

| 数据来源 | gaze | openness | pupil | wide/squint |
|---|---:|---:|---:|---:|
| 5 点 quick | 可选，默认只校准 | 0 | 0 | 0 |
| 9 点 gaze | 1 | 0 | 质量足够才 1 | 0 |
| right_down hardcase | 1 | 0 | 0 | 0 |
| open/half/closed/blink | 0 | 1 | 质量足够才 1 | 可选 |
| pupil calibration | 0 | 0 | 1 | 0 |

这一点是后续训练不再倒退的关键：app 采集到什么，就只监督对应 head。

## 不新增复杂 UI 的改造方式

优先复用现有页面：

- Calibration 页面继续完成 5 点/9 点和眼皮采集；
- Sessions 页面显示 session 审核结果；
- 训练包导出可以作为 session 操作，不必单独做一个复杂向导；
- 模型选择按钮继续存在，但数据来源改成 registry；
- VRCFT/SR 分工在 About/Settings 里说明，不引导用户同时占 expression slot。

建议最小新增能力：

1. `Export training package`：对当前 session 打 zip；
2. `Use as runtime calibration`：把当前 session 的 quick calibration 写到 runtime 配置；
3. `Copy package path`：方便用户手动上传 Google Drive/OneDrive。

如果坚持不加按钮，也可以先把导出放在 Sessions 页的现有 session item 操作里。

## 后续工程优先级

P0:

- 把 app 硬编码 run 路径改成 model registry；
- 明确默认只输出 VRCFT eye slot；
- 保留 SR/其它 module 的 face/expression slot；
- 文档化 5 点/9 点、right_down、open/closed 的语义；
- 训练包导出不包含个人绝对路径。

P1:

- app package -> manifest builder 直连；
- `validate_capture_package.py`；
- subject/wear/session 字段强制存在；
- acceptance report 回写 model package；
- runtime openness curve 配置化，避免每次为 VRC 表现改代码。

P2:

- model registry UI；
- model package 下载/校验；
- per-user adapter/profile 管理；
- 多 session audit 可视化。

## 工程判断

Dream Air 单一头显下，app 侧校准是必须保留的。它解决的是同一硬件、不同佩戴、不同人的小分布偏移。

但 app 校准不能替代模型训练：

- right_down 完全学不到，必须补训练 hardcase；
- openness 曲线形状怪，优先调 runtime curve，然后再补 eyelid 数据；
- pupil weak label 只能辅助，不能让它支配共享 encoder；
- 5 点校准能修 offset/gain，不能证明模型具有边缘 reach。

下一代路线应该是：

```text
通用 Dream Air 模型
  + app 5/9 点本地校准
  + per-wear runtime adapter
  + app 导出的 hardcase/eyelid/pupil package
  + head-specific retrain/fine-tune
```

这样 app 和模型解耦，但 app 的校准、采集、眼皮追踪都会真实服务模型。
