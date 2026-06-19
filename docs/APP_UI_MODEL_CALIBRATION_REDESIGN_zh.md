# DreamAirTracking App UI、模型加载与三类校准重设计

本文针对当前截图里的 app 问题做设计收敛。目标不是继续堆调试按钮，而是把 app 做成一个给 Dream Air 用户可用的工具：

- 下载什么模型，就加载什么模型；
- gaze / eyelid / pupil 三个校准模块都能同时服务本地校准和后续训练；
- 普通用户只看到当前校准状态和一个 `Start calibration`；
- 历史 session、watcher、details、review 手工按钮都退到开发者/内部流程。

## 总结决策

1. 首页模型选择必须改成 `Installed model packages`，不再显示 Round2/Round8/Round9 这种训练内部名字。
2. 模型下载/导入后生成本地 package registry，app 根据用户选中的 package id 精确加载对应 ONNX、metadata、runtime defaults。
3. 普通 UI 只显示真实导入的模型包；没有模型包时下拉留空，并显示“没有模型包”。
4. Calibration 页保留三个模块：`Gaze`、`Eyelid`、`Pupil`。切到哪个模块，就显示该模块当前参数和一个醒目的大校准按钮。
5. `Accept / Retry / Mark bad` 从普通 UI 删除。校准流程内部自动做质量 gate；失败就在状态里提示重新开始。
6. `Details` 删除。Pipeline diagnostics 移到开发者诊断页或隐藏开发者模式。
7. Sessions 页不作为普通用户主导航。普通用户只看到“当前已保存校准”；历史 session 保留在本地用于开发者导出训练包。
8. About 页删掉长说明，只保留开发者卡片、头像、当前头显选择。
9. 图标使用开发者自己的模型头像，和 About 页头像保持一致。

## 软件图标

当前决定：不用概念头显图标，直接使用开发者自己的模型头像作为 app 图标和 About 头像。

```text
docs/assets/branding/developer-avatar-app-icon-1024.png
docs/assets/branding/developer-avatar-app-icon-256.png
docs/assets/branding/developer-avatar-app-icon-44.png
```

预览：

![DreamAirTracking developer avatar icon](assets/branding/developer-avatar-app-icon-256.png)

实现原则：

- 图标和 About 页都使用同一张开发者头像，减少品牌割裂；
- app icon 输出为 Windows 需要的 ico、44、150、store、wide、splash 尺寸；
- 头像资产保留在 app 仓库内，不依赖用户临时目录；
- 后续如果要发布到 GitHub，只需要替换 `Assets/DeveloperAvatar.png` 和同名 icon 输出资产。

已替换的 app 资产：

```text
src/DreamAirTracking.App/Assets/AppIcon.ico
src/DreamAirTracking.App/Assets/DeveloperAvatar.png
src/DreamAirTracking.App/Assets/Square44x44Logo.png
src/DreamAirTracking.App/Assets/Square44x44Logo.scale-200.png
src/DreamAirTracking.App/Assets/Square150x150Logo.png
src/DreamAirTracking.App/Assets/Square150x150Logo.scale-200.png
src/DreamAirTracking.App/Assets/StoreLogo.png
src/DreamAirTracking.App/Assets/Wide310x150Logo.png
src/DreamAirTracking.App/Assets/Wide310x150Logo.scale-200.png
```

## 模型下载与加载

你说得对：用户下载了哪个模型，app 就应该加载哪个模型。训练内部 round 名字不应该出现在用户 UI。

### 模型包结构

```text
%LOCALAPPDATA%/DreamAirTracking/models/
  dreamair-main-20260619/
    model.onnx
    metadata.json
    runtime_defaults.json
    acceptance.json
    model_card.md
    sha256.txt
```

repo 内只保留 schema 和 example：

```text
models/model_registry.schema.json
models/model_registry.example.json
```

用户机器上的实际 registry：

```text
%LOCALAPPDATA%/DreamAirTracking/models/model_registry.json
```

开发工作区 fallback registry：

```text
<repo>/models/model_registry.json
```

### 加载顺序

```text
selectedModelPackageId
  -> user model registry
  -> repo model registry
  -> explicit local path
  -> legacy runs fallback, only in developer mode
```

普通用户模式不做 legacy fallback 充数。导入什么模型包，就显示那个模型包的 `displayName`；没有导入模型包时：

```text
Model package: [No model package]
Status: No model package installed
Start Eye Tracking: disabled
```

普通 UI 只显示：

```text
Dream Air Main 2026-06
Dream Air Stable
Dream Air Experimental
```

不显示：

```text
Round9 + Round8 expression
Split Round2/Round8 single ONNX
Legacy two-channel
```

这些内部名字只留在 model card 和 developer mode。

### 首页改法

当前图一的问题是模型下拉框还在显示训练历史。应改成：

```text
Model
[Dream Air Main 2026-06       v]
Status: Installed, accepted, gaze/openness/pupil
[Start Eye Tracking] [Stop]
```

如果没有安装模型：

```text
Model
No model installed
[Install model package]
```

下载完成后：

1. 解压 package；
2. 校验 sha256；
3. 写入 registry；
4. 选择该 package id；
5. runtime 启动时按该 package 加载 ONNX/metadata/defaults。

## 三类校准的统一模型

三个模块都遵循同一个结构：

```text
Calibration module
  -> local runtime calibration
  -> saved current profile
  -> optional training package rows
```

保存产物：

```text
%LOCALAPPDATA%/DreamAirTracking/calibrations/
  current/
    gaze_calibration.json
    eyelid_calibration.json
    pupil_calibration.json
    calibration_profile.json
  sessions/
    20260619_184500_gaze/
    20260619_185200_eyelid/
    20260619_190100_pupil/
```

普通用户只看 `current`。`sessions` 是内部记录，用于导出训练包，不在主 UI 大列表里刷屏。

## Calibration 页整体结构

顶部：

```text
Calibration
[ Gaze ] [ Eyelid ] [ Pupil ]
```

左侧主区域：当前模块的校准画面和 live preview。

右侧参数卡：只显示当前模块最重要的参数。

底部：

```text
[        Start gaze calibration        ]
```

校准按钮必须是页面主动作，使用大号 accent button，不再放成右下角很小的 command bar item。

没有：

```text
Accept
Retry
Mark bad
Details
Use latest
Capture training
```

这些动作改为内部流程：

- 质量通过 -> 自动保存当前模块 calibration；
- 质量失败 -> 状态提示“重新开始校准”；
- 训练包导出 -> 校准完成后显示 `Export training package`，普通用户不需要进 Sessions；
- 使用最新 -> 保存成功后自动成为最新。

## 傻瓜式采集和导出

普通用户不需要理解 manifest。客户端要把“采集什么个人数据”翻译成三个 Cali：

| 用户遇到的问题 | 选择哪个 Cali | 采集到的个人数据 | 是否适合训练 |
|---|---|---|---|
| 视线方向不准 | `Gaze` | center/up/down/left/right，9 点时包含四个 diagonal | 是，训练 `gaze_xy` |
| 右下/左下/角落看不到 | `Gaze` + 9 点 | `right_down`、`left_down`、`right_up`、`left_up` hardcase | 是，训练 edge gaze |
| 眨眼不明显、闭眼不够 | `Eyelid` | open/half/closed/blink 眼皮形状 | 是，训练 `openness_lr` |
| 眼睛形状挤压 | `Eyelid` | 半闭、闭眼、自然眨眼曲线 | 是，训练 openness 曲线，必要时修 runtime |
| 瞳孔不稳定 | `Pupil` | pupil runtime 状态和 quality gate | 仅质量通过时作为 weak pupil |

推荐公开模型采集流程：

```text
1. 戴好 Dream Air
2. 打开 BrokenEye，确认 /eye/left 和 /eye/right 正常
3. 打开 DreamAirTracking
4. Calibration -> Gaze -> 9-point -> Start gaze calibration
5. Calibration -> Eyelid -> Start eyelid calibration
6. 如果 pupil 是问题，再 Calibration -> Pupil -> Start pupil calibration
7. 点 Export training package
8. 得到 DreamAirTrackingCapture_*.zip
9. 用户自己上传 Google Drive / OneDrive
10. 在 GitHub issue 里贴链接和问题描述
```

最小可用包：

```text
Gaze 9-point 一次
Eyelid 一次
notes 写清楚失败方向或眼皮问题
```

推荐适配包：

```text
同一人摘戴 2-3 次
每次都做 Gaze 9-point
至少一次 Eyelid
有问题再补 Pupil
```

导出按钮逻辑：

```text
Calibration complete
  -> Export training package 按钮出现
  -> 打包 session.json / labels.jsonl / pairs.csv / metrics.json / frames/
  -> 输出到 %LOCALAPPDATA%/DreamAirTracking/capture_packages/
```

不接受：

```text
单独截图
只有 CSV 行数
BrokenEye 没有 live stream 的采集
没有 accepted labels 的 session
```

## Gaze 校准

### 用户用途

校准当前模型在当前佩戴下的 gaze offset/gain/quick affine。

### 训练用途

生成 gaze hardcase / 9 点训练样本：

```text
gaze_weight = 1
openness_valid = 0
pupil_valid = quality gated
wide_valid = 0
squint_valid = 0
```

### UI 参数

只显示：

```text
Model: Dream Air Main 2026-06
Mode: 5-point / 9-point
Last saved: 2026-06-19 18:45
Offset: x, y
Gain: x, y
Edge check: right_down pass/fail
```

默认普通用户用 5 点；开发者或高级开关才显示 9 点。

### Start calibration 行为

当前 tab 是 `Gaze` 时：

1. 检查 BrokenEye stream；
2. 检查模型 runtime；
3. 跑 5 点或 9 点；
4. 自动质量审核；
5. 保存 `gaze_calibration.json`；
6. 写入 current profile；
7. 将 session 标记为可导出训练包。

## Eyelid 校准

图三当前功能有意义，但 UI 形态不对。

### 当前作用

现在的 Eyelid 页有三类功能混在一起：

- 加载 openness calibration；
- 做 eyelid calibration；
- 捕获 eyelid training；
- live 显示左右眼 openness。

这些功能本身有用，但不该作为三个按钮暴露给普通用户。

### 新设计

Eyelid 和 gaze 一样，也是一键校准 + 自动产出训练样本。

协议：

```text
open
half_open
closed
blink
optional_squint
```

runtime 输出：

```text
eyelid_calibration.json
open baseline
closed floor
curve gamma / s-curve
blink fall/rise timing
```

训练输出：

```text
gaze_weight = 0
openness_valid_left/right = 1
pupil_valid = only when open and quality good
wide/squint_valid = only if explicitly captured
```

### UI 参数

只显示：

```text
Last saved: time
Open: left/right median
Closed: left/right median
Curve: gentle / normal / firm
Blink: detected / not detected
```

删除路径输入框。路径应该由 app 自动管理。

### VRC 表现原则

Eyelid 默认只服务眼皮 openness，不默认抢 expression shape。

闭眼 shape “挤压”问题优先通过 runtime curve 解决：

- open 区更稳定；
- 中段更平滑；
- 尾部更容易闭严；
- 不把 wide/squint 默认送到 expression slot。

## Pupil 校准

图四当前太像调试页，不像用户校准页。

### 当前作用

当前 Pupil 页混了：

- pupil assist；
- raw pupil diameter；
- VRCFT eye shape 开关；
- output mode；
- calibration file；
- live gauge。

这些大多是开发调试项。对普通用户，pupil 只需要知道“当前是否可用”和“是否保存了 calibration”。

### 新设计

Pupil 校准也应接入 workflow：

协议：

```text
open_center
open_left
open_right
open_up
open_down
optional_low_openness
```

runtime 输出：

```text
pupil_calibration.json
left/right diameter baseline
quality gate
min openness for pupil
weak-label confidence
```

训练输出：

```text
gaze_weight = 0
openness_valid = 0
pupil_valid_left/right = quality gated
```

### UI 参数

只显示：

```text
Last saved: time
Mode: model pupil / off
Quality: left/right
Diameter: left/right normalized
Gate: min openness
```

隐藏：

```text
Expression pupil shrink from wide
VRCFT eye shapes: wide / squint
Pupil output mode
Pupil calibration file path
```

这些保留到 developer mode。

## Review / Diagnostics 区域

图五里的 Pipeline Diagnostics 对开发者有用，但对校准用户噪音太大。

### 保留的内容

在校准页只保留一个简短状态：

```text
BrokenEye: live / missing
Runtime: live / stopped
VRCFT: connected / missing
```

当前模块参数另放在右侧参数卡。

### 删除的内容

```text
Details expander
Session: not started
Captured pairs
Valid pairs
FT/v2 pupil
Pupil gate aliases
长段 watcher 文案
```

这些移到 developer diagnostics 页面。

## 底部按钮

图六只保留：

```text
[Start calibration]
```

按钮行为取决于当前 tab：

```text
Gaze   -> start gaze calibration
Eyelid -> start eyelid calibration
Pupil  -> start pupil calibration
```

不再显示：

```text
Accept
Retry
Mark bad
```

原因：

- 当前已经是模型 runtime，不适合让用户逐点手工判定；
- 手工按钮会把训练标签和用户校准混在一起；
- 质量审核应该自动做，不通过就提示重新开始；
- 训练导出由后台 session metadata 决定。

## Sessions 页

图七当前像开发者审计列表，不适合普通用户。

你说得对：普通使用应该是“人校准一次，保存一次当前校准”，而不是看到几十条记录。

### 普通用户改法

导航中删除 `Sessions`，或改成 `Profile`。

页面只显示：

```text
Current calibration

Gaze
  saved time
  model id
  offset/gain

Eyelid
  saved time
  open/closed stats

Pupil
  saved time
  quality

[Export training package]
```

`Export training package` 导出的是最近一次完整 profile 对应的 session package。

### 开发者模式

开发者模式才显示：

```text
Session history
Audit summary
Weak stages
Export package per session
```

这部分不进普通主导航。

## About 页

图八删掉当前长文案。

新 About 只保留：

```text
DreamAirTracking

Developer
[avatar]
Sumirui / UikaMisumi

Headset
[Dream Air v]

Runtime
BrokenEye -> DreamAirTracking -> VRCFT eye slot
Face/expression handled by SR or another module
```

头像：

- 使用用户提供的头像图作为 developer profile asset；
- 不把头像当 app icon；
- 放在 app 本地资源，例如 `Assets/DeveloperAvatar.png`。

头显选择：

现在只有：

```text
Dream Air
```

可以做成 disabled ComboBox 或普通 label。未来如果扩展其它头显，再变成可选。

## 页面合并后的导航

建议主导航：

```text
Home
Calibration
Profile
Diagnostics  (developer mode)
About
```

普通用户默认隐藏：

```text
Diagnostics
Session history
Watcher
Training internals
Legacy model presets
```

## 数据保存模型

每次校准保存两层数据：

### 当前 profile

```text
%LOCALAPPDATA%/DreamAirTracking/calibrations/current/
  gaze_calibration.json
  eyelid_calibration.json
  pupil_calibration.json
  calibration_profile.json
```

runtime 永远读取 current。

### 内部 session

```text
%LOCALAPPDATA%/DreamAirTracking/calibrations/sessions/20260619_184500_gaze/
  manifest.json
  session.json
  labels.jsonl
  pairs.csv
  frames/
```

训练包从 session 导出，但普通用户不需要看历史列表。

## 实现优先级

P0:

- 首页模型下拉框改成 installed model package；
- legacy round 名字只留 developer fallback；
- Calibration 页右侧改为当前模块参数卡；
- 删除 Details；
- 底部只保留 Start calibration；
- About 页改成开发者头像 + Dream Air 头显选择；
- Sessions 从普通导航移除或改成 Profile。

P1:

- 建立 `CalibrationModule` 抽象：`gaze / eyelid / pupil`；
- `Start calibration` 根据当前 module 分发；
- 每个 module 保存 current calibration；
- 每个 module 同时生成训练包需要的 masks；
- `Export training package` 导出 current profile 对应 session。

P2:

- 模型下载管理器；
- sha256 校验；
- model package install/update/remove；
- user registry + repo registry 双层解析；
- developer diagnostics 独立页面。

## 最终用户工作流

```text
1. 打开 app
2. 选择已安装模型
3. Start Eye Tracking
4. Calibration -> Gaze -> Start calibration
5. Calibration -> Eyelid -> Start calibration
6. Calibration -> Pupil -> Start calibration, optional
7. Profile -> Export training package, optional
```

用户看见的是校准和当前状态；开发者拿到的是可训练的数据包。

这和我们之前定下的训练原则一致：app 校准服务 runtime，同时每次有效校准都能变成 head-specific、mask 明确的训练数据，而不是再把所有 head 混在一起训练。
