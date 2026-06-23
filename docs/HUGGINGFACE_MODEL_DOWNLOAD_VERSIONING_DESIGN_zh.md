# Hugging Face 模型下载与版本控制设计

日期：2026-06-23  
范围：DreamAirTracking App 首页/设置页模型下载、Hugging Face 模型源、模型版本控制、本地切换、下载失败提示

## 目标

当前用户需要手动去 Hugging Face 下载模型 repo zip，再复制到：

```text
%LOCALAPPDATA%\DreamAirTracking\models\
```

下一步要把这个流程放进 App：

1. 首页可以直接安装当前公开模型。
2. 设置页可以管理模型源、检查更新、安装/更新模型。
3. 下载失败必须给用户明确提示，不能静默失败。
4. 下载完成后仍保留模型切换功能。
5. 新版本模型不能覆盖掉旧模型，避免更新失败或新模型退化时用户无法回退。

Hugging Face Hub 本身是 Git-based repository，文件可以按 revision/commit 下载。官方 `huggingface_hub` 的下载也强调本地缓存是 version-aware 的。DreamAirTracking 不需要完整引入 Python `huggingface_hub`，但应采用同样原则：**远端 main/current 可以变化，本地安装必须 pin 到明确 commit/version**。

参考：

- Hugging Face Hub 下载说明：`https://huggingface.co/docs/huggingface_hub/en/guides/download`
- Hugging Face Hub cache/version-aware 说明：`https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download`
- Hugging Face Hub 是 Git-based repo：`https://huggingface.co/docs/hub/en/index`

## 现有模型源

当前公开模型：

```text
repoId: Sumirui/dreamairtracking-dreamair-main-current
repoType: model
url: https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current
archive: https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/archive/main.zip
```

当前本地安装后应有：

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

这套布局短期可以兼容，但新的下载器内部应改成“不可变 package 目录 + 生成 registry”。

## 用户界面设计

### 首页：Model package 区域

首页继续保留模型切换下拉框，不改成单一模型。

无模型时：

```text
Model package: [No model package]
Status: No model package installed

[Install Dream Air model]
```

有模型时：

```text
Model package: [Dream Air Main 2026-06 stable v]
Status: selected / running / update available

[Start Eye Tracking] [Stop Eye Tracking] [Refresh] [Check model update]
```

下载中：

```text
Downloading Dream Air Main...
42%  model.onnx

[Cancel]
```

下载失败：

```text
Model download failed
Could not reach Hugging Face. Your current installed model was not changed.

Details: connection timeout / 403 / checksum mismatch / disk full
[Retry] [Open manual install instructions]
```

下载成功：

```text
Dream Air Main 2026-06 installed

[Use this model] [Keep current model]
```

关键交互原则：

- 如果 runtime 正在运行，不自动切换模型。
- 如果用户点击 `Use this model`，runtime 停止时直接切换；runtime 运行中则提示需要重启 eye tracking。
- 下载失败不能清空当前 registry。
- 下载成功后刷新下拉框，用户可以在所有已安装模型之间切换。

### 设置页：Models 管理

设置页新增 `Models` 区域。

```text
Models

Source
  Hugging Face repo: Sumirui/dreamairtracking-dreamair-main-current
  Channel: stable/current

Installed
  Dream Air Main 2026-06 stable     active
  Dream Air Main 2026-06 test2      installed
  Dream Air Expression Auxiliary    installed

[Check for updates] [Download latest] [Open model folder]
```

每个 installed item 展示：

```text
Display name
Model id
Version
Source commit
Installed time
Outputs: gaze/openness/pupil/expression
Status: active / installed / broken / missing files

[Set default] [Remove] [Open folder]
```

设置页用于管理，首页用于快速安装和切换。

## 本地目录设计

建议从“直接把 HF repo 内容复制到 models 根目录”升级为：

```text
%LOCALAPPDATA%\DreamAirTracking\models\
  model_registry.json
  model_sources.json
  active_model.json
  downloads\
  packages\
    dreamair-main-current@2026.06.23+bad587a\
      package.json
      source.json
      model.onnx
      metadata.json
      runtime_defaults.json
      acceptance.json
      model_card.md
    dreamair-expression-current@2026.06.23+bad587a\
      package.json
      source.json
      model.onnx
      metadata.json
      runtime_defaults.json
      acceptance.json
      model_card.md
```

兼容层：

- `model_registry.json` 仍然放在 `%LOCALAPPDATA%\DreamAirTracking\models\model_registry.json`。
- app 启动时从 `packages\*\package.json` 生成 registry。
- 如果发现旧布局 `dreamair-main-current\model.onnx`，仍然读取，但标记为 `legacyLayout=true`。

## 远端版本控制

不要把 `main` 当成本地版本号。

远端可以有这些概念：

```text
repoId: Sumirui/dreamairtracking-dreamair-main-current
channel: stable / current / experimental
branch: main
commit: Hugging Face Git commit hash
modelVersion: 2026.06.23-main.1
packageVersion: 1
```

下载时流程：

1. 请求 HF repo metadata，解析当前 `main` 对应 commit。
2. 下载该 commit 对应的 zip 或逐文件下载。
3. 从 repo 内读取 `model_registry.json`、`metadata.json`、`acceptance.json`、`sha256.txt`。
4. 生成本地不可变安装目录：

```text
packages\<modelId>@<modelVersion>+<shortCommit>\
```

5. 写入 `source.json`。
6. 校验通过后，最后一步才更新 `model_registry.json`。

`source.json` 示例：

```json
{
  "schema": "dream_air_tracking.model_source.v1",
  "sourceType": "huggingface",
  "repoId": "Sumirui/dreamairtracking-dreamair-main-current",
  "repoType": "model",
  "channel": "stable",
  "requestedRevision": "main",
  "resolvedCommit": "bad587a0000000000000000000000000000000000",
  "downloadedAtUtc": "2026-06-23T08:00:00Z",
  "downloadUrl": "https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/archive/bad587a0000000000000000000000000000000000.zip",
  "verified": true
}
```

`package.json` 示例：

```json
{
  "schema": "dream_air_tracking.model_package.v1",
  "id": "dreamair-main-current",
  "displayName": "Dream Air Main",
  "deviceFamily": "Dream Air",
  "role": "main",
  "modelVersion": "2026.06.23-main.1",
  "packageVersion": 1,
  "architecture": "Siamese MobileNetV3-small",
  "runtime": "predict_live_multitask",
  "outputs": ["gaze_xy", "openness_lr", "pupil_lr", "confidence"],
  "files": {
    "onnx": "model.onnx",
    "metadata": "metadata.json",
    "runtimeDefaults": "runtime_defaults.json",
    "acceptance": "acceptance.json",
    "modelCard": "model_card.md"
  }
}
```

## model_registry 生成规则

`model_registry.json` 是 app runtime 使用的索引，不应该手写覆盖用户状态。

生成规则：

1. 扫描 `packages\*\package.json`。
2. 只加入完整且校验通过的 package。
3. 用 `active_model.json` 决定 default。
4. registry 里写绝对路径或相对 `%LOCALAPPDATA%\DreamAirTracking\models` 的路径。
5. 下载新版本不删除旧版本。

示例：

```json
{
  "schema": "dream_air_tracking.model_registry.v1",
  "models": [
    {
      "id": "dreamair-main-current@2026.06.23-main.1",
      "displayName": "Dream Air Main 2026-06",
      "deviceFamily": "Dream Air",
      "role": "main",
      "architecture": "Siamese MobileNetV3-small",
      "runtime": "predict_live_multitask",
      "onnx": "packages/dreamair-main-current@2026.06.23-main.1+bad587a/model.onnx",
      "metadata": "packages/dreamair-main-current@2026.06.23-main.1+bad587a/metadata.json",
      "runtimeDefaults": "packages/dreamair-main-current@2026.06.23-main.1+bad587a/runtime_defaults.json",
      "acceptance": "packages/dreamair-main-current@2026.06.23-main.1+bad587a/acceptance.json",
      "outputs": ["gaze_xy", "openness_lr", "pupil_lr", "confidence"],
      "default": true
    }
  ]
}
```

## 下载实现设计

### 组件

新增 app service：

```text
src/DreamAirTracking.App/Services/HuggingFaceModelDownloadService.cs
src/DreamAirTracking.Core/Models/ModelPackageInstaller.cs
src/DreamAirTracking.Core/Models/ModelPackageValidator.cs
src/DreamAirTracking.Core/Models/ModelRegistryWriter.cs
```

职责：

| 组件 | 职责 |
|---|---|
| `HuggingFaceModelDownloadService` | 网络请求、进度、取消、错误分类 |
| `ModelPackageInstaller` | 解压到 temp、移动到 packages、原子更新 |
| `ModelPackageValidator` | 检查必需文件、sha256、metadata/schema |
| `ModelRegistryWriter` | 根据 installed packages 重写 registry |

### 下载策略

短期最稳：

```text
GET https://huggingface.co/<repoId>/archive/<revision>.zip
```

其中 `<revision>` 不应永远写 `main`。流程应先解析 commit，再下载 commit zip。

如果解析 commit 暂时没实现，第一版可以：

```text
下载 main.zip
读取 repo 内 metadata/source
安装时记录 requestedRevision=main
```

但 UI 必须标记：

```text
Source revision: main at download time
```

中期改进：

- 使用 HF API 获取 repo metadata。
- 用 commit hash 下载 archive。
- 支持 range/resume。
- 支持只下载需要的文件，而不是整个 zip。

### 原子安装

不能边下边覆盖已安装模型。

流程：

```text
downloads\<guid>.zip
downloads\<guid>.extracting\
downloads\<guid>.validated\
packages\<packageName>.installing\
packages\<packageName>\
model_registry.json.tmp
model_registry.json
```

只有最后 rename registry 时才影响 app 选择列表。

失败清理：

```text
删除 downloads\<guid>*
保留 packages\*
保留旧 model_registry.json
保留 active_model.json
```

## 网络失败与提示

错误必须分类，不要只显示 `Exception.Message`。

| 场景 | 用户提示 | 行为 |
|---|---|---|
| 无网络 / DNS 失败 | `Could not connect to Hugging Face.` | 保留当前模型，显示 Retry |
| 超时 | `Download timed out. Your current model was not changed.` | 保留当前模型，允许 Retry |
| 403/401 | `This model source requires access permission.` | 提示检查 repo/token，当前公开模型不应出现 |
| 404 | `Model source or revision was not found.` | 提示源配置可能过期 |
| 下载中断 | `Download was interrupted.` | 删除 temp，允许 Retry |
| 磁盘不足 | `Not enough disk space to install the model.` | 保留当前模型，提示释放空间 |
| zip 解压失败 | `Downloaded model package is corrupted.` | 删除 temp，允许 Retry |
| sha256 mismatch | `Model checksum mismatch. The package was not installed.` | 删除 temp，不更新 registry |
| 必需文件缺失 | `Model package is missing model.onnx / metadata.json / runtime_defaults.json.` | 不安装 |
| registry 无效 | `Downloaded registry is invalid.` | 不安装 |

首页提示应短，设置页可展开详情。

日志写入：

```text
%LOCALAPPDATA%\DreamAirTracking\model_download.log
%LOCALAPPDATA%\DreamAirTracking\model_download_status.json
```

`model_download_status.json` 示例：

```json
{
  "state": "failed",
  "repoId": "Sumirui/dreamairtracking-dreamair-main-current",
  "revision": "main",
  "stage": "download",
  "errorCode": "network_timeout",
  "message": "Download timed out. Your current model was not changed.",
  "currentModelPreserved": true,
  "timeUtc": "2026-06-23T08:00:00Z"
}
```

## 模型切换规则

模型切换仍以本地 registry 为准。

下拉框显示：

```text
Dream Air Main 2026-06 stable
Dream Air Main 2026-06 experimental
Dream Air Main 2026-06 legacy
Dream Air Expression Auxiliary 2026-06
```

切换行为：

1. 用户选择模型。
2. 保存到 `active_model.json` 或现有 `bridge_launch_options.json`。
3. 如果 runtime 停止，下一次启动使用新模型。
4. 如果 runtime 正在运行，弹出：

```text
Restart eye tracking to use this model?
[Restart now] [Later]
```

不允许：

- 下载完成后自动覆盖运行中的模型。
- 删除当前 active model。
- 远端检查失败时清空模型列表。

## 更新规则

`Check for updates`：

1. 读取本地 active package 的 `source.json`。
2. 查询同 repo/channel 的远端最新 commit 或 `model_feed.json`。
3. 比较：

```text
local modelVersion
local resolvedCommit
remote modelVersion
remote resolvedCommit
```

4. 如果远端不同，显示：

```text
Update available: Dream Air Main 2026-06 stable build 2
[Download update]
```

5. 下载后作为新 package 安装。
6. 用户自己选择是否切换。

版本比较优先级：

```text
modelVersion > packageVersion > resolvedCommit
```

## Hugging Face repo 内推荐新增文件

当前 HF repo 可以继续保留 `model_registry.json`。建议新增：

```text
dreamair_model_feed.json
```

示例：

```json
{
  "schema": "dream_air_tracking.hf_model_feed.v1",
  "deviceFamily": "Dream Air",
  "channel": "stable",
  "updatedAtUtc": "2026-06-23T08:00:00Z",
  "packages": [
    {
      "id": "dreamair-main-current",
      "displayName": "Dream Air Main",
      "modelVersion": "2026.06.23-main.1",
      "role": "main",
      "required": true,
      "path": "dreamair-main-current/package.json"
    },
    {
      "id": "dreamair-expression-current",
      "displayName": "Dream Air Expression Auxiliary",
      "modelVersion": "2026.06.23-expression.1",
      "role": "expression",
      "required": false,
      "path": "dreamair-expression-current/package.json"
    }
  ]
}
```

好处：

- app 不需要猜 repo 里有哪些包。
- 可以区分 main/expression/experimental。
- 可以显示更新说明。

## 安全与隐私

1. App 只从默认公开 repo 下载模型，不上传用户数据。
2. 不在 App 内要求 Hugging Face token。
3. 如未来支持 private repo/token，token 必须走 Windows Credential Manager，不写入明文 json。
4. 校准 capture package 仍由用户手动导出并手动上传，不自动上传。

## 实施计划

### Phase 1：最小可用下载

目标：用户能在首页一键下载公开模型。

实现：

- 首页新增 `Install Dream Air model`。
- 下载 HF `main.zip` 到 temp。
- 解压并复制到 `%LOCALAPPDATA%\DreamAirTracking\models\`。
- 校验 `model_registry.json`、`model.onnx`、`metadata.json`。
- 下载失败显示明确 InfoBar。
- 成功后刷新模型下拉框。

验收：

```text
无模型 -> 点击 Install -> 成功出现模型下拉项
断网 -> 明确失败提示，Start Eye Tracking 仍 disabled
下载中断 -> 不留下半安装 registry
```

### Phase 2：不可变 package 与版本 pin

目标：更新模型不覆盖旧模型。

实现：

- 引入 `packages\<id>@<version>+<commit>\`。
- 写入 `source.json` 和 `package.json`。
- registry 从 packages 生成。
- 支持多个版本切换。

验收：

```text
下载新版后旧版仍在
用户可切回旧版
runtime 运行中不会被自动换模型
```

### Phase 3：检查更新

目标：App 可以告诉用户 HF 上有新版。

实现：

- 设置页新增 `Check for updates`。
- 查询远端 feed 或 repo metadata。
- 比较本地 `source.json` 和远端 revision。
- 显示 update available。

验收：

```text
远端无变化 -> up to date
远端有变化 -> update available
网络失败 -> current model preserved
```

### Phase 4：下载体验完善

目标：下载可取消、可重试、错误可诊断。

实现：

- 进度条。
- Cancel。
- status json。
- log file。
- 错误分类。
- Open manual install instructions。

## 第一版 UI 文案

首页无模型：

```text
No model package installed.
Download the public Dream Air model from Hugging Face to enable eye tracking.
```

下载按钮：

```text
Install Dream Air model
```

下载失败：

```text
Model download failed.
Could not connect to Hugging Face. Your current installed model was not changed.
```

下载成功：

```text
Model installed.
Select it from Model package, then start eye tracking.
```

有更新：

```text
Model update available.
Download installs a new local version. Your current model will stay available.
```

## 不做的事情

第一版不做：

- 不支持用户输入任意 HF repo。
- 不支持 private HF token。
- 不自动上传校准数据。
- 不下载私有 test set。
- 不删除旧模型。
- 不在 runtime 运行中自动切换模型。

## 与现有 workflow 的关系

训练侧输出仍然是 model package：

```text
model.onnx
metadata.json
runtime_defaults.json
acceptance.json
model_card.md
package.json
```

发布侧上传到 Hugging Face：

```text
Sumirui/dreamairtracking-dreamair-main-current
```

App 侧只负责：

```text
download -> validate -> install -> registry -> switch -> runtime
```

训练数据、私有测试集、用户 capture package 不进入 App 自动下载流程。
