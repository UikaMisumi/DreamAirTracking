# Hugging Face 模型目录与下载界面整改设计

日期：2026-06-23  
背景：当前首页直接放 `Install Model` 并立刻下载，体验和工程边界都不对。实测 `https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/archive/main.zip` 返回 404，而 HF API 和单文件 `resolve` 可用。因此需要把“查询模型版本”和“下载指定模型”拆开。

## 结论

模型下载不应该是首页 runtime command bar 上的一个按钮。

正确结构应该是：

```text
Home / Eye Tracking
  只负责选择“已经下载好的模型”
  只负责 Start / Stop / Refresh runtime

Models / Model Library
  负责从 Hugging Face 拉取可用模型版本
  展示模型名字、版本、commit、更新时间、文件状态
  用户确认后才下载指定版本
  下载成功后写入本地 installed model registry
```

换句话说：

```text
远端 Hugging Face 模型目录
  -> 用户选择一个版本下载
  -> 本地 installed model packages
  -> 首页 Model package 下拉框选择本地模型
```

首页不直接“盲下最新模型”。

## 当前问题复盘

### 1. 首页按钮职责不对

现在 `Install Model` 和 `Start Eye Tracking` 放在同一个 CommandBar。用户看到的是 runtime 操作，但点击后实际发生的是网络下载、解压、安装模型。

这会造成几个问题：

- 用户不知道会下载哪个模型。
- 用户不知道模型版本。
- 用户不知道是否有多个版本可选。
- 下载失败后错误直接挤在 runtime 页面里。
- 下载和模型切换逻辑混在一起。

### 2. 下载 URL 不对

实测：

```text
GET https://huggingface.co/api/models/Sumirui/dreamairtracking-dreamair-main-current
-> 200

GET https://huggingface.co/api/models/Sumirui/dreamairtracking-dreamair-main-current/refs
-> 200

GET https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/resolve/main/model_registry.json
-> 200

GET https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/archive/main.zip
-> 404
```

所以不能再依赖 `archive/main.zip`。第一版应改成：

```text
1. 用 HF API 查 repo 信息和 refs。
2. 用 /resolve/<revision>/<file> 按文件下载。
3. 文件清单来自 model_registry.json 或 dreamair_model_feed.json。
```

### 3. 缺少“远端 catalog”

用户需要先看到：

```text
Dream Air Main
Version: 2026.06.19
Revision: 8afaa8c00e137bb1aac1ba727930be675bbbfba9
Updated: 2026-06-19
Role: main
Outputs: gaze/openness/pupil/confidence
Status: not installed / installed / update available
```

然后再点下载。

## 新 UI 设计

### 导航

新增一个顶层页面：

```text
Dashboard
Calibration
Diagnostics
Models
About
Settings
```

或者短期放在 Settings 中的独立大区，但建议最终做 `Models` 独立页。

理由：

- 下载模型是一个完整 workflow，不是 runtime 的附属按钮。
- 未来要支持多个模型、版本、删除、校验、更新，Settings 会变得太挤。
- 首页应该保持“运行”语义干净。

## 首页最终形态

首页只显示本地已安装模型。

```text
Eye Tracking

Model package  [Dream Air Main 2026-06 stable v]
Status         selected / running / no model package installed

[Start Eye Tracking] [Stop Eye Tracking] [Refresh]
```

无模型时：

```text
No model package installed.
Go to Models to download a Dream Air model.

[Open Models]
```

首页不再有：

```text
[Install Model]
[Cancel Download]
下载进度条
下载失败大红条
```

首页只保留一个轻入口 `Open Models`，而不是直接下载。

## Models 页面设计

### 顶部：远端源

```text
Models

Source: Hugging Face
Repo:   Sumirui/dreamairtracking-dreamair-main-current

[Refresh remote list]
```

刷新远端列表后显示：

```text
Remote models

Dream Air Main
  Version: 2026.06.19
  Revision: 8afaa8c
  Updated: 2026-06-19 12:13 UTC
  Role: main
  Status: Not installed
  [Download]

Dream Air Expression Auxiliary
  Version: 2026.06.19
  Revision: 8afaa8c
  Role: expression
  Status: Not installed
  [Download]
```

### 中部：下载进度

下载时：

```text
Downloading Dream Air Main 2026.06.19
metadata.json
38%
[Cancel]
```

失败时：

```text
Download failed
Could not download model.onnx from Hugging Face.
Current installed models were not changed.

Details: 404 / timeout / network error / checksum mismatch
[Retry] [Copy details]
```

成功时：

```text
Installed Dream Air Main 2026.06.19
It is now available in Home -> Model package.

[Set as default] [Open Home]
```

### 底部：本地已安装模型

```text
Installed models

Dream Air Main 2026-06 stable
  Revision: 8afaa8c
  Path: %LOCALAPPDATA%\DreamAirTracking\models\packages\...
  Status: default
  [Set default] [Open folder]

Dream Air Main 2026-06 previous
  Revision: ...
  Status: installed
  [Set default] [Open folder]
```

第一版可以暂时不做删除按钮，避免误删用户还能用的模型。

## Hugging Face 查询流程

### 1. 获取 repo metadata

```text
GET https://huggingface.co/api/models/Sumirui/dreamairtracking-dreamair-main-current
```

需要读取：

```text
id
sha
lastModified
siblings[].rfilename
siblings[].size
```

`sha` 用作当前远端 revision。

### 2. 获取 refs

```text
GET https://huggingface.co/api/models/Sumirui/dreamairtracking-dreamair-main-current/refs
```

用途：

- 展示 branches/tags。
- 后续支持 stable/beta/experimental。

第一版可以只用 `main` 和 metadata 里的 `sha`。

### 3. 下载 manifest

优先读取：

```text
https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/resolve/<sha>/dreamair_model_feed.json
```

如果没有 feed，fallback：

```text
https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/resolve/<sha>/model_registry.json
```

`dreamair_model_feed.json` 是推荐新增的远端目录文件，用于展示模型名字和版本。

示例：

```json
{
  "schema": "dream_air_tracking.hf_model_feed.v1",
  "repoId": "Sumirui/dreamairtracking-dreamair-main-current",
  "channel": "stable",
  "revision": "8afaa8c00e137bb1aac1ba727930be675bbbfba9",
  "packages": [
    {
      "id": "dreamair-main-current",
      "displayName": "Dream Air Main",
      "modelVersion": "2026.06.19",
      "role": "main",
      "files": [
        "dreamair-main-current/model.onnx",
        "dreamair-main-current/metadata.json",
        "dreamair-main-current/runtime_defaults.json",
        "dreamair-main-current/acceptance.json",
        "dreamair-main-current/model_card.md"
      ]
    }
  ]
}
```

### 4. 按文件下载

不要下载 zip。逐文件下载：

```text
GET https://huggingface.co/<repoId>/resolve/<sha>/<filePath>
```

例如：

```text
https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/resolve/8afaa8c00e137bb1aac1ba727930be675bbbfba9/model_registry.json
https://huggingface.co/Sumirui/dreamairtracking-dreamair-main-current/resolve/8afaa8c00e137bb1aac1ba727930be675bbbfba9/dreamair-main-current/model.onnx
```

好处：

- 可展示当前下载到哪个文件。
- 可准确知道缺哪个文件。
- 不依赖 zip archive endpoint。
- 后续可做断点续传/校验/只更新变化文件。

## 本地安装流程

### 第一版目录

先保持兼容现有 registry：

```text
%LOCALAPPDATA%\DreamAirTracking\models\
  model_registry.json
  dreamair-main-current\
    model.onnx
    metadata.json
    runtime_defaults.json
    acceptance.json
    model_card.md
```

但下载过程必须用 staging：

```text
%LOCALAPPDATA%\DreamAirTracking\models\.downloads\<operationId>\
  remote.json
  model_registry.json
  dreamair-main-current\
    ...
```

只有全部文件下载和校验通过后，才复制到正式目录。

### 第二版目录

后续升级成不可变版本目录：

```text
%LOCALAPPDATA%\DreamAirTracking\models\
  model_registry.json
  packages\
    dreamair-main-current@2026.06.19+8afaa8c\
      package.json
      source.json
      model.onnx
      metadata.json
      runtime_defaults.json
      acceptance.json
      model_card.md
```

## 模型选择规则

首页下拉框只读本地 registry：

```text
%LOCALAPPDATA%\DreamAirTracking\models\model_registry.json
```

下载页完成安装后：

1. 重写本地 `model_registry.json`。
2. 通知/刷新首页下拉框。
3. 不自动 start runtime。
4. 如果 runtime 正在运行，不自动换模型。

用户必须在首页显式选择模型。

## 错误提示设计

### 远端列表失败

```text
Could not load model list from Hugging Face.
Check your network connection or try again later.
```

不影响本地已安装模型。

### manifest 缺失

```text
The Hugging Face repo does not contain a model feed or model_registry.json.
No model was downloaded.
```

### 单文件 404

```text
Could not download dreamair-main-current/model.onnx.
The selected model version is incomplete on Hugging Face.
```

### 网络中断

```text
Download interrupted.
Current installed models were not changed.
```

### 校验失败

```text
Downloaded package failed validation.
Missing runtime_defaults.json.
Current installed models were not changed.
```

## 代码整改计划

### Step 1：回滚首页直接下载按钮

从 HomePage 移除：

```text
Install Model
Cancel Download
ModelDownloadInfoBar
```

改成无模型时显示：

```text
[Open Models]
```

### Step 2：新增 ModelsPage

新增：

```text
src/DreamAirTracking.App/Pages/ModelsPage.xaml
src/DreamAirTracking.App/Pages/ModelsPage.xaml.cs
```

导航栏新增：

```text
Models
```

### Step 3：重写下载服务职责

当前 `HuggingFaceModelDownloadService.InstallDefaultModelAsync()` 改掉。

新的接口：

```csharp
Task<RemoteModelCatalog> RefreshCatalogAsync(CancellationToken ct);
Task<ModelDownloadResult> DownloadPackageAsync(RemoteModelPackage package, IProgress<ModelDownloadProgress> progress, CancellationToken ct);
IReadOnlyList<InstalledModelPackage> LoadInstalledPackages();
```

不要再有“默认直接下载 latest”的 API。

### Step 4：新增远端数据结构

```text
RemoteModelCatalog
RemoteModelPackage
RemoteModelFile
InstalledModelPackage
```

### Step 5：逐文件下载

替换：

```text
/archive/main.zip
```

为：

```text
/api/models/<repoId>
/api/models/<repoId>/refs
/resolve/<sha>/<file>
```

### Step 6：发布前验证

必须验证：

```text
打开 Models 页面
Refresh remote list 成功
能看到 Dream Air Main / version / commit
点击 Download
文件落地到 %LOCALAPPDATA%
Home 页面下拉框出现下载后的模型
Start Eye Tracking 不因下载失败被误启用
断网/错误源时本地 registry 不变
```

## 第一版验收标准

第一版完成后，用户流程必须是：

```text
1. 打开 app
2. 进入 Models
3. 点击 Refresh remote list
4. 看到 Hugging Face 上的模型名字和版本
5. 点击某个版本的 Download
6. 下载完成
7. 回到 Home
8. 在 Model package 下拉框选择刚下载好的模型
9. Start Eye Tracking
```

不能再是：

```text
Home 上直接 Install Model
```

也不能再是：

```text
点击后 app 自己猜 latest 并下载
```

## 当前代码需要修正的点

当前已经做过的 `Install Model` 第一版应视为临时实现，需要整改：

```text
src/DreamAirTracking.App/Pages/HomePage.xaml
src/DreamAirTracking.App/Pages/HomePage.xaml.cs
src/DreamAirTracking.App/Pages/SettingsPage.xaml
src/DreamAirTracking.App/Pages/SettingsPage.xaml.cs
src/DreamAirTracking.App/Services/HuggingFaceModelDownloadService.cs
```

保留有价值部分：

- 进度回调类型 `ModelDownloadProgress`
- 下载失败不破坏旧模型的原则
- temp/staging 后再安装的原则

删除或替换：

- 首页直接 `Install Model`
- `InstallDefaultModelAsync`
- `/archive/main.zip`

## 参考

- Hugging Face Hub repo 是 Git-based repository。
- Hugging Face Hub API 可提供 repo metadata、refs、siblings。
- 单文件下载应使用 `/resolve/<revision>/<path>`。

本地实测结论：

```text
api/models -> 200
api/models/<repo>/refs -> 200
resolve/main/model_registry.json -> 200
archive/main.zip -> 404
```
