# DreamAirTracking 发布包启动问题整改计划

日期：2026-06-23  
范围：Windows zip 发布包、WinUI 启动资源、v0.1.1 错误整改、v0.1.2 修复发布

## 结论

`v0.1.0` 下载包打不开，不是用户运行方式错误，也不是模型文件问题，而是发布包漏掉了 WinUI 编译后的应用 XAML 资源。

本地编译输出目录可以运行，是因为 `bin\x64\Release\...\win-x64\` 里存在：

- `DreamAirTracking.App.pri`
- `App.xbf`
- `MainWindow.xbf`
- `Pages\*.xbf`

但 `v0.1.0` zip 里这些文件数量为：

```text
xbf=0
DreamAirTracking.App.pri=0
```

所以用户解压后双击 `DreamAirTracking.App.exe`，程序在 `MainWindow.InitializeComponent()` 阶段找不到 XAML 资源并崩溃，表面现象就是“没反应”。

## 对 v0.1.1 的错误复盘

`v0.1.1` 已经修到了“发布包带 XAML 资源”这一点，但同时混入了不该做的改动：

1. 改掉了原来的顶层 UI。
   - 原本是 `TitleBar + NavigationView + MicaBackdrop`。
   - v0.1.1 被改成了手写左侧 Button 导航。
   - 这不是启动修复的必要条件，属于错误扩大修改范围。

2. Release 包体积从约 59 MB 增加到约 105 MB。
   - 主要原因是 `PublishTrimmed` 被改成全局 `False`。
   - 关 trimming 可能能避免部分 JSON/reflection 问题，但它不是“打不开”的直接根因。
   - 这类兼容性改动应该单独评估，不能混在启动修复里。

3. 调试过程中的临时降级改动进入了正式提交。
   - 临时移除 `MicaBackdrop`、`TitleBar`、`NavigationView` 可以用于定位。
   - 但定位完成后应该回滚 UI，只保留发布资源修复。

## 正确整改方向

v0.1.2 只做最小必要修复：

1. 恢复 v0.1.0/v0.1.1 之前设计好的顶层 UI。
   - 恢复 `MainWindow.xaml` 到 `deabfe4` 的结构。
   - 恢复 `MainWindow.xaml.cs` 到 `deabfe4` 的导航逻辑。
   - 保留原有 `TitleBar`、`NavigationView`、`MicaBackdrop`、图标逻辑。

2. 只保留发布包资源修复。
   - 在 publish 后复制：
     - `$(OutDir)**\*.xbf`
     - `$(OutDir)$(AssemblyName).pri`
   - 目标目录保持原相对路径，例如 `Pages\CalibrationPage.xbf` 仍在 `Pages\` 下。

3. 还原 Release trimming 策略。
   - 恢复：
     - Debug: `PublishTrimmed=False`
     - Release: `PublishTrimmed=True`
   - 如果后续发现 runtime JSON/reflection 被裁剪，再单独开一个 issue 或 commit 处理。
   - 不允许把“包体积翻倍”作为启动修复的副作用带进 release。

4. 保持 app asset 复制完整。
   - 可以保留 `Assets\**\*.*` 的 publish copy。
   - 这只影响头像/icon 等资源是否漏拷，风险低，且符合当前 app 设计。

## 具体执行步骤

### Step 1：回滚错误 UI 改动

从 `deabfe4` 恢复：

```text
src/DreamAirTracking.App/MainWindow.xaml
src/DreamAirTracking.App/MainWindow.xaml.cs
```

不要改页面内部逻辑，不要改模型加载逻辑，不要改校准逻辑。

### Step 2：保留发布资源补丁

在 `DreamAirTracking.App.csproj` 保留如下目标：

```xml
<Target Name="CopyWinUICompiledXamlToPublish" AfterTargets="Publish" Condition="'$(PublishDir)' != ''">
  <ItemGroup>
    <WinUICompiledXaml Include="$(OutDir)**\*.xbf" />
    <WinUIAppPri Include="$(OutDir)$(AssemblyName).pri" />
  </ItemGroup>
  <Copy
    SourceFiles="@(WinUICompiledXaml)"
    DestinationFiles="@(WinUICompiledXaml->'$(PublishDir)%(RecursiveDir)%(Filename)%(Extension)')"
    SkipUnchangedFiles="true" />
  <Copy
    SourceFiles="@(WinUIAppPri)"
    DestinationFolder="$(PublishDir)"
    SkipUnchangedFiles="true" />
</Target>
```

### Step 3：恢复包体积策略

把 trimming 恢复为：

```xml
<PublishTrimmed Condition="'$(Configuration)' == 'Debug'">False</PublishTrimmed>
<PublishTrimmed Condition="'$(Configuration)' != 'Debug'">True</PublishTrimmed>
```

如果 trimming 后出现运行时 JSON/reflection 错误，再用精确的 trim descriptor、source generator 或代码改造处理，而不是直接关闭整个 Release trimming。

### Step 4：重新发布 v0.1.2

发布目录：

```text
artifacts\publish\DreamAirTracking-v0.1.2-win-x64
```

压缩包：

```text
artifacts\dist\DreamAirTracking-v0.1.2-win-x64.zip
artifacts\dist\DreamAirTracking-v0.1.2-win-x64.zip.sha256
```

## 验收标准

v0.1.2 必须同时满足：

1. zip 内存在应用 XAML 资源。

```text
DreamAirTracking.App.pri exists
App.xbf exists
MainWindow.xbf exists
Pages\HomePage.xbf exists
Pages\CalibrationPage.xbf exists
```

2. 解压后的 exe 可以启动。

检查项：

```text
AliveAfter4s=True
MainWindowTitle=DreamAirTracking
CrashLogChanged=False
```

3. 顶层 UI 和原设计一致。

必须保留：

```text
TitleBar
NavigationView
MicaBackdrop
Dashboard / Calibration / Diagnostics / About / Settings navigation
```

4. zip 体积不能异常翻倍。

目标：

```text
接近 v0.1.0 的 59 MB
允许因补入 .xbf/.pri 和 assets 略微增加
不接受无理由增加到 100 MB 级别
```

5. v0.1.1 release 标记为问题版本。

GitHub release 文案需要明确：

```text
v0.1.1 was a startup investigation build and changed shell UI unintentionally.
Use v0.1.2 or newer.
```

## 不允许再犯的流程问题

1. 定位实验和正式修复必须拆开。
2. UI 降级只能作为临时实验，不能跟随发布。
3. 启动修复只解决启动根因，不顺手改设计、不顺手改包体积策略。
4. 每个 release 前必须检查 zip 内容，而不是只看 `dotnet publish` 成功。
5. release 前必须从 zip 解压目录启动测试，不能只测 `bin` 或 IDE 运行。

## 当前建议

下一步先按本文件执行 v0.1.2：

1. 回滚 `MainWindow.xaml` 和 `MainWindow.xaml.cs` 到原 UI。
2. 保留 `.xbf/.pri` publish 复制。
3. 恢复 Release trimming。
4. 重新 build、从 zip 解压目录启动验证。
5. 发布 v0.1.2，并在 v0.1.1 release 上写明弃用。
