# Attention-Analysis

> 2026-08-26｜GitHub 仓库已统一命名为 `Attention-Analysis`；当前默认维护分支为 `nvidia-cuda`，AMD/DirectML 路线使用 `amd-DirectML`。

> 本 checkout 为 `amd-DirectML` package `0.2.0`：YOLO26n 与 RITnet 已改用 ONNX Runtime DirectML，正式组合冻结为 YOLO fixed batch=8 + RITnet fixed batch=16；当前正在进行 RITnet full-class 遗漏信息补充分析。

## 当前状态｜2026-08-26

- 原正式 NIR 全量分析已经完成。
- 当前正式 NIR 主链：FocusWave v3.1.3 phase windows → **逐帧 YOLO26n** → ROI → RITnet batch inference → 指标/QC 输出。
- AMD 正式 runtime：`runtime/nir-formal/`；正式结果保存在仓库外 `D:\_AttentionData\Beijing-NIR\amd-directml`。
- AMD 正式组合：YOLO26n 640×640 / FP32 / DirectML / fixed batch=8；RITnet 640×400 / FP32 / DirectML / fixed batch=16。
- RITnet 原正式运行已经执行完整四分类分割，但旧 `eyes.csv` 没有完整落盘 sclera / iris / pupil / visible ocular 等结构信息，因此当前使用 post-hoc full-class extension 补齐这些变量，不重新运行 YOLO、不覆盖旧正式产物。
- `sub-031` 的 full-class extension 已完成验证；当前补充全量从 `sub-032` 继续。
- 当前正式实验版本为 FocusWave v3.1.3，正式阶段包含 `block1`、`block2` 两个 B block。
- 当前正式 Behavior 分析已经按最终 BB 版本建立：`configs/behavior_formal.yaml` → `scripts/sart_formal_analysis.py` → `src/attention_pipeline/behavior_formal/`。
- 旧 v3.0 BBB 行为分析仍保留独立历史可执行入口，不作为当前正式口径。

## AMD 每次打开新终端：直接从这里开始

当前 AMD 工作副本：

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML
```

每次重新打开 PowerShell / VS Code Terminal 后执行：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
conda activate D:\CondaEnvs\nir-amd

git switch amd-DirectML
git pull --ff-only

cd runtime\nir-formal
python run_pipeline.py check-env
```

`git pull --ff-only` 如果提示本地存在未提交修改，不要强制覆盖；先执行 `git status --short --branch` 检查。

## AMD RITnet full-class 补充全量｜当前从 sub-032 继续

补充分析入口：

```text
runtime/nir-formal/run_ritnet_fullclass_batch.py
```

它根据原正式结果保存的 `frame_idx` 和 ROI 坐标重新裁剪相同眼 ROI，只重跑冻结的 **RITnet 640×400 / FP32 / fixed batch=16 / DirectML**。不会重新运行 YOLO，也不会修改原 `eyes.csv`、`frames.csv` 等历史正式产物。

### 1. 自动生成 sub-032 及以后实际存在的被试列表

在 `runtime\nir-formal` 下执行：

```powershell
$subjects = Get-ChildItem "D:\_AttentionData\Beijing-NIR\amd-directml" -Directory |
  ForEach-Object {
    if ($_.Name -match '^(sub-(\d{3}))_formal_') {
      [PSCustomObject]@{
        Subject = $matches[1]
        Number  = [int]$matches[2]
      }
    }
  } |
  Where-Object { $_.Number -ge 32 } |
  Sort-Object Number |
  Select-Object -ExpandProperty Subject -Unique

$subjectArg = $subjects -join ","
$subjectArg
```

最后一行会先打印本次选中的被试，例如：

```text
sub-032,sub-033,sub-034,...
```

### 2. 先 dry-run 检查选择是否正确

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "$subjectArg" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

### 3. 正式从 sub-032 开始继续全量

确认 dry-run 无误后：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "$subjectArg" `
  --device 0 `
  --postprocess-workers 4
```

补充分析结束后的批处理总表：

```text
D:\_AttentionData\Beijing-NIR\amd-directml\ritnet_fullclass_batch_summary.json
```

更完整的变量、输出、QC 图片和 completion identity 说明见：

- [`runtime/nir-formal/README.md`](runtime/nir-formal/README.md)
- [`runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md`](runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md)

## 快速入口

| 我想找 | 入口 |
|---|---|
| AMD 新终端启动 + 当前 full-class 补充全量 | 本 README 上方操作段 |
| 正式 NIR 运行包 | [`runtime/nir-formal/`](runtime/nir-formal/) |
| RITnet full-class 补充指标方法与 QC 口径 | [`runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md`](runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md) |
| 项目整体结构与数据流 | [`docs/010-overview/`](docs/010-overview/) |
| NIR 方法、当前入口与历史路线 | [`docs/020-nir/`](docs/020-nir/) |
| 当前 BB 行为分析与历史 BBB | [`docs/030-behavior/`](docs/030-behavior/) |
| RGB 保留接口 | [`docs/040-rgb/`](docs/040-rgb/) |
| 技术选择与路线变更原因 | [`docs/050-decisions/`](docs/050-decisions/) |
| 日期型研究过程 | [`docs/工作记录/`](docs/工作记录/) |
| 仓库长期工作规则 | [`AGENTS.md`](AGENTS.md) |

完整文档导航见 [`docs/README.md`](docs/README.md)。

## 关键资产关系

```text
datasets/
    ↓
training/
    ↓
trained weights
    ↓
runtime/nir-formal/
    ↓
正式 NIR 全量分析
    ↓
RITnet full-class post-hoc extension
```

其中：

- `datasets/` 保存冻结训练/标注数据与 provenance；
- `training/` 保存本项目 YOLO 训练过程与结果；
- `runtime/nir-formal/` 保存正式分析所需的冻结 YOLO/RITnet 权重、RITnet 运行源码、配置和执行入口；
- `scripts/sart_formal_analysis.py` 与 `src/attention_pipeline/behavior_formal/` 是当前 FocusWave v3.1.3 BB 行为分析；
- `scripts/sart_bbb_v3_0_analysis.py` 与 `src/attention_pipeline/behavior_bbb_v3_0/` 是明确标记的历史 BBB 可执行复现；
- `tests/` 保存仓库级自动化测试；`runtime/nir-formal/tests/` 保存正式 runtime 自包含测试；
- 已淘汰的第三方模型源码、历史候选模型和阶段性 artifacts 通过 `docs/工作记录/`、决策记录、tags 和 Git 历史追溯。

更完整说明见 [`docs/010-overview/013-仓库资产与复现关系.md`](docs/010-overview/013-仓库资产与复现关系.md)。

## 正式原始数据根

正式原始数据分布在两个逻辑数据目录：`正式实验` 与 `Data`。它们位于两块外接存储设备上，而 Windows 可能根据连接顺序把两块盘分配为 `E:` 或 `F:`，因此不能把某个逻辑目录与某个盘符永久绑定。

当前 Behavior 与 NIR 配置统一声明四个候选路径：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

运行时会忽略不存在的候选根，并在所有有效根中发现被试。若同一被试在多个有效根中出现重复正式数据，程序应拒绝静默选取并报告重复数据。

最终正式 Behavior/NIR 均从 `sub-031` 及以后进入 v3.1.3 两-block 口径。典型正式行为文件为 `sub-XXX_/beh/sub-XXX_Block1_B_beh.csv` 与 `sub-XXX_/beh/sub-XXX_Block2_B_beh.csv`。

## 分支状态

`nvidia-cuda` 是 GitHub default，保存已完成正式全量分析的 NVIDIA/CUDA 基线。`amd-DirectML` 已从该冻结基线开始实际改造，当前 package 为 `0.2.0`。

tracking 时代已冻结为 tag `v0.8-tracking`；正式 NIR 全量完成阶段为 `v0.9-nir-formal`；历史 BBB 行为分析为 `behavior-bbb-v3.0`。旧开发分支已删除，新开发不再从 tracking 路线继续。

## 历史与 provenance

日期型工作记录和历史研究文档不追溯改写。旧文档中的“候选 / 待准入 / 准备全量 / YOLO + tracking + RITnet / BBB”等表述代表当时阶段，不代表当前正式流程。

原根目录《项目总览与架构》的 2026-08-23 快照完整保留在：

[`docs/010-overview/014-2026-08-23项目总览与架构历史快照.md`](docs/010-overview/014-2026-08-23项目总览与架构历史快照.md)
