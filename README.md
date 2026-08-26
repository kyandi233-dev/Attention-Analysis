# Attention-Analysis

> 2026-08-26｜GitHub 仓库已统一命名为 `Attention-Analysis`；长期维护按硬件收口：AMD/DirectML 使用 `amd-DirectML`，NVIDIA/CUDA 使用 `nvidia-cuda`。

> 本 checkout 为 `amd-DirectML`：当前同一硬件主线已经包含 **NIR + Behavior + NIR-Behavior + RGB**。AMD NIR 正式组合仍冻结为 YOLO fixed batch=8 + RITnet fixed batch=16；RGB Face 已冻结为 Py-Feat 2.1.1 scientific core + ONNX Runtime DirectML，并完成第一档工程提速与 sub-031 representative dry-run。

## 当前状态｜2026-08-26

- 原正式 NIR 全量分析已经完成。
- 当前正式 NIR 主链：FocusWave v3.1.3 phase windows → **逐帧 YOLO26n** → ROI → RITnet batch inference → 指标/QC 输出。
- AMD 正式 runtime：`runtime/nir-formal/`；正式结果保存在仓库外 `D:\_AttentionData\Beijing-NIR\amd-directml`。
- AMD 正式组合：YOLO26n 640×640 / FP32 / DirectML / fixed batch=8；RITnet 640×400 / FP32 / DirectML / fixed batch=16。
- RITnet 原正式运行已经执行完整四分类分割，但旧 `eyes.csv` 没有完整落盘 sclera / iris / pupil / visible ocular 等结构信息，因此当前使用 post-hoc full-class extension 补齐这些变量，不重新运行 YOLO、不覆盖旧正式产物。
- `sub-031` 的 full-class extension 已完成验证；当前补充全量从 `sub-032` 继续。
- 当前正式实验版本为 FocusWave v3.1.3，正式阶段包含 `block1`、`block2` 两个 B block。
- 当前正式 Behavior 分析已经按最终 BB 版本建立：`configs/behavior_formal.yaml` → `scripts/sart_formal_analysis.py` → `src/attention_pipeline/behavior_formal/`；默认正式输出已迁至仓库外 `D:\_AttentionData\Beijing-Behavior\formal-v1`。
- `src/attention_pipeline/nir_behavior/` 已提供 NIR ↔ SART trial/probe/phase 对齐、coverage/QC/diagnostics 与 stimulus visual covariates；默认输出位于仓库外 NIR analysis 目录。
- **RGB 已不再只是保留接口**：`src/attention_pipeline/rgb/`、`configs/rgb_analysis.yaml`、RGB Motion/Pose/Face scripts/tests 与相关决策文档已经进入 `amd-DirectML`。
- AMD RGB Face：Py-Feat 2.1.1 Detectorv2 scientific core + ONNX Runtime DirectML；formal cadence 冻结为 timestamp-driven 15 Hz。第一档 direct-AVI + prefetch + RetinaFace B8 → pending multitask B16 优化在 sub-031 3600 帧达到约 **29.15 fps**，sub-031 window-aware primary/eyelid dry-run 达到 3600/3600 primary 与 eye-geometry coverage。
- RGB 正式/测试输出根统一位于仓库外 `D:\_AttentionData\Beijing-RGB`；Git pull/切分支不会覆盖既有仓库外分析结果。
- RGB 仍处于正式化收尾：sub-033 gap stress、blink event / `perclos80_proxy` 最终规则、full-video formal runner 与 completion/resume/QC 尚待冻结。
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

RGB Face / Pose / Motion 使用各自 RGB Conda 环境，具体环境与当前 dry-run/验证命令见 `docs/040-rgb/` 与 `scripts/README.md`；不要为了运行 RGB 修改 NIR 的 `nir-amd` 环境。

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
| 当前 BB Behavior 正式分析 | [`docs/030-behavior/`](docs/030-behavior/) |
| NIR ↔ SART 下游对齐 | `src/attention_pipeline/nir_behavior/` + `scripts/nir_behavior_alignment.py` |
| **当前 RGB Motion / Pose / Face 科学层与方法** | **[`docs/040-rgb/`](docs/040-rgb/)** |
| RGB 当前配置 | [`configs/rgb_analysis.yaml`](configs/rgb_analysis.yaml) |
| RGB 运行/验证脚本 | [`scripts/`](scripts/) |
| 技术选择与路线变更原因 | [`docs/050-decisions/`](docs/050-decisions/) |
| 日期型研究过程 | [`docs/工作记录/`](docs/工作记录/) |
| 仓库长期工作规则 | [`AGENTS.md`](AGENTS.md) |

完整文档导航见 [`docs/README.md`](docs/README.md)。

## 关键资产关系

```text
datasets/ + training/
        ↓
NIR runtime ──────────────┐
        ↓                 │
正式 NIR / full-class     │
        ↓                 │
NIR × Behavior            │
                          │
Behavior formal ──────────┤
                          │
RGB Motion/Pose/Face ─────┘
        ↓
后续 trial / block / probe / multimodal analysis
```

其中：

- `datasets/` 保存冻结训练/标注数据与 provenance；
- `training/` 保存本项目 YOLO 训练过程与结果；
- `runtime/nir-formal/` 保存正式分析所需的冻结 YOLO/RITnet 权重、RITnet 运行源码、配置和执行入口；
- `scripts/sart_formal_analysis.py` 与 `src/attention_pipeline/behavior_formal/` 是当前 FocusWave v3.1.3 BB 行为分析；
- `src/attention_pipeline/nir_behavior/` 是 NIR × Behavior 共享科学层；
- `src/attention_pipeline/rgb/` 与 `configs/rgb_analysis.yaml` 是 RGB Motion/Pose/Face 共享科学层与硬件配置入口；
- `scripts/sart_bbb_v3_0_analysis.py` 与 `src/attention_pipeline/behavior_bbb_v3_0/` 是明确标记的历史 BBB 可执行复现；
- `tests/` 保存仓库级自动化测试；`runtime/nir-formal/tests/` 保存正式 runtime 自包含测试；
- 已淘汰的第三方模型源码、历史候选模型和阶段性 artifacts 通过 `docs/工作记录/`、决策记录、tags 和 Git 历史追溯。

更完整说明见 [`docs/010-overview/013-仓库资产与复现关系.md`](docs/010-overview/013-仓库资产与复现关系.md)。

## 正式原始数据根

正式原始数据分布在两个逻辑数据目录：`正式实验` 与 `Data`。它们位于两块外接存储设备上，而 Windows 可能根据连接顺序把两块盘分配为 `E:` 或 `F:`，因此不能把某个逻辑目录与某个盘符永久绑定。

当前 Behavior、NIR 与 AMD RGB discovery 配置按各模块需求声明候选根；NIR/Behavior 当前常用候选为：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

运行时会忽略不存在的候选根，并在所有有效根中发现被试。若同一被试在多个有效根中出现重复正式数据，程序应拒绝静默选取并报告重复数据。

最终正式 Behavior/NIR 均从 `sub-031` 及以后进入 v3.1.3 两-block 口径。典型正式行为文件为 `sub-XXX_/beh/sub-XXX_Block1_B_beh.csv` 与 `sub-XXX_/beh/sub-XXX_Block2_B_beh.csv`。

## 分支状态

长期硬件主线目标为：

- `amd-DirectML`：AMD/DirectML 的 NIR + Behavior + RGB 完整工作线；
- `nvidia-cuda`：NVIDIA/CUDA 的 NIR + Behavior + RGB 完整工作线。

`rgb-dev` 与 `rgb-nvidia-cuda` 当前仅作为开发期/历史保险分支，待 NVIDIA CUDA Face runner、跨硬件 parity、sub-033 gap stress、blink/PERCLOS proxy 与 full-video formal runner 收口后再决定删除。`amd-DirectML-ritnet512` 已不在远端分支列表。

tracking 时代已冻结为 tag `v0.8-tracking`；正式 NIR 全量完成阶段为 `v0.9-nir-formal`；历史 BBB 行为分析为 `behavior-bbb-v3.0`。日期型工作记录与历史决策不追溯改写。