# NIR Formal GPU 分析包

> 2026-08-24 02:16（Asia/Shanghai）｜当前 NVIDIA/CUDA 正式运行分支为 `nvidia-cuda`；旧 GPU/tracking 开发状态由 Git 历史与 tags 保留，正式实验时间语义以 FocusWave release `v3.1.3` 为准。

> 2026-08-23：该 runtime 已用于正式全量分析。目录由历史名称 `nir-yolo-tracking-ritnet-v1` 重命名为 `nir-formal`；正式模式默认逐帧 YOLO，不使用 tracker。

**新电脑从零配置、CUDA/PyTorch/OpenCV/RITnet 排错、Git 拉取、YAML 选人和正式批处理的完整说明，请优先看：**  
[docs/010-nir/08-22-04-NIR新电脑GPU环境配置与正式批处理运行指南.md](../../docs/010-nir/08-22-04-NIR新电脑GPU环境配置与正式批处理运行指南.md)

这个目录同时保留两条入口：

- `run`：原短视频诊断模式，仍可复现 `none / KCF / CSRT`；
- `formal`：正式分析模式，默认逐帧 YOLO + RITnet batch + phase 时间窗。

## 正式分析范围

`formal` 默认只接受 **sub-031 及以后**。sub-030 及以前属于旧三 Block 结构，目前不进入这套正式分析。

FocusWave v3.1.3 默认分析阶段：

1. `baseline`：真正静息的 180 秒。`baseline_start` 在确认页出现前记录，因此实际窗口按 `baseline_stop - duration → baseline_stop` 计算；
2. `instructions`：`instructions → practice_start`，包括两张实验说明页面，可作为额外放松/基线候选状态独立保存；
3. `practice`：优先使用 `SART_*_Practice_run*.csv` 的 `absolute_onset_time` 定位真实练习 trial，排除 321 倒计时和练习结果页；
4. `block1`：`Block1` 的 `block_start → block_stop`；
5. `block2`：`Block2` 的 `block_start → block_stop`。

初始摄像头/坐姿调整、cover、Block 间强制休息、休息后的 NIR 重新调整、结算页和尾部空录默认不分析。

正式 phase 必须同时依赖：

```text
sub-XXX_/beh/master_timeline.csv
sub-XXX_/nir/sub-XXX_nir_timestamps.csv
```

Practice 精确定位还会读取 `beh/*Practice_run*.csv`。如果 Practice CSV 缺失，会显式记录 timeline fallback，而不是伪装成 trial 精确对齐。

## 正式 GPU 流程

```text
选取 baseline / instructions / practice / block1 / block2 帧
        ↓
每帧 YOLO26n 眼框（GPU，不使用 tracker）
        ↓
按画面 x 坐标排序 frame_left / frame_right
        ↓
扩展原始眼 ROI（不先缩到 320×160）
        ↓
多个 ROI 组成 RITnet batch
        ↓
每个 raw crop 直接一次 resize 到 640×400
        ↓
RITnet 分割
        ↓
mask 映射到稳定的 320×160 分析坐标
        ↓
瞳孔 ellipse / area / confidence / QC
```

因此删除的是旧流程中的 **320×160 → 640×400 二次图像缩放**，不是删除 320×160 的分析坐标标准。新旧 pupil center / diameter 仍以同一 320×160 坐标体系输出。

## Batch 与 FP32 / FP16

默认配置：

```yaml
ritnet:
  batch_size: 16
  precision: fp32
```

`batch_size=16` 表示一次最多把约 16 个眼 ROI 交给 RITnet GPU；它和 `FP16` 名字里的 16 没有关系。

`FP` 是 floating point（浮点数）：

- `FP32`：32-bit 浮点，数值精度更高，作为默认科研基准；
- `FP16`：16-bit 浮点，数值精度较低，但 RTX GPU 通常能更快、占更少显存。本实现使用 CUDA autocast mixed precision，不把整个模型永久强制 `.half()`。

默认不写精度参数就是 FP32：

```powershell
python .\run_pipeline.py formal --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi"
```

测试 FP16：

```powershell
python .\run_pipeline.py formal `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --ritnet-precision fp16
```

测试 batch 32：

```powershell
python .\run_pipeline.py formal `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --ritnet-batch-size 32
```

本 pipeline 不把 `CPU + FP16` 当作正式运行组合。如果显式写：

```powershell
--device cpu --ritnet-precision fp16
```

程序会直接报错，要求使用 CPU+FP32 或 CUDA+FP16。这样避免科研运行中发生静默精度回退。CPU 不是“只能 FP32”，而是这里没有把 CPU half-precision 当作需要验证和支持的优化目标。

## Formal 命令

先检查环境：

```powershell
python .\run_pipeline.py check-env
```

只查看当前正式分析范围内（sub-031+）的视频：

```powershell
python .\run_pipeline.py discover --formal-only
```

正式单被试，默认 `batch=16 + FP32`：

```powershell
python .\run_pipeline.py formal `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi"
```

也可以按被试编号 + 数据根目录指定，不用手写完整视频路径：

```powershell
python .\run_pipeline.py formal --subject sub-033 --root "F:\正式实验"
```

只测试部分 phase：

```powershell
python .\run_pipeline.py formal `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --phases baseline,instructions,block1
```

`formal` 默认 `tracker=none`，即每帧 YOLO。KCF/CSRT 仍保留在 `run` 中用于复现实验：

```powershell
python .\run_pipeline.py run `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --duration-sec 60 --tracker kcf --redetect-interval 10
```

## 多被试批处理：用 YAML 指定谁要跑

`config.yaml` 已包含：

```yaml
batch:
  subjects:
    include: []
    exclude: []
  device: "0"
  continue_on_error: true
  skip_completed: true
  output_root: "outputs/formal"
```

选择规则：

- `include: []`：处理 `F:/正式实验` 和 `E:/Data` 下发现的全部 **sub-031+**；
- `include` 非空：只处理列出的被试；
- `exclude`：无论 include 是否为空，都排除这里列出的被试。

例如只跑 031、033、056：

```yaml
batch:
  subjects:
    include:
      - "sub-031"
      - "sub-033"
      - "sub-056"
    exclude: []
  device: "0"
  continue_on_error: true
  skip_completed: true
  output_root: "outputs/formal"
```

然后先预览，不真正运行：

```powershell
python .\run_formal_batch.py --dry-run
```

确认后正式顺序运行：

```powershell
python .\run_formal_batch.py
```

也可临时覆盖 YAML，不改文件：

```powershell
python .\run_formal_batch.py --subjects sub-031,sub-033,sub-056
```

批处理是**串行**的，同一时间只跑一个被试，避免两个任务争抢同一张 GPU。默认 `skip_completed: true`：如果对应运行目录已有 `summary.json`，会跳过该被试；失败时默认记录错误并继续下一名。批次结果写到 `outputs/formal/batch_run_summary.json`。

## 另一台已配置 GPU 电脑

如果该电脑已经有之前验证过的 `eye-ai` Conda 环境，优先沿用它，不要先重新安装/覆盖 PyTorch 或 OpenCV。拉取当前整理主线后进入 runtime 目录：

```cmd
conda activate D:\conda_envs\eye-ai
cd /d D:\NIR_Analysis\Attention-Analysis
git fetch origin
git switch nvidia-cuda
git pull
cd /d D:\NIR_Analysis\Attention-Analysis\runtime\nir-formal
python run_pipeline.py check-env
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

如果仓库放在其他目录，只需要改 `cd` 路径。数据路径仍来自 `config.yaml`：`F:/正式实验` 和 `E:/Data`。

模型权重随这个 runtime 目录保存在 `models/` 中；正常 Git clone/pull 会一并得到 YOLO 与 RITnet 权重。

## 输出

正式输出保留主要文件：

```text
frames.csv
eyes.csv
summary.json
run_manifest.json
overlays/
phase_windows.json
```

`frames.csv` / `eyes.csv` 包含 `phase`、`phase_segment`、`phase_time_ms` 等字段。`summary.json` 包含每个 phase 的状态统计；`run_manifest.json` 明确保存最终生效的 phase、batch、precision、device、YOLO-every-frame 等参数，避免 YAML 与命令行覆盖后产生复现歧义。

正式模式包含分阶段耗时字段，包括 `decode_ms`、`yolo_ms`、`roi_crop_ms`、`ritnet_attributed_ms`、`overlay_write_ms`。RITnet 是跨帧 batch，因此 `frame_processing_ms` 是批量成本按眼睛分摊后的成本归因；真实性能以整段 `elapsed_sec / processing_fps` 为准。

## Overlay

正式默认：

```yaml
output:
  overlay_stride: 3000
```

30 FPS 下约每 100 秒保存一张，并且每个 phase 的第一帧也会保存一张 QC 图。默认不保存 ROI；只有显式 `--save-rois` 才写出 ROI 文件。

## 当前运行状态与历史准入检查

截至 2026-08-23，这套 runtime 已执行正式全量分析。当前整理工作不把它重新描述为“production candidate”。

08-22 开发阶段曾设定以下准入/回归检查，作为历史技术核对项继续保留：

- batch16 FP32 与旧 scalar FP32 的 pupil / missing 一致性；
- batch16 FP16 与 batch16 FP32 的 pupil center、diameter、missing 差异；
- phase_windows 与 FocusWave v3.1.3 的 timeline / Practice CSV 对齐；
- 一名完整被试的显存稳定性、速度和输出完整性。

这些条目用于保留开发过程和后续复核依据，不再作为“是否允许启动全量分析”的当前状态描述。默认正式精度仍为 FP32。
