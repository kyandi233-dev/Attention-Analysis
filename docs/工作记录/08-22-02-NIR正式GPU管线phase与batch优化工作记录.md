# 08-22-02｜NIR正式GPU管线 phase 与 batch 优化工作记录

> 2026-08-22（Asia/Shanghai）｜开发分支 `codex/nir-formal-gpu-v3`。本轮只修改正式 NIR production candidate，不直接合并现有默认分支；运行验证通过后再决定合并目标。

## 用户批准的正式范围

- 真正执行的 pipeline：`runtime/nir-yolo-tracking-ritnet-v1/`。
- 正式实验语义以 **FocusWave release v3.1.3** 为准，不以当前 `stable-msmf` 分支代替。
- 当前正式分析从 **sub-031** 开始；sub-030 及以前为旧三 Block 结构，暂不进入本轮正式分析。
- 默认正式阶段：`baseline`、`instructions`、`practice`、`block1`、`block2`。
- `baseline` 保留真正静息 180 秒；`instructions` 也保留为独立可分析阶段，后续可作为放松/基线候选状态，但不与 180 秒静息段混成同一标签。
- 初始设备/坐姿调整、cover、Block 间休息、休息后的 NIR 重新调整、结算页和尾部空录默认排除。
- 正式模式默认逐帧 YOLO，不使用 KCF/CSRT；tracking 代码保留在诊断 `run` 模式以便复现历史 benchmark。
- RITnet 默认 `batch_size=16`、`precision=fp32`；FP16 仅作为 CUDA mixed-precision 候选，需与 FP32 做同段一致性比较后才能考虑冻结。
- overlay 默认每 3000 帧一张；ROI 默认不落盘。
- 本轮不更新 `nir-yolo-tracking-ritnet-v1.zip` 或其 SHA256。

## FocusWave v3.1.3 时间语义

正式程序在 `baseline_start` 后仍先显示“请确认是否调整好坐姿 / 即将开始静坐测试”，被试按空格后才进入 180 秒真正静息。因此 `baseline_start → baseline_stop` 会比 180 秒长；正式 NIR 的 `baseline` 窗口按：

```text
baseline_stop - duration → baseline_stop
```

其中 duration 优先读取 `baseline_stop.detail` 的 `duration=...s`，默认 180 秒。

`instructions` marker 在两张实验说明图片出现前记录，`practice_start` 在练习 321 倒计时前记录，因此：

```text
instructions = instructions → practice_start
```

可作为独立分析阶段。

Practice 不直接用 `practice_start → practice_end`，因为该范围包含 321 倒计时和练习结束页。优先读取 `beh/*Practice_run*.csv` 的 `absolute_onset_time`，取真实 trial 范围；Practice CSV 缺失时才显式 fallback 到 timeline，并在 phase source 中标记 fallback。

正式 Block 的 `block_start` 位于 321 倒计时之后、`run_single_block` 之前，所以：

```text
block1 = Block1 block_start → block_stop
block2 = Block2 block_start → block_stop
```

## FP32 / FP16 口径

FP = floating point（浮点数表示）。

- FP32：32-bit 浮点，数值精度更高，当前科研基准与默认值；
- FP16：16-bit 浮点，数值精度较低，但 RTX GPU 的吞吐和显存效率通常更好；本实现采用 CUDA autocast mixed precision，而不是把整个模型永久 `.half()`。

`batch_size=16` 与 `FP16` 的 16 无关：前者是一次送入 GPU 的 ROI 数量，后者是浮点表示位宽。

本 pipeline 不支持 `CPU + FP16` 作为正式组合。若用户显式选择 `--device cpu --ritnet-precision fp16`，程序应明确报错，而不是静默回退 FP32。目的不是宣称 CPU 不能表示 FP16，而是避免科研运行中“请求参数”和“实际参数”不一致。

## 本轮文件修改

### `runtime/nir-yolo-tracking-ritnet-v1/config.yaml`

- `tracking.method: none`；
- 新增 `ritnet.batch_size: 16`；
- 新增 `ritnet.precision: fp32`；
- 新增 `formal.focuswave_release: v3.1.3`；
- 新增 `formal.min_subject_number: 31`；
- 正式 phases 固定为 baseline / instructions / practice / block1 / block2；
- baseline 默认 180 秒；Practice trial 默认 1150 ms；
- `output.overlay_stride: 3000`。

### 新增 `runtime/nir-yolo-tracking-ritnet-v1/phase_windows.py`

- 读取 `beh/master_timeline.csv`；
- 将 baseline/instructions/practice/block1/block2 映射到 NIR `unix_ms → frame_idx`；
- baseline 去掉确认页；
- Practice 优先使用 Practice CSV trial onset；
- 对时间窗重叠、frame-index gap、缺失 block marker 做 fail-fast；
- 输出可审计的 phase source。

### `runtime/nir-yolo-tracking-ritnet-v1/ritnet_runtime.py`

- 新增 `infer_batch()`；
- raw eye crop 直接一次 resize 到 640×400；
- 多 ROI 一次 GPU forward；
- batched GPU→CPU transfer；
- 支持 `fp32 / fp16`，FP16 使用 CUDA autocast；
- `CPU + FP16` 明确拒绝；
- segmentation mask 最终仍映射到 320×160 分析坐标后拟合瞳孔椭圆，保持既有 pupil 坐标尺度。

### `runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py`

保留旧 `run` 诊断入口，同时新增 `formal`：

- sub-031 前直接拒绝进入当前正式模式；
- 每帧 YOLO；
- phase-aware 非连续视频读取，只解码需要分析的阶段；
- raw ROI 排队组成 RITnet batch；
- `--ritnet-batch-size` / `--ritnet-precision` 可覆盖 YAML；
- 新增 `phase`、`phase_segment`、`phase_time_ms`；
- 新增 `phase_windows.json`；
- manifest 保存 `effective_parameters`，明确最终生效的 mode/tracker/batch/precision/device/phases；
- 增加 decode / YOLO / ROI / RITnet attributed / overlay 分项耗时；
- formal summary 增加 phase 级状态统计。

### `runtime/nir-yolo-tracking-ritnet-v1/tests/test_phase_windows.py`

新增 synthetic 时间轴测试：

- baseline_start 确认页不会进入真正 baseline；
- instructions 正确落在 instructions → practice_start；
- practice 用真实 trial onset，排除倒计时和结果页；
- Block1/2 marker 映射；
- 请求不存在的 block 时失败。

### `README.md` / `run_examples.ps1`

同步正式命令、phase、FP32/FP16、batch、sub-031 边界和输出语义。

## 最终预期运行结果

默认正式命令：

```powershell
python .\run_pipeline.py formal --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi"
```

等价核心参数：

```text
FocusWave release = v3.1.3
subject >= 31
phases = baseline,instructions,practice,block1,block2
YOLO = every frame
tracker = none
RITnet batch = 16
RITnet precision = FP32
overlay_stride = 3000
```

输出继续保留：

```text
frames.csv
eyes.csv
summary.json
run_manifest.json
overlays/
```

新增：

```text
phase_windows.json
```

## 尚未完成 / 停止点

当前代码已写入开发分支，但还没有在目标 RTX 4060 Laptop 上运行真实 NIR benchmark，因此**不得合并生产分支、不得全量跑所有被试**。

下一停止门：

1. 在目标 Anaconda 环境执行 `check-env` 与 phase 解析冒烟；
2. 同一段真实 NIR 比较旧 scalar FP32 与新 batch16 FP32；
3. 同一段比较 batch16 FP32 与 batch16 FP16；
4. 检查 pupil center / diameter / found/missing 与 phase frame window；
5. 完整跑 1 名 sub-031+ 被试，确认显存、速度、CSV/manifest/overlay 完整；
6. 上述通过后再设计正式 E/F 批量入口与恢复/跳过已完成机制，并决定是否合并。
