# 重建任务窗恢复流程

本流程只用于原始行为 `master_timeline.csv` 缺失、但行为 Block CSV 和模态绝对时间戳仍可审计的场次。它不是正式完整时间线的替代品，结果状态使用 `recovery_complete`，不会被正式 batch 的 `complete` 跳过逻辑接收。

## 时间对齐原则

恢复边界来自外部重建时间线中的 `reconstructed_block_start` / `reconstructed_block_stop`，单位为绝对 Unix 毫秒。NIR 和毫米波分别读取自己的时间戳文件，再独立映射到各自的帧号：

- NIR：`*_nir_timestamps.csv` 的帧号和 Unix 毫秒；
- 毫米波：`*_mmwave_timestamps.csv` 的帧号和 Python/Unix 毫秒列；
- RGB 只用于审计有效时段，不能把 RGB 帧号当作 NIR 或毫米波帧号。

原始视频、原始时间戳和毫米波 NPZ 均不裁切、不覆盖。长 NIR 录制只通过时间窗口选择任务帧，避免把异常收尾录制混入任务分析。

## 生成 NIR/mmWave 窗口清单

```powershell
python tools/recover_reconstructed_task_windows.py `
  --subject sub-099 `
  --timeline D:/Project/厚粲杯/11_数据/derived/sub099_timeline_reconstruction_v1/sub099_reconstructed_timeline.csv `
  --nir-timestamps J:/Data/sub-099_/nir/sub-099_nir_timestamps.csv `
  --mmwave-timestamps J:/Data/sub-099_/mmwave/sub-099_mmwave_timestamps.csv `
  --output D:/Project/厚粲杯/11_数据/derived/sub099_recovery_windows_v1
```

该命令只生成 `recovery_windows.csv` 和 `recovery_manifest.json`。清单同时报告两个模态的帧范围、时间戳覆盖、帧号缺口和连续性。

## NIR 恢复推理

NIR 使用外部时间线、只选择两个正式任务 Block，并写入正式目录之外的 recovery 根目录：

```powershell
python runtime/nir-formal/run_pipeline.py formal `
  --video J:/Data/sub-099_/nir/sub-099_nir.avi `
  --device 0 `
  --backend pytorch-cuda `
  --ritnet-precision fp32 `
  --ritnet-batch-size 16 `
  --yolo-batch-size 8 `
  --phases block1,block2 `
  --recovery-timeline D:/Project/厚粲杯/11_数据/derived/sub099_timeline_reconstruction_v1/sub099_reconstructed_timeline.csv `
  --output D:/Project/厚粲杯/11_数据/01_Attention-Analysis_nvidia-cuda_recovery_NIR
```

成功结果的 `completion.json` 为 `recovery_complete`，并带有 `recovery_mode=true`、外部时间线路径和两个 Block 的 `phase_windows.json`。它不能写入或冒充 `01_Attention-Analysis_nvidia-cuda_formal_NIR` 中的正式完整结果。

## 毫米波恢复边界

毫米波下游分析必须读取同一份 `recovery_windows.csv` 中 `modality=mmwave` 的窗口，或等价地使用重建 Block 的绝对时间范围再映射毫米波自己的时间戳。不得使用 RGB 帧号、NIR 帧号或行号替代毫米波时间戳。若窗口内有毫米波帧号缺口，结果必须记录缺口并降级为受限/探索性，不得静默插值为完整连续信号。

## 资格限制

重建任务窗只能支持行为定义的 block1/block2 和探针前窗口。它不能恢复：

- 原始 baseline 起止；
- practice 的完整页面事件；
- 三模态启动/停止事件；
- 被裁切或缺失的原始行为时间线。

因此 `sub-099` 的恢复结果可用于受限任务窗质量和特征探索，但不能写成完整 formal baseline 或完整跨模态时间线结果。
