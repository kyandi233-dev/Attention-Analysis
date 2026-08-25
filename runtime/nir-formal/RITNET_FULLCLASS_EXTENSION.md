# NVIDIA CUDA｜RITnet Full-Class Post-hoc Extension

## 目的

历史正式 NIR 已完成 YOLO 眼框与 RITnet 瞳孔提取，但 `eyes.csv` 只保留了 pupil 派生量。当前补全阶段不重新运行 YOLO，而是复用每行已经保存的 `video + frame_idx + roi_x1/y1/x2/y2`，从原 AVI 重建同一眼睛 ROI，再运行冻结的 RITnet 四分类。

```text
source eyes.csv + original AVI
        ↓
exact saved ROI
        ↓
640×400 + gamma + CLAHE
        ↓
RITnet ONNX FP32 fixed-b16 / CUDAExecutionProvider
        ↓
0 background / 1 sclera / 2 iris / 3 pupil
        ↓
320×160 structural metrics + pupil/iris normalization + QC
```

本扩展不使用 FP16，也不降低到 512×320 / 384×240。

## 与 AMD 结果的统一口径

NVIDIA 扩展使用 `models/ritnet-b16-fp32.onnx`。该 ONNX 与 AMD full-class 扩展使用同一冻结模型 SHA256：

```text
1933f44f483b350e17249a37b4a2ebe8b5e32f83fc8c1eb1a21c27e96477e621
```

AMD 使用 DirectML，NVIDIA 使用 ONNX Runtime CUDA；模型、输入分辨率、FP32、batch=16、预处理、类别码和 320×160 analysis coordinates 保持一致。

NVIDIA 历史正式 `eyes.csv` 的 pupil 主要来自 PyTorch CUDA。为了避免在主 `pupil/iris` 比值中把“旧 PyTorch pupil”与“新 ONNX iris”混在一起，NVIDIA full-class 正式输出中的 pupil 与 iris 几何都从同一 ONNX hard-label map 重新计算。原 `eyes.csv` 的 pupil 列仍完整保留用于 provenance；`--validate-pupil` 可额外请求 pupil probability 并审计新旧 pupil parity。

## 主要分析指标

主瞳孔尺度指标固定为：

```text
fullclass_pupil_to_iris_diameter_ratio
```

定义为：

\[
PIR_D=\frac{\sqrt{a_p b_p}}{\sqrt{a_i b_i}}
\]

其中 pupil 使用 class 3 拟合椭圆；`iris_outer = class 2 OR class 3`，因此虹膜外边界不会因为 pupil hole 而被错误当成缺失区域。`a,b` 为椭圆两轴。

辅助保留：ellipse-area ratio、contour-area ratio、pupil/iris center offset、四分类面积/比例、iris fill、connected-components、ROI edge touch、ocular aperture 等。正式 pupil/iris 分析优先使用 `fullclass_normalization_valid == True` 的帧。

## Ocular aperture 的解释边界

`fullclass_ocular_aperture_ratio_median` / `p90` 基于：

```text
ocular = sclera OR iris OR pupil
```

在可见眼球中间 80% 横向范围内计算逐列垂直高度，并用 ocular bbox width 归一化。median 更稳健，p90 更接近最大开口区域。

它不是 EAR，也不能直接命名为 blink 或 PERCLOS；当前用途是眼睛开合/QC候选量，后续可与 RGB MediaPipe EAR/blink 结果交叉验证。

## Sparse QC

固定 QC sampling：

- 每 3000 帧约保存一次（30 FPS 下约 100 秒）；
- 每个 phase/segment 保留 first / middle / last；
- 每个 phase 对 `roi_clipped`、`ritnet_missing`、`normalization_invalid`、`ocular_fragmented` 各最多补 2 个异常例。

每个抽样眼睛保存：

```text
*_labels.png
*_overlay.png
```

并写 subject-numbered `*_qc_index.csv`，记录 phase、frame、unix_ms、eye、reason 与图像文件。

## Subject-numbered 输出

以 `sub-031` 为例：

```text
sub-031_ritnet_fullclass_v1-2-fast-qc.csv
sub-031_ritnet_fullclass_v1-2-fast-qc_summary.json
sub-031_ritnet_fullclass_v1-2-fast-qc_manifest.json
sub-031_ritnet_fullclass_v1-2-fast-qc_completion.json
sub-031_ritnet_fullclass_v1-2-fast-qc_qc_index.csv
sub-031_ritnet_fullclass_v1-2-fast-qc_qc/
```

被试身份同时存在于文件夹、文件名和 CSV `subject` 列。

## 时间映射

full-class CSV 原样保留 source `eyes.csv` 的：

```text
phase
phase_segment
frame_idx
video_time_ms
unix_ms
phase_time_ms
```

行为 trial 对齐应作为独立下游步骤，通过 NIR `unix_ms` 与行为绝对时间戳匹配；不需要、也不应该为了 trial 对齐再次运行 RITnet。

## 推荐运行顺序

先更新并测试：

```powershell
git switch nvidia-cuda
git pull --ff-only
cd runtime\nir-formal
python -m pytest tests -q
```

先查看完整正式 source runs：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

首次在 NVIDIA 上建议对一名完整被试做 parity + speed 验证：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --subjects "sub-031" `
  --device 0 `
  --postprocess-workers 4 `
  --validate-pupil
```

审计完成后，正式 production 不加 `--validate-pupil`：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4
```

production 请求 labels-only，减少不需要的 probability 输出传输；CPU decode/crop/preprocess 与 CUDA inference 流水重叠，四分类后处理由小型 worker pool 并行执行。
