# 08-26-12｜RGB Face 第一档提速实现

## 背景

Face backend 已冻结为 Py-Feat 2.1.1 Detectorv2 scientific core + ONNX Runtime DirectML，正式采样率已冻结为 timestamp-driven 15 Hz。`sub-031` formal dry-run 已成功抽取 5 个代表窗口，共 3600 帧。

用户要求优先进行不改变科学定义的第一档工程提速，并参考 NIR 正式 AMD runtime 之前的提速经验。

## NIR 可直接借鉴的关键实现

NIR 后期提速不只是更换硬件或单模型 batch，而是把上游和下游 batch 解耦：YOLO 固定 batch 8；每帧产生的 eye ROI 先进入 `pending`，跨多个 YOLO batch 累积；只有达到 RITnet fixed batch 16 才调用一次下游模型，最后不足 16 才 flush。

RGB 原 validated real-300 runner 虽配置 `retinaface_batch=8`、`multitask_batch=16`，但每个 RetinaFace batch 通常只有约 8 张单脸 chip，因此 multitask 实际大多以约 batch 8 调用，没有充分利用 Gate 1 中 batch 16 的最优吞吐。

## 第一档 candidate

新增：

- `scripts/face_formal_dryrun_directml_v02.py`
- `scripts/face_compare_pyfeat_runs.py`

### 1. Direct AVI

candidate 直接依据 dry-run manifest 的 `video_frame_position` 从原始 AVI 解码，不再依赖 JPEG95 round-trip。窗口内相邻采样点使用顺序 `grab/read`；跨较大窗口边界时 seek 到下一目标位置。

正式 full-cohort 本来就计划 direct AVI，因此这也是从 test-only JPEG 入口向正式 runner 结构靠拢。

### 2. Decode / preprocess prefetch

独立 reader thread 负责：

```text
AVI selected-frame decode
→ BGR→RGB
→ RetinaFace FP32 NCHW preprocess
→ bounded queue
```

主线程只负责 DirectML 与后续 CPU processing。队列默认仅预取 2 个 RetinaFace batch，控制内存占用。

reader 使用 `stop_event + timeout queue put`，确保主线程异常时不会因为满队列导致 `join()` 死锁。

### 3. NIR-style downstream pending batch

RetinaFace 继续 batch 8，不改变已验证配置。

检测得到的 face chips 不再每 8 帧立即调用 multitask，而是跨 RetinaFace batches 放入 pending：

```text
RetinaFace B8
→ face chips ┐
RetinaFace B8
→ face chips ├→ pending >= 16 → multitask B16
...           ┘
```

最后剩余不足 16 的尾 batch 才 partial flush。

这与 NIR `YOLO B8 → pending ROI → RITnet B16` 的结构一致。

## 明确不做的优化

本轮不做：

- RetinaFace 降频；
- bbox tracking 替代部分 detector calls；
- 输入分辨率变化；
- 模型替换/量化；
- AU/mesh/blendshape 字段裁剪；
- 15 Hz 改动。

因此本轮属于工程调度优化，不应改变科学测量定义。

## 为什么仍需 parity

candidate 直接读取 AVI，而 reference dry-run 使用 JPEG quality 95。即使模型图与处理定义完全不变，去除 JPEG round-trip 后输入像素也可能产生小幅差异。因此必须做一次同 3600 时点 A/B：

1. legacy JPEG reference；
2. optimized direct-AVI candidate；
3. 比较 face count、bbox IoU、AU、emotion、V/A、pose、gaze、blendshapes、mesh。

只有速度提升且 parity 可接受，v0.2 candidate 才能晋升为正式 Face runner 基础。

## 推荐实机命令

### A. legacy reference（一次性）

```powershell
python scripts/face_formal_dryrun_directml.py `
  --sample-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031" `
  --model-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml-v01-reference"
```

### B. optimized v0.2 candidate

```powershell
python scripts/face_formal_dryrun_directml_v02.py `
  --sample-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031" `
  --model-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml-v02"
```

### C. parity

```powershell
python scripts/face_compare_pyfeat_runs.py `
  --reference-raw "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml-v01-reference\pyfeat_dml_raw.parquet" `
  --candidate-raw "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml-v02\pyfeat_dml_raw.parquet" `
  --output "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml-v02\optimization_parity.json"
```

## 回传重点

优先检查/回传：

- legacy `pyfeat_dml_real300_manifest.json`；
- candidate `pyfeat_dml_formal_dryrun_v02_manifest.json`；
- `optimization_parity.json`。

速度比较应使用两者都不含 parquet write 的 pipeline wall/FPS；candidate manifest 同时单独记录 parquet write 与 including-write FPS。

## 当前状态

**First-tier optimization: Implemented, not yet accepted.**

等待 `sub-031` 3600-frame A/B speed + parity。通过后再把 optimized direct-AVI/prefetch/pending-batch 结构作为正式 full-cohort runner 基础；第二档 detector cadence/tracking optimization 暂不进入。
