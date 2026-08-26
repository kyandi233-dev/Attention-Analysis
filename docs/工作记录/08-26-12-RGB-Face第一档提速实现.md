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

因此本轮属于工程调度优化，不改变科学测量定义。

## sub-031 3600 帧 A/B 实机结果

### Legacy JPEG95 reference

- 3600 input frames；
- 3629 detected/output face rows；
- RetinaFace B8；multitask 配置 B16，但旧 loop 实际通常仅处理当前 RetinaFace batch 产生的约 8 张 face chips；
- pipeline wall = 212.1708 s；
- throughput = **16.9675 fps**；
- multitask DML = 42.7643 s。

### Optimized direct-AVI v0.2

- 3600 input frames；
- 3630 detected/output face rows；
- RetinaFace calls = 450（严格 B8）；
- multitask full-B16 calls = **226**；partial calls = **1**；
- faces sent to multitask = 3630；
- pipeline wall before parquet = **123.4865 s**；
- throughput before parquet = **29.1530 fps**；
- including parquet write = **28.6060 fps**；
- multitask DML = **19.0804 s**。

相对 legacy reference：

- throughput = **1.718×**；
- pipeline wall 降低约 **41.8%**；
- 即使包含 parquet 写盘，仍为 **1.686×**；
- multitask DML 时间降低约 **55.4%**，证明跨 RetinaFace batch 的 pending-B16 设计真正吃到了 B16 吞吐。

reader thread wall=122.2945 s，而整个 pipeline wall=123.4865 s。reader timing 与主线程 DML/CPU 工作发生重叠，因此不能把各 stage 简单相加重构 wall；这一结果本身也说明 prefetch 已把视频读取/预处理与主推理链显著重叠。

## Parity

reference 是 JPEG quality 95 test frames，candidate 直接读取原 AVI，因此输入像素并非 bitwise identical。此 parity 的目的不是要求完全相等，而是确认移除 JPEG round-trip 与工程调度没有造成不可接受的科学输出漂移。

结果：

- reference rows=3629，candidate rows=3630，matched rows=3629；
- 3600 帧 face-count agreement = **0.9997222**，即仅 1 帧 face count 不同；
- bbox mean IoU = **0.995838**，min IoU = **0.940041**；
- FaceScore Pearson≈0.99490；
- AU20 MAE≈0.00781，Pearson≈0.99784；
- emotion7 MAE≈0.00624，Pearson≈0.99826；
- V/A MAE≈0.00784，Pearson≈0.99843；
- pose6d MAE≈0.00218，Pearson≈0.9999969；
- gaze MAE≈0.00493，Pearson≈0.99783；
- blendshape MAE≈0.00161，Pearson≈0.999734；
- normalized 478×3 mesh MAE≈0.000457，Pearson≈0.9999959；
- original-frame mesh XY MAE≈0.0637 px，Pearson≈0.99999948。

解释边界：candidate 是直接读原 AVI；reference 经过 JPEG95 有损编码。上述小幅非零差异不能解释为模型 graph 或科学定义改变。相反，formal runtime 本来就应优先使用原 AVI，避免把测试期 JPEG 压缩引入正式测量链。

## 决策

**First-tier optimization: Accepted.**

正式 Face runner 基础从现在起采用：

```text
original AVI
→ timestamp-driven 15 Hz selected frames
→ reader/preprocess prefetch
→ RetinaFace DirectML B8
→ decode/NMS + 1.2 square-reflect crop
→ cross-RetinaFace pending face chips
→ multitask DirectML full B16 when available
→ full scientific raw outputs
→ parquet
```

旧 `face_formal_dryrun_directml.py` 与 real-300 runner 保留作为历史/reference，不删除、不覆盖。

下一阶段 tracking / primary-face / eyelid derived 应以 optimized v0.2 raw 为输入。第二档 detector cadence / tracking-based detector skipping 仍暂不进入，除非后续正式全量成本仍需要进一步压缩。
