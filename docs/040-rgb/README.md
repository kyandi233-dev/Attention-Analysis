# RGB

> 2026-08-26（Asia/Shanghai）｜当前 NVIDIA RGB 并行开发工作线为 `rgb-nvidia`。`nvidia-cuda` 是 NVIDIA 综合线，目前有人正在使用，本轮同步不直接修改它。两条分支仍属于同一个 Attention-Analysis 项目。

> 分支关系见 [`../010-overview/015-并行分支与同步约定.md`](../010-overview/015-并行分支与同步约定.md)。日期型工作记录保留历史原文，不追溯改写。

RGB 当前主线为 **Face + Pose + Motion**，目标是形成可与 NIR、SART Behavior 按 `unix_ms` / trial / probe window 对齐的连续行为测量。rPPG / HR / HRV 不属于当前正式 RGB 主线。

当前硬盘分工：NVIDIA 工作站连接剩余约 72 名被试的数据盘，AMD 工作站连接另一块约 44 名被试的数据盘。因此 NVIDIA representative Face Gate 使用 `sub-130`，不能默认使用 AMD 盘上的 `sub-031` / `sub-033`。

## 当前状态

| 模块 | 当前路线 | NVIDIA 状态 |
|---|---|---|
| Face scientific definition | Py-Feat 2.1.1 Detectorv2 scientific core | **与 AMD 共享科学定义** |
| Face cadence | timestamp-driven 15 Hz | **Accepted** |
| CUDA dry-run runner | native Py-Feat / PyTorch CUDA | **`face_formal_dryrun_cuda.py` 已实现；待 sub-130 实机 parity** |
| Face batching | CUDA native batch | RTX 5070 最优 batch 待实测；不机械照搬 AMD DirectML B8/B16 |
| Primary / eyelid | temporal tracking + 478 mesh + EAR / aperture-iris / eyeBlink | 共享 derived 逻辑已存在；最终事件规则待冻结 |
| Face QC | 478 mesh + eyes/iris + bbox/track/metrics | 共享 QC 已同步 |
| Pose | MediaPipe Pose 10 Hz | 共享 science/QC/features 已同步 |
| Motion | OpenCV frame-difference motion | 共享 science/QC/review 已同步 |
| full-video formal CUDA runner | original AVI + completion/resume | **尚未实现/验收** |
| body_motion_energy | body ROI 内像素运动 | 尚未实现 |

## NVIDIA 与 AMD 的科学等价关系

执行后端可以不同，但 scientific contract 必须一致：

```text
AMD:    Py-Feat scientific core → ONNX Runtime DirectML
NVIDIA: Py-Feat 2.1.1 native     → PyTorch CUDA
```

两边共同保持：

- timestamp-driven Face 15 Hz；
- RetinaFace / Detectorv2 检测语义；
- bbox/crop 与 canonical pose/gaze 定义；
- 所有 detected faces 先保留；
- 20 AU、7 emotion、V/A、gaze、6DoF pose、478 mesh、68 compatibility landmarks、blendshapes 全保留；
- identity 不属于正式 scientific core；
- primary tracking / EAR / eyelid derived / subject provenance 与 QC 语义一致。

CUDA 可以改变 device scheduling 和最优 batch，但不得删减字段、改变采样时间点或改变变量定义。

## 为什么不做 sub-031 ↔ sub-130 逐帧 parity

两台机器连接不同数据盘，因此不同被试之间不能做 row-wise parity。当前验证链为：

```text
AMD 已完成：同输入 Py-Feat CPU reference ↔ DirectML parity
NVIDIA 待完成：同一 sub-130 sample Py-Feat CPU reference ↔ native PyTorch CUDA parity
```

如果以后需要真正的 cross-device parity，只复制同一小份 representative sample 到另一台机器即可。

## 当前 NVIDIA Face Gate

代表被试：

```text
sub-130
```

五个 representative windows 合计约 240 秒，15 Hz，预期约 3600 帧：baseline start/end、Block1 middle、interblock middle、Block2 middle。

当前顺序：

```text
sub-130 15 Hz sample
→ native Py-Feat CPU reference
→ native PyTorch CUDA
→ field-level parity
→ tracking / eyelid / QC
→ NVIDIA representative gap-stress
→ original-AVI full-video CUDA runner
→ cohort batch / resume
```

CUDA dry-run runner：

```text
scripts/face_formal_dryrun_cuda.py
```

它要求 `py-feat==2.1.1` 与可用 CUDA GPU，并关闭 identity scientific branch。

## 与 AMD 当前正式化进度的同步边界

AMD `rgb-amd` 已进入 full-span formal runner 实机验收阶段。该进展说明共享科学层、正式 output schema 与 orchestration 设计已经继续向前，但 **AMD DirectML runner 不能直接当 NVIDIA CUDA runner 使用**。

可以同步到 NVIDIA 的是：

- full-span timestamp frame selection 逻辑；
- subject output 命名与 manifest/completion 设计；
- Motion/Pose 正式包装思路；
- raw-first / flag-first 信息保留原则；
- tracking / eyelid derived 科学规则。

仍需 NVIDIA 独立实现/验收的是：

- native PyTorch CUDA full-video Face executor；
- RTX 5070 batch / memory / throughput；
- NVIDIA cohort completion/resume；
- CUDA-specific runtime manifest。

## 当前数据与输出位置

NVIDIA 原始数据根：

```text
J:\Data
```

RGB 输出位于仓库外：

```text
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB
```

当前 RGB 开发 branch：

```text
rgb-nvidia
```

`nvidia-cuda` 当前有人在使用，本轮不直接修改。

## 开发前优先阅读

1. [`../010-overview/015-并行分支与同步约定.md`](../010-overview/015-并行分支与同步约定.md)；
2. [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)；
3. [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)；
4. [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)；
5. [`046-NVIDIA-CUDA-RGB运行路线.md`](046-NVIDIA-CUDA-RGB运行路线.md)；
6. [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)；
7. [`../050-decisions/055-RGB-Face-15Hz采样频率冻结.md`](../050-decisions/055-RGB-Face-15Hz采样频率冻结.md)；
8. [`../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md`](../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md)。

AMD-specific DirectML backend/optimization decisions继续作为 scientific provenance 参考，但不替代 NVIDIA CUDA 的实机验收。

## 当前续接

```text
sub-130 3600-frame CPU↔CUDA parity
→ tracking / eyelid / QC
→ NVIDIA gap-stress representative
→ original-AVI full-video CUDA runner
→ completion / resume
→ NVIDIA RGB cohort queue
```
