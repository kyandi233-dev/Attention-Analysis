# RGB

> 2026-08-26（Asia/Shanghai）｜RGB 共享科学层已从独立开发线选择性同步进入长期 NVIDIA 硬件主线 `nvidia-cuda`。本页描述当前有效状态；日期型工作记录保持历史原文，不追溯改写。

RGB 当前主线为 **Face + Pose + Motion**，目标是形成可与 NIR、SART Behavior 按 `unix_ms` / trial / probe window 对齐的连续行为测量，而不是输出单一 Attention Score。rPPG / HR / HRV 不属于当前正式 RGB 主线。

## 当前状态

| 模块 | 当前路线 | NVIDIA 状态 |
|---|---|---|
| Face scientific definition | Py-Feat 2.1.1 Detectorv2 scientific core | **与 AMD 共享科学定义；CUDA formal runtime 待实现/验收** |
| Face cadence | timestamp-driven 15 Hz | **Accepted** |
| Face batching | RetinaFace B8 / multitask B16 | 科学/工程目标已冻结；CUDA 实机 batch 仍需 benchmark |
| Primary / eyelid | temporal tracking + 478 mesh + EAR / aperture-iris / eyeBlink | 共享 derived 逻辑已进入主线；最终事件阈值待冻结 |
| Face QC | full-face 478 mesh + eyes/iris highlight + bbox/track/metrics | **v03 单层黑字版本已同步** |
| Pose | MediaPipe Tasks Pose Landmarker | 共享 10 Hz science/QC/features 已进入主线 |
| Motion | OpenCV frame-difference motion energy | 共享 science/QC/review 已进入主线 |
| body_motion_energy | body ROI 内像素运动 | 待统一正式视频读取阶段实现 |

## NVIDIA Face 当前边界

当前 `nvidia-cuda` 已拥有 RGB sampling、schema、tracking/eyelid derived、QC、Pose/Motion 等共享科学层，但**不能把“代码已同步”写成“CUDA Face 正式推理已验证完成”**。

NVIDIA Face 正式执行器沿用 NIR 双硬件原则：

```text
AMD:    Py-Feat scientific core → ONNX Runtime DirectML
NVIDIA: Py-Feat 2.1.1 native     → PyTorch CUDA
```

两边必须保持相同：

- 15 Hz timestamp target grid；
- RetinaFace threshold / decode / NMS；
- bbox expansion=1.2 与 square/reflection crop 语义；
- canonical pose / gaze 定义；
- raw schema 与 multi-face 保留规则；
- primary tracking / EAR / eyelid derived / QC；
- subject/timestamp/provenance 字段。

NVIDIA CUDA runner 实现后，必须使用同一 `sub-031` 3600 timestamp 集与 AMD DirectML 做跨后端 parity，不能仅凭模型名称相同直接进入全量。

## 仍未完成的正式化 Gate

1. NVIDIA native PyTorch/CUDA Face runner；
2. `sub-031` AMD DirectML ↔ NVIDIA CUDA parity；
3. `sub-033` timestamp / capture-gap stress test；
4. continuous primary-face gates 最终 Accepted；
5. blink event threshold；
6. `perclos80_proxy` 最终窗口、分母与 QC；
7. direct full-video formal runner + completion/resume；
8. Face/Pose/Motion/body_motion_energy 统一视频读取与正式 schema 收口。

在这些 Gate 完成前，不进行 NVIDIA RGB Face 正式全量。

## 当前数据与输出位置

当前 NVIDIA 工作站原始数据根：

```text
J:\Data
```

RGB 默认正式/测试输出位于 Git 仓库外：

```text
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB
```

Git pull、切换分支或代码合并不会覆盖上述仓库外结果；重新运行具体 runner 时仍需遵循 completion/resume/versioned-output 规则。

## 开发前优先阅读

1. [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)
2. [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)
3. [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)
4. [`046-NVIDIA-CUDA-RGB运行路线.md`](046-NVIDIA-CUDA-RGB运行路线.md)
5. [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)
6. [`../050-decisions/055-RGB-Face-15Hz采样频率冻结.md`](../050-decisions/055-RGB-Face-15Hz采样频率冻结.md)
7. [`../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md`](../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md)

AMD-specific `054-RGB-Face-Backend冻结.md` 与 `057-RGB-Face第一档工程优化冻结.md` 是 DirectML 实机决策，不作为 NVIDIA CUDA 当前验收记录；其历史研究过程仍可在工作记录/Git 历史追溯。

## 当前续接

```text
实现 NVIDIA PyTorch/CUDA Face runner
→ sub-031 AMD↔NVIDIA parity
→ sub-033 gap stress
→ blink / perclos80_proxy 冻结
→ full-video formal runner
→ 双硬件正式运行
```

最新硬件主线收口记录见：

- `docs/工作记录/08-26-16-硬件主线收口与共享科学层同步.md`
- `docs/工作记录/08-26-17-硬件主线Behavior-RGB同步阶段完成.md`
