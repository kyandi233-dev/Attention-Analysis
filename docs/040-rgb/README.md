# RGB

> 2026-08-26（Asia/Shanghai）｜RGB 共享科学层已从独立开发线选择性同步进入长期 NVIDIA 硬件主线 `nvidia-cuda`。本页描述当前有效状态；日期型工作记录保持历史原文，不追溯改写。

RGB 当前主线为 **Face + Pose + Motion**，目标是形成可与 NIR、SART Behavior 按 `unix_ms` / trial / probe window 对齐的连续行为测量，而不是输出单一 Attention Score。rPPG / HR / HRV 不属于当前正式 RGB 主线。

当前硬盘分工需要特别注意：**NVIDIA 工作站连接的是当前剩余约 72 名被试的数据盘，AMD 工作站连接另一块约 44 名被试的数据盘。** 因此 NVIDIA representative Face Gate 使用 `sub-130`，不能再把 `sub-031` 作为 NVIDIA 本机默认代表被试。

## 当前状态

| 模块 | 当前路线 | NVIDIA 状态 |
|---|---|---|
| Face scientific definition | Py-Feat 2.1.1 Detectorv2 scientific core | **与 AMD 共享科学定义；native PyTorch/CUDA dry-run runner 已实现，待 sub-130 实机 parity** |
| Face cadence | timestamp-driven 15 Hz | **Accepted** |
| Face batching | Py-Feat native Detectorv2 batch | CUDA 最优 batch 需在 RTX 5070 实测；AMD DirectML 的 B8/B16 不强制照搬 |
| Primary / eyelid | temporal tracking + 478 mesh + EAR / aperture-iris / eyeBlink | 共享 derived 逻辑已进入主线；最终事件阈值待冻结 |
| Face QC | full-face 478 mesh + eyes/iris highlight + bbox/track/metrics | **v03 单层黑字版本已同步** |
| Pose | MediaPipe Tasks Pose Landmarker | 共享 10 Hz science/QC/features 已进入主线 |
| Motion | OpenCV frame-difference motion energy | 共享 science/QC/review 已进入主线 |
| body_motion_energy | body ROI 内像素运动 | 待统一正式视频读取阶段实现 |

## NVIDIA Face 当前边界

NVIDIA Face 正式执行器沿用 NIR 双硬件原则：

```text
AMD:    Py-Feat scientific core → ONNX Runtime DirectML
NVIDIA: Py-Feat 2.1.1 native     → PyTorch CUDA
```

两边必须保持相同 scientific contract：15 Hz timestamp target grid、RetinaFace/Detectorv2 语义、bbox/crop 口径、canonical pose/gaze、raw schema、multi-face 保留、primary tracking、EAR/eyelid derived、subject/timestamp/provenance 等。

### 为什么 NVIDIA 不再写“sub-031 AMD↔NVIDIA 逐帧 parity”

`sub-031` 当前位于 AMD 所连接的数据盘，而 NVIDIA 当前使用另一块包含剩余约 72 名被试的数据盘。因此 NVIDIA 代表被试改为：

```text
sub-130
```

不同被试之间不能做逐帧数值 parity。当前更严谨的验证链为：

```text
AMD：Py-Feat CPU reference ↔ DirectML parity（已完成）
NVIDIA：同一 sub-130 3600 帧 Py-Feat CPU reference ↔ native PyTorch CUDA parity（待运行）
                         ↓
两边共同锚定 Py-Feat 2.1.1 Detectorv2 scientific contract
```

如果以后需要真正的 AMD↔NVIDIA 同帧 cross-device parity，只需复制一小份 representative sample（例如 300/3600 帧）到另一台机器，不需要移动整块正式数据盘。

## NVIDIA 当前 Face dry-run Gate

正式 representative：`sub-130`，五个时间窗合计约 240 秒，15 Hz，预期约 3600 个采样时点：

- baseline start 30 s；
- baseline end 30 s；
- Block1 middle 60 s；
- interblock middle 60 s；
- Block2 middle 60 s。

当前 Gate 顺序：

1. 用 RGB 主环境生成 `sub-130` 3600 帧 timestamp-driven sample；
2. 在 Py-Feat native CPU reference 上跑同一批 sample；
3. 在 RTX 5070 native PyTorch CUDA 上跑同一批 sample；
4. 比较 coverage / face count / bbox / AU / emotion / V-A / pose / gaze / mesh / blendshapes；
5. 通过后再跑 tracking / eyelid / QC；
6. 再进入 direct-AVI full-video runner 与正式队列。

CUDA runner：

```text
scripts/face_formal_dryrun_cuda.py
```

它明确要求 `py-feat==2.1.1`、`torch.cuda.is_available()==True`，并显式关闭 identity scientific branch。

## 仍未完成的正式化 Gate

1. `sub-130` native CPU ↔ CUDA parity；
2. NVIDIA representative timestamp/capture-gap stress（具体被试从 NVIDIA 当前数据盘选择，不再硬编码 AMD 的 `sub-033`）；
3. continuous primary-face gates 最终 Accepted；
4. blink event threshold；
5. `perclos80_proxy` 最终窗口、分母与 QC；
6. direct full-video formal runner + completion/resume；
7. Face/Pose/Motion/body_motion_energy 统一视频读取与正式 schema 收口。

在这些 Gate 完成前，不进行 NVIDIA RGB 正式全量。

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

1. [`../../README.md`](../../README.md)：当前 NVIDIA 总操作手册；
2. [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)；
3. [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)；
4. [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)；
5. [`046-NVIDIA-CUDA-RGB运行路线.md`](046-NVIDIA-CUDA-RGB运行路线.md)；
6. [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)；
7. [`../050-decisions/055-RGB-Face-15Hz采样频率冻结.md`](../050-decisions/055-RGB-Face-15Hz采样频率冻结.md)；
8. [`../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md`](../050-decisions/056-RGB-Face-Primary与眼睑派生规则.md)。

AMD-specific `054-RGB-Face-Backend冻结.md` 与 `057-RGB-Face第一档工程优化冻结.md` 是 DirectML 实机决策，不作为 NVIDIA CUDA 当前验收记录；其历史研究过程仍可在工作记录/Git 历史追溯。

## 当前续接

```text
sub-130 3600-frame sample
→ native Py-Feat CPU reference
→ native PyTorch CUDA
→ CPU↔CUDA parity
→ tracking / eyelid / QC
→ NVIDIA gap-stress representative
→ blink / perclos80_proxy 冻结
→ direct-AVI full-video formal runner
→ NVIDIA 72-subject RGB queue
```

最新硬件主线收口记录见 `docs/工作记录/`；历史 `sub-031`/`sub-033` 记录继续保留为 AMD/开发阶段 provenance，不追溯改写。
