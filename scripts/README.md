# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量需要继续保留、可直接重跑的历史验证入口。正式 NIR runtime 仍位于 `runtime/nir-formal/`。

当前 NVIDIA RGB 开发 branch 为 `rgb-nvidia`；`nvidia-cuda` 是正在使用的综合线，本轮不直接修改。两者属于同一个 Attention-Analysis 项目。分支关系见 `docs/010-overview/015-并行分支与同步约定.md`。

## 当前入口索引

| 脚本 | 定位 | 用途 |
|---|---|---|
| `sart_formal_analysis.py` | 当前 | FocusWave v3.1.3 最终 BB Behavior |
| `nir_behavior_alignment.py` | 当前 | NIR × Behavior Unix-ms / trial / probe 对齐 |
| `build_stimulus_visual_table.py` | 当前 | SART 视觉协变量与报告图 |
| `rgb_analysis.py` | 当前，共享 RGB | RGB audit / gaps / Motion / Pose / Face sampling / QC |
| `face_formal_dryrun_sample.py` | 当前，共享 RGB | timestamp-driven 15 Hz representative Face sample |
| `face_formal_dryrun_cuda.py` | **当前，NVIDIA CUDA Gate** | 同一 sample 上运行 Py-Feat 2.1.1 native PyTorch CUDA |
| `face_derive_tracking_eyelid_v02.py` | 当前，共享 RGB | tracking / primary / EAR / aperture-iris / eyeBlink derived |
| `face_qc_visualize_v03.py` | 当前，共享 RGB | 478 mesh + eyes/iris + primary/secondary + metrics QC |
| `face_benchmark_pyfeat.py` | reference | Py-Feat native CPU reference / historical benchmark |
| `sart_bbb_v3_0_analysis.py` | 历史、可执行 | FocusWave v3.0 BBB 行为分析重跑 |

## NVIDIA RGB 当前科学定义

```text
Py-Feat 2.1.1 Detectorv2 scientific core
+ timestamp-driven 15 Hz
+ native PyTorch CUDA execution
+ complete raw scientific schema
+ shared primary tracking / eyelid derived / QC
```

当前 NVIDIA representative 使用：

```text
sub-130
```

原因是 NVIDIA 工作站连接的是剩余约 72 名被试的数据盘，而 AMD 工作站连接另一块约 44 名被试的数据盘。不同被试之间不能做逐帧 parity。

## 当前 CUDA Gate

当前已实现：

```text
face_formal_dryrun_cuda.py
```

正确验证链：

```text
sub-130 15 Hz representative sample
→ 同一 sample native Py-Feat CPU reference
→ 同一 sample native PyTorch CUDA
→ field-level parity
→ tracking / eyelid / QC
```

而不是：

```text
sub-031 AMD output ↔ sub-130 NVIDIA output
```

CUDA runner 已存在，但 **sub-130 实机 parity 结果尚未完成**，所以现在还不能把 NVIDIA RGB Face 写成正式 full-video 已冻结。

## AMD formal runner 的共享进展

AMD `rgb-amd` 已经进入完整正式时间段 runner 验收阶段，包含：

```text
full-span Face frame preparation
Motion/Pose formal wrapper
original-AVI DirectML Face runner
continuous tracking/eyelid formal derive
single-subject orchestration
subject manifests / completion semantics
```

这些成果中，frame manifest、output schema、Motion/Pose orchestration、tracking/eyelid 与 completion/resume 设计可以同步到 NVIDIA；**DirectML Face executor 不能直接当 CUDA executor**。

NVIDIA 需要单独实现/验收：

```text
original AVI
→ timestamp-driven 15 Hz
→ native Py-Feat / PyTorch CUDA
→ same raw schema
→ same derived
```

并记录 RTX 5070 的 batch、throughput、GPU memory 与 CUDA runtime provenance。

## 当前推荐执行顺序

```text
sub-130 3600-frame CPU↔CUDA parity
→ tracking / eyelid / QC
→ NVIDIA gap-stress representative
→ original-AVI full-video CUDA runner
→ single-subject orchestration
→ cohort completion / resume
→ NVIDIA RGB cohort queue
```

## Behavior / NIR-Behavior

这些模块仍属于同一项目，不因为 branch 名称是 `rgb-nvidia` 就被排除。当前 NVIDIA 具体数据根与输出根应继续遵循 NVIDIA 环境配置，不直接复制 AMD 路径。

## 历史与当前状态边界

历史工作记录、旧 `rgb-nvidia-cuda` 名称、早期 `sub-031` 设想继续作为 provenance 保留，不追溯改写。当前状态以 `docs/040-rgb/README.md`、`docs/040-rgb/046-NVIDIA-CUDA-RGB运行路线.md` 和本页为准。

`nvidia-cuda` 当前有人在使用，本轮文档同步没有直接修改该分支。
