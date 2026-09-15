# NIR / Ocular｜近红外生产与眼部正式分析

> **状态：CURRENT（当前入口）**  
> 最后核验：2026-09-15  
> 正式科学分析：`codex/formal-analysis-v2-portable`  
> NIR producer（生产端）：`amd-DirectML` / `nvidia-cuda-v8`  
> 方法与正式结果：`FocusWave-Formal-Analysis@main`

本目录同时保留 NIR（近红外）生产、早期下游验证和当前 Ocular（眼部）正式分析文档。2026-08-27 以前 README 中“PIR 数值已知错误、只能做 pipeline validation（管线验证）、20_formal_statistics 禁止”等表述属于历史阶段，**已经被后续 G1 真实测量审计、表示冻结和正式 Ocular 科学输出取代**。

当前科学问题不再是“旧 PIR 能否进入正式分析”，而是：在 NIR 瞳孔与 RGB（可见光视频）眨眼可用范围内，怎样形成可追溯、时间合法且质量受控的 Ocular 特征，并分析其与 Q1/Q2、任务进程和近期行为的关系，再将冻结特征交给监督学习。

## 1. 当前 Ocular 数据结构

Ocular 由两个科学特征家族组成：

```text
NIR 瞳孔
  ├─ hard R_seg = pupil / (pupil + iris)
  ├─ level（总体水平）
  ├─ variability（波动）
  ├─ linear slope（线性变化）
  └─ quadratic curvature（二次曲率）

RGB 眨眼
  ├─ blink rate（眨眼频率）
  └─ 作为 NIR 瞳孔 blink-artifact mask（眨眼伪迹掩码）的辅助来源
```

RGB 提供眨眼并不把眨眼归为 Movement（动作）；眨眼属于 Ocular。NIR 与 RGB 是设备/来源，Ocular 才是科学模态。

## 2. 当前样本与可用性

正式总体仍是 **61 名参与者、116 场、2,320 probes（思维探针）**。NIR source manifest（来源清单）当前覆盖 **109 场**；7 个治理场次为结构性 NIR availability（可用性）缺失，不是 G1 QC（质量控制）淘汰，也不改变 cohort membership（队列成员资格）。同时具备 NIR 瞳孔与 RGB 眨眼辅助清洗条件的场次为 108 场。

第一轮冻结后的主要 Ocular 特征覆盖为：

| 特征 | 有限值 / 2,320 | 参与者 | 场次 |
|---|---:|---:|---:|
| hard R_seg 瞳孔总体水平 | 1,936 | 61 | 108 |
| hard R_seg 瞳孔 MAD（中位数绝对偏差） | 1,936 | 61 | 108 |
| 瞳孔线性变化 | 1,857 | 61 | 107 |
| 瞳孔二次曲率 | 1,855 | 61 | 107 |
| 眨眼频率 | 2,200 | 60 | 110 |

这些数字表示 Ocular 的真实可估计结构，不得据此把总体样本改写为 109 场或 108 场。

## 3. 当前冻结的瞳孔表示与清洗规则

第一轮正式主表示为 hard segmentation ratio（硬分割瞳孔—虹膜区域内的瞳孔面积比例）：

```text
R_seg,hard = N_pupil / (N_pupil + N_iris)
```

该量是无量纲相对面积比例，不是瞳孔直径/虹膜直径。基于椭圆拟合的几何平均直径 `D_geom = sqrt(d1*d2)` 保留为 cross-representation sensitivity（跨表示敏感性），soft R_seg（软分割比例）保留为 segmentation sensitivity（分割方法敏感性）。

正式瞳孔波动主表示为 MAD（中位数绝对偏差）；SD（标准差）保留为敏感性表示。联合清洗使用 RGB 眨眼事件前 200 ms、结束后 200 ms 的 buffer（缓冲）与 NIR 自身 QC 无效时段合并，被排除区间保持缺失，不插值。

每个 probe 前 30 s 固定划分为 **15 个 2 s bins（时间箱）**，bin 内使用有效值中位数，空 bin 保持缺失。linear slope（线性斜率）和 quadratic curvature（二次曲率）要求前后半段均存在真实数据，且首末有效时间点至少相隔 **20 s**。

## 4. 当前正式代码入口

| 入口 | 当前作用 |
|---|---|
| `scripts/nir_pupil_blink_measurement_audit.py` | NIR 瞳孔 × RGB 眨眼 G1 测量审计 |
| `scripts/nir_ocular_g1_freeze_support.py` | G1 表示、时间支持和同步冻结支持 |
| `scripts/build_ocular_science_output_frozen.py` | 物化冻结后的 Ocular probe 特征、coverage（覆盖）和 handoff（交接） |
| `scripts/run_ocular_postfreeze_analysis.py` | 冻结后 Q1/Q2、任务进程、近期行为等解释性分析 |
| `scripts/build_ocular_headmotion_sensitivity_set.py` | 构建 Ocular × head-motion（头部运动）敏感性集合 |
| `scripts/summarise_ocular_headmotion_sensitivity.py` | 汇总头部运动敏感性结果 |

当前正式下游不需要为了得到 Ocular 结果重新跑 YOLO/RITnet。producer 重跑、环境恢复和硬件实现应切到 `amd-DirectML` 或 `nvidia-cuda-v8`，并遵循相应 `runtime/nir-formal/` 文档。

## 5. 当前正式资产层级

正式本地 Ocular science output（科学输出）以 `FormalScience/Ocular/` 为下游权威，包括：

```text
ocular_science_output_manifest.json
ocular_feature_handoff.csv
ocular_feature_coverage.csv
ocular_probe_features_wide.csv
```

其中 `ocular_probe_features_wide.csv` 保持 **2,320 probe canonical identity（规范身份全集）**；结构性 NIR 缺失以缺失值/状态表达，而不是删除这些 probe。

G1 权威输入及大型固定时间箱表属于本地大资产，不应复制进 GitHub。它们的路径、覆盖、冻结证据和结果身份以 `FocusWave-Formal-Analysis/main/国赛报告/完整结果/3-Ocular眼部结果与资产.md` 为准。

## 6. 旧 10/11/12 管线现在是什么角色

历史流程：

```text
10_analysis_ready
→ 11_analysis_tables
→ 12_pipeline_validation
→ 20_formal_statistics
```

保存了从旧 PIR 时代形成的数据契约、绘图和验证思路，对 provenance（来源追踪）仍有价值；但它**不是当前 Ocular 正式分析的状态机**。旧文件中“当前只能运行 12_pipeline_validation”“PIR 修复前不得进入正式统计”等限制只描述当时阶段。

当前正式结果以 G1 测量审计、冻结后的 Ocular science output 和 post-freeze analysis（冻结后分析）为准。`scripts/nir_pipeline_validation.py` 等旧验证入口可以用于历史复现或工程诊断，不得据其输出覆盖当前正式结果。

## 7. NIR producer 与科学分析的边界

NIR producer 负责：

```text
原始 NIR 视频
→ 眼框/ROI
→ RITnet 分割
→ 当前 schema 测量产物
→ 可追溯时间戳与 manifest
```

正式科学分析负责：

```text
可追溯生产产物
→ G1 measurement audit（测量审计）
→ 瞳孔×眨眼清洗与 probe 前窗口
→ 表示冻结
→ Ocular 科学输出
→ 解释性分析与监督学习
```

NVIDIA 当前最终 NIR producer 是 `nvidia-cuda-v8`；历史 `nvidia-cuda` 最终生产状态已经由 archive tag（归档标签）冻结，不再是长期分支。AMD 当前 producer 为 `amd-DirectML`。

## 8. 阅读顺序

如果目的是理解**当前 Ocular 科学分析**，优先：

```text
本 README
→ FocusWave-Formal-Analysis/main/分析设计/1.16.2、1.16.14、1.16.15(b)、1.16.16、1.16.18、1.16.19
→ FocusWave-Formal-Analysis/main/国赛报告/完整结果/3-Ocular眼部结果与资产.md
→ 当前 scripts / src 实现
```

如果目的是**重新生产 NIR 测量**，再进入相应 producer 分支的 `runtime/nir-formal/README.md`。不要从 8 月 27 日的旧 validation-only（仅验证）文档反推当前科学方法。