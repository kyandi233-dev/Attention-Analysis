# RGB / Movement｜可见光生产与动作、眨眼正式分析

> **状态：CURRENT（当前入口）**  
> 最后核验：2026-09-15  
> 正式科学分析：`codex/formal-analysis-v2-portable`  
> RGB producer（生产端）：`rgb-amd` / `rgb-nvidia`  
> 方法与正式结果：`FocusWave-Formal-Analysis@main`

本目录保存 RGB（可见光视频）生产、历史 Face/Pose/Motion 工程开发，以及当前 Movement（动作）和 Ocular（眼部眨眼）下游所需文档。2026-08-26 README 中大量“full-video runner 待收口”“44/72 人硬件队列”“NVIDIA representative 待验证”等内容属于当时 producer 开发阶段，已经不能代表当前正式科学分析状态。

当前必须先区分：**RGB 是设备/来源命名空间，不是科学模态。** RGB 产物当前主要服务两个科学方向：

```text
RGB 可见光视频
├─ 身体运动 / 姿态 ─────────────→ Movement（动作）
├─ 眨眼事件 ───────────────────→ Ocular（眼部）
├─ 眨眼时间段 ─────────────────→ NIR 瞳孔伪迹辅助清洗
└─ 曝光 / 亮度 / 全局画面变化 ─→ device-support / QC（设备支持 / 质量控制）
```

Face（面部）、Pose（姿态）、Motion（运动）是工程提取模块；正式报告中的科学解释应落到 Movement、Ocular 或 QC 角色，而不是把摄像头模块名称直接当成心理变量。

## 1. 当前 Movement 正式状态

Movement P4 已正式关闭。正式总体为 **61 名参与者、116 场、2,320 probes（思维探针）**；RGB 动作主要指标在 **115 场、2,300 个 probe** 上有有限值，全部 61 名参与者至少保留一个有效动作场次。

第一轮 Movement 正式主指标为 **整体身体动作强度中位数**。它是无量纲相对指标，只描述窗口内整体身体动作多少，不能解释为真实物理位移、速度或能量。

| RGB 派生信息 | 当前科学角色 |
|---|---|
| 整体身体动作强度 | Movement 第一轮正式主指标 |
| 横向姿态方向 | Movement sensitivity（敏感性） |
| 纵向姿态方向 | Movement sensitivity（敏感性） |
| 相对径向方向分数 | QC / sensitivity；不解释为真实靠近/远离 |
| 曝光变化 | device QC（设备质量控制） |
| 全局画面运动 | device QC |
| 画面平均亮度 | device QC |
| blink rate（眨眼频率） | Ocular，不属于 Movement |

当前 Movement 已完成任务进程、Q1/Q2、近期行为关系、姿态方向敏感性、科研图和监督学习 handoff（特征交接）。正式结果见 `FocusWave-Formal-Analysis/main/国赛报告/完整结果/4-Movement动作结果与资产.md`。

## 2. RGB 眨眼当前角色

RGB 眨眼具有两层用途：

1. 作为 Ocular 独立科学特征，形成 probe 前 blink rate（眨眼频率）；
2. 为 NIR 瞳孔提供 blink-artifact mask（眨眼伪迹掩码），用于联合清洗。

眨眼科学上始终归入 Ocular。当前瞳孔联合清洗使用眨眼前 200 ms、结束后 200 ms 的 buffer（缓冲）；缺失 RGB 眨眼来源保持 missing（缺失），不得解释成 zero blink（零眨眼）。具体规则见 [`../020-nir/README.md`](../020-nir/README.md) 和 Formal Ocular 结果总账。

## 3. 当前正式代码入口

### RGB / Movement 下游

| 入口 | 当前用途 |
|---|---|
| `scripts/rgb_55_analysis.py` | 当前 RGB 5.5 正式分析表和模型生产入口 |
| `scripts/build_movement_science_output.py` | 将既有 RGB 5.5 结果物化为 Movement 科学输出、coverage（覆盖）和 handoff |
| `scripts/rgb_formal_downstream.py` | RGB 正式下游接口 |
| `scripts/rgb_formal_motion_pose.py` | Motion / Pose（运动 / 姿态）正式化入口 |
| `scripts/rgb_formal_report.py` | RGB 正式报告辅助输出 |

### producer / 工程复现

`scripts/rgb_analysis.py`、`face_*`、`run_rgb_formal_subject.ps1` 等仍用于 producer（生产端）、工程验证、Face/Pose/Motion 提取和 QC。它们保留价值，但不是当前正式科学结论的总入口。

需要重新生产 RGB 测量时，按硬件切到 `rgb-amd` 或 `rgb-nvidia`；需要解释科学结果时，回到 `codex/formal-analysis-v2-portable` 和 Formal 当前结果文件。

## 4. 当前正式数据关系

当前 Movement 不重新跑视频 producer 来“寻找结果”，而是消费已经存在的 RGB 5.5 正式资产：

```text
RGB producer
→ 21_analysis_tables_5.5
→ rgb_probe_pre30s_strict_features.csv
→ 既有解释性模型源表
→ build_movement_science_output.py
→ FormalScience/Movement/
→ 监督学习 feature handoff
```

`rgb_probe_pre30s_strict_features.csv` 保持 2,320 probe 规范身份骨架；主要动作字段在真实可用范围内产生有限值。唯一 RGB 缺失场次不改变 116 场治理队列。

Movement 正式科学输出包括 `movement_feature_handoff.csv`、`movement_feature_coverage.csv`、science-output manifest（科学输出清单）和对应图/模型源表。正式资产位置与证据等级以 Formal 结果总账为准。

## 5. 已修正的 RT-CV 口径

Movement 近期行为链接曾消费旧备份行为表，其中 RT-CV（反应时变异系数）使用 `rt_cv_min_n = 20`。当前唯一正式口径已经统一为 **30 s 窗口内至少 2 次有效正确 Go 反应，即 `rt_cv_min_n = 2`**；旧 `=20` 属试错产物。

RGB 5.5 已按当前行为表重新运行相应链接，Movement 重新物化后，实质变化只出现在 RT-CV 相关模型行，整体 Movement 科学结论和 feature handoff 未改变。旧产物保留为 provenance（来源追踪），不得再次被当作当前输入。

## 6. 旧 Face/Pose/Motion Gate 现在是什么角色

早期 README 曾列出 timestamp stress（时间戳压力测试）、primary-face gate（主要人脸闸门）、blink threshold（眨眼阈值）、PERCLOS、full-video runner 等未完成事项。这些内容记录的是 RGB producer 的开发史，不再是判断“Movement 是否有正式结果”的门槛。

当前判断顺序是：

```text
producer 是否形成可追溯正式资产
→ 下游字段/时间/QC 是否满足科学合同
→ Movement / Ocular 是否已完成正式物化与分析
→ 是否进入监督学习和报告
```

因此不能因为某份 8 月 producer 文档仍写“Gate 未完成”，就推翻 9 月已经完成的 Movement P4 或 Ocular 眨眼结果。若确需重跑 producer，再单独核对当前 `rgb-amd` / `rgb-nvidia` 的运行文档和环境。

## 7. 当前分支职责

| 分支 | 角色 |
|---|---|
| `codex/formal-analysis-v2-portable` | **当前正式科学分析权威线** |
| `rgb-amd` | AMD RGB producer |
| `rgb-nvidia` | NVIDIA RGB producer |

两台硬件可能连接不同原始数据盘，机器上的本地队列规模只用于执行调度，不是总体样本定义。正式样本、缺失和 coverage 必须从治理清单和当前正式输出读取。

## 8. 阅读顺序

理解当前 RGB 科学作用时：

```text
本 README
→ FocusWave-Formal-Analysis/main/分析设计/1.16.7、1.16.10、1.16.12、1.16.15(a)、1.16.19、1.16.23
→ FocusWave-Formal-Analysis/main/国赛报告/完整结果/4-Movement动作结果与资产.md
→ Ocular 需要时再读 ../020-nir/README.md
→ 当前 scripts / src 实现
```

只有在需要重新生产视频级特征、恢复环境或检查硬件实现时，才进入 `rgb-amd` / `rgb-nvidia` 的 producer 操作手册。