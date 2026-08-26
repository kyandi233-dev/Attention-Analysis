# NIR

## 2026-08-26 当前 NIR 状态

正式 NIR 全量分析已经执行。当前有效入口不是早期 PuReST / faceparts 候选链，而是已经运行过的：

```text
runtime/nir-formal/
```

当前正式流程：FocusWave v3.1.3 phase windows → 逐帧 YOLO26n 眼框 → ROI → RITnet batch inference → 正式输出。

**当前首要入口：** [08-23-01-NIR正式分析当前入口与资产映射.md](08-23-01-NIR正式分析当前入口与资产映射.md)

**眨眼解释边界：** [021-眨眼检测边界与RITnet派生开合度.md](021-眨眼检测边界与RITnet派生开合度.md)。当前 runtime 不把 `ritnet_missing`/`yolo_missing` 解释为眨眼；`fullclass_ocular_aperture_ratio_median/p90` 是基于 RITnet 四分类可见眼球几何的连续开合度辅助指标，不等同于 EAR、blink 或 PERCLOS。

**正式分析设计：** [022-2026-08-25-NIR正式分析设计与待验证项.md](022-2026-08-25-NIR正式分析设计与待验证项.md)。当前北京正式样本按 116 人处理；瞳孔正式分析需以虹膜作同帧几何参照，行为/Probe/问卷与 NIR 的时间窗、动态特征、按键 QC、混合模型和机器学习边界统一记录在该文件中。

**NIR × Behavior 数据契约：** [024-2026-08-26-NIR行为对齐原型与数据契约.md](024-2026-08-26-NIR行为对齐原型与数据契约.md)。`sub-031` 已完成 v1.2 最终验收；当前正式下游实现为 **`nir-behavior-v1.2 / schema 2`，prototype 已冻结**。已冻结 trial/probe 长表、PIR/OAR 字段语义、Block 边界截断与内部 NIR 缺失的区分、coverage report 与 completion/provenance 规则。`subjects.include=[sub-031]` safety gate 暂时保留，直到更多 full-class 被试完成、准备进入 cohort alignment 时再解除。

**刺激视觉协变量：** [025-2026-08-26-SART刺激视觉协变量重建.md](025-2026-08-26-SART刺激视觉协变量重建.md)。当前 `stimulus-visual-v1.0` 按 FocusWave formaltest 的真实绘制规则重建 `刺激.png + 9种水果×3种size` 的 27 个正式条件，并生成数字 linear-sRGB relative luminance、RMS contrast 及相对 mask 的亮度步长；该表用于控制条件间相对视觉差异，不等同于光度计校准的 cd/m²。

**44 人全量分析资料入口：** [027-44人全量分析数据边界与资料清单.md](027-44人全量分析数据边界与资料清单.md)

> 下方 08-16、08-17、08-21、08-22 内容按当时研究阶段保留。其中“候选 / 待实测 / 停止门 / 尚未全量”等措辞是历史状态，不代表当前状态。

## 当前入口

| 入口 | 作用 | 当前状态 |
|---|---|---|
| [08-23-01-NIR正式分析当前入口与资产映射.md](08-23-01-NIR正式分析当前入口与资产映射.md) | 当前 branch、runtime、模型、配置、运行命令与历史路径映射 | **当前正式入口** |
| [021-眨眼检测边界与RITnet派生开合度.md](021-眨眼检测边界与RITnet派生开合度.md) | `fullclass_ocular_aperture_ratio_*` 的真实定义、blink/PERCLOS 解释边界与人工验证要求 | **当前开合度定义** |
| [022-2026-08-25-NIR正式分析设计与待验证项.md](022-2026-08-25-NIR正式分析设计与待验证项.md) | 116 人正式 NIR 的变量、行为对齐、时间窗、统计/ML、QC 与待冻结项 | **当前分析设计** |
| [024-2026-08-26-NIR行为对齐原型与数据契约.md](024-2026-08-26-NIR行为对齐原型与数据契约.md) | `nir-behavior-v1.2 / schema 2` 的输入、时间对齐、trial/probe 表、coverage、provenance 与 prototype 验收 | **当前冻结对齐契约** |
| [025-2026-08-26-SART刺激视觉协变量重建.md](025-2026-08-26-SART刺激视觉协变量重建.md) | 27 个正式 SART 画面重建、relative luminance/contrast/mask-delta 定义与运行方式 | **当前视觉协变量入口** |
| `runtime/nir-formal/` | AMD 分支为 ONNX Runtime DirectML package `0.1.1`；NVIDIA 复现在 `nvidia-cuda` | **当前 AMD runtime** |
| `runtime/nir-formal/INSTALL.md` | 新电脑从零配置当前正式 runtime | **当前安装入口** |
| `src/attention_pipeline/nir/` | 项目级 NIR 可复用源码及保留的历史评价逻辑 | 保留；不等同于正式 runtime |
| `scripts/` | 仓库级可执行脚本 | 当前脚本入口见目录实际文件与根 README |
| [08-22-04-NIR新电脑GPU环境配置与正式批处理运行指南.md](08-22-04-NIR新电脑GPU环境配置与正式批处理运行指南.md) | RTX 新电脑从零配置、CUDA/PyTorch/OpenCV/RITnet 排错过程 | 历史环境配置手册 |
| [08-16-03-NIR历史多算法环境与迁移说明.md](08-16-03-NIR历史多算法环境与迁移说明.md) | 4 ROI × 4 pupil 阶段的环境/迁移说明 | 历史多算法部署手册 |

## 历史 08-16 核心入口

以下内容描述 08-16 当时的研究状态。部分脚本、`artifacts/` 和候选模型后来已经从当前 `nvidia-cuda` 删除；表中路径用于理解历史记录，不表示 current branch 仍应存在这些资产。

| 入口 | 作用 | 当时状态 / 当前追溯方式 |
|---|---|---|
| `scripts/roi_faceparts.py` | 特写专用 ROI：直接检测 eye bbox（不依赖完整人脸） | 当时新候选；当前通过 Git/tag 追溯 |
| [08-16-01-NIR特写ROI调研与选型.md](08-16-01-NIR特写ROI调研与选型.md) | 三后端全灭后调研 RITnet/DeepVOG/Iris/faceparts/自训练，选定 faceparts | 历史总结 |
| [08-16-02-NIR算法封装与可移植.md](08-16-02-NIR算法封装与可移植.md) | 4 ROI + 4 瞳孔封装、画面分型、模型归集、迁移 | 历史总结 |
| [08-16-03-NIR历史多算法环境与迁移说明.md](08-16-03-NIR历史多算法环境与迁移说明.md) | 当时完整多算法环境与跨机迁移说明 | 历史环境说明 |
| `configs/formal.yaml` | 当时正式 ROI 与 PuReST 候选配置 | 历史兼容配置；不是 current NIR config |
| `artifacts/truth-528/` 等 | 当时人工真值、benchmark、sequence、ROI selection 阶段产物 | 当前 branch 已删除；由工作记录和 Git 历史追溯 |

## 工作记录跳转

| 文件 | 内容 |
|---|---|
| [08-16 NIR正式管线审计](../工作记录/08-16-01-NIR正式管线审计工作记录.md) | 项目/配置/正式成像与迁移依赖审计 |
| [08-16 PuReST算法逻辑审计](../工作记录/08-16-02-PuReST算法逻辑审计工作记录.md) | 论文—C++—绑定—v2调用逻辑 |
| [08-16 正式管线修复](../工作记录/08-16-03-NIR正式管线修复工作记录.md) | 高严重度修复、状态与测试 |
| [08-16 阶段4与5修复复核](../工作记录/08-16-04-NIR阶段4与5修复复核工作记录.md) | 修复后六算法/调优/连续序列数字 |
| [08-16 正式ROI入围检查](../工作记录/08-16-05-NIR正式ROI入围检查工作记录.md) | 60时点、12组图、三后端硬门失败 |
| [08-16 ROI模型拟合对话](../工作记录/08-16-07-ROI模型拟合对话.md) | DeepVOG/RITnet/pye3d 分层 + 自训练检测器教程（对话原文） |
| [08-16 非全脸ROI管线设想对话](../工作记录/08-16-08-非全脸ROI管线设想对话.md) | RITnet 528眼实测、DeepVOG/Iris 对比、faceparts 调研（对话原文） |
| [08-14 阶段4历史记录](../工作记录/08-14-NIR阶段4六算法基准工作记录.md) | 修复前历史快照；数字已被08-16复核取代 |
| [08-14 阶段5历史记录](../工作记录/08-14-NIR阶段5连续序列工作记录.md) | 修复前历史快照；恢复/插值数字已被08-16复核取代 |
| [011 历史审计计划](011-2026-08-13-NIR现状审计与分阶段实施计划.md) | 最早只读审计快照；按历史文件规则保留原文件名 |

## 历史算法判断｜08-16

- PuReST 当时是瞳孔连续检测候选，PuRe 用于抽样质控/诊断。
- PuReST 内部搜索不等于严格 px 控制；正式接受范围由 Python 后置门执行。
- 原始椭圆 `size/angle` 用于绘图、mask 和光度；canonical 长短轴交换时角度同步旋转 90°。
- 每帧×眼别保留完整成功/失败行；插值只写副轨，不跨闭眼、无脸、ROI失败或断点。
- 当时三种 ROI 失败的原因是输入域为双眼特写，而候选模型都以完整人脸为前提。
- faceparts 当时作为特写 ROI 新候选；训练域 RGB，与 NIR 存在域差。
- 当时比较的瞳孔算法包括 PuReST / RITnet / DeepVOG / Iris。

## 历史停止点｜08-16

当时已经把 4 个 ROI 后端 + 4 个瞳孔算法封装成可移植包，但尚未选择正式路线。该停止点后来由 NIR 专用 YOLO26n 眼框检测 + RITnet 正式 runtime 路线取代；原记录继续保留用于追踪方法演变。

## 历史状态｜2026-08-21

> 2026-08-21 17:30（Asia/Shanghai）｜眼框训练已启动，正式分析准备工作将在 2026-08-23 晚回珠海后继续。

- 北京已采集一百多名正式被试；单段视频约 25 分钟。
- 当时眼框模型在 AMD RX 6750 GRE 电脑上使用 CPU 训练 100 epochs。
- 当时计划的正式处理架构为 `YOLO + tracking + RITnet`。
- 当时尚未冻结 tracking 算法、重新检测频率、RITnet 参数和全量运行资源预算。

## 08-17 YOLO 眼框训练文件

| 文件 | 用途 | 历史状态 |
|---|---|---|
| [08-17-01-NIR-YOLO眼框训练计划.md](08-17-01-NIR-YOLO眼框训练计划.md) | NIR 专用 eye 检测、可选跟踪、标准 ROI、瞳孔分析衔接 | 训练计划，已进入历史 |
| [08-17-02-NIR离线眼框标注指南.md](08-17-02-NIR离线眼框标注指南.md) | 离线抽帧、单类 eye 框标注、YOLO 标签、人工质量清单 | 标注操作说明 |
| [08-17-03-NIR远程YOLO训练指南.md](08-17-03-NIR远程YOLO训练指南.md) | 新电脑/远程机器需要的文件、训练命令、输出归档和迁移检查 | 训练操作说明 |
| [../工作记录/08-22-01-NIR-YOLO测试评估与跨机验证工作记录.md](../工作记录/08-22-01-NIR-YOLO测试评估与跨机验证工作记录.md) | yolo26n 本机 test 评价与跨机验证计划 | 历史工作记录 |
| `runtime/nir-yolo-tracking-ritnet-v1.zip` | 当时生成的 YOLO26n + CSRT/KCF + ROI + RITnet GPU 试跑包 | 历史事实；压缩包已于 08-23 仓库整理时删除 |
