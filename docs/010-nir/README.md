# 00｜NIR目录与映射

## 2026-08-23 当前状态

正式 NIR 全量分析已经完成。当前 NIR 主线不再是 08-16 时的 full-face ROI 候选，也不再是 08-21/08-22 的“准备跨机短视频验证”状态；这些内容继续保留在本文件后半部分作为研究过程记录。

当前已经核验的路线血缘为：

```text
NIR video
    ↓
YOLO26n eye bbox
    ↓
周期性重新检测 + tracking
    ↓
单眼 ROI
    ↓
RITnet segmentation
    ↓
逐帧 / 逐眼 QC 与时序输出
```

当前仓库中完整 YOLO + tracking + RITnet 实现已经定位到：

```text
runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py
```

该 portable package 内的 `models/nir-eye-yolo26n-best.pt` 与 `yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt` 为同一 Git blob；`models/ritnet-best_model.pkl` 与仓库 `models/RITnet-master/best_model.pkl` 也为同一 Git blob。因此 runtime 确实冻结了后来正式路线所需的两个模型，不是另一套候选权重。

需要保留一个 provenance 限制：runtime package 在 2026-08-22 创建时仍被定义为 GPU 短视频准入包。当前 Git 分支没有保存全量正式运行的最终 `run_manifest.json` / 输出目录，也没有 2026-08-22 之后的 committed 工作记录。因此目前可以确认“正式全量已经完成”和“完整实现/模型血缘在哪里”，但不能从 package 默认配置反推最终 full-run 的 tracker、重检测间隔、ROI expansion 等冻结参数。

## 当前核心入口

| 入口 | 作用 | 当前状态 |
|---|---|---|
| `runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py` | YOLO26n + CSRT/KCF + ROI + RITnet 的完整 portable 实现 | 已核验实现；package 创建时为准入版本 |
| `runtime/README.md` | frozen runtime、ZIP、SHA256 与当前项目状态的边界说明 | 当前 runtime 总入口 |
| `yolotrain/README.md` | YOLO 数据划分、100 epochs 训练、validation/test provenance | 当前训练资产入口 |
| `models/README.md` | RITnet、历史外部模型与候选模型角色说明 | 当前模型资产入口 |
| `src/attention_pipeline/nir/` | ROI 几何、正式时间轴及历史 review / benchmark / sequence 代码 | 保留的 NIR 核心与历史实现 |
| `scripts/00-目录与映射.md` | 数据集、模型评价、历史 ROI/pupil 比较与诊断入口 | 当前脚本索引 |
| [08-22-01-NIR-YOLO测试评估与跨机验证工作记录.md](../工作记录/08-22-01-NIR-YOLO测试评估与跨机验证工作记录.md) | 冻结 test 评价与 portable package 建立过程 | 当前分支最新 committed NIR 工作记录 |

## YOLO26n 当前已核验结果

冻结 test 为 7 名被试、85 张图片、169 个 eye boxes。2026-08-22 工作记录记录：Ultralytics 原生 test precision=0.9754、recall=0.9645、mAP50=0.9913、mAP50-95=0.6589；阈值 0.40 由 val 选择后冻结到 test。

当时完整机器可读评价产物曾写入 `artifacts/yolo-eye-evaluation/yolo26n_eye_100epoch/`，但当前分支的 `artifacts/` 已不再保存该目录。因此 test 数值有 committed 工作记录和评价脚本支持，但原 machine-readable evaluation artifact 当前存在归档缺口。

## 仍需补齐的复现证据

如果之后能从实际全量运行电脑或分析输出盘找到以下轻量文件，应补回仓库而不必上传全量 CSV/视频：

- 最终 full-run `run_manifest.json` 或等价参数摘要；
- 实际运行命令；
- tracker 与重新检测间隔；
- ROI expansion；
- RITnet 运行门控/阈值；
- 模型 SHA256；
- 输出字段版本与正式输出根目录说明。

在这些证据找到前，不自行创造“final.yaml”或把 08-22 runtime 默认值改称最终参数。

---

# 历史状态与方法演化

以下部分保留 08-16 至 08-22 的阶段性判断，用于解释正式路线如何从早期 ROI/瞳孔算法比较逐步转向自训练 YOLO 眼框 + tracking + RITnet。其“待实测”“未冻结”“准备继续”等措辞均应按对应日期理解，不代表 2026-08-23 当前状态。

> 08-16（Asia/Shanghai）｜NIR核心修复与历史复核完成；正式画面分型为完整脸（sub-013/9504）与双眼特写（sub-011/012/016），特写下三种完整人脸ROI全灭。已把 4 个 ROI 后端 + 4 个瞳孔算法全封装成可移植包（faceparts 为特写新候选），待数据接入实测。

## 08-16 核心入口

| 入口 | 作用 | 当时状态 |
|---|---|---|
| `src/attention_pipeline/nir/` | ROI几何、正式时间轴、连续评价、插值副轨与报告 | 活动核心实现 |
| `scripts/00-目录与映射.md` | 可运行脚本、环境和入口→核心跳转 | 当时脚本唯一入口 |
| `scripts/roi_faceparts.py` | 特写专用 ROI：直接检测 eye bbox（不依赖完整人脸） | 新候选，待数据实测 |
| [08-16-01-NIR特写ROI调研与选型.md](08-16-01-NIR特写ROI调研与选型.md) | 三后端全灭后调研 RITnet/DeepVOG/Iris/faceparts/自训练，选定 faceparts | 本目录总结 |
| [08-16-02-NIR算法封装与可移植.md](08-16-02-NIR算法封装与可移植.md) | 4 ROI + 4 瞳孔封装、画面分型、模型归集、迁移 | 本目录总结 |
| `SETUP.md`（根目录） | 环境/迁移/运行命令 | 跨模块可移植说明 |
| `configs/formal.yaml` | 正式ROI与PuReST候选配置 | `blocked_no_candidate_passed_identity_gate`，未冻结生产参数 |
| `artifacts/truth-528/` | 528眼人工真值 | 当时保留供新算法复测 |
| `artifacts/benchmark-axis-fix-review/` | 轴角修复后的阶段4/4b复核 | 当时权威单帧复核 |
| `artifacts/sequence-adapter-review/` | 新PuReST适配层历史序列复核 | 当时权威连续历史复核 |
| `artifacts/roi-selection-sub011-block1-2min/` | 正式60时点与12组入围图 | 当时阻断证据 |

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
| [08-19 眼框数据集抽帧](../工作记录/08-19-01-NIR眼框数据集抽帧工作记录.md) | 自训练 NIR 眼框数据集建立 |
| [08-21 正式数据分析交接](../工作记录/08-21-01-NIR正式数据分析交接与批量推理准备工作记录.md) | YOLO 训练完成前后的正式批量推理准备 |
| [08-22 YOLO测试评估与跨机验证](../工作记录/08-22-01-NIR-YOLO测试评估与跨机验证工作记录.md) | 冻结 test、portable runtime 与跨机计划 |

## 08-16 当时的算法判断

- PuReST仍是瞳孔连续检测候选，PuRe只用于抽样质控/诊断。
- PuReST内部搜索不等于严格px控制；正式接受范围由Python后置门执行。
- 原始椭圆`size/angle`用于绘图、mask和光度；canonical长短轴交换时角度同步旋转90°。
- 每帧×眼别保留完整成功/失败行；插值只写副轨，不跨闭眼、无脸、ROI失败或断点。
- 正式三种ROI失败的原因是输入域为双眼特写，而候选模型都以完整人脸为前提；不是简单调`corner_span`可解决。
- 新增 faceparts（`yolo-face-parts-detector`）为特写 ROI 候选：直接检测 eye bbox、不依赖完整人脸；训练域 RGB，与 NIR 有域差，须实测。
- 瞳孔算法 4 路：PuReST（venv-pupil）/ RITnet（torch）/ DeepVOG（需 keras/tf）/ Iris（仅全脸，输出虹膜非瞳孔）。

## 08-16 当时停止点

当时已把 4 个 ROI 后端 + 4 个瞳孔算法全封装成可移植包，本机不选最优；计划在数据接入后先实测 faceparts 特写检出率，再定稿生产路径。这个停止点后来已经被 NIR 专用自训练 YOLO 眼框路线取代。

## 2026-08-21 当时 NIR 工作状态

> 2026-08-21 17:30（Asia/Shanghai）｜眼框训练已启动，正式分析准备工作将在 2026-08-23 晚回珠海后继续。

- 北京已采集一百多名正式被试；单段视频约 25 分钟。
- 当时眼框模型在 AMD RX 6750 GRE 电脑上使用 CPU 训练 100 epochs。
- 计划中的正式处理架构为 `YOLO + tracking + RITnet`，目标是在 NVIDIA 电脑上完成批量推理。
- 当时尚未冻结 tracking 算法、重新检测频率、RITnet 参数和全量运行资源预算；这些项目计划以短视频基准为准。

## 08-17 至 08-22 YOLO 眼框训练与迁移文件

| 文件 | 用途 | 历史状态 |
|---|---|---|
| [08-17-01-NIR-YOLO眼框训练计划.md](08-17-01-NIR-YOLO眼框训练计划.md) | 从 Causa 方法中保留的必要步骤：NIR 专用 eye 检测、可选跟踪、标准 ROI、PuReST 衔接 | 训练计划；后续实际使用 YOLO26n |
| [08-17-02-NIR离线眼框标注指南.md](08-17-02-NIR离线眼框标注指南.md) | 离线抽帧、单类 eye 框标注、YOLO 标签、人工质量清单 | 标注操作说明 |
| [08-17-03-NIR远程YOLO训练指南.md](08-17-03-NIR远程YOLO训练指南.md) | 新电脑/远程机器需要的文件、训练命令、输出归档和迁移检查 | 训练操作说明 |
| [../工作记录/08-22-01-NIR-YOLO测试评估与跨机验证工作记录.md](../工作记录/08-22-01-NIR-YOLO测试评估与跨机验证工作记录.md) | yolo26n 本机 test 评价与跨机短视频/RITnet 验证计划 | 本机静态 test 已完成；该记录当时仍将跨机视频写为后续 |
| `runtime/nir-yolo-tracking-ritnet-v1.zip` | 可整体迁移的 YOLO26n + CSRT/KCF + ROI + RITnet GPU package | 08-22 冻结 package；当前仍保留原 ZIP/SHA256，不直接改写 |
