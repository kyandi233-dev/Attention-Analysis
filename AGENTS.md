# AGENTS.md｜Attention-Analysis 仓库规则

本文件是仓库级长期工作约束。项目当前已完成正式 NIR 全量分析；后续默认任务是结构整理、可复现维护、结果复核与必要的增量开发，不得重新把项目描述为“准备进入正式分析”。

## 当前状态

- 当前整理主线：`main`。
- 正式 NIR runtime：`runtime/nir-formal/`。
- 正式 NIR 流程已经全量运行：FocusWave v3.1.3 phase windows → 逐帧 YOLO26n 眼框 → ROI → RITnet batch inference → 指标/QC 输出。
- YOLO26n 100 epochs 训练产物：`training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt`。
- 正式 runtime 内冻结副本：`runtime/nir-formal/models/nir-eye-yolo26n-best.pt`。
- 正式分析输出放在仓库外独立分析目录，不把全量结果堆回 Git 仓库。

## 必读入口

开始仓库工作前优先读取：

1. `README.md`
2. `项目总览与架构.md`
3. `docs/README.md`
4. 对应模块目录的 `README.md`
5. `runtime/nir-formal/README.md`（涉及正式 NIR 运行口径）
6. `runtime/nir-formal/INSTALL.md`（涉及新电脑安装/迁移）
7. 最新日期型工作记录（仅在需要追溯决策时）

历史文档中的“候选 / 待准入 / 准备全量 / 尚未冻结”等表述只代表当时状态；当前状态以上述入口为准。

## 文件与历史保护规则

- 日期型工作记录、研究过程证据和历史方法文档属于科研 provenance，原则上不追溯改写。
- 移动、重命名和补充当前索引可以执行，但必须同步当前有效引用。
- 删除、合并或覆盖历史内容必须先获得用户明确许可；已经单独授权的删除项除外。
- 不为了目录“整齐”而改动算法逻辑、参数或正式分析结果。

## 命名规则

- 仓库和子目录的导航/说明文件统一使用 `README.md`，不再用 `00-目录与映射.md` 只为排序。
- 独立文档不使用无语义的 `000-` / `00-` 前缀。
- 只有确实表达阶段、阅读顺序、实验顺序或日期的文件才保留数字前缀。
- 日期型历史工作记录保持既有命名，不批量美化。
- `scripts/` 当前保持扁平，不为分类而拆目录。

## 目录职责

- `src/attention_pipeline/`：项目自身可复用 Python 源码。
- `scripts/`：命令入口、诊断与历史比较脚本。
- `tools/`：独立辅助工具，不属于 `attention_pipeline` Python 包；`tools/labelimg/` 保存可重建的 LabelImg 标注环境定义与补丁。
- `datasets/`：原始训练/标注数据资产。
- `training/nir-eye-yolo/`：YOLO 眼框训练工作区、固定划分与训练结果。
- `models/external/`：第三方源码/算法仓库。
- `models/historical/`：曾参与历史候选/比较、但不属于当前正式 NIR pipeline 的独立模型文件。
- `runtime/nir-formal/`：当前正式可迁移 NIR 运行包，包含正式冻结权重和运行依赖说明。
- `runtime/legacy/`：旧环境快照，仅供历史复现，不作为当前正式 runtime 依赖。
- `artifacts/`：已提交的历史评估/QC/审批证据，不是正式全量输出；当前有意保持扁平。
- `docs/`：方法、架构、决策、模态文档和历史工作记录。
- `configs/`：仓库级行为/历史配置；当前正式 NIR 使用 `runtime/nir-formal/config.yaml`，`configs/formal.yaml` 仅为历史兼容配置。
- `tests/`：仓库级回归；`runtime/nir-formal/tests/`：正式运行包内部最小自检。
- `venv-labelimg/`：当前暂保留的本机历史虚拟环境；不视为长期可移植资产，未经用户明确许可不得删除。

## 运行与复现原则

- 当前正式 NIR runtime 必须尽量自包含；换电脑时以 `runtime/nir-formal/INSTALL.md` 为安装入口，以 `runtime/nir-formal/README.md` 为运行与科研口径入口。
- 不依赖个人电脑绝对路径作为唯一可复现机制；绝对数据根目录应由配置或命令行覆盖。
- 运行参数、模型权重和 phase 语义变化必须显式记录，不允许静默改变科研口径。
- 准确率/QC 阶段不得用插值掩盖失败；观测、缺失、拒绝和插值保持可区分。
