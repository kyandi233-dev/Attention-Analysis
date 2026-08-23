# AGENTS.md｜Attention-Analysis 仓库规则

本文件是仓库级长期工作约束。项目当前已完成正式 NIR 全量分析；后续默认任务是结构整理、可复现维护、结果复核与必要的增量开发，不得重新把项目描述为“准备进入正式分析”。

## 当前状态

- 当前唯一维护主线：`main`。
- 正式 NIR runtime：`runtime/nir-formal/`。
- 正式 NIR 流程已经全量运行：FocusWave v3.1.3 phase windows → 逐帧 YOLO26n 眼框 → ROI → RITnet batch inference → 指标/QC 输出。
- CSRT/KCF 等 ROI tracking 不属于当前正式主链；tracking 时代通过 Git 历史和 `history/tracking-era-2026-08` 追溯。
- YOLO26n 100 epochs 训练产物：`training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt`。
- 正式 runtime 冻结模型：`runtime/nir-formal/models/nir-eye-yolo26n-best.pt` 与 `runtime/nir-formal/models/ritnet-best_model.pkl`。
- 正式 runtime 冻结最终实验阶段：FocusWave v3.1.3、正式被试编号下限 31、两个正式 B block。
- 旧 v3.0 BBB SART 分析包是历史结果，不是当前最终行为分析口径。
- 正式分析输出放在仓库外独立分析目录，不把全量结果堆回 Git 仓库。

## 必读入口

开始仓库工作前优先读取：

1. `README.md`
2. `docs/README.md`
3. `docs/010-overview/README.md`
4. 对应模块目录的 `README.md`
5. `runtime/nir-formal/README.md`（涉及正式 NIR 运行口径）
6. `runtime/nir-formal/INSTALL.md`（涉及新电脑安装/迁移）
7. `docs/050-decisions/`（涉及路线变化或采纳/放弃理由）
8. 最新日期型工作记录（仅在需要追溯执行过程时）

历史文档中的“候选 / 待准入 / 准备全量 / BBB”等表述只代表当时状态；当前状态以上述入口为准。

## 文件与历史保护规则

- `docs/工作记录/` 中的日期型工作记录属于科研 provenance，默认永久保留；不得因目录整理、内容过时或已有新版而擅自删除、合并、压缩或改写。
- 删除任何工作记录必须针对具体文件获得用户明确许可，不能把一般性的“整理/清理”授权解释为删除工作记录。
- 研究过程证据和历史方法文档原则上不追溯改写。
- 移动、重命名和补充当前索引可以执行，但必须同步当前有效引用。
- 删除、合并或覆盖其他历史内容必须先获得用户明确许可；已经单独授权的删除项除外。
- 不为了目录“整齐”而改动算法逻辑、参数或正式分析结果。

## 命名与编号规则

- 数字编号主要用于 `docs/` 中需要人工阅读的说明文档，用来表达所属模块、阅读顺序、来源位置和快速定位。例如 `021-...` 属于 NIR，`051-...` 属于 decisions。
- `README.md` 是目录入口，不编号。
- `docs/工作记录/` 使用固定格式：`MM-DD-NN-标题.md`，不写年份；同一天按 `01`、`02`、`03` 递增。
- 已存在的日期型工作记录保留其历史命名与顺序，不批量美化或改成模块编号。
- Python、PowerShell、配置、测试、模型、数据文件和普通运行脚本默认不加数字前缀。只有文件本身存在真实运行顺序、阶段顺序或明确重要性顺序时才编号。
- 不使用 `00-`、`000-` 仅仅为了让文件排在前面；无顺序逻辑的目录说明统一使用 `README.md`。
- `scripts/` 保持扁平，不为分类而拆目录。

## docs 结构与职责

- `docs/010-overview/`：当前系统架构、模态关系、仓库资产与复现关系。
- `docs/020-nir/`：NIR 当前方法、运行入口和该模态需要保留的历史说明。
- `docs/030-behavior/`：行为分析入口；旧 v3.0 BBB SART 包作为历史结果保留，当前 BB 分析需按最终版本重建。
- `docs/040-rgb/`：RGB 保留接口与状态；当前关闭。
- `docs/050-decisions/`：关键技术/研究决策。
- `docs/工作记录/`：日期型实际执行过程与 provenance；原则上不追溯改写。

当前方法只保留一个 canonical 说明位置；其他文档需要提及时链接过去，不复制第二份“当前版本”。

## 当前一级目录职责

- `src/attention_pipeline/`：项目自身可复用 Python 源码。
- `scripts/`：当前仓库级任务入口；已淘汰候选脚本不继续堆在主线。
- `tools/`：独立辅助工具，不属于 `attention_pipeline` Python 包。
- `datasets/`：训练/标注数据资产与 provenance。
- `training/nir-eye-yolo/`：YOLO 眼框训练工作区、固定划分与训练结果。
- `runtime/nir-formal/`：当前正式可迁移 NIR 运行包，包含正式冻结权重、RITnet 运行源码和运行依赖说明。
- `runtime/legacy/`：旧环境快照，仅供历史复现，不作为当前正式 runtime 依赖。
- `configs/`：仓库级预实验/历史配置；当前正式 NIR 使用 `runtime/nir-formal/config.yaml`。
- `tests/`：仓库级回归；`runtime/nir-formal/tests/`：正式运行包内部最小自检。
- `venv-labelimg/`：当前暂保留的本机历史虚拟环境；未经用户明确许可不得删除。

根 `models/` 和 `artifacts/` 已按用户授权从 `main` 删除。第三方历史模型、候选权重和阶段性输出的用途、来源、淘汰与删除原因统一由 `docs/工作记录/`、decision record 和 Git 历史保存，不重新创建空壳目录。

## 运行与复现原则

- 当前正式 NIR runtime 必须尽量自包含；换电脑时以 `runtime/nir-formal/INSTALL.md` 为安装入口，以 `runtime/nir-formal/README.md` 为运行与科研口径入口。
- 不依赖个人电脑绝对路径作为唯一可复现机制；绝对数据根目录应由配置或命令行覆盖。
- 运行参数、模型权重和 phase 语义变化必须显式记录，不允许静默改变科研口径。
- 准确率/QC 阶段不得用插值掩盖失败；观测、缺失、拒绝和插值保持可区分。
- 当前行为分析必须与最终 v3.1.3 BB 实验数据一致；在重新审计前，不得把旧 BBB 分析结果冒充最终正式行为分析。
