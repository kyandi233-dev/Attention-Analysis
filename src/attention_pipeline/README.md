# 00｜源码目录与映射

## 当前状态（2026-08-23）

`src/attention_pipeline/` 保存项目中可复用的 Python 核心逻辑，但它是多个研究阶段逐步演化形成的源码包。当前不能把包内现有 `cli.py` 或旧 `nir/` 模块自动等同于“正式 NIR 全量分析最终入口”。

项目现实状态是：**YOLO26n 眼框模型训练和正式 NIR 全量分析已经完成**；当前仓库整理工作的目标之一，就是把历史阶段代码、正式行为分析代码、NIR benchmark/sequence 代码与最终全量流程的真实职责重新标清。

## 顶层结构

| 组件 | 当前职责 |
| --- | --- |
| `cli.py` | 早期统一 CLI；主要连接 approval-gate、`behavior/`、NIR review / benchmark / sequence 等历史阶段流程。**不是已核验的正式 NIR 全量最终入口。** |
| `config.py` | 配置加载与路径解析 |
| `contracts.py` | 跨模块基础契约 / 常量 |
| `io.py` | 通用 I/O 辅助 |
| `metadata.py` | 运行 metadata / source id 等 |
| `protocol.py` | 正式实验协议与被试/文件结构相关逻辑 |
| `validation.py` | 文件、时间轴和实验环境验证 |
| `behavior/` | 较早建立的行为证据层：逐试次提取、block 指标、时间窗证据及阶段性 reporting |
| `behavior_formal/` | 后续建立的**正式 BBB SART 行为分析**：extract、metrics、stats、figures、report |
| `nir/` | NIR 历史与可复用核心：ROI 几何、人工 review、六算法 benchmark、sequence/evaluation、正式数据时间轴辅助 |

当前包中没有独立的 `rgb/` 或 `cross_modal/` 运行模块；对应内容目前主要仍属于文档/后续分析层，而不是这里的已实现 Python 子包。

## `behavior/` 与 `behavior_formal/`

这两个目录当前不能仅因为名字相近就直接合并。

`behavior/` 是较早阶段建立的行为证据与阶段性报告体系，当前 `cli.py` 仍直接依赖：

```text
behavior.extract
behavior.evidence
behavior.reporting
```

`behavior_formal/` 则是后续独立形成的正式 BBB SART 行为分析模块，包含：

```text
extract.py
metrics.py
stats.py
figures.py
report.py
```

因此目前先保留两套路径。后续是否合并，需要先比较数据输入、指标口径、调用入口和已有文档引用，不能仅根据目录命名判断为重复实现。

## `nir/` 当前实际边界

当前 `nir/` 中可见模块包括：

| 文件 | 主要职责 |
| --- | --- |
| `formal.py` | 正式实验 NIR 视频 / timestamp / behavior timeline 的路径与时间窗对齐 |
| `roi.py` | 眼 ROI 几何、仿射、椭圆与 IoU 等可复用函数 |
| `metrics.py` | NIR 基础指标函数 |
| `review.py` | ROI / 人工真值 / representative review 等历史审批与数据构建流程 |
| `benchmark.py` | 历史六算法单帧 benchmark 与参数评估 |
| `sequence.py` | 历史连续序列构建与检测流程 |
| `sequence_eval.py` | 历史 sequence 状态机、插值、连续性评估与报告 |

需要特别注意：当前 `nir/` 包目录中本身**没有以独立模块形式实现 YOLO26n eye detector、CSRT/KCF tracking、RITnet segmentation 和 full-video orchestration**。这些功能的完整 portable implementation 已经核验位于：

```text
runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py
```

该 runner 实际实现：

```text
YOLO26n
→ tracking / 周期重检测 / tracker 失败回退
→ 单眼 ROI 扩展与裁剪
→ RITnet
→ frames.csv / eyes.csv / summary.json / run_manifest.json
```

并支持 `--full-video`。其打包 YOLO 与训练目录正式 `best.pt` 的 Git blob 完全相同，打包 RITnet 与 `models/RITnet-master/best_model.pkl` 也完全相同。

因此现在可以确定的是：**正式路线的完整 YOLO + tracking + RITnet 代码血缘已经定位到 runtime portable package，而不是旧 `src/attention_pipeline/cli.py`。**

仍未完全闭合的是另一层 provenance：当前 GitHub 分支尚未找到最终正式全量运行输出中的 `run_manifest.json` 或等价命令记录，因此不能仅依据 2026-08-22 portable package 的默认参数，反推最终全量实际冻结的 tracker、重检测间隔、ROI 扩展等参数。

## 当前整理原则

1. 不把旧 approval-gate 状态重新描述成当前项目状态。
2. 不把 `cli.py` 自动称为正式生产入口。
3. 不因 `behavior/` 与 `behavior_formal/` 名字相近就直接合并或删除。
4. 不因正式路线已经确定就删除 benchmark / review / sequence 等历史研究代码。
5. `runtime/nir-yolo-tracking-ritnet-v1/` 作为冻结 portable implementation 保持独立，不为了“统一源码”直接搬入 `src/`。
6. 下一步如继续追正式 NIR provenance，应优先寻找最终 full-run 的 `run_manifest.json` / 命令 / 输出目录，而不是继续猜测入口。

---

## 历史说明

> 2026-08-13 15:35（Asia/Shanghai）｜当时源码按通用契约、行为证据和 NIR 测量分层，RGB/跨模态没有运行实现。

原始阶段索引当时主要记录：

- `cli.py`：统一命令与审批门；
- `config.py`：配置加载和摘要；
- `protocol.py` / `validation.py`：正式协议、被试、文件、时间轴和双环境验证；
- `behavior/`：逐试次、block 和窗口证据；
- `nir/`：ROI、PERCLOS、代表性抽样、盲标、六算法基准与参数调优。

该记录作为源码演化历史保留，但不再代表 2026-08-23 的完整源码结构和项目阶段。
