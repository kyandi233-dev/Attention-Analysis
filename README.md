# Attention-Analysis

## 当前项目状态

> 2026-08-23｜正式 NIR 全量分析已经完成。当前工作重点是仓库整理、研究资产归档、路径规范化与复现入口维护，不是“准备进入正式 NIR pipeline”。

- 一百多名正式被试的数据采集已经完成。
- NIR 双眼 ROI 数据集 v1 已完成并用于训练 YOLO26n；YOLO26n 100 epochs 训练与冻结 test 评价已经完成。
- 冻结 test：7 名被试、85 张图、169 个眼框；mAP50 = 0.9913，mAP50-95 = 0.6589。
- YOLO 眼框权重 `best.pt` 已实际用于后续流程；它不是未来待训练资产。
- 正式 NIR 流程已经完成全量运行。

当前 NIR 主流程：

```text
NIR video
    ↓
YOLO 周期性重新检测眼睛 bbox
    ↓
tracking / 上一帧 ROI 更新中间帧
    ↓
裁剪双眼 ROI
    ↓
RITnet 瞳孔 / 虹膜分割
    ↓
时序 QC 与指标提取
    ↓
行为时间窗 / NIR 指标输出
```

## 当前入口

- 项目总览与架构：[`项目总览与架构.md`](项目总览与架构.md)
- 文档总目录：[`docs/README.md`](docs/README.md)
- NIR 文档：[`docs/010-nir/README.md`](docs/010-nir/README.md)
- 方法说明：[`docs/060-methods/README.md`](docs/060-methods/README.md)
- 架构说明：[`docs/070-architecture/README.md`](docs/070-architecture/README.md)
- 决策记录：[`docs/080-decisions/README.md`](docs/080-decisions/README.md)
- 脚本索引：[`scripts/README.md`](scripts/README.md)
- 源码索引：[`src/attention_pipeline/README.md`](src/attention_pipeline/README.md)
- runtime 说明：[`runtime/README.md`](runtime/README.md)
- 历史证据产物：[`artifacts/README.md`](artifacts/README.md)
- 历史工作记录：[`docs/工作记录/README.md`](docs/工作记录/README.md)

## 当前已核验的 YOLO + tracking + RITnet 实现

完整 frozen implementation 当前位于：

```text
runtime/NIR-formal/run_pipeline.py
```

`runtime/NIR-formal/` 包含 YOLO26n → CSRT/KCF tracking → ROI → RITnet 的完整处理链，并支持 `--full-video`。包内 YOLO 权重与 `yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt` 为同一 Git blob；包内 RITnet 权重与 `models/RITnet-master/best_model.pkl` 也为同一 Git blob。

需要区分“实现已经核验”和“最终运行参数已经核验”。当前 Git 分支仍没有保存正式全量运行的最终 `run_manifest.json` / 全量输出目录，因此不能从 runtime 默认配置反推全量最终使用的 tracker、重检测间隔和 ROI 扩展比例。

## 配置与历史资产边界

`configs/formal.yaml` 是早期正式 ROI 入围/候选阶段留下的历史兼容配置，不代表当前正式流程的唯一配置源。

`runtime/NIR-formal/config.yaml` 记录 2026-08-22 portable package 建立时的默认准入参数；在没有 final full-run manifest 的情况下，同样不能直接当作全量最终冻结参数表。

`artifacts/` 保存有限、可审计的过程证据；其中 `artifacts/archive/` 保存已经封存的历史 Gate、preview、ROI comparison、smoke test 等证据。正式全量输出本体继续保存在仓库外部。

## 仓库整理原则

1. 当前正式路线与历史候选路线在命名和入口文档中明确分开。
2. `src/attention_pipeline/` 保存可复用核心代码；`scripts/` 保存命令入口、诊断和历史比较工具；frozen runtime 单独保留。
3. 没有真正编号序列意义的目录入口使用 `README.md`，不使用 `00-` / `000-` 伪序号。
4. 真正按阶段或文档序列组织的正文保留连续编号，例如 `061/062/063`、`071`、`081` 与正式 SART 的 `000/001/002`。
5. `docs/工作记录/` 保留研究过程与历史决策，不为了适配当前目录名批量改写历史正文。
6. `venv-labelimg/` 当前仍保留；后续是否从 Git 移除另行决定。
