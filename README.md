# Attention-Analysis

## 当前项目状态

> 2026-08-23｜正式 NIR 全量分析已经完成。当前工作重点是仓库整理、研究资产归档、路径规范化与复现入口维护，不是“准备进入正式 NIR pipeline”。

- 一百多名正式被试的数据采集已经完成。
- NIR 双眼 ROI 数据集 v1 已完成并用于训练 YOLO26n；YOLO26n 100 epochs 训练与冻结 test 评价已经完成。
- 冻结 test：7 名被试、85 张图、169 个眼框；mAP50 = 0.9913，mAP50-95 = 0.6589。
- YOLO 眼框权重 `best.pt` 已实际用于后续流程；它不是未来待训练资产。
- 正式 NIR 流程已经完成全量运行。当前仓库整理的目标是把正式运行资产、历史候选方案、诊断脚本和工作记录区分清楚，避免旧文档把项目状态退回到“候选/准备阶段”。

当前 NIR 主流程的逻辑为：

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

- 项目总览与架构：[`000-项目总览与架构.md`](000-项目总览与架构.md)
- 文档总目录：[`docs/00-目录与映射.md`](docs/00-目录与映射.md)
- NIR 文档：[`docs/010-nir/`](docs/010-nir/)
- 脚本索引：[`scripts/00-目录与映射.md`](scripts/00-目录与映射.md)
- 源码索引：[`src/attention_pipeline/00-目录与映射.md`](src/attention_pipeline/00-目录与映射.md)
- runtime 说明：[`runtime/README.md`](runtime/README.md)
- 历史工作记录：[`docs/工作记录/`](docs/工作记录/00-目录与映射.md)

## 当前已核验的 YOLO + tracking + RITnet 实现

当前仓库已经核验到完整实现位于：

```text
runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py
```

这个 frozen portable package 包含完整的 YOLO26n → CSRT/KCF tracking → ROI → RITnet 处理链，并支持 `--full-video`。包内 `nir-eye-yolo26n-best.pt` 与 `yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt` 是同一个 Git blob；包内 RITnet 权重与 `models/RITnet-master/best_model.pkl` 也是同一个 Git blob。

需要区分“实现已经核验”和“最终运行参数已经核验”。该 runtime 包在 2026-08-22 创建时仍定位为跨机短视频准入包；当前 Git 分支没有保存正式全量运行生成的最终 `run_manifest.json` / 全量输出目录，也没有 2026-08-22 之后的 committed 工作记录。因此目前不能从 package 默认配置反推全量最终使用的 tracker、重检测间隔和 ROI 扩展比例。

## 配置与历史资产边界

仓库根目录的 `configs/formal.yaml` 是早期正式 ROI 入围/候选阶段留下的兼容配置，文件内部仍记录旧的 full-face ROI 阻断状态。它用于历史复现和旧脚本兼容，**不代表已经完成全量 NIR 分析时的当前项目状态，也不应被当成当前正式流程的唯一配置源**。

`runtime/nir-yolo-tracking-ritnet-v1/config.yaml` 则记录 portable package 在 2026-08-22 建立时的默认准入参数。它比旧 `configs/formal.yaml` 更接近后来 YOLO + tracking + RITnet 路线，但在没有 final full-run manifest 的情况下仍不能直接当作全量最终冻结参数表。

同理，`scripts/`、`artifacts/`、`models/` 中保留了多轮 ROI 候选、基准测试、诊断和历史模型。整理原则是保留研究 provenance，不因当前路线已经确定而删除历史证据；需要删除任何历史文件时单独确认。

## 仓库整理原则

1. 当前正式路线与历史候选路线必须在命名和入口文档中明确分开。
2. `src/attention_pipeline/` 保存可复用核心代码；`scripts/` 保存命令入口、诊断和历史比较工具；frozen portable runtime 单独保留，不强拆进入 `src/`。
3. `docs/工作记录/` 保留研究过程与历史决策，不批量改写历史正文。
4. 训练数据集、训练权重、第三方模型、runtime 包和历史 artifacts 作为研究资产保留并注明角色。
5. 输出数据继续与代码仓库分离，避免把全量正式分析产物直接纳入 Git；但关键运行 manifest / 参数摘要应在能够核验后补充为轻量 provenance。
