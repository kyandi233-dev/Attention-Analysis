# 00｜scripts 目录与映射

> 2026-08-23｜正式 NIR 全量分析已经完成。`scripts/` 当前保存的是多个阶段积累下来的命令入口、诊断、评价和历史比较工具；它不是第二套核心算法库，也不能仅根据旧脚本名称判断当前正式路线。

## 目录职责

- `src/attention_pipeline/`：可复用核心逻辑，以及较早阶段的 NIR review / benchmark / sequence 实现。
- `scripts/`：可直接运行的命令入口、诊断、评价、历史比较和一次性研究工具。
- `runtime/nir-yolo-tracking-ritnet-v1/`：2026-08-22 冻结的可移植 YOLO26n + tracking + RITnet 实现包；其中 `run_pipeline.py` 是当前仓库已核验到的完整该路线实现。
- `tests/`：回归测试，不处理正式数据。
- `artifacts/`：有限历史证据目录；当前分支不保存全量正式输出。
- `docs/工作记录/`：历史计划、执行过程与决策证据。

当前不为了目录整齐把 `scripts/` 拆成多层子目录，因为大量历史命令和文档已经引用这些路径；除非后续确认收益明显，否则保持扁平更利于复现。

## 当前项目事实

正式 NIR 主链已经实际完成全量运行，其逻辑为：

```text
NIR video
    ↓
YOLO 周期性重新检测双眼 bbox
    ↓
tracking / 上一帧 ROI 更新中间帧
    ↓
裁剪双眼 ROI
    ↓
RITnet 瞳孔 / 虹膜分割
    ↓
时序 QC 与指标输出
```

`best.pt` 已经存在并使用过；不是“待训练”或“待生产准入”的未来资产。

当前仓库已经核验到完整 YOLO + tracking + RITnet 实现位于：

```text
runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py
```

该脚本支持 YOLO、CSRT/KCF、周期重检测、ROI 扩展、RITnet、逐帧/逐眼状态、时间戳和 `--full-video`；runtime 中的 YOLO 权重与 `yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt` 为同一 Git blob，RITnet 权重也与 `models/RITnet-master/best_model.pkl` 为同一 Git blob。

但是，runtime 包创建时仍被定义为跨机短视频准入包。当前分支没有保存正式全量运行产生的最终 `run_manifest.json` 或全量输出目录，也没有 2026-08-22 之后的 committed 工作记录可用于反推出最终运行参数。因此不能把包内默认 `CSRT / 10帧重检测 / 30%横向扩展 / 45%纵向扩展` 自动写成全量最终冻结参数。

## 现有脚本分类

| 脚本 | 当前角色 | 说明 |
|---|---|---|
| `run_all_backends.py` | 历史统一比较入口 | 用于 ROI 后端 × 瞳孔算法组合比较，不代表当前全量正式主链 |
| `run_roi_selection.py` | 历史 ROI 选型总控 | 早期 ROI 入围/尺度/门控比较 |
| `roi_mediapipe.py` | 历史 ROI 候选 | 基于 MediaPipe FaceLandmarker 的完整人脸路线 |
| `roi_yunet.py` | 历史 ROI 候选 | YuNet 人脸检测路线 |
| `roi_yolo.py` | 历史 ROI 候选 | 旧 YOLO-face bbox 路线，不是后续训练的 YOLO26n 双眼模型 |
| `roi_faceparts.py` | 历史 ROI 候选 | face-parts 特写候选路线 |
| `roi_common.py` | 共享脚本实现 | 为多种历史 ROI 入口提供公共处理逻辑；可复用核心逻辑应逐步保持在 `src/` |
| `roi_compare.py` | 历史/诊断 | 少量帧 ROI 候选对比，不是正式全量入口 |
| `nir_sequence_detect.py` | 瞳孔/序列适配器 | PuRe/PuReST 等历史序列检测适配 |
| `deepvog_pupil.py` | 历史比较适配器 | DeepVOG 瞳孔分割比较 |
| `iris_landmark.py` | 历史比较适配器 | MediaPipe iris 路线，输出语义与瞳孔不同 |
| `nir_detect_batch.py` | 历史基准 | 阶段 4 六算法单帧基准与调参 |
| `compare_nir_history_review.py` | 历史复核 | 比较历史修复前后产物 |
| `gate1_contract_check.py` | 回归/协议检查 | 早期 Gate1 契约检查，不产生当前正式全量结果 |
| `evaluate_yolo_eye_test.py` | YOLO 评价 | 对训练后的眼框模型执行 val/test 评价；属于模型评价工具 |
| `build_formal_experiment_recommendation.py` | 行为报告工具 | 根据行为输出生成正式实验程序修改建议 |
| `extract_eye_dataset.py` | 数据集工具 | 眼框训练数据抽取/整理相关工具 |

## 配置使用注意

`configs/formal.yaml` 是早期 ROI 入围/候选阶段的历史兼容配置，文件中保留 candidate/blocked 状态。旧脚本为了复现可以继续引用它，但它不能用来判断当前项目是否完成正式分析，也不应被默认解释为全量正式运行时的最终配置。

同样，`runtime/nir-yolo-tracking-ritnet-v1/config.yaml` 是 2026-08-22 portable package 的准入默认配置。除非找到正式 full-run 的 manifest 或运行命令，否则它只能说明“包当时如何默认运行”，不能证明“全量最终如何运行”。

## 历史工具为什么继续保留

本项目经历过多轮 ROI、瞳孔算法、序列连续性和模型选择比较。即使某条路线后来被淘汰，相关脚本仍然可以回答“为什么没选它”“当时如何比较”“某个历史 artifact 如何生成”等复现问题。因此整理原则是：

1. 保留历史工具及其路径。
2. 在索引中明确它属于当前路线还是历史路线。
3. 核心共享逻辑继续收敛到 `src/attention_pipeline/`。
4. 不因正式路线确定就删除旧脚本。
5. 如将来确需移动/重命名，先检查文档和代码引用；涉及删除旧路径时先取得用户明确同意。
