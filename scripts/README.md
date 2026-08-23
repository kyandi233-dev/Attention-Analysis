# Scripts

`scripts/` 保存仓库级任务入口、评估工具、诊断工具和历史方法复现脚本。它**不是当前正式 NIR 全量分析的运行入口**。

当前正式 NIR 运行入口：

```text
runtime/nir-formal/
```

当前正式 NIR 主链：

```text
FocusWave v3.1.3 phase windows
→ 逐帧 YOLO26n 眼框
→ eye ROI
→ RITnet batch inference
→ QC / metrics / formal outputs
```

## 当前仍有明确用途的脚本

| 脚本 | 用途 | 当前定位 |
|---|---|---|
| `extract_eye_dataset.py` | 从正式 NIR 数据抽取眼框训练数据 | 数据集 provenance / 可复现 |
| `evaluate_yolo_eye_test.py` | 对冻结 test 执行 YOLO 眼框评估 | 模型评估 / 可复现 |
| `sart_formal_analysis.py` | 正式 SART 行为分析入口 | 当前行为分析 |

这些脚本可以继续存在于主线，因为它们仍直接对应当前项目的数据、模型或正式行为分析链路。

## 回归、审计与历史复现工具

| 脚本 | 用途 | 当前定位 |
|---|---|---|
| `gate1_contract_check.py` | 早期协议、时间轴、几何和行为契约检查 | 回归 / 历史审计 |
| `compare_nir_history_review.py` | 比较历史 NIR 阶段结果 | 历史复核 |
| `roi_check.py` | 早期 ROI 检查 | 历史诊断 |
| `roi_compare.py` | 多种 ROI 候选的离散比较 | 历史诊断 |
| `nir_detect_batch.py` | 阶段4六算法单帧基准 | 历史 benchmark |
| `nir_sequence_detect.py` | PuRe / PuReST 连续序列适配 | 历史方法复现 |
| `build_formal_experiment_recommendation.py` | 生成早期正式实验修改建议 | 历史行为报告生成 |

这些文件不代表当前正式运行路线。保留它们的价值主要是复核旧实验、解释历史工作记录或验证旧结论。

## 已退出当前正式路线的算法脚本

以下脚本属于 2026-08 中旬 ROI / pupil 多算法候选阶段：

| 脚本 | 历史角色 | 当前状态 |
|---|---|---|
| `roi_mediapipe.py` | MediaPipe 完整人脸 ROI 候选 | 已退出正式路线 |
| `roi_yunet.py` | YuNet 完整人脸 ROI 候选 | 已退出正式路线 |
| `roi_yolo.py` | YOLO-face bbox ROI 候选 | 已退出正式路线 |
| `roi_faceparts.py` | YOLO face-parts 特写 ROI 候选 | 已退出正式路线 |
| `deepvog_pupil.py` | DeepVOG 瞳孔分割候选 | 已退出正式路线 |
| `iris_landmark.py` | MediaPipe iris 关键点候选 | 已退出正式路线 |
| `run_roi_selection.py` | 多 ROI 候选选择总控 | 已退出正式路线 |
| `run_all_backends.py` | ROI × pupil 多算法统一调度 | 已退出正式路线 |
| `roi_common.py` | 上述历史 ROI 脚本共享实现 | 历史支持代码 |

这些脚本暂不删除。后续先完成第三方模型源码与历史依赖审计，再逐项判断：

1. 是否仍需要在 `main` 中直接运行；
2. 是否只需由历史冻结版本保存；
3. 是否已有工作记录、decision 和历史版本足以复现其研究结论。

任何删除仍需用户针对相应文件/文件组明确授权。

## RITnet 的位置

RITnet 与上述淘汰候选不同：它属于**当前正式 NIR 主链**。

正式运行所需的冻结实现位于：

```text
runtime/nir-formal/ritnet/
runtime/nir-formal/ritnet_runtime.py
runtime/nir-formal/models/ritnet-best_model.pkl
```

因此 RITnet 不放入 `tools/`。`tools/` 用于 LabelImg 一类独立辅助工具，而 RITnet 是正式分析 runtime 的直接算法依赖。

`models/external/ritnet/` 中的完整上游仓库是否继续保留，将在确认 runtime 已保存必要源码、license、来源和版本 provenance 后单独决定。

## 与历史模型资源的关系

早期候选脚本曾依赖 `models/external/` 和 `models/historical/` 中的第三方源码或模型。后续如果从 `main` 删除已淘汰第三方模型源码，必须同步审计这些脚本；不能在不说明的情况下留下看似可运行、实际上已经缺少依赖的入口。

## 历史版本

旧 `codex/v2-YOLO+Tracking+RInet` 开发线已额外冻结为：

```text
history/tracking-era-2026-08
```

该历史 ref 用于保存 tracking 时代的完整仓库快照。当前维护与新开发仍只在 `main` 进行。

## 旧目录说明

`scripts/00-目录与映射.md` 是 08-16 多算法候选阶段的历史目录说明，其中部分“当前方向”“待选型”等表述已经过时。它暂时保留，不作为当前入口；本文件 `README.md` 为当前 scripts 导航。
