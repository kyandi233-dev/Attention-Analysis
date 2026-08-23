# 08-23-02｜scripts审计与Tracking历史版本冻结工作记录

> 文档性质：仓库整理工作记录  
> 日期：08-23  
> 当天序号：02  
> 当前维护主线：`main`  
> 状态：进行中

## 一、本轮目标

本轮继续执行仓库整理计划，重点处理两件事：

1. 在不丢失旧 YOLO + Tracking + RITnet 仓库状态的前提下，将 tracking 时代从当前维护主线中退出；
2. 审计 `scripts/`，区分当前仍有直接用途的脚本、历史复现/审计脚本，以及已经退出当前正式路线的算法候选脚本。

本轮不删除工作记录，不删除历史脚本，不删除历史模型源码。

## 二、Tracking 历史版本冻结

当前正式 NIR 主线已经改为：

```text
FocusWave v3.1.3 phase windows
→ 逐帧 YOLO26n
→ eye ROI
→ RITnet batch inference
→ QC / metrics / formal outputs
```

CSRT / KCF 等 ROI tracking 不再属于正式主链。

旧 `codex/v2-YOLO+Tracking+RInet` 分支仍包含一批 `main` 没有的独立提交，因此不能在没有稳定历史入口的情况下直接删除。

当前 GitHub 连接器没有创建 tag 的写接口。为避免旧版本丢失，本轮创建：

```text
history/tracking-era-2026-08
```

该历史分支直接从旧 `codex/v2-YOLO+Tracking+RInet` 创建，用于冻结 tracking 时代的完整仓库快照。

### 后续分支处理原则

理想最终状态仍是：

```text
main
+ 历史 tag（冻结旧版本）
```

如果后续在 GitHub UI 为该历史快照创建正式 tag，则可以再删除 `history/tracking-era-2026-08` 和原 `codex/v2-YOLO+Tracking+RInet` 分支，使 Branch 页面最终只保留 `main`。

`codex/nir-formal-gpu-v3` 已确认没有 `main` 不包含的独立提交，因此不需要另设历史快照；后续可直接删除分支。

## 三、scripts 当前问题

当前 `scripts/` 并不是一套需要依次执行的 19 个正式步骤，而是项目开发过程中逐步积累的脚本工具箱，混合了：

- 当前数据/模型/行为分析入口；
- 评估和回归工具；
- 08-16 阶段 ROI / pupil 多算法候选；
- PuRe / PuReST / DeepVOG / Iris 等历史方法适配器；
- 历史比较和报告生成工具。

因此“scripts 数量很多”主要来自历史研究路线累积，而不是当前正式 pipeline 本身复杂。

当前正式 NIR 运行入口不在 `scripts/`，而在：

```text
runtime/nir-formal/
```

## 四、scripts 初步分类结果

### 1. 当前仍有明确直接用途

- `extract_eye_dataset.py`：眼框训练数据抽取与数据集 provenance。
- `evaluate_yolo_eye_test.py`：冻结 YOLO test 的模型评估复现。
- `sart_formal_analysis.py`：正式 SART 行为分析入口。

### 2. 回归 / 审计 / 历史复现

- `gate1_contract_check.py`
- `compare_nir_history_review.py`
- `roi_check.py`
- `roi_compare.py`
- `nir_detect_batch.py`
- `nir_sequence_detect.py`
- `build_formal_experiment_recommendation.py`

这些脚本不应被误认为当前正式 NIR 主入口，但仍可能用于核对旧结果、解释历史工作记录或复现阶段性结论。

### 3. 已退出当前正式路线的算法候选脚本

- `roi_mediapipe.py`
- `roi_yunet.py`
- `roi_yolo.py`
- `roi_faceparts.py`
- `deepvog_pupil.py`
- `iris_landmark.py`
- `run_roi_selection.py`
- `run_all_backends.py`
- `roi_common.py`

这些脚本目前暂时保留。后续是否从 `main` 删除，要与第三方模型源码清理一起判断，避免留下已经缺依赖的伪入口。

## 五、scripts 当前入口调整

本轮新增：

```text
scripts/README.md
```

作为当前 scripts 导航，并明确：

- 正式 NIR 入口位于 `runtime/nir-formal/`；
- 各脚本的当前状态；
- 历史算法脚本不代表当前正式路线；
- 后续删除旧模型源码时必须同步处理相关历史脚本。

原：

```text
scripts/00-目录与映射.md
```

暂时保留。该文件记录 08-16 多算法候选阶段，其中“当前方向 / 待选型”等内容已经过时，因此不再作为当前入口。由于它是已有文件，本轮未擅自删除。

## 六、RITnet 源码位置结论

RITnet 与已经淘汰的 DeepVOG、PuReST、YuNet 等不同，它属于当前正式 NIR 主链。

当前正式 runtime 已包含：

```text
runtime/nir-formal/ritnet/License.md
runtime/nir-formal/ritnet/densenet.py
runtime/nir-formal/ritnet_runtime.py
runtime/nir-formal/models/ritnet-best_model.pkl
```

因此正式运行所需的冻结 RITnet 源码继续放在 `runtime/nir-formal/` 最合理，不应移入 `tools/`。

`tools/` 用于独立辅助工具；RITnet 则是正式 runtime 的直接算法依赖。

当前 `runtime/nir-formal/ritnet/` 已保留 license 和正式运行所需核心网络源码。下一步仍需确认并补充：

- 上游项目来源 URL；
- 使用的上游版本 / commit；
- 如果 runtime 中源码经过裁剪或修改，应记录与上游的差异。

这些 provenance 信息补齐后，可以再判断 `models/external/ritnet/` 的完整上游仓库是否有必要继续保留。

## 七、后续整理顺序

下一步建议按以下顺序继续：

1. 审计 `runtime/nir-formal/ritnet/` 的来源、版本和修改记录；
2. 审计 `models/external/` 与 `models/historical/`，建立“当前需要 / 历史可删 / 必须保留”的清单；
3. 将历史脚本与旧模型依赖对应起来，再决定哪些历史脚本可以退出 `main`；
4. 审计 `artifacts/` 内每个目录，区分真正需要长期保存的研究证据和可重新生成的临时输出；
5. 最后再执行经用户明确批准的删除。

工作记录本身不参与上述删除整理。
