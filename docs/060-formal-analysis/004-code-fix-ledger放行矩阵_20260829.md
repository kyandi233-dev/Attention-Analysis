# code-fix-ledger 放行矩阵（2026-08-29）

本文件把 `FocusWave-Formal-Analysis@codex/code-fix-ledger` 的代码缺陷条目映射到 `Attention-Analysis` 正式主线实现与测试。它只记录代码合同，不记录真实实验结果，也不授权真实 44-session 全量运行。

| Ledger | 缺陷/合同 | Attention-Analysis 正式实现 | 放行测试/门 | 状态边界 |
|---|---|---|---|---|
| 2.1 | NIR 临时运行不完整、模型奇异仍可能被当成功 | `nir_analysis_ready/pupil_only.py` → `nir_formal_analysis/pupil_tables.py` → `nir_pipeline_validation/pupil_validation.py`; `scientific_contract.audit_model_result` | `test_nir_formal_failure_gates.py`, `test_nir_pupil_validation_contract.py`, staged NIR regression | 模型不可估计写 `not_estimable`/失败表；不得用空成功表替代 |
| 2.2 | current/future brightness 污染刺激前窗口 | `nir_pipeline_validation.attach_visual_with_temporal_gate`; current/previous visual 分轨 | `test_nir_temporal_brightness_gate.py` | pre-stimulus current visual 必须拒绝/缺失；previous only if strictly previous |
| 2.3 | 绕过 analysis-ready/analysis-tables/Figure01–10 | pupil-only staged manifests + `nir_validate_pupil_formal.py`; Figure01–10 suite | `test_nir_pupil_validation_contract.py`, `test_nir_analysis_ready.py`, `test_nir_formal_analysis.py`, `test_nir_pipeline_validation.py`, `test_nir_publication_suite.py` | validation 不能直接读取 production；阶段 completion/manifest 缺失即拒绝 |
| 2.4 | Go omission 与 No-Go commission 混分母 | `behavior_formal/science_v3.py` 的 separate opportunity/outcome contract | `test_behavior_science_v3_contract.py` | Go/No-Go 各自机会数、分母与 SDT 输入必须透明 |
| 2.5 | 多尺度 RT/robust 指标、重复层级不足 | `science_v3.canonical_metrics`, probe sensitivity, cycle/block/session aggregates | `test_behavior_science_v3_contract.py`, `test_behavior_science_v3_extensions.py` | mean/median/SD/MAD/IQR/CV/Theil–Sen slope + omission/commission/d′/c/β；44/38/6 分轴 |
| 2.6 | Q1/Q2 重复测量模型不正确 | Q1 `MNLogit` explicit reference + participant-clustered covariance; Q2 `OrdinalGEE` by repeat participant | behavior science-v3 contract tests | Q1 名义四类不施加顺序；Q2 无 cluster 不得作为正式有序模型 |
| 2.7 | B1–B2、错误轨迹、重复组混淆 | session-internal B1/B2 pair → participant-cluster bootstrap; centered error trajectories; block×cycle GEE | `test_behavior_science_v3_extensions.py` | 两场重复只能描述/配对，不作广义 reliability/validity 声明 |
| 2.8 | 预测基线、森林图单位、QC 分母、候选证据不透明 | `behavior_formal/reporting_contract.py`; participant-disjoint folds; NIR participant-exclusive prediction | `test_behavior_reporting_contract.py`, `test_nir_pupil_validation_contract.py` | 必须报告 majority baseline、balanced accuracy、AUROC/PR-AUC；混单位禁止同轴；QC 必带分母；候选必须有 evidence source/weight/rule |
| 2.9 | mmWave loadability/缺失状态与正式报告边界 | `formal_multimodal_v2.yaml` 只记录 external producer ingest/governance；fusion merge contract | formal config/adapter gates | **生产算法权威仍在 `greenboo26/focuswave-multimodal-attention-analysis`**；本仓库不得复制 producer engine。当前治理口径只记录 44 registered / 39 loadable / 33 groups / 5 structural missing；missing ≠ 0 ≠ success |

## 跨模态融合键

`src/attention_pipeline/formal_analysis/merge.py` 与 `configs/formal_multimodal_v2.yaml` 必须一致支持：

- trial: `repeat_participant_id, session_id, block_id, trial_id`
- probe: `repeat_participant_id, session_id, block_id, probe_id, window_name`
- cycle: `repeat_participant_id, session_id, block_id, cycle_bin`
- block: `repeat_participant_id, session_id, block_id`
- session: `repeat_participant_id, session_id`
- participant_group: `repeat_participant_id`

所有键先走 canonical dtype/time normalization，再检查缺失和重复；不能用 dtype coercion 后产生的重复主键继续 merge。

## 发布门

PR 只有在以下条件同时满足后才能合回 `codex/formal-analysis-v2-portable`：

1. formal adapter + ledger contract tests 全绿；
2. pupil-only staged NIR regression 全绿；
3. 当前仓库 baseline 与 formal NIR runtime tests 全绿；
4. PR 相对 portable 为纯前进，不落后基线；
5. 不运行真实 44-session 正式数据，不以合成/CI 测试宣称科学效应或测量有效性。

PR 合并后，正式下游只继续维护 `codex/formal-analysis-v2-portable`；历史/废弃分支按照 `003-正式分析V2分支收口与历史分支处置_20260829.md` 与 NIR 退役清单处理。
