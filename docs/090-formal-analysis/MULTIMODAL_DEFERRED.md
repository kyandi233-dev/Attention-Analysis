# 多模态融合当前状态：DEFERRED / NOT RELEASE READY

> 当前多模态融合**暂时搁置**。不要因为 `configs/formal_multimodal_v2.yaml`、merge helper 或历史融合脚本存在，就把它描述为已完成正式融合。

## 为什么暂时搁置

当前单模态科学层仍在收口：Behavior 的 omission taxonomy、问卷/participant identity 与 endpoint freeze 需要真实数据验证；NIR pupil-only candidate endpoint 也需要真实数据 scientific freeze；RGB raw-first producer 已存在于历史 RGB 分支，但正式 downstream 正在迁入 `codex/formal-analysis-v2-portable`。在这些输入契约未冻结前继续扩展 fusion，会把上游变化传播成大量重复返工。

## 当前不完善之处

1. `configs/formal_multimodal_v2.yaml` 仍包含旧的模态/身份口径，不能视为最终 release config；
2. `src/attention_pipeline/formal_analysis/merge.py`、`join_keys.py` 等提供合并基础设施，但没有证明 Behavior/NIR/RGB/mmWave 当前最终 schema 全部通过真实数据 parity；
3. RGB downstream endpoint 尚未真实数据冻结；
4. modality missingness、不同采样率/窗口的支持度、视觉/头动等 confound 的最终融合策略尚未完成；
5. 参与者级重复测量推断与 participant-exclusive prediction 虽有公共约束，但尚未在完整多模态表上完成 representative smoke / 44-session release run；
6. 最终 multimodal feature registry、冗余控制、模型失败表、外部验证/交叉验证和正式图表 contract 尚未完成闭环。

## 相关文件位置（后期从这里恢复）

- 配置：`configs/formal_multimodal_v2.yaml`
- 合并基础：`src/attention_pipeline/formal_analysis/merge.py`
- 合并键/时间规范：`src/attention_pipeline/formal_analysis/join_keys.py`
- cohort/identity：`src/attention_pipeline/formal_analysis/cohort.py`、`identity_questionnaire.py`
- provenance：`src/attention_pipeline/formal_analysis/provenance.py`
- 当前 Behavior 正式 runner：`scripts/sart_formal_analysis.py`
- 当前 NIR 正式 runner：`scripts/nir_formal_pipeline.py`
- RGB 正式 downstream：`scripts/rgb_formal_pipeline.py`（迁入后以此为准）

历史 RGB producer 资产仍主要位于 `rgb-amd`、`rgb-nvidia` 和 `codex/rgb-nvidia-formal-pipeline-v1`，不得整体 merge 回正式主线；只允许审计后逐项迁移所需 schema/逻辑。

## 重新开启融合前的放行门

必须至少满足：三个单模态 analysis-ready/table schema 冻结；participant_key/cohort parity 真实验证；每个单模态 candidate endpoint 决策表完成；模态 missingness/coverage 明确；严格防止 probe/post-event future leakage；推断与预测分离；participant-exclusive folds；targeted pytest + representative real-session smoke；最终 fusion provenance manifest 可追溯。

当前状态字段：`deferred_not_release_ready`。