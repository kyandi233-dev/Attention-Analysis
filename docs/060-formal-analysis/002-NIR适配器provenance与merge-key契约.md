# NIR legacy adapter provenance 与 merge-key 契约

> **当前定位更新（2026-08-30）：** 本文件记录 `formal_multimodal_analysis.py nir-adapt` 这条历史 CSV adapter 的 provenance、join dtype 与 merge-key 合同。它仍保留工程审计价值，但**不是当前 staged NIR 的 production-authoritative 入口**。当前权威 NIR 链为 `scripts/nir_formal_pipeline.py -> configs/nir_analysis_ready.yaml -> configs/nir_formal_analysis.yaml -> pupil-only validation`；总合同见 `001`，修复后复审见 `009`。

## 1. legacy adapter 的运行时 provenance

legacy `nir-adapt` 输出 manifest 不从科学配置或历史记录复制固定 commit。运行时分别从实际执行代码 checkout 与实际 evidence checkout 解析完整 40 位 Git `HEAD`，并记录解析方法、仓库根、origin（若存在）与工作树是否 clean。默认要求两个 checkout 都是 clean；checkout 不存在、Git 无法解析、HEAD 不是完整 SHA 或存在未提交改动时，该 legacy adapter 在创建输出目录前 fail closed，不得写入一个看似成功但实际不可审计的 commit。

证据 checkout 通过环境变量 `ATTENTION_FORMAL_EVIDENCE_REPO` 指向；变量本身只提供路径，不提供 commit。`configs/formal_multimodal_v2.yaml` 不保存固定 evidence commit，也没有固定 SHA fallback。legacy NIR source manifest 中已有的 `source_pipeline_version`、`source_commit`、`source_selection_reason` 与输入路径继续作为 source provenance 保留。

上述机制只证明 legacy adapter 的可追溯性，不允许它替换当前 staged pupil-only analysis-ready chain。

## 2. merge-key dtype / 时间契约

跨表键统一通过 `formal_analysis.join_keys.normalize_known_join_dtypes` 进入规范化入口。标识键使用字符串语义并去除首尾空白，空字符串视为缺失；离散整数键使用整数语义，非整数值拒绝；时间键统一为 UTC epoch milliseconds 整数语义。

数值时间默认单位为毫秒；若源表以秒保存，调用者必须显式声明 `time_units`，程序不根据数值大小猜单位。带时区 datetime 会转换到 UTC；无时区 datetime 必须显式给出 `naive_timezone`，否则失败。数值时间和 datetime 文本混在同一列时失败。required merge key 在规范化后仍缺失时失败；重复键检查在规范化后执行，因此规范化后的碰撞也会被发现。

`formal_analysis.merge.merge_modalities`、legacy NIR frame adapter，以及当前 pupil-only library 暴露的行为时间连接包装层可复用这一 dtype/time 合同。但**复用 join helper 不等于 historical merge scaffold 已被放行为正式 multimodal fusion**。

## 3. 当前 NIR 科学与 availability 边界

当前 governed queue 快照为 44 sessions / 38 participant groups / 6 double-session repeat groups。NIR 生产提取在当前队列中已有 44/44 source 完成记录；这只代表当前 NIR source availability，不代表 pupil 测量效度、QC、正式统计或其他模态也必须 44/44。

participant/session identity 与 modality availability 必须分开：RGB 或 mmWave 缺失不能改变 NIR/Behavior participant mapping；反之，某个未来 NIR session 缺失时也只能记录 NIR `source_missing/not_estimable`，不能从全项目 governed cohort 删除该 session。

NIR 正式主线保持 pupil-only：

- 不恢复 PIR、`pupil_to_iris`、`iris_outer`；
- 不从 `hard_iris_fraction` / `soft_iris_fraction` 重建虹膜几何；
- producer 已有 `fullclass_ocular_aperture_ratio_median/p90` 可保留为 eye-opening QC candidate；
- OAR 不是 PIR、iris diameter、MediaPipe EAR、blink event 或 PERCLOS；
- OAR 当前不自动成为 formal physiological endpoint。

当前权威 manifest/config 语义已经拆成：

```text
PIR / iris geometry refused
ocular-aperture QC preserved
ocular-aperture formal endpoint = false
```

旧 `pir_oar_allowed` / `pir_oar_refused` 混合字段只允许作为历史 provenance 出现，不得再次定义当前科学合同。

## 4. historical merge scaffold 的边界

旧 merge scaffold 仍保留 `repeat_participant_id` 等历史 key 以维持兼容测试。当前 canonical participant grouping 是 `participant_group_id`；`session_id` 是场次 locator，不能被当作 participant。

多模态 fusion 当前 `disabled_deferred`。因此 `merge-audit` 只能检查 schema/key/dtype 问题，不能授权 production merge，也不能生成“正式融合已经完成”的结论。

未来真正做 paired multimodal analysis 时应：

```text
governed cohort
-> per-modality availability masks
-> explicit common-available subset
-> participant-disjoint validation
-> preprocessing inside training folds
-> report coverage and performance together
```

不得默认采用 complete-case 方式永久缩减整个项目 cohort。

## 5. PR #27 与 canonical pupil-only library

`codex/formal-analysis-v2-portable` 已吸收 PR #27 中需要的 canonical `src/attention_pipeline/nir_pupil_only/` library。PR #27 与当前 formal 分支已分叉，因此不整支合并、不复制其旧 CLI/config/fixture，也不把其中历史固定 evidence commit 逻辑恢复到当前主线。

当前 staged NIR 通过自己的权威入口消费 pupil-only contract；legacy adapter 仅保留历史可追溯性和兼容测试。

## 6. 回归验证范围

当前相关自动化验证覆盖：

- provenance / dirty checkout fail-closed；
- known join dtype 与时间单位规范化；
- participant-group identity invariants；
- staged pupil-only contract；
- OAR QC 保留与 PIR/iris geometry 拒绝；
- package-level pupil-only validation authority；
- authoritative NIR manifest 不再传播 `pir_oar_*` 混合语义。

这些测试是代码/合同验证，不读取本地完整正式原始数据，也不能替代 representative real-data smoke、44-session 全量运行或科学结论。
