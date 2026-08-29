# NIR 适配器阻塞修复：运行时 provenance 与 merge-key 契约

状态：本文件补充并在冲突处取代 `001-正式多模态V2路径与分析契约.md` 中关于 NIR 当前完成状态与固定 evidence commit 的旧表述。本修复只处理 NIR 下游适配器的 provenance 与跨表 merge key；不修改行为窗口生产器或毫米波 producer。

## 1. 运行时 provenance

正式 `nir-adapt` 输出 manifest 不再从科学配置或历史记录复制固定 commit。运行时必须分别从实际执行代码 checkout 与实际 evidence checkout 解析完整 40 位 Git `HEAD`，并记录解析方法、仓库根、origin（若存在）与工作树是否 clean。默认要求两个 checkout 都是 clean；checkout 不存在、Git 无法解析、HEAD 不是完整 SHA 或存在未提交改动时，`nir-adapt` 在创建输出目录前 fail closed，不得写入一个看似成功但实际不可审计的 commit。

证据 checkout 通过环境变量 `ATTENTION_FORMAL_EVIDENCE_REPO` 指向；变量本身只提供路径，不提供 commit。`configs/formal_multimodal_v2.yaml` 不保存固定 evidence commit，也没有固定 SHA fallback。NIR source manifest 中已有的 `source_pipeline_version`、`source_commit`、`source_selection_reason` 与输入路径继续作为 source provenance 保留；frame-level 输出继续保留 source、geometry、analysis-domain、uncertainty、temporal 等 QC 字段，run manifest 明确记录这些 QC 轨道仍被保留。

## 2. merge-key dtype / 时间契约

跨表键统一通过 `formal_analysis.join_keys.normalize_known_join_dtypes` 进入规范化入口。标识键（包括 `phase_segment`、`repeat_participant_id`、`session_id` 等）使用字符串语义并去除首尾空白，空字符串视为缺失；离散整数键使用整数语义，非整数值拒绝；时间键（包括 `unix_ms`、`absolute_onset_time`、`next_trial_onset_time` 等）统一为 UTC epoch milliseconds（UTC Unix 纪元毫秒）整数语义。

数值时间默认单位为毫秒；若源表以秒保存，调用者必须显式声明 `time_units`，程序不根据数值大小猜单位。带时区 datetime 会转换到 UTC；无时区 datetime 必须显式给出 `naive_timezone`，否则失败。数值时间和 datetime 文本混在同一列时失败。required merge key 在规范化后仍缺失时失败；重复键检查在规范化后执行，因此 `1` / `"1"`、`" sub-001 "` / `"sub-001"` 等规范化后碰撞也会被发现。

`formal_analysis.merge.merge_modalities`、正式 NIR frame adapter，以及从 PR #27 已吸收的 canonical pupil-only library 对外暴露的行为时间连接入口都使用这一契约；后者通过薄包装层规范化后再调用原 interval join，不复制第二套 pupil-only 实现。

## 3. 当前科学与数据边界

当前 NIR 生产提取状态为 44/44 session 完成，因此 44 场都可作为下游输入候选；这只表示生产可用性，不表示测量效度、QC 门或正式统计已经通过。当前聚合口径为 44 个 session、38 个当前匿名分析组，其中 6 个为双场重复组；这些计数是当前 cohort manifest 的状态，不是代码常量，不上传任何具体匿名组 ID 或身份映射。未来约 72 场接入后必须重新执行全量参与者映射、分组与分折，当前 38 组不得当作最终参与者数。

NIR 正式主线保持 pupil-only，并保留 source/QC provenance；不得恢复 PIR、`iris_outer` 或从 `hard_iris_fraction` / `soft_iris_fraction` 重建虹膜几何。OAR 仅可作为 eye opening / eyelid candidate 或 QC；它不是 PIR、blink rate、blink event 或 PERCLOS，也不得由 iris fraction 重建。engineering validation 只能表述为工程契约验证，不能升级为测量效度或正式统计结论。

行为探针窗口边界修复是独立门控，本修复不修改其代码或替代其验收。毫米波契约同样是独立门控：当前只有 39/44 可加载，且没有 ECG/RSP 外部参考验证；本修复不改变这些状态，也不允许由本次 NIR 工程测试推导毫米波生理效度。

## 4. PR #27 迁移边界

`codex/formal-analysis-v2-portable` 已通过提交 `549ca0458fade3de6cf0b995e326f41286d3b03d` 吸收 PR #27 的 canonical `src/attention_pipeline/nir_pupil_only/` library。PR #27 与当前 formal 分支已分叉，因此本修复不整支合并、不 cherry-pick 其旧 CLI/config/fixture，也不复制其中固定 evidence commit 的 manifest 逻辑；仅在当前 formal 分支已吸收的 library 上增加共享 merge-key 规范化包装层。

## 5. 回归验证范围

测试只使用临时 Git 仓库和合成 DataFrame，不读取本地原始实验数据、不包含身份映射或真实匿名组键，也不运行 44 场正式数据。覆盖真实 HEAD 解析、dirty/missing checkout fail-closed、固定 hash 不再进入正式配置/脚本、混合 dtype、显式秒→毫秒、时区统一、无时区拒绝、混合时间表达拒绝、规范化后的真实 join 成功、规范化后重复/缺失失败、NIR pupil-only 与 OAR/PIR 边界，以及 pupil—behavior interval join 的 phase/time 类型一致性。
