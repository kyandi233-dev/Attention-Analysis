# NIR Timestamp Mapping Recovery Result

状态：validation_ready；本报告只记录映射与 phase-window 验证，不上传原始 NIR 或行级时间戳。

## Canonical mapping

NIR timestamp CSV 第一列保留为 source capture counter，AVI 内部 frame index 使用有效 timestamp 行的顺序编号。capture counter gap 不再自动解释为 AVI frame gap。

## Session results

| subject | old status | capture gap | timestamp-time gap | AVI frames | valid timestamp rows | old phase mapping | sequential phase mapping | AVI gap | recovery status | minimal recovery |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| sub-056 | complete | 0 (0 frames) | 3588 | 50199 | 50199 | True | True | False | CONTROL_NO_REGRESSION | not_run (None frames; ) |
| sub-057 | complete | 0 (0 frames) | 3509 | 50036 | 50036 | True | True | False | CONTROL_NO_REGRESSION | not_run (None frames; ) |
| sub-058 | complete | 0 (0 frames) | 3516 | 51230 | 51230 | True | True | False | CONTROL_NO_REGRESSION | not_run (None frames; ) |
| sub-100 | failed | 12 (23 frames) | 3258 | 47682 | 47682 | False | True | False | RECOVERED | smoke_complete (32 frames; pytorch-cuda) |
| sub-178 | failed | 221 (390 frames) | 3990 | 54890 | 54890 | False | True | False | RECOVERED | smoke_complete (32 frames; pytorch-cuda) |

## Limits

本轮 mapping-only validation 未运行完整 YOLO/RITnet 全量分析，也未验证 Probe alignment 的下游数值，因此该字段记录为 not_checked_in_mapping_only_validation。sub-100 与 sub-178 已各完成 32 帧、pytorch-cuda、block1/block2 smoke recovery validation，确认 runtime 使用 sequential AVI frame mapping；恢复 session 如需进入后续 fullclass，必须在隔离 recovery 输出根中进行完整恢复运行并检查 completion/QC。

sub-099 不属于本任务；其 master_timeline 缺失问题不因 timestamp mapping 修复而改变。
