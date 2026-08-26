# NIR Timestamp Mapping Recovery Result

状态：recovery_review_ready；本报告只记录映射、恢复产物与下游对齐验证，不上传原始 NIR 或行级时间戳。

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

## Full recovery and Probe alignment

在隔离正式 recovery 输出根中，sub-100 与 sub-178 均完成完整 NIR video recovery、RITnet full-class extension 和 Probe alignment，未重跑原有 69 个 complete session。

| subject | formal video completion | fullclass rows | fullclass backend | QC images | Probe alignment | probe rows | missing eye blocks |
|---|---|---:|---|---:|---|---:|---|
| sub-100 | complete, 39205/39205 | 68180/68180 | onnxruntime-cuda, cuda:0 | 142 | complete | 160 | none |
| sub-178 | complete, 43080/43080 | 84810/84810 | onnxruntime-cuda, cuda:0 | 158 | complete | 160 | none |

两场恢复均保留 fullclass CSV、summary、manifest、QC index/PNG、alignment trial/probe outputs 和 coverage 文件。Probe alignment 报告中的 coverage、边界截断和内部 NIR gap 为描述性 QC，不在本任务中冻结排除阈值。

本轮 mapping-only validation 的 32 帧 smoke 已进一步由上述完整恢复覆盖；恢复过程未发现 AVI decode/frame gap。capture counter gap 与 timestamp-time gap 仍作为独立质量信息保留，不解释为 AVI 缺帧。

sub-099 不属于本任务；其 master_timeline 缺失问题不因 timestamp mapping 修复而改变。

当前 cohort 计数因此由 69/72 增至潜在 71/72。下一步应重新生成 matched NIR cohort，并单独评估是否按扩大后的 cohort 重跑 NIR v1；本任务不自动重跑既有 68-session/1360-probe NIR v1 结果。
