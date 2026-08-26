# 08-26-13｜RGB Face dry-run primary 窗口分段修复

## 1. 触发原因

`sub-031` 第一档优化后的 Py-Feat raw 已完成 tracking / eyelid v0.1 派生。摘要显示：

- detected face rows = 3630；
- track count = 8；
- multi-face frames = 30；
- `primary_track=7`；
- `primary_face_present_fraction=0.25`；
- `eye_valid_fraction=0.25`；
- primary track 7 在一个任务窗口覆盖 900 帧；另一个任务窗口的 track 5 同样覆盖 900 帧；
- open-eye reference 因 primary track 不覆盖 baseline，被迫使用 `all_valid_top30_median_fallback`。

这不是 Py-Feat 或窗口内部 tracking 失败，而是 dry-run 输入本身由 5 个相隔很远的连续窗口拼成：baseline start、baseline end、Block1 middle、interblock middle、Block2 middle。v0.1 tracker 的 `max_track_gap_ms=2000` 不可能跨几十秒到数分钟的人为省略区间维持同一个 temporal track ID，因此同一个 participant 在不同 dry-run window 中必然获得不同 segment ID。

## 2. 诊断

当前摘要中两个任务窗口各有一个 900/900 帧稳定轨迹，FaceScore 中位数约 0.9996；同时全体只有 30 个 multi-face frames。这更符合“每个连续窗口内 participant 稳定存在，另有少量短暂第二人脸”，而不是主轨迹在窗口内部频繁碎裂。

因此 v0.1 的错误在于：

```text
5 个不连续 dry-run windows
→ 当成一条连续时间序列 track
→ 最后只允许一个 global primary_track
```

这会把任意一个 900 帧任务窗口错误地当成整个 3600 帧 dry-run 的唯一 participant，导致 coverage=0.25，并使 baseline normalization 失去真实 baseline。

## 3. 修复原则

新增：

```text
scripts/face_derive_tracking_eyelid_v02.py
```

v0.2 对 dry-run 明确采用 window-aware 语义：

1. 每个 `dryrun_window` 内独立执行原 v0.1 temporal bbox tracker；
2. 每个窗口的 local track ID 映射成全局唯一 segment ID，但不声称时间追踪跨过了被人为省略的区间；
3. 每个窗口独立选择长期占用率最高的 primary-face segment；tie 仍按 median FaceScore、median bbox area；
4. 5 个窗口选出的 primary segments 在眼睑派生层视为同一个逻辑 participant 流；
5. baseline open-reference 因此可以重新使用 baseline_start + baseline_end 的真实 baseline 样本。

正式 full-video runner 不使用这一“跨窗口逻辑合并”语义。正式源视频是连续时间流，仍采用连续 temporal tracking，再从任务期长期 occupancy 选择主被试。

## 4. 为什么不直接把不同窗口 track ID 硬连接

窗口之间缺失的真实视频长达几十秒至数分钟，因此不存在足够的 temporal evidence 去证明两个 track ID 是同一条连续轨迹。v0.2 只在 dry-run 评估层把“每个窗口的 dominant participant segment”作为逻辑 primary，不伪造跨缺失区间的 tracking continuity。

这也保留了后续 QC 的透明性：`face_tracks.parquet` 仍能看到每个窗口自己的 segment ID；`eye_features.parquet` 才把这些窗口 primary segment 汇总成被试层连续测量样本集合。

## 5. 当前眼睑信号的初步状态

v0.1 只覆盖单个 900 帧 primary segment，因此其分布暂不用于冻结阈值。但已有信号说明数据不是常数：

- EAR mean median ≈ 0.299；
- EAR 1% quantile ≈ 0.170；
- native eyeBlink mean median ≈ 0.089；
- native eyeBlink mean 95% ≈ 0.302；
- native eyeBlink mean 99% ≈ 0.393。

这些分布存在明显闭眼尾部，但 blink event threshold、minimum consecutive samples、bilateral agreement、merge gap 仍保持未冻结状态。

## 6. v0.2 通过预期

重新运行 v0.2 后优先检查：

1. `tracking.primary_frame_coverage` 应从 0.25 显著恢复，理想情况下接近 1.0；
2. 每个 dry-run window 应有一个 primary segment，且任务窗口应接近 100% coverage；
3. `eyelid.primary_face_present_fraction` / `eye_valid_fraction` 应同步恢复；
4. open-reference source 应优先变为 `baseline_top30_median`，而不是 fallback；
5. 30 个 multi-face frames 中 primary segment 不应切换到短暂第二人脸；
6. 后续用 annotated QC keyframes / clip 视觉确认 5 个窗口选择的是同一 participant。

## 7. 当前状态

**Window-aware primary fix: Implemented, waiting for sub-031 v0.2 rerun.**

不需要重新运行 Py-Feat。直接读取已经保存的：

```text
D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml-v02\pyfeat_dml_raw.parquet
```

重新计算 tracking / eyelid 即可。
