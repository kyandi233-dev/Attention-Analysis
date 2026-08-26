# 08-26-15｜RGB Face sub-031 primary/眼睑 QC 通过与可视化

## 1. sub-031 window-aware 修复结果

输入：optimized direct-AVI Py-Feat raw（3600 sampled frames，15 Hz）+ `face_derive_tracking_eyelid_v02.py`。

结果：

- detected face rows：3630；
- multi-face frames：30；
- 5 个 intentional dry-run windows 均成功选出 primary segment；
- primary frames：3600 / 3600；
- primary frame coverage：1.0；
- eye valid fraction：1.0；
- baseline open-reference 正确来自 900 个 baseline valid samples 的 `baseline_top30_median`；
- baseline start 的 primary track 完整覆盖 450/450 帧；同时存在 3 条短 secondary tracks（18、11、1 帧）；
- baseline end、Block1 middle、interblock middle、Block2 middle 均只有一个完整主轨迹。

因此上一版 0.25 primary coverage 已确认是“5 个不连续 dry-run window 却只允许一个 global track ID”的测试设计伪影，不是 Py-Feat/mesh/tracking 在连续片段中的失败。

## 2. 当前结论边界

sub-031 的 primary-face window-level selection 与眼睑 geometry 可以判定通过，但 `056-RGB-Face-Primary与眼睑派生规则.md` 暂不整体改为 Accepted：

- 仍需 sub-033 timestamp/capture-gap stress dry-run；
- 仍需对 sub-031 的 30 个 multi-face frames 做视觉检查；
- 仍需对 minimum-EAR / high-eyeBlink frames 做视觉检查；
- blink event threshold、bilateral rule、minimum samples、merge gap 与 `perclos80_proxy` 仍不冻结。

## 3. QC 可视化入口

新增：

```text
scripts/face_qc_visualize.py
```

该脚本只读取：

- `face_tracks.parquet`；
- `eye_features.parquet`；
- dry-run frame/sample manifests；
- 原始 RGB AVI。

不会重新运行 Py-Feat。

输出：

- `face_qc_contact_sheet.jpg`：最多 12 帧，覆盖 multi-face、最低 EAR、最高 native eyeBlink；
- `face_qc_multiface_clip.mp4`：围绕 multi-face sampled-frame 中位位置的约 10 s 标注片段；
- `face_qc_blink_extreme_clip.mp4`：围绕最低 EAR sampled frame 的约 6 s 标注片段；
- `face_qc_visualization_summary.json`。

标注包括：

- primary / other bbox + track id + FaceScore；
- primary eye/iris mesh topology；
- EAR；
- native eyeBlink；
- normalized eye openness；
- aperture/iris；
- pose / gaze；
- top emotion；
- dry-run window / phase / unix timestamp / face count。

## 4. sub-031 运行命令

```powershell
python scripts/face_qc_visualize.py `
  --tracks "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\derived-v03\face_tracks.parquet" `
  --eye "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\derived-v03\eye_features.parquet" `
  --sample-manifest "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\sub-031_face-dryrun_manifest.json" `
  --frame-manifest "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\sub-031_face-dryrun_frames.csv" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\qc-v01" `
  --fps 15
```

## 5. 下一 Gate

1. 查看 sub-031 contact sheet 与两个短视频，确认 secondary face 出现时 primary bbox/track 不跳人，并确认低 EAR / 高 eyeBlink 确实对应可见闭眼；
2. 若视觉 QC 通过，按同样 optimized direct-AVI + window-aware derived 流程跑 sub-033；
3. sub-033 通过后，冻结正式 continuous primary tracking/QC 规则，再讨论 blink event threshold 与 `perclos80_proxy`。
