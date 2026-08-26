# 08-26-11｜RGB Face 正式 QC 可视化要求

> 2026-08-26｜分支：`rgb-dev`｜承接 Face backend=Py-Feat DirectML、15 Hz 正式采样与 primary-face/eyelid dry-run。

## 1. 用户新增要求

正式 RGB Face 全量分析除 raw / derived / summary 数据外，应额外落盘**少量、可人工快速复核的带标注示意图或短视频**。目标不是保存大量重复图像，而是为方法说明、QC、异常定位和后续展示保留直观证据。

## 2. 推荐输出层级

### A. 每被试轻量 keyframe contact sheet

从少量代表性时点抽取带 overlay 的静态帧，建议覆盖 baseline、Block1、interblock、Block2，以及必要时的 multi-face / track reacquisition / eyelid closure 事件。建议每被试约 8–16 张，不做大规模逐帧图片落盘。

Overlay 至少包含：

- primary face bbox 与 `face_track_id`；
- 非 primary face 的 bbox（若存在），与 primary 用不同样式；
- FaceScore；
- 眼睛/虹膜关键 mesh；
- gaze；
- head pose；
- top AU / emotion（用于可视 QC，不代替 raw 数据）；
- `eyeBlinkLeft/Right`、EAR / aperture-iris / closure proxy；
- timestamp、phase/block 与必要的 QC flag。

### B. 每被试一个很短的固定 QC clip

建议正式全量每被试自动输出 1 个约 10–20 s 的带标注短视频，优先选 Block1 或 Block2 中间稳定片段。该视频只作为快速检查：bbox/track 是否稳定、mesh 是否贴合、gaze/head-pose 方向是否合理、blink/closure 曲线是否与肉眼一致。

### C. 异常触发 clip（条件式）

仅当出现以下情况时再额外落盘短片，避免无意义存储：

- multi-face；
- primary track reacquisition / fragmentation；
- 连续低 FaceScore / face missing；
- 极端 blink/closure/perclos proxy；
- timestamp/capture gap 附近；
- 其他正式 QC 规则触发。

## 3. 成本与实现边界

这些可视化必须从已经保存的 raw / derived 输出和原 AVI 重建，不重新运行 Py-Feat 模型。因此：

- 不增加昂贵模型推理；
- 不把 overlay 图像当作科学 raw 数据；
- 可视化可在正式推理完成后单独生成；
- 正式 raw schema 仍按信息保留原则完整落盘；
- 图/视频仅作 QC、方法展示和论文示意。

## 4. 当前 dry-run 额外发现

`sub-031` 15 Hz dry-run 已生成 3600 帧。抽样本身有效，但当前 `capture_gap_before` 定义为 sampled `capture_frame_idx` 差值 >3，会把 15 Hz timestamp-nearest sampling 在 30 fps 源视频上的正常 4-frame 跳跃误标为 capture gap。正式化前需要把 QC 改成区分：

1. `window_boundary_before`：dry-run 人为分段；
2. `source_position_step`：采样源帧跳步；
3. `capture_index_missing_before`：capture index 相对于 video source position 的真实额外缺失；
4. `temporal_gap`：真实时间连续性异常。

现有 3600 张 dry-run 图片无需重抽；该问题只影响 QC 字段解释，不影响已选帧或后续 Py-Feat 推理。

## 5. 正式输出建议命名

```text
sub-XXX_face_raw.parquet
sub-XXX_face_tracking.parquet
sub-XXX_face_eye_features.parquet
sub-XXX_face_qc_keyframes.jpg
sub-XXX_face_qc_clip.mp4
sub-XXX_face_qc_events_manifest.csv
sub-XXX_manifest.json
```

如异常触发额外 clip，可采用：

```text
sub-XXX_face_qc_event_multiface_001.mp4
sub-XXX_face_qc_event_track-reacquire_001.mp4
sub-XXX_face_qc_event_closure_001.mp4
```

## 6. 当前状态

本记录只冻结“正式分析要保留轻量 annotated QC 可视化”这一要求；具体选帧规则、视频长度和 overlay 版式在 representative dry-run 后实现并确认，不阻塞当前 sub-031 Py-Feat dry-run 推理。
