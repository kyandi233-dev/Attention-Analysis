# 056｜RGB Face Primary-Face 与眼睑派生规则

**Status: Proposed / Representative Dry-Run**

## 1. 决策范围

本文件不重新讨论 Face backend。Backend 已在 `054-RGB-Face-Backend冻结.md` 冻结为 Py-Feat 2.1.1 Detectorv2 scientific core + DirectML；Face 采样频率已在 `055-RGB-Face-15Hz采样频率冻结.md` 冻结为 timestamp-driven 15 Hz。

本文件只定义 dry-run 中需要验证的：

1. multi-face temporal tracking；
2. primary-face 选择；
3. raw / derived 分层；
4. EAR / eye aperture / iris normalization；
5. blink / PERCLOS-like 指标的命名与止损边界。

## 2. Raw 层：所有检测到的人脸都保留

正式 raw 不允许因为当前只分析主被试就提前删除其他人脸。每个检测 row 至少保留：

### 身份与时间

- `subject`；
- `video_frame_position`；
- `capture_frame_idx`；
- `unix_ms`；
- `target_unix_ms`；
- `sample_error_ms`；
- `dt_ms`；
- `phase` / `block`；
- trial / condition / probe / behavior context；
- timestamp/capture gap flags。

### RetinaFace / tracking

- `face_count`；
- `face_rank`；
- `FaceScore`；
- bbox；
- 5-point landmarks；
- `face_track_id`；
- `primary_face`；
- track-match diagnostic：IoU / normalized center distance / scale ratio / match score；
- `multi_face` / `primary_face_missing` / related QC flags。

### Py-Feat native scientific outputs

完整保存当前 DirectML real-300 已验证输出：

- 20 AU probability；
- 7 emotion probability；
- valence / arousal；
- raw + canonical gaze；
- raw + canonical 6DoF pose；
- normalized 478×3 mesh；
- original-frame 478×3 mesh；
- 68-point compatibility view；
- 52 blendshapes，包括 `eyeBlinkLeft` / `eyeBlinkRight`。

Raw 层不根据 head pose、FaceScore、multi-face、blink 或当前统计假设删除 row。

## 3. Tracking：先 track，后选 primary

禁止使用：

```text
每帧 FaceScore 最大者 = participant
```

因为主试短暂入镜时可能发生身份跳转。

Dry-run v0.1 采用 temporal bbox continuity：

- active track gap ≤ 2000 ms；
- IoU ≥ 0.05 **或** normalized center displacement ≤ 0.75；
- absolute log area ratio ≤ 0.80；
- match score = 0.70×IoU + 0.20×exp(-center_distance_norm) + 0.10×exp(-scale_log_ratio)；
- 同一 frame 的 track/detection 一对一 greedy assignment；
- 不满足 gate 的 detection 创建新 track。

这些数值是 representative dry-run 参数，不是最终 Accepted QC threshold。

## 4. Primary-face：用任务期长期占用选人

Primary track 不按单帧 confidence 选择。v0.1：

1. 先统计 Block1 + Block2 每个 track 的 unique frame occupancy；
2. 选择 occupancy 最大的 track；
3. tie 时按 median FaceScore；
4. 再 tie 时按 median bbox area。

理由：正式视频的真实 participant 应在两个 SART block 中长期存在；主试/其他人即使在 baseline / transition 短暂出现，也不应覆盖任务期主轨迹。

Raw 仍保留其他轨迹；`primary_face` 只是标记。

## 5. 眼睑 derived：同时保留 native 与 geometry

### 5.1 Native blink signal

Raw 中已有：

- `eyeBlinkLeft`；
- `eyeBlinkRight`。

Derived convenience：

- `native_eyeBlink_mean`。

不在 raw 层阈值化。

### 5.2 EAR

使用 MediaPipe-compatible 478 topology 的 6-point eye geometry：

- right eye：33, 160, 158, 133, 153, 144；
- left eye：362, 385, 387, 263, 373, 380。

EAR：

\[
EAR = \frac{\|p_2-p_6\| + \|p_3-p_5\|}{2\|p_1-p_4\|}
\]

保存：

- `ear_left`；
- `ear_right`；
- `ear_mean`。

### 5.3 Eye aperture / iris diameter

478-point mesh 同时包含 iris ring：

- right iris：469–472；
- left iris：474–477。

保存：

- `eye_aperture_px_left/right`；
- `iris_diameter_px_left/right`；
- `aperture_iris_left/right/mean`。

其中 iris diameter 取同侧 4 个 iris-ring 点的最大 pairwise 2D distance；aperture 为两个对应 upper/lower eyelid distance 的均值。

与 EAR 相比，`aperture_iris` 使用 iris 尺寸作为 scale reference，更接近“眼睑相对眼球尺度”的开度 proxy；但它仍不是经过人工标定的 pupil-occlusion percentage。

## 6. 个体基线归一化

考虑个体眼裂大小、相机距离、眼镜与面部几何差异，不直接跨被试对 raw EAR / aperture 使用同一个绝对阈值。

v0.1 open reference：

```text
baseline valid aperture/iris
→ 取 top 30%
→ 取该集合 median
```

如果 baseline 有效样本不足 30 个，则 fallback 到全分析可用样本 top 30% median，并明确记录 source。

得到：

```text
eye_openness_norm = aperture_iris / open_reference
closure_fraction = clip(1 - min(eye_openness_norm, 1), 0, 1)
```

## 7. PERCLOS-like 指标必须明确叫 proxy

经典 PERCLOS 常被定义为一段时间内眼睑 ≥80% 闭合的时间比例，但文献中对“eye opening percentage / closure threshold”的具体几何实现并不完全一致。

因此当前只生成：

- `closure80_proxy_left/right`：`eye_openness_norm <= 0.20`；
- 未来 rolling 字段命名固定为 `perclos80_proxy_*`；
- 默认候选 rolling window=60 s。

在没有人工标定 / representative visual review 前，不把该 proxy 写成未经限定的 `PERCLOS`。

## 8. Blink event threshold 暂不冻结

15 Hz 已足以支持 blink event / coarse duration 的 temporal sampling，但代表被试之间 EAR / `eyeBlink*` 分布还没检查。

因此 dry-run v0.1 只保存连续信号和 quantiles，不先冻结：

- EAR threshold；
- eyeBlink blendshape threshold；
- minimum consecutive samples；
- bilateral agreement rule；
- event merge gap。

这些规则必须在 sub-031 + gap/multi-face stress subject 的连续输出上查看后再 Accepted。

## 9. 与 NIR 的关系

NIR 已保存 `fullclass_ocular_aperture_ratio_median/p90` 等基于 RITnet 可见眼球区域的开度 proxy。它和 RGB EAR 不是同一个定义，不直接互相替代。

后续可以把：

- RGB EAR / aperture-iris / eyeBlink；
- NIR ocular aperture ratio / ocular fraction / iris/sclera visibility

作为跨模态 convergent evidence，用于 blink/eye-closure QC，而不是把其中任一方强行当另一方的 ground truth。

## 10. Representative dry-run 通过条件

至少检查：

1. source nominal fps ≥15 Hz；
2. timestamp-driven sample error / gaps 正常；
3. Face coverage；
4. multi-face frame 数；
5. track fragmentation；
6. Block1+Block2 primary track occupancy；
7. primary track 在 baseline / transition 主试入镜时不跳人；
8. EAR / aperture-iris valid fraction；
9. native eyeBlink / EAR / normalized openness distribution；
10. 极端 head pose / glasses / partial eye closure 是否产生明显假信号。

推荐第一批：

- `sub-031`：当前 Face/Pose/Motion reference subject；
- `sub-033`：已知存在 capture/timestamp gap，作为时间轴 stress subject。

如这两人没有覆盖明显低 Face 质量，再从 dataset-level sparse QC 选第三个低质量 subject；不为了凑人数盲目跑第三个。
