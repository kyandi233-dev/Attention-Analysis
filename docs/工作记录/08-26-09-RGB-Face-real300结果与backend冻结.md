# 08-26-09｜RGB Face real-300 结果与 backend 冻结

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-07` / `08-26-08`。本记录基于同一 `sub-031` 连续 30 秒、300 帧真实输入的 DirectML end-to-end 与 CPU-reference parity，冻结 Face **backend**。本轮不同时冻结 fps 与 primary-face 规则。

## 1. 输入与比较边界

共同输入：

```text
D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031\sub-031_face-continuous_frames.csv
```

两候选均使用 300 个相同 benchmark_index / image_path。既有 CPU reference 没有重跑。

比较维度：

1. coverage / face-count / alignment；
2. 与 CPU reference 的逐字段 parity；
3. AMD DirectML 真实 end-to-end；
4. 可保留的 raw scientific outputs；
5. 工程复杂度与后续正式 pipeline 价值。

## 2. LibreFace real-300

### Coverage

- expected frames：300；
- fresh alignment：300/300；
- gaze feature：300/300；
- alignment valid fraction：1.0。

### Fresh CPU-side prep

```text
fresh alignment                 16.7409261 s
MediaPipe gaze feature          2.4506129 s
CPU preprocess total           19.1915390 s
```

### DirectML heads + full component-summed pipeline

```text
AU input preprocess             1.7380195 s
AU DML inference                0.3298515 s
expression input preprocess     1.6378049 s
expression DML inference        0.4306281 s
gaze DML inference              0.0090089 s
DML stage total                 4.1453129 s
component-summed end-to-end    23.3368519 s
throughput                     12.8552 frames/s
```

### Parity

Fresh alignment 与旧 CPU reference 完全一致：

- alignment success flag agreement = 1.0；
- head pose：MAE/RMSE/max_abs = 0；
- 478×3 MediaPipe landmarks：MAE/RMSE/max_abs = 0。

Learned heads：

- AU intensity（3600 values）：MAE ≈ `6.48e-08`，Pearson ≈ 1；
- AU detection（3600 values）：MAE = 0，完全一致；
- gaze（600 values）：MAE ≈ `0.000330`，Pearson ≈ 1；
- expression label agreement = 1.0。

结论：LibreFace DirectML parity 可视为通过，没有数值阻断。

## 3. Py-Feat real-300

### Coverage / detection

- CPU reference rows：300；
- DML rows：300；
- matched face rows：300；
- CPU detected frames：300；
- DML detected frames：300；
- face-count mismatch：0。

RetinaFace bbox：

```text
mean IoU    0.999999524
median IoU  0.999999569
min IoU     0.999998789
```

FaceScore MAE ≈ `1.99e-09`。

### DirectML real end-to-end

当前 batch：RetinaFace=8，multitask=16。

```text
image read + preprocess         4.4861607 s
RetinaFace DML                  6.5468100 s
decode/NMS/crop                 3.4820963 s
multitask preprocess            0.2321732 s
multitask DML                   2.3267958 s
postprocess                     0.2539945 s
actual raw-frame end-to-end    17.3494252 s
throughput                     17.2916 frames/s
```

### Scientific parity

- 68-point compatibility landmarks：MAE ≈ 0.0256 px，Pearson ≈ 0.99999995；
- 20 AU（6000 values）：MAE ≈ 0.002526，Pearson ≈ 0.999681，Spearman ≈ 0.999514；
- 7 emotion probability（2100 values）：MAE ≈ 0.002581，Pearson ≈ 0.999700；
- emotion top-class agreement：299/300 = 99.67%；
- valence/arousal（600 values）：MAE ≈ 0.003995，Pearson ≈ 0.999910；
- 6DoF pose（1800 values）：MAE ≈ 0.000461，Pearson ≈ 0.99999994；
- gaze（900 values）：MAE ≈ 0.001208，Pearson ≈ 0.999789；
- original-frame 478×3 mesh（430200 values）：MAE ≈ 0.01623，Pearson ≈ 0.999999996；
- 52 blendshapes（15600 values）：MAE ≈ 0.000792，Pearson ≈ 0.999909。

这些漂移与当前 OpenCV CPU-side crop/resize 相对 PyTorch reference 的插值实现差异一致，量级很小；没有出现 coverage、bbox、科学输出结构或相关性层面的阻断。

结论：Py-Feat DirectML real-300 parity **通过**。

## 4. AMD 下速度结论发生翻转

此前 CPU reference：

- Py-Feat ≈ 0.528 image/s；
- LibreFace ≈ 4.452 image/s；
- LibreFace CPU 约快 8.43×。

但 AMD DirectML real-300：

- Py-Feat：17.2916 frames/s；
- LibreFace：12.8552 frames/s（component-summed）。

因此 Py-Feat 实测吞吐约为 LibreFace 的 **1.345×**，即约快 **34.5%**；同样 300 帧耗时从 23.3369 s 降到 17.3494 s，约减少 **25.7%**。

原因不是 LibreFace learned heads 慢，而是其正式路线的 MediaPipe alignment + gaze feature CPU prep 占据主要时间；Py-Feat 的 RetinaFace 与 multitask 已完整进入 DirectML 后，整体链路反而更快。

## 5. 信息完整性

### Py-Feat 保留

- RetinaFace decoded bbox + score；
- RetinaFace decoded 5-point landmarks；
- 20 AU probabilities；
- 7 emotion probabilities；
- valence / arousal；
- raw + canonical gaze；
- raw + canonical 6DoF pose；
- normalized 478×3 mesh；
- original-frame 478×3 mesh；
- dlib-68 compatibility view；
- 52 blendshapes；
- multi-face rows / face_rank；
- frame provenance。

### LibreFace 保留

- alignment/headpose/landmarks；
- 12 AU intensity + raw probability；
- 12 AU detection + raw probability；
- expression；
- gaze；
- 1404 gaze features。

Py-Feat 在可直接获得且后续可能有科学价值的 raw 输出上明显更完整。

## 6. Backend 决策

**冻结 Face backend：Py-Feat 2.1.1 Detectorv2 scientific core + ONNX Runtime DirectML。**

正式 AMD 路线：

```text
raw RGB frame
→ RetinaFace R34 ONNX / DML
→ priors + decode + NMS
→ isotropic square-pad crop, expand=1.2, reflection padding
→ 256 → 224 + ImageNet normalize
→ multitask scientific core ONNX / DML
→ full raw scientific outputs + canonical convenience fields
```

当前 validated batch：

- RetinaFace：8；
- multitask：16。

LibreFace 2.0 **不删除**，保留为：

- 独立 reference / cross-model uncertainty 证据；
- 将来 Py-Feat 某个被试出现异常时的备选验证路线；
- 旧 benchmark / visual review / provenance 的历史记录。

不再把 LibreFace 作为正式全量主 backend。

## 7. 当前仍未冻结的内容

本记录只冻结 backend。以下继续留待正式 pipeline 定义：

1. Face 正式采样 fps；
2. multi-face 时 primary-face 规则；
3. 正式 raw parquet 的最终字段名 / subject-id duplication / manifest schema；
4. QC flag 与 derived feature 规则；
5. body motion energy 与统一视频读取的整合方式。

因此 `053-RGB分析路线与开发边界.md` 整体仍可保持 Proposed，直到这些正式边界全部收口。

## 8. 下一步

停止 Face backend benchmark / DirectML compatibility 开发，进入正式 pipeline 工程化：

1. 把当前 `face_real_directml_pyfeat.py` 的 validated contract 整合进正式 RGB runtime；
2. 同时实现 primary-face / multi-face provenance；
3. 确定正式 fps；
4. 与 Pose / Motion / body_motion_energy 做统一视频读取与时间轴落盘；
5. 先在 representative subjects 做 schema/QC dry-run，再启动 44 被试全量。

除非后续出现新的阻断性实机证据，不再重新打开 Py-Feat vs LibreFace backend 竞争。