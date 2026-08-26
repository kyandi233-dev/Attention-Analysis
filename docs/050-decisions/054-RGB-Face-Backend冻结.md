# 054｜RGB Face Backend 冻结

**Status: Accepted**

## 决策

RGB Face 正式 AMD backend 冻结为：

**Py-Feat 2.1.1 Detectorv2 scientific core + ONNX Runtime `DmlExecutionProvider`**。

当前 validated model split / batch：

- RetinaFace R34 ONNX：batch 8；
- multitask scientific core ONNX：batch 16。

LibreFace 2.0 不删除，保留为 reference / cross-model uncertainty / fallback validation 路线，但不再作为 44 被试正式全量主 backend。

## 证据

同一 `sub-031` 连续 30 秒 / 300 帧 real-input 验证：

### Coverage

Py-Feat：300/300 CPU reference rows、300/300 DML rows、300 matched face rows，face-count mismatch=0；bbox mean IoU=0.999999524，min IoU=0.999998789。

LibreFace：300/300 fresh alignment，300/300 gaze features，alignment valid fraction=1.0。

### CPU-reference parity

Py-Feat：

- AU20 MAE≈0.002526，Pearson≈0.999681；
- emotion7 MAE≈0.002581，Pearson≈0.999700，top-class agreement=99.67%；
- V/A MAE≈0.003995，Pearson≈0.999910；
- pose6d MAE≈0.000461，Pearson≈0.99999994；
- gaze MAE≈0.001208，Pearson≈0.999789；
- 478×3 original-frame mesh MAE≈0.01623 px，Pearson≈0.999999996；
- 52 blendshapes MAE≈0.000792，Pearson≈0.999909。

LibreFace：

- fresh headpose / landmarks 与 CPU reference 完全一致；
- AU intensity MAE≈6.48e-08；
- AU binary detection 完全一致；
- gaze MAE≈0.000330；
- expression label agreement=1.0。

两者 parity 都通过。

### AMD real end-to-end

Py-Feat raw-frame actual end-to-end：17.3494 s / 300，**17.2916 fps**。

LibreFace fresh CPU prep + DirectML heads component-summed：23.3369 s / 300，**12.8552 fps**。

Py-Feat 吞吐约为 LibreFace 的 1.345×（约快 34.5%）；300 帧总耗时约少 25.7%。

### 信息完整性

Py-Feat 一次性保留 RetinaFace bbox/score/5-point、20 AU、7 emotion、V/A、raw+canonical gaze、raw+canonical 6DoF pose、478 normalized/original mesh、68-point compatibility view、52 blendshapes、multi-face rows 与 frame provenance。

LibreFace 保留 alignment/headpose/landmarks、12 AU intensity/detection 与 raw probability、expression、gaze、1404 gaze features。

在 parity 已通过的前提下，Py-Feat 同时具有更高 AMD real end-to-end throughput 与更完整 raw scientific schema，因此作为正式 backend。

## 正式链条

```text
raw RGB
→ RetinaFace R34 ONNX / DirectML
→ priors + decode + NMS
→ expand=1.2 isotropic square / reflection crop
→ 256 → 224 full-field resize + ImageNet normalize
→ multitask scientific core ONNX / DirectML
→ full raw outputs + canonical convenience fields
```

## 不属于本决策的内容

本决策只冻结 backend，不同时冻结：

- 正式 Face fps；
- primary-face / multi-face rule；
- 正式 raw schema 的最终列名；
- QC / derived feature 规则；
- body_motion_energy / Pose / Motion 统一视频读取方式。

这些仍在 `053-RGB分析路线与开发边界.md` 的整体 Proposed 阶段继续收口。

详细实机记录见：

`docs/工作记录/08-26-09-RGB-Face-real300结果与backend冻结.md`。