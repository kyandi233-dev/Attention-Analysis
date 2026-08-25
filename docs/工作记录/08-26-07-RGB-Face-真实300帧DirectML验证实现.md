# 08-26-07｜RGB Face 真实 300 帧 DirectML 验证实现

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-06-RGB-Face-PyFeat-DirectML-v03诊断收口.md`。本记录定义 Gate 0/1 通过后的 real-input 300 帧验证。既有 CPU reference 不重跑；Face backend 仍未冻结。

## 1. 阶段起点

当前已经具备：

- LibreFace 2.0 Gate 0：PASS；
- LibreFace DirectML Gate 1：PASS；
- Py-Feat 2.1.1 Gate 0：PASS；
- Py-Feat RetinaFace + multitask DirectML Gate 1：PASS；
- 两边均已通过实际 ORT profile 证明存在 DML kernel、没有观察到 CPU kernel fallback；
- 既有同一 300 帧 CPU reference 已经保存，禁止为了本阶段重复运行 CPU learned-model benchmark。

因此本阶段只回答两个新问题：

1. ONNX / DirectML 在**真实输入和完整前后处理**下与既有 CPU reference 是否保持可接受 parity；
2. 从真实 RGB frame 到完整科学输出的 AMD end-to-end 性能如何。

Synthetic Gate-1 images/s 不再用于 backend 决策。

## 2. 固定共同输入

继续使用现有连续 30 s / 300 sample：

```text
D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031\
```

其中共享 manifest：

```text
sub-031_face-continuous_frames.csv
```

两候选必须使用完全相同的 `benchmark_index` / `image_path`，不得重新抽帧。

## 3. 新输出目录

本阶段不覆盖原 CPU reference，也不覆盖 Gate 0/1：

```text
D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\
├── libreface-prep\
├── libreface\
├── pyfeat\
└── parity\
```

所有脚本会自行创建所需子目录。

## 4. 为什么 LibreFace 分成 CPU prep + DirectML heads

LibreFace 当前 Python reference 并不是所有环节都属于 learned GPU model：

```text
raw frame
→ get_aligned_image / MediaPipe alignment + head pose + landmarks   [CPU reference side]
→ aligned image
├─ AU joint learned model                                           [DirectML]
├─ expression learned model                                         [DirectML]
└─ MediaPipe refine_landmarks=True → 468 xyz → 1404 features        [CPU]
   → gaze MLP                                                       [DirectML]
```

为了既保持当前 reference 语义，又不把旧 CPU model inference 时间混进新结果，本阶段：

1. 在 `attention-face-libreface` fresh 重跑**必要 CPU 前处理**；
2. 不调用旧 PyTorch AU / expression / gaze learned heads；
3. 把 fresh CPU prep 的实测时间写入 prep manifest；
4. 在独立 DirectML 环境消费这些 prep 输出；
5. LibreFace end-to-end 报告明确标为 `component_summed_end_to_end`：fresh CPU prep process + DirectML process 各组件时间之和，不伪装成单进程 wall-clock。

这不是重复 CPU benchmark；它是新 DirectML pipeline 必须承担的真实 CPU 前处理成本。

## 5. 为什么 Py-Feat 从 raw frame 直接运行

Py-Feat 的正式候选链条可以直接在 DirectML runtime 中实现：

```text
raw RGB 720×1280
→ RGB [0,255] - [123,117,104]
→ RetinaFace R34 ONNX / DML
→ priors + bbox / 5-landmark decode
→ pre-NMS > 0.02
→ NMS IoU 0.4
→ final score >= 0.5
→ isotropic square-pad crop, expand=1.2, reflection padding, 256×256
→ full-field 256→224
→ ImageNet normalize
→ multitask scientific core ONNX / DML
→ canonical pose/gaze/mesh convenience decode
```

当前 batch 候选来自 clean Gate 1：

- RetinaFace：`8`；
- multitask：`16`。

两个模型不要求强制使用同一个 batch。

### Py-Feat interpolation 边界

当前 DirectML runtime 不安装 PyTorch/torchvision；CPU-side crop/resize 使用 OpenCV 实现与 Py-Feat 2.1.1 相同的几何定义。几何 contract 已严格对应：中心、`max(w,h)*1.2`、不 clamp、reflection padding、256 chip、全场 224 resize。

但 OpenCV bilinear/remap 与 PyTorch `grid_sample` / torchvision antialias resize 可能存在小数值差异。因此：

- bbox/coverage parity 主要验证 RetinaFace ONNX + decode/NMS；
- 如果 bbox parity 很高、但 multitask scientific outputs 出现系统性小漂移，首先检查 interpolation/preprocess，不直接判定模型或 DirectML 数值失败；
- 若为了逼近 PyTorch preprocessing 需要明显增加工程复杂度，而 LibreFace 已满足科学需求，则允许在此止损并选择 LibreFace。

## 6. 信息保留：本轮一次性保存，不先过滤

### LibreFace

fresh prep 保存：

- alignment success/error；
- aligned image path；
- headpose JSON；
- landmarks JSON；
- MediaPipe refine-landmarks 468×xyz 展平后的 1404 gaze features；
- gaze feature success/error。

DirectML 保存：

- AU intensity raw probability；
- AU detection raw probability；
- AU intensity=`probability*5`；
- AU detection=`probability>=0.5`；
- 8-class expression scores + argmax label；
- gaze yaw/pitch。

### Py-Feat

保存：

- frame provenance / benchmark index；
- multi-face `face_rank`；
- RetinaFace decoded raw bbox + confidence；
- RetinaFace decoded 5-point landmarks；
- square crop affine convenience bbox；
- 20 AU probabilities；
- 7 emotion probabilities；
- valence / arousal；
- raw gaze `[yaw,pitch]` + canonical gaze；
- raw 6-element pose + canonical `[Pitch,Roll,Yaw,X,Y,Z]`；
- normalized 478×3 mesh；
- original-frame convenience 478×3 mesh；
- Py-Feat canonical dlib-68 compatibility landmark view；
- 52 blendshapes。

Identity 按 Gate-0 scientific-core 决策继续排除：本项目为单被试实验视频测量，identity embedding 不参与当前科学问题，也不值得为此扩展 DirectML pipeline。

## 7. 执行命令

### 7.1 拉取当前代码

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-dev"
git pull --ff-only
git status --short --branch
```

### 7.2 LibreFace fresh CPU prep

```powershell
conda activate "D:\CondaEnvs\attention-face-libreface"

python scripts/face_real_prepare_libreface.py `
  --benchmark-dir "D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\libreface-prep"
```

主要输出：

```text
libreface_dml_alignment.parquet
libreface_dml_gaze_features.npy
libreface_dml_gaze_feature_index.parquet
libreface_dml_prep_manifest.json
aligned\...
```

### 7.3 LibreFace DirectML learned heads

```powershell
conda activate "D:\CondaEnvs\attention-face-directml"

python scripts/face_real_directml_libreface.py `
  --prep-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\libreface-prep" `
  --model-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\libreface" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\libreface" `
  --batch-size 16
```

主要输出：

```text
libreface_dml_au_intensity_probability.parquet
libreface_dml_au_detection_probability.parquet
libreface_dml_au_intensity.parquet
libreface_dml_au_detection.parquet
libreface_dml_expression.parquet
libreface_dml_gaze.parquet
libreface_dml_real300_manifest.json
```

### 7.4 Py-Feat raw-frame DirectML

仍在：

```text
D:\CondaEnvs\attention-face-directml
```

运行：

```powershell
python scripts/face_real_directml_pyfeat.py `
  --benchmark-dir "D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031" `
  --model-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\pyfeat" `
  --retinaface-batch 8 `
  --multitask-batch 16
```

主要输出：

```text
pyfeat_dml_raw.parquet
pyfeat_dml_real300_manifest.json
```

`pyfeat_dml_raw.parquet` 是宽表，完整保留本记录第 6 节字段；不要根据文件宽度提前删列。

### 7.5 LibreFace parity

```powershell
python scripts/face_real_parity_v02.py `
  --candidate libreface `
  --benchmark-dir "D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031" `
  --prep-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\libreface-prep" `
  --dml-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\libreface" `
  --output "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\parity\libreface_parity.json"
```

### 7.6 Py-Feat parity

```powershell
python scripts/face_real_parity_v02.py `
  --candidate pyfeat `
  --benchmark-dir "D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031" `
  --dml-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\pyfeat" `
  --output "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\parity\pyfeat_parity.json"
```

## 8. Parity 报告范围

### Py-Feat

自动报告：

- CPU/DML detected-frame coverage；
- 每帧 face count mismatch；
- 同一帧多脸按最高 bbox IoU 贪心匹配，不静默丢 multi-face；
- bbox IoU；
- FaceScore；
- 68-point compatibility landmarks；
- AU20；
- emotion7 probability + top-class agreement；
- valence/arousal；
- 6DoF pose；
- gaze；
- original-frame 478 mesh；
- 52 blendshapes。

DML-only retention fields（raw RetinaFace 5-point、raw pose/gaze、normalized mesh）因旧 CPU Fex 未保存同等 raw representation，不强行伪造 parity；它们仍保留供未来复用/QC。

### LibreFace

自动报告：

- fresh alignment vs saved CPU alignment success flag；
- headpose JSON 中共同 numeric path 的误差；
- landmarks JSON 中共同 numeric path 的误差；
- AU intensity；
- AU binary detection；
- gaze；
- expression label agreement（若旧 CPU table 存在可识别 label column）。

新保存的 raw AU probabilities 没有旧 CPU raw probability reference，因此只保留，不伪造 parity。

## 9. 判定原则与止损线

本阶段不预先设一个脱离变量尺度的统一 MAE 数字。先按层次判断：

1. **coverage / face count / bbox 或 alignment** 是否一致；
2. 若输入几何一致，再看 AU / emotion / VA / pose / gaze / landmarks / mesh 的 MAE、max abs、Pearson/Spearman；
3. category 输出另外看 label/top-class agreement；
4. speed 使用各自 real-input manifest 的 end-to-end 定义，不拿 Gate-1 model-core images/s 混入；
5. 如果 Py-Feat bbox 很一致但 scientific output 仅有小幅系统性漂移，优先判定是否由 OpenCV vs PyTorch interpolation 引起；
6. 如果解决 Py-Feat preprocessing parity 需要继续显著增加工程复杂度，而 LibreFace parity/coverage/速度已经满足正式分析，则在此停止继续优化 Py-Feat，选择 LibreFace；
7. 如果 Py-Feat parity 正常且 end-to-end 已达到可接受水平，则因其信息完整性更强，继续保留为正式候选。

**只有 real-300 parity + real end-to-end 都完成后，才冻结 Face backend / fps / primary-face 规则。**

## 10. 本轮结束后需要回传的文件

优先只上传 5 个 summary JSON，不需要上传宽 parquet：

```text
libreface-prep\libreface_dml_prep_manifest.json
libreface\libreface_dml_real300_manifest.json
pyfeat\pyfeat_dml_real300_manifest.json
parity\libreface_parity.json
parity\pyfeat_parity.json
```

如果某个 parity 报告暴露异常，再针对性检查对应 parquet；不先上传全部大文件。

本轮不删除/覆盖任何历史 CPU benchmark、Gate profile 或 diagnostic 输出。