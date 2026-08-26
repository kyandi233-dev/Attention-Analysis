# 08-26-10｜RGB Face 15 Hz、Primary Tracking、眼睑指标 Dry-Run 实现

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-09-RGB-Face-real300结果与backend冻结.md`。

## 1. 起点

上一阶段已完成：

- Py-Feat / LibreFace Gate 0/1；
- 同一 300 帧 real-input parity；
- AMD raw-frame end-to-end；
- Face backend 冻结为 Py-Feat 2.1.1 Detectorv2 scientific core + DirectML。

本轮不再比较 backend，而是进入全量前正式化：

1. 冻结 Face sampling cadence；
2. 设计 multi-face tracking / primary-face；
3. 定义 raw vs derived schema；
4. 把 blink / EAR / aperture / PERCLOS-like 指标放进可验证的 derived 层；
5. 先对 representative subjects dry-run，再进入 44-subject full run。

## 2. Face sampling 冻结为 timestamp-driven 15 Hz

正式 Face 频率：**15 Hz**。

理由：

- 典型 blink 约 100–400 ms / 150–400 ms；
- 10 Hz 仅约 100 ms/sample，短 blink 可能只有 1–2 个点；
- 15 Hz≈66.7 ms/sample，常见 150–400 ms blink 通常约 2–6 个点；
- 本项目只要求 blink event / coarse duration / slower closure statistics，不宣称精细 eyelid kinematics；
- current Py-Feat AMD real end-to-end≈17.29 fps，因此 15 Hz 在当前机器上工程上可行；
- 30 Hz 会近似翻倍 Face 计算量，而且仍不等于 high-frame-rate blink kinematics。

正式采样必须按 `unix_ms` 目标网格，不使用 frame modulo；capture/timestamp gap 必须保留在时间轴中。

对应 Accepted decision：

`docs/050-decisions/055-RGB-Face-15Hz采样频率冻结.md`

## 3. Primary-face 设计

Raw 层先保留所有检测到的 faces，不每帧只取 FaceScore 最大者。

v0.1 track gate：

- max gap=2000 ms；
- IoU≥0.05 或 normalized center distance≤0.75；
- abs log area ratio≤0.80；
- continuity score=0.70 IoU + 0.20 exp(-center distance) + 0.10 exp(-scale log ratio)；
- frame 内 greedy one-to-one assignment。

Primary 选择：

- 在 Block1+Block2 中统计 track occupancy；
- occupancy 最大者为 participant；
- tie：median FaceScore → median bbox area。

这些 gate 尚未 Accepted，要在 baseline/interblock 多脸场景和 sub-033 gap 场景 dry-run 后检查。

对应 Proposed decision：

`docs/050-decisions/056-RGB-Face-Primary与眼睑派生规则.md`

## 4. Raw 与 derived

### Raw（昂贵推理结果）

继续完整保留：

- frame/time/behavior identity；
- all face rows；
- RetinaFace bbox/score/5-point；
- 20 AU；
- 7 emotion；
- V/A；
- raw+canonical gaze；
- raw+canonical pose；
- normalized/original-frame 478 mesh；
- 68-point compatibility view；
- all 52 blendshapes，包括 `eyeBlinkLeft/Right`；
- tracking / primary-face / QC flags。

### Derived（可从 raw 重建）

- EAR left/right/mean；
- eyelid aperture px；
- iris diameter px；
- aperture / iris ratio；
- native eyeBlink mean convenience signal；
- individual open-eye reference；
- normalized eye openness；
- closure fraction；
- provisional closure80 proxy。

Blink event threshold 暂不冻结。

## 5. EAR / iris topology

使用 retained 478-point mesh：

- MediaPipe right eye：33,160,158,133,153,144；
- MediaPipe left eye：362,385,387,263,373,380；
- right iris ring：469–472；
- left iris ring：474–477。

EAR：

\[
EAR=(d(p2,p6)+d(p3,p5))/(2d(p1,p4))
\]

iris diameter：同侧 4 iris-ring 点最大 pairwise 2D distance。

eye aperture：两组 upper/lower eyelid distance 均值。

## 6. Individual open reference

v0.1：

```text
baseline valid aperture/iris
→ top 30%
→ median
```

baseline 有效样本少于 30 时，fallback 到 all-valid top30 median，并把 source 写进 summary。

`closure80_proxy` 只表示 normalized openness≤0.20；在人工/representative calibration 前不直接称为经典 PERCLOS。

## 7. Representative dry-run sample

每个被试约 4 min continuous windows：

- baseline start 30 s；
- baseline end 30 s；
- Block1 middle 60 s；
- interblock middle 60 s；
- Block2 middle 60 s。

15 Hz 理论约 3600 samples / subject。

第一批：

- sub-031：reference；
- sub-033：capture/timestamp-gap stress。

## 8. 新增代码

- `src/attention_pipeline/rgb/face_formal_dryrun_sample.py`
- `scripts/face_formal_dryrun_sample.py`
- `scripts/face_formal_dryrun_directml.py`
- `scripts/face_derive_tracking_eyelid.py`

配置 `configs/rgb_analysis.yaml` 已更新：

- `face.selected_backend=pyfeat`；
- `face.inference_fps=15.0`；
- RetinaFace batch=8；
- multitask batch=16；
- `formal_dryrun` windows；
- primary-face pilot gates；
- eyelid raw/derived contract。

## 9. sub-031 执行命令

先拉代码：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-dev"
git pull --ff-only
```

### Step A：抽取 dry-run continuous frames

```powershell
conda activate "D:\CondaEnvs\attention-rgb"

python scripts/face_formal_dryrun_sample.py --subject sub-031
```

预期目录：

```text
D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\
```

### Step B：冻结 Py-Feat backend / DirectML

```powershell
conda activate "D:\CondaEnvs\attention-face-directml"

python scripts/face_formal_dryrun_directml.py `
  --sample-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031" `
  --model-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml"
```

### Step C：tracking + eyelid derived

```powershell
python scripts/face_derive_tracking_eyelid.py `
  --raw "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml\pyfeat_dml_raw.parquet" `
  --frame-manifest "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\sub-031_face-dryrun_frames.csv" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\derived"
```

sub-033 重复同样流程，只替换 subject/path。

## 10. 本轮回传文件

sub-031 先回传：

```text
sub-031_face-dryrun_manifest.json
directml/pyfeat_dml_real300_manifest.json
derived/tracking_eyelid_summary.json
```

如果 tracking summary 显示 multi-face / fragmentation，再针对性读取 `face_tracks.parquet`；如果 eyelid 分布异常，再读取 `eye_features.parquet` 或生成少量 review frames。不要一开始上传宽 raw parquet。

## 11. Dry-run acceptance

检查：

- 15 Hz sample count / dt / sample error；
- source fps 是否支持；
- Face coverage；
- multi-face；
- track count / fragmentation；
- primary task occupancy；
- primary 是否在 baseline/interblock 跳到 experimenter；
- EAR/aperture-iris valid fraction；
- native eyeBlink / EAR distributions；
- head pose / glasses / partial closure 下的异常。

通过后才：

1. Accepted primary-face gate；
2. 冻结 blink event rule；
3. 冻结 `perclos80_proxy` rolling implementation；
4. 实现 direct-AVI formal 15 Hz runner；
5. 与 Pose/Motion/body_motion_energy 统一正式视频读取；
6. 进入 44-subject full run。
