# RGB

> 2026-08-26｜当前 AMD RGB 正式工作线为 `rgb-amd`。正式目标是：**从 baseline 开始连续到 Block2 结束，完整保存 Face、Pose、Motion 三类 raw 信息；能从 raw 重建的 tracking、眼睑、blink/PERCLOS、Pose features、QC 与统计聚合全部后移。**

## 最重要的两个入口

- **正式方法、算法与参数依据**：[`049-RGB正式分析方法与参数依据.md`](049-RGB正式分析方法与参数依据.md)
- **当前实现与全量状态快照**：[`410-RGB当前状态与全量执行总结_20260826.md`](410-RGB当前状态与全量执行总结_20260826.md)

049 用于长期稳定的方法说明；410 用于记录截至当前的实现、实机状态、AMD/NVIDIA 差异与下一步。

## 当前正式配置

| 模块 | 当前路线 | 状态 |
|---|---|---|
| Face | Py-Feat 2.1.1 scientific core + ONNX Runtime DirectML | 已用于正式 full-span raw |
| Face cadence | timestamp-driven 15 Hz | 已冻结 |
| Face runtime batch | RetinaFace B32 + multitask B64 | 当前默认；性能参数 |
| Face threshold | 0.5 | 正式检测规则 |
| Pose | MediaPipe Pose Landmarker Lite 10 Hz | 已验证 |
| Motion | OpenCV frame difference full FPS | 已验证 |
| 单被试正式总控 | Motion + Pose + Face 三线独立 reader 并行 | 已通过 full-span validator |
| raw validator | `rgb_formal_validate.py` → `sub-XXX_manifest.json` | raw-only 完成判定已生效 |
| cohort batch / resume | eligible 自动队列、已完成跳过、失败继续 | **已进入正式全量** |
| tracking / primary face | 从 Face raw 后续重建 | 不阻挡全量 |
| EAR / eyelid / blink / PERCLOS | 从 478 mesh + blendshape 后续派生 | 不阻挡全量 |
| Pose features | 从 Pose landmarks 后续派生 | 不阻挡全量 |
| QC / statistical aggregation | downstream | 不阻挡全量 |

`sub-036` 已完成正式 full-span 验证：

```text
completion_status = complete
extraction_complete = true
qc_pass = null
issues = []
warnings = []
```

其中 `qc_pass=null` 表示 QC 尚未执行，不是 QC 失败。

## 正式分析范围

```text
baseline start
→ instructions / practice / transition
→ Block1
→ inter-block transition
→ Block2
→ Block2 stop
```

当前通过 `baseline_stop - 180 s` 得到 baseline start。中间阶段连续保留，不在 raw extraction 中提前删除。

## 正式数据流

```text
1. face_formal_prepare.py
   生成 timestamp-driven 15 Hz Face frame schedule

2. 三条 raw 并行
   ├─ Motion full FPS
   ├─ Pose 10 Hz landmarks
   └─ Face 15 Hz Py-Feat / DirectML

3. rgb_formal_validate.py
   只验证三条 raw + manifest 完整性

4. sub-XXX_manifest.json
   extraction_complete = true
```

实验性 SharedDecode 已做过 AMD 实机速度测试；当前没有显示出比三条独立 reader 更好的墙钟吞吐，因此不作为正式 cohort 默认路径。

## Raw 信息保留

昂贵模型尽量只运行一次。

**Face raw** 保留所有 detected faces、no-face placeholder、bbox/confidence、5-point landmarks、20 AU、7 emotion、valence/arousal、gaze、6DoF head pose、478 mesh、68-point compatibility landmarks、native blendshapes（含 `eyeBlinkLeft/Right`）以及完整时间/行为身份。

**Pose raw** 保留全部返回 pose 的 33 landmarks、normalized/world coordinates、visibility/presence、pose bbox、多 pose/no-pose 状态。

**Motion raw** 保留正式区间 full-FPS 的真实时间、亮度、相邻帧差、motion energy、changed-pixel ratio 与 capture/timestamp gap。

因此下游可以在不重跑 Py-Feat / MediaPipe 的情况下计算 tracking、主脸、EAR、眼睑开度、虹膜标准化、blink、PERCLOS、Pose motion、trial/block/probe/sliding-window 特征等。详细边界见 044、049、410。

## 正式核心输出

```text
D:\_AttentionData\Beijing-RGB\sub-XXX\
```

```text
sub-XXX_face_frames.csv
sub-XXX_face_prepare_manifest.json
sub-XXX_face_raw.parquet
sub-XXX_face_raw_manifest.json

sub-XXX_pose_landmarks.parquet
sub-XXX_pose_manifest.json

sub-XXX_motion_raw.parquet
sub-XXX_motion_manifest.json

sub-XXX_manifest.json
```

以下属于可重建 downstream derived，不参与 extraction complete：

```text
face_tracks.parquet
eye_features.parquet
face_derived_manifest.json
pose_features.parquet
pose_features_manifest.json
```

cohort 级：

```text
rgb_inventory.csv
cohort_status.csv
cohort_manifest.json
```

## 当前正式运行

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd"
git pull --ff-only

$env:ATTENTION_FACE_MODEL_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"

powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1
```

正常续跑不要使用 `-Force`。完整 subject manifest 会自动 skip；已完成 raw branch 会自动 resume/skip。

## 文档索引

| 编号 | 内容 |
|---|---|
| 041 | RGB 分析目标与数据流 |
| 042 | 面部分析工具与 Benchmark |
| 043 | 姿态与运动量分析方法 |
| 044 | RGB 输出 Schema 与信息保留原则 |
| 045 | AMD RGB 开发环境与运行指令 |
| 049 | **RGB 正式分析方法与参数依据** |
| 410 | **RGB 当前状态与全量执行总结** |

后续新增文档继续按 `410 → 411 → 412 ...` 编号，不使用 `050`。

## 当前优先级

```text
AMD cohort raw 全量
→ 补失败/缺失被试
→ NVIDIA sub-130 Gate + CUDA batch benchmark
→ AMD DirectML ↔ NVIDIA native Py-Feat parity
→ tracking / Pose features / EAR / blink / PERCLOS
→ cohort QC
→ trial / block / probe / sliding-window 与统计分析
```
