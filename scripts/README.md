# Scripts

当前 branch 为 `rgb-amd`。RGB 正式任务是从 baseline 开始连续到 Block2 结束，完整抽取 Face、Pose、Motion；QC 与统计派生后移。

## RGB 当前正式入口

| 脚本 | 用途 |
|---|---|
| `face_formal_prepare.py` | 生成正式 15 Hz Face 帧清单 |
| `rgb_formal_motion_pose.py` | 正式 Motion full-fps + Pose 10 Hz |
| `face_formal_directml.py` | original AVI → Py-Feat DirectML Face raw |
| `face_formal_derive.py` | tracking / primary face / eyelid derived |
| `rgb_formal_validate.py` | 最终完整性检查，生成 `sub-XXX_manifest.json` |
| `run_rgb_formal_subject.ps1` | 单被试一条命令跑完整正式链 |
| `run_rgb_formal_cohort.ps1` | cohort 自动队列、skip complete、失败继续、状态汇总 |

其余 benchmark / QC / parity 脚本继续保留用于验证和 provenance，不是当前全量运行入口。

## 单被试正式链

```text
face_formal_prepare.py
→ rgb_formal_motion_pose.py
→ face_formal_directml.py
→ face_formal_derive.py
→ rgb_formal_validate.py
```

推荐直接使用总控：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd"

$env:ATTENTION_FACE_MODEL_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"

powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-031
```

总控自动使用：

```text
D:\CondaEnvs\attention-rgb
D:\CondaEnvs\attention-face-directml
```

最终只有 `rgb_formal_validate.py` 通过后才算抽取完成，并生成：

```text
D:\_AttentionData\Beijing-RGB\sub-XXX\sub-XXX_manifest.json
```

其中：

```text
extraction_complete = true
qc_pass = null
```

表示“数据已经完整抽取”，不等于“QC 已通过”。

## cohort 正式运行

`sub-031` 实机验收通过后：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1
```

运行逻辑：

```text
RGB audit
→ 读取 rgb_inventory.csv
→ 只跑 analysis_eligible=True
→ 已完成被试自动 skip
→ 未完成被试运行单被试总控
→ 某人失败：写入 cohort_status.csv，继续下一人
→ 最后生成 cohort_manifest.json
```

只跑指定被试：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1 `
  -Subjects sub-031,sub-032,sub-033
```

## 正式输出

```text
D:\_AttentionData\Beijing-RGB\sub-XXX\
```

主要文件：

```text
sub-XXX_face_frames.csv
sub-XXX_face_prepare_manifest.json
sub-XXX_face_raw.parquet
sub-XXX_face_raw_manifest.json
sub-XXX_face_tracks.parquet
sub-XXX_eye_features.parquet
sub-XXX_face_derived_manifest.json
sub-XXX_motion_raw.parquet
sub-XXX_motion_manifest.json
sub-XXX_pose_landmarks.parquet
sub-XXX_pose_manifest.json
sub-XXX_pose_features.parquet
sub-XXX_pose_features_manifest.json
sub-XXX_manifest.json
```

cohort 级输出：

```text
D:\_AttentionData\Beijing-RGB\rgb_inventory.csv
D:\_AttentionData\Beijing-RGB\cohort_status.csv
D:\_AttentionData\Beijing-RGB\cohort_manifest.json
```

## 其他当前入口

| 脚本 | 用途 |
|---|---|
| `rgb_analysis.py` | RGB audit / gaps / pilot QC 开发入口 |
| `sart_formal_analysis.py` | 当前 BB Behavior |
| `nir_behavior_alignment.py` | NIR × Behavior 对齐 |
| `build_stimulus_visual_table.py` | SART 视觉协变量与报告图 |

当前不要因为 blink/PERCLOS/body motion 尚未最终冻结而阻挡 RGB raw 全量抽取。Face raw、Pose landmarks 和 Motion raw 保存完整后，这些规则都可以后续重算。
