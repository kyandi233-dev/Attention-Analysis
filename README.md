# Attention-Analysis｜AMD RGB 工作线

> 当前 branch：`rgb-amd`。这是 Attention-Analysis 的 AMD RGB 并行开发工作线，不是独立项目。分支关系见 [`docs/010-overview/015-并行分支与同步约定.md`](docs/010-overview/015-并行分支与同步约定.md)。

## 当前工作目录

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd
```

每次开始工作：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd"
git status --short --branch
git fetch origin --prune
git pull --ff-only
```

当前 branch 应为：

```text
rgb-amd
```

## RGB 当前目标

正式 RGB 只做一件事：**从 baseline 开始连续到 Block2 结束，把 Face、Pose、Motion 能稳定获得的数据完整抽取并落盘。** QC、blink/PERCLOS、body motion 和统计聚合可以后续直接基于已保存结果完成，不阻挡当前全量抽取。

当前正式链：

```text
Face 15 Hz：Py-Feat 2.1.1 scientific core + ONNX Runtime DirectML
Pose 10 Hz：MediaPipe Pose Landmarker
Motion full-fps：OpenCV frame difference
```

## 当前状态

| 模块 | 状态 |
|---|---|
| Face formal | 已实现 |
| Pose formal | 已实现 |
| Motion formal | 已实现 |
| 单被试总控 | **已实现** |
| 被试最终完整性验证 + `sub-XXX_manifest.json` | **已实现** |
| cohort batch / resume | **已实现，待实机验收** |
| blink / PERCLOS / body_motion_energy | 后续派生，不阻挡抽取 |

## 单被试运行

环境保持隔离：

| 任务 | Conda 环境 |
|---|---|
| RGB audit / Motion / Pose / validation | `D:\CondaEnvs\attention-rgb` |
| Face ONNX Runtime DirectML | `D:\CondaEnvs\attention-face-directml` |
| Py-Feat reference / ONNX export | `D:\CondaEnvs\attention-face-pyfeat` |

先设置 Face 模型目录：

```powershell
$env:ATTENTION_FACE_MODEL_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"
```

然后运行一个被试：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-031
```

总控顺序为：

```text
face_formal_prepare.py
→ rgb_formal_motion_pose.py
→ face_formal_directml.py
→ face_formal_derive.py
→ rgb_formal_validate.py
```

只有最终 validation 通过后，该被试才会生成：

```text
D:\_AttentionData\Beijing-RGB\sub-XXX\sub-XXX_manifest.json
```

其中 `extraction_complete=true` 表示抽取完整；它与后续 QC 是否通过是两件事。

## cohort 全量运行

`sub-031` 实机验收通过后直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1
```

cohort runner 会：

```text
刷新 rgb_inventory.csv
→ 只选择 analysis_eligible=True 的正式被试
→ 已有完整 sub-XXX_manifest.json 的自动跳过
→ 未完成的继续运行
→ 单个被试失败时记录错误并继续下一人
→ 持续更新 cohort_status.csv
→ 最后生成 cohort_manifest.json
```

需要只跑指定被试时可以使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1 `
  -Subjects sub-031,sub-032,sub-033
```

## 输出

正式输出统一位于：

```text
D:\_AttentionData\Beijing-RGB
```

单被试主要文件：

```text
sub-XXX_face_frames.csv
sub-XXX_face_raw.parquet
sub-XXX_face_tracks.parquet
sub-XXX_eye_features.parquet
sub-XXX_motion_raw.parquet
sub-XXX_pose_landmarks.parquet
sub-XXX_pose_features.parquet
sub-XXX_manifest.json
```

cohort 级文件：

```text
rgb_inventory.csv
cohort_status.csv
cohort_manifest.json
```

Git pull / switch / merge 不管理这些正式结果。

## 文档入口

| 内容 | 入口 |
|---|---|
| RGB 当前状态和正式运行 | [`docs/040-rgb/README.md`](docs/040-rgb/README.md) |
| RGB 输出 Schema / 信息保留 | [`docs/040-rgb/044-RGB输出Schema与信息保留原则.md`](docs/040-rgb/044-RGB输出Schema与信息保留原则.md) |
| AMD RGB 环境与指令 | [`docs/040-rgb/045-RGB开发环境与运行指令.md`](docs/040-rgb/045-RGB开发环境与运行指令.md) |
| scripts 索引 | [`scripts/README.md`](scripts/README.md) |
| 历史工作记录 | [`docs/工作记录/`](docs/工作记录/) |

## 现在的执行顺序

```text
sub-031 全程实机验收
→ 修实际运行错误（如果有）
→ cohort 全量抽取
→ 根据 cohort_status 补失败被试
→ 再处理 QC / blink / PERCLOS / body motion / 统计派生
```
