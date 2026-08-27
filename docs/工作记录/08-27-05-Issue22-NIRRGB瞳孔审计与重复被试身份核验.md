# Issue #22：NIR–RGB 瞳孔审计与重复被试身份核验

## 总结

本次工作完成 Issue #22 的代表性 `sub-031` 工程审计，并将重复参与者身份字段纳入本地 paired-sample 与 summary 契约。审计只读现有 NIR full-class CSV、RGB Face raw Parquet、SART 问卷重复登记表和北京被试信息 Excel，不修改正式 NIR 下游表、不进入正式统计，也不把相关性或 p 值解释成科学发现。完整审计评论、身份修正、repeat-registry/relations follow-up 和 correction validation pilot follow-up 已提交并复核于 GitHub Issue #22（comments `5437786976`、`5437908758`、`5438500943`、`5439108697`）。

最终生成：

- `scripts/multimodal_pupil_audit.py`：可重复运行的只读审计入口，当前版本支持通过 `--repeat-registry` 读取外部非 PII 重复被试 registry。
- `scripts/multimodal_pupil_correction_pilot.py`：validation-only 的 M0–M3 校正候选比较入口；支持 NIR same-camera bbox、RGB outer-eye/inner-canthus、Pitch 敏感性和 baseline-only fit。
- `tests/test_multimodal_pupil_audit.py`：mesh 索引、采样时间点和时间轴统计的定向测试。
- `artifacts/multimodal_pupil_audit/summary.json`：本次审计摘要、schema/provenance、对齐统计、关系审计和 blocker。
- `artifacts/multimodal_pupil_audit/sample_schema.json`：paired sample 字段 schema。
- `artifacts/multimodal_pupil_audit/sub-031_paired_sample.csv`：1000 个 NIR 唯一时间点、1971 行双眼/配对样本；该目录被 `.gitignore` 忽略，不进入 Git。
- `artifacts/multimodal_pupil_audit/beijing_repeat_participants.csv`：仅根据北京被试信息表全部非空日期 sheet 整理的 32 组重复被试关系（2 次 10 组、3 次 18 组、4 次 4 组）；只保留实验编号、重复次数、匹配依据和来源 sheet，不写入姓名、电话、身份证或学号。

## 原计划

1. 核验实际 checkout、Issue #22、中央 multimodal/identity contract 与本地 SART 重复被试登记表。
2. 对 `sub-031` 审计 NIR pupil-only 字段、RGB Face/Pose/mesh 字段、实际 `unix_ms` 对齐、缺口和候选 nuisance proxy。
3. 将 `site`、`session_id`、`session_key`、local linkage、identity status、provisional repeat ID、repeat session count、match basis 和 global repeat ID（如有）纳入 paired sample，并支持外部 repeat registry join。
4. 使用现有 Python 环境进行小范围测试和真实数据回归，保存本地摘要。
5. 将 Issue #22 要求的 A–G 证据与未解决 blocker 回贴 GitHub。

## 执行与决策过程

### 仓库和工作边界

实际可用 checkout 是：

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-analysis-validation
branch: analysis/multimodal-integration
HEAD: 22af8f6
```

另一个 `Attention-Analysis-amd-DirectML-old` checkout 当前位于 `rgb-amd`，且存在 stale/detached worktree，未作为本次写入目标。Issue #22 所需的审计脚本写入当前 `analysis/multimodal-integration` checkout；没有执行 destructive Git 操作。

中央 `greenboo26/focuswave-multimodal-attention-analysis` 的 identity contract 明确要求中央 reconciliation 后才产生 authoritative `global_repeat_participant_id`；中央公开树中未发现可用于本次 `sub-031` 的 participant map。因此本地只保留可审计的 provisional 字段，不生成 global ID。

### 输入和 schema

NIR 输入为：

```text
D:\_AttentionData\Beijing-NIR\amd-directml\sub-031_formal_v3.1.3_yolo_b16_fp32\sub-031_ritnet_fullclass.csv
```

该文件为 104 列、81,830 行；包含 `fullclass_pupil_*` pupil/ellipse/center/ROI 字段、`fullclass_iris_outer_*` 眼虹膜外边界字段及 normalization/QC 字段。审计的 pupil 主候选使用 `fullclass_pupil_geom_mean_diameter`，有效性要求 `fullclass_pupil_found`、`fullclass_pupil_fit_valid` 和正的有限直径同时满足。`fullclass_pupil_to_iris_*`、`fullclass_iris_outer_*` 与 `fullclass_normalization_valid` 明确标为 iris-dependent，不作为 pupil-only 主输入。

RGB 输入为：

```text
D:\_AttentionData\Beijing-RGB\sub-031\sub-031_face_raw.parquet
```

原始输出 23,270 行、23,240 个唯一 primary-face 时间点、3162 列。审计保留 FaceRect、canonical `Pitch/Roll/Yaw/X/Y/Z`、`gaze_pitch/gaze_yaw` 和原始 478 点 mesh。虹膜直径遵循既有实现：右眼 `[469,470,471,472]`、左眼 `[474,475,476,477]`，各眼取 4 点之间的最大二维距离，再取双眼均值。脸框尺度使用 `sqrt(FaceRectWidth*FaceRectHeight)`。

### 时间轴和配对

配对使用 NIR/RGB 实际 `unix_ms`，不使用 RGB `target_unix_ms`。nearest join 容差为 1000 ms，同时报告实际 delta 分布并建议未来窗口先冻结为 50 ms：

| 指标 | 结果 |
|---|---:|
| NIR 行数 | 81,830 |
| NIR 唯一时间点 / 估计频率 | 41,538 / 26.56 Hz |
| RGB primary 唯一时间点 / 估计频率 | 23,240 / 14.86 Hz |
| 1 s 容差内配对行 | 81,830 / 81,830（100%） |
| `abs(delta_ms)` 中位数 / P95 / P99 | 17 / 36 / 45 ms |
| `abs(delta_ms) <= 40 ms` | 97.57% |
| `abs(delta_ms) <= 50 ms` | 99.92% |
| NIR `>200 ms` / `>1000 ms` timeline gaps | 6 / 5 |
| RGB `>200 ms` / `>1000 ms` timeline gaps | 2 / 1 |

这说明代表性 session 的相机时间戳可用于工程配对；它不等价于已经完成跨设备物理标定或正式 multimodal model contract。

### NIR pupil-only 和 RGB nuisance 审计

NIR 中 `fullclass_pupil_found` 为 72,256 行，`fullclass_pupil_fit_valid` 为 72,237 行，审计定义的有效 pupil 行为 72,237 行；`fullclass_normalization_valid` 仅 60,294 行。pupil geometric-mean diameter 的 q10/median/q90 为 11.83/14.54/16.97（当前 NIR 值仍处于已知 invalid 边界，不能作生理结论）。

RGB 478-mesh/Face 派生字段全量可计算。RGB bilateral iris diameter 的 median/P10/P90/CV 为 14.17/13.67/14.58 px、0.0286；500 点确定性均匀子样本的 median/P10/P90/CV 为 14.16/13.64/14.57 px、0.0290。连续有效帧虹膜直径绝对差的 median/P95 为 0.054/0.416 px。按 `abs(Yaw)` 的低/高 20% 分组，中位数比为 1.017；按 `abs(Pitch)` 的低/高 20% 分组，中位数比为 0.954。这里的分组差异只作为工程敏感性证据保存，不解释为眼睛真实变化或姿态校正效果。

使用有效 pupil 行的关系审计保留了 log pupil 与 Face/eye scale、Pose/gaze、ellipse ratio、pupil center 和大/小头部旋转分组的统计。所有输出都写入 summary 的 `relations` 节，并带有 `interpretation_boundary`；不得在正式报告中直接引用方向或 p 值。

### 重复被试身份

本地来源：

```text
D:\AAAWORK\07-竞赛\厚璨杯\021-analysisplan\SART\问卷\问卷分析\subject_repeat_registry.csv
```

登记表有 146 条登记、81 个匿名 `participant_key`，其中 39 个 participant key 对应多次登记、65 条记录标记为 repeat、10 条记录标记为 cross-stage repeat；同时有 29 条 identity conflict flag。登记表的 identity evidence basis 是本地 `phone_digits` 规则，但原始电话/问卷内容没有复制到代码、paired sample 或 Issue 评论。

新增核验来源：

```text
D:\AAAWORK\07-竞赛\厚璨杯\020-Experiment\北京被试信息表.xlsx
```

该表四个日期页共有 152 条数据行，其中 149 条有实验编号；字段包括“编号（主试填写）”、姓名、手机号、身份证/学号。以实验编号 `031` 对应 `sub-031`，并在内存中对手机号、身份证/学号做规范化比对（原文不写入任何输出），得到 `031`、`059`、`068` 属于同一名本地 participant。匹配依据为：`031` 与 `059` 共享手机号，`031` 与 `068` 同时共享手机号和身份证/学号。

第二个用户提供的 `被试信息.xlsx` 不是第一张表的副本，而是汇总表：有效 149 行，字段为电话号码、学号和“参与次数”，其中 `sub-031` 的身份在该表中匹配到 3 行，参与次数均为 3，与 `031/059/068` 三个实验记录相互印证。两张表的手机号集合交集为 63，身份证/学号集合交集为 46；它们不是同一 schema，不能按行号合并。

因此 paired sample 更新为本地 provisional repeat identity：

```text
site = Beijing
session_id = sub-031
session_key = Beijing:sub-031
identity_status = local_repeat_confirmed_provisional
local_participant_linkage_key = local_xlsx_<sha256-prefix>
local_repeat_participant_id = provisional_xlsx_<sha256-prefix>
global_repeat_participant_id = null
current_session_repeat_status = repeat_participant_confirmed_locally
```

`059`、`068` 的身份匹配已确认，但当前 NIR/RGB 输出根目录没有发现这两个 session 的数据，因此没有把它们与 `sub-031` 合并成一条时间序列。不能把本地 provisional key 冒充中央 authoritative ID；后续全局分析仍需 central identity reconciliation。

### 全量重复 participant 与实验编号清单

按 `被试信息.xlsx` 的 `参与次数` 字段，身份汇总包含 81 个 participant：42 人参加 1 次、14 人参加 2 次、22 人参加 3 次、3 人参加 4 次。因此按该字段定义，重复 participant 为 39 组，重复发生次数（超过首次的 visit）为 67 次。

第一张实验登记表中有 149 个实验 session，其中 120 个 session 填有手机号或身份证/学号，可直接形成 62 个身份组；其中 32 个身份组出现多个实验编号。与第二张汇总表交叉后，30 组的实验编号数量与 `参与次数` 完全一致：

```text
2次：039-108；044-057；048-075；050-165；093-174；104-109；117-145；134-162；137-150

3次：031-059-068；033-084-096；035-083-099；036-074-116；043-082-127；
     045-129-163；046-085-168；051-124-166；053-091-133；054-088-126；
     061-073-081；070-153-171；071-110-158；086-128-167；087-106-160；
     089-131-175；090-114-146；119-144-154

4次：041-064-076-102；052-056-138-170；067-143-156-164
```

以下两组不是一致结果，必须保留为冲突：

```text
034-125：第一张实验登记表显示 2 个 session，但第二张表的参与次数写为 1；
105-107-147-148：第一张实验登记表显示 4 个 session，但第二张表的参与次数写为 3。
```

第二张汇总表另有 8 个重复 participant（5 组标记参加 2 次、3 组标记参加 3 次）无法通过当前第一张实验登记表中的手机号/身份证/学号回到实验编号；这 8 组不能被安全地补写实验编号。故“39 组重复 participant”是第二张汇总表的全量结果，“30 组完全一致 + 2 组冲突”是目前能回到第一张实验编号表的可核验结果。

### 外部 repeat registry join 与关系摘要补充

为避免后续正式分析脚本写死 `031/059/068` 等映射，新增 `read_repeat_registry_csv()` 和命令行参数 `--repeat-registry`。该参数只接受非 PII registry，要求 `local_repeat_participant_id`、`experiment_ids`、`session_count`，并拒绝姓名、电话、身份证/学号列；同一个实验编号若映射到多个 local group 会直接报错。使用：

```powershell
D:\CondaEnvs\attention-rgb\python.exe scripts/multimodal_pupil_audit.py `
  --output-dir artifacts/multimodal_pupil_audit `
  --repeat-registry artifacts/multimodal_pupil_audit/beijing_repeat_participants.csv
```

本次 join 结果为 `sub-031 → beijing_xlsx_repeat_001 → 031/059/068`，`global_repeat_participant_id = null`；因此该 local group 可用于后续 participant-level 分组，但不能被当作中央权威 ID。生成的 paired sample 和 summary 已记录 registry 路径、join key、匹配状态、同组实验编号、repeat session count、repeat visits beyond first 和 identity match basis。

现有 `relations` 摘要（`n=72,237` 个有效 pupil 且完成 RGB 配对）为：`rgb_eye_outer_corner_distance_px` 的 overall/baseline/task Spearman r 分别为 `-0.418/-0.178/-0.392`；`rgb_iris_center_distance_px` 为 `-0.415/-0.164/-0.391`；`rgb_eye_inner_canthus_distance_px` 为 `-0.328/-0.135/-0.277`；`rgb_iris_diameter_px` 为 `-0.306/-0.201/-0.272`；`rgb_face_bbox_scale_px` 为 `-0.154/+0.074/-0.143`。这些只是工程关系审计，不是科学效应结论。

结合当前全帧可用率、眼角/虹膜结构的解剖刚性、姿态分层稳定性和 RGB 虹膜直径稳定性（全量 CV `0.0286`，有效率 `1.0`），当前只提出待冻结前验证的候选：`rgb_eye_outer_corner_distance_px` 作为首选解剖尺度 proxy，`rgb_iris_center_distance_px` 作为备选内部眼部尺度 proxy，`rgb_iris_diameter_px` 作为 QC/敏感性 proxy。`rgb_face_bbox_scale_px` 暂不作为首选，因为 baseline 与 task 的关系方向不一致、姿态敏感性和解释边界更弱。最终校正公式、训练/验证分组和正式统计仍待 NIR 几何修复、跨 session 验证与 analysis contract 冻结后决定。

### Correction validation pilot

根据 Issue #22 最新要求，新增 `scripts/multimodal_pupil_correction_pilot.py`，只做 validation-only 测量学比较，不读取 Behavior/Probe/ML outcome，不重跑 NIR/RGB 模型。pilot 自动读取外部 repeat registry 的指定 local group，逐 session 查找已有输出；本次 `beijing_xlsx_repeat_001` 的 `031` 有完整 NIR/RGB 输出，`059`、`068` 两个 session 的现成输出均不存在，因此 cross-session consistency 明确记为 `not_estimable`，没有启动补跑，也没有拼接三条时间序列。

pilot 比较：

- M0：uncorrected `log(NIR pupil geom mean diameter)`；
- M1：NIR same-camera YOLO bbox geometric scale 的 baseline-only residualization；
- M2a：RGB outer-eye distance 的 baseline-only residualization；
- M2b：RGB inner-canthus distance 的 baseline-only residualization；
- M3：NIR bbox geometric scale + RGB `Pitch` 的 baseline-only sensitivity model。

NIR bbox 不跨眼平均，分别保留 `frame_left` 和 `frame_right`。`sub-031` 的 bbox geometric scale 指标为：left baseline/task CV `0.0387/0.0743`、successive-difference median/P95 `1.538/6.680 px`；right baseline/task CV `0.0432/0.0403`、successive-difference median/P95 `1.862/9.292 px`。完整结果写入：

```text
artifacts/multimodal_pupil_correction_pilot/sub-031_correction_pilot_summary.json
artifacts/multimodal_pupil_correction_pilot/repeat_group_beijing_xlsx_repeat_001_cross_session_summary.json
```

在当前单个 session 的 pilot 中，M2a 将校正后 pupil 与 RGB outer-eye 的 rho 从 M0 的 `-0.418` 降至 `-0.067`，与 RGB inner-canthus 的 rho 为 `-0.323`，M1 校正后与 NIR bbox 的 rho 为 `-0.516`，M3 校正后与 Pitch 的 rho 为 `+0.517`；这些结果只说明单 session 的测量学行为，不能作为最终方法选择。M2a 的 baseline CV 为 `0.1363`，M0 为 `0.1370`，successive-difference median 分别为 `0.511` 和 `0.504 px`，尚未显示稳定性全面改善。故当前不冻结校正公式，最终选择仍需 `031/059/068` 的现成跨 session 输出或后续授权的数据补充。

### Provenance 和风险

两个实际数据 manifest 都记录了 schema/model/config/hash 等信息，但没有记录生成代码的 Git commit。因此摘要将 `nir_git_commit`、`rgb_git_commit` 明确写成 `unrecorded_in_manifest`，而不是从 stale command path 或其他 checkout 推断。

当前 blocker：

1. NIR/RGB 生成 Git commit 未记录，精确代码 provenance 仍待补齐。
2. SART 派生 registry 中 `031` 当前行的 repeat flag 仍为 0，而 Excel 个人信息匹配显示其属于 3 次实验记录；两种本地来源需要在正式 identity registry 中做冲突 reconciliation。
3. 中央公开 identity contract 存在，但当前可见树中没有可供本次 join 的 authoritative participant map。
4. `X/Y/Z` 的单位与坐标语义未由本地输出契约确认，只能作为 raw numeric diagnostic。
5. 当前 NIR/PIR 数值已知不适合正式科学解释；本任务没有修复它们。
6. NIR pupil pixels 与 RGB iris/face pixels 属于跨相机 nuisance proxy，不能直接解释为校准后的物理瞳孔尺度。

## 最终决策结果

- 已完成 Issue #22 的 `sub-031` A–G 工程审计，并将身份规则加入 paired-sample/summary schema。
- 已生成本地忽略 artifact；没有上传 NIR/RGB 原始文件、问卷内容、电话或任何 participant-level raw identity rows。
- 已新增只读脚本、脚本索引和 3 个定向单元测试；未修改正式 NIR analysis-ready、analysis tables 或 formal statistics。
- 当前已确认 `sub-031` 属于本地重复 participant，但尚无 authoritative global ID；后续任何 train/test/CV/bootstrap/mixed/GEE 必须等待中央身份 reconciliation，并以自然 participant/authoritative repeat ID 分组；重复 session 不得按独立 participant 计数。

## 校验结果

```text
D:\CondaEnvs\attention-rgb\python.exe -m pytest tests/test_multimodal_pupil_audit.py -q
6 passed

D:\CondaEnvs\attention-rgb\python.exe -m pytest tests/test_multimodal_pupil_audit.py tests/test_multimodal_pupil_correction_pilot.py -q
9 passed

D:\CondaEnvs\attention-rgb\python.exe -m py_compile scripts/multimodal_pupil_audit.py
passed
```

真实数据命令：

```powershell
D:\CondaEnvs\attention-rgb\python.exe scripts/multimodal_pupil_audit.py `
  --output-dir artifacts/multimodal_pupil_audit
```

## 已完成 / 未完成 / 待确认

### 已完成

- Issue #22 代表性 session 的 NIR/RGB schema、实际字段、时间配对和工程统计审计。
- 本地 SART repeat registry 的聚合核验。
- 北京被试信息 Excel 与按参与次数汇总 Excel 的身份字段交叉核验。
- 全量重复 participant 数量、参与次数分布、实验编号组和跨表冲突核验。
- 根据北京被试信息表单独生成非 PII 的重复被试 CSV。
- paired sample 中 identity/provenance 字段的保守落位。
- 本地脚本、schema、summary、测试和工作记录。

### 未完成

- 没有完成 central identity reconciliation；已有 local provisional repeat ID，但尚无 authoritative global repeat ID。
- 没有修复已知错误的 NIR/PIR geometry，也没有进入正式统计。
- 已完成 `sub-031` 的 correction validation pilot；由于 `sub-059`、`sub-068` 无现成 NIR/RGB 输出，跨场次一致性暂不可估计，未启动补跑。
- 没有把本地 ignored artifact 提交到 Git。

### 待确认

- 为 NIR 与 RGB producer manifests 补写真实 producing Git commit。
- 对 Excel 与 SART 派生 registry 的 `031` repeat flag 差异进行正式 reconciliation，并保留两种来源的 provenance。
- 从中央身份 reconciliation 产出 authoritative `global_repeat_participant_id` 后，再冻结 multimodal train/test/CV 和重复 session 的统计分组。
- 由模型/设备负责人确认 `X/Y/Z` 单位、坐标系和是否可作为尺度校正变量。

## 外部交付状态

已通过 GitHub connector 提交完整的 A–G 审计评论及身份修正 follow-up，并通过再次读取 Issue #22 评论列表确认新增评论：

```text
https://github.com/kyandi233-dev/Attention-Analysis/issues/22#issuecomment-5437786976
https://github.com/kyandi233-dev/Attention-Analysis/issues/22#issuecomment-5437908758
```

## 本轮继续验证

用户要求从“多模态瞳孔分析”对话继续未完成的 correction validation pilot。本轮先读取该对话末尾要求，再在本地既有 checkout 和 artifacts 上复核，而不是向对话或 Issue 发送新消息。新增测试使用 `D:\CondaEnvs\attention-rgb`（`pyarrow 25.0.1`）运行，共 9 passed；`nir-amd` 不包含 `pyarrow`，不作为本 pilot 的运行环境。

实际运行：

```text
python scripts/multimodal_pupil_correction_pilot.py
pilot_version = issue22-multimodal-pupil-correction-pilot-v1
validation_only = true
outcome_data_used = false
fit_scope = baseline_only_for_M1_M2a_M2b_M3; M0_unadjusted_reference
available_sessions = 031
missing_sessions = 059, 068
cross_session_status = not_estimable
```

`beijing_xlsx_repeat_001` 的 `031/059/068` registry join 仍然保留；本轮确认 `031` 的 NIR 81,830 行、RGB primary 23,240 行、paired 81,830 行，有效 pupil 72,237 行，其中 `abs(delta_ms) <= 50 ms` 为 72,180 行。M0、M1、M2a、M2b、M3 均在 `031` 上完成测量学输出；`059`、`068` 没有现成 NIR 或 RGB 输出，所以没有伪造 cross-session 指标，也没有启动数据补跑。

本轮未读取 Behavior/Probe/ML outcome，未运行 NIR/RGB 模型，未修改正式分析表，未冻结校正公式。结果仍是 measurement/QC evidence only；后续 blocker 是取得合法的 `059/068` 成对 session 输出并完成 participant-level cross-session validation，同时等待中央 authoritative participant identity reconciliation。
