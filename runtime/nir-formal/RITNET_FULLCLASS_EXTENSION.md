# RITnet Full-Class：AMD 最终正式管线（v8）

## 1. 唯一正式口径

当前 AMD/DirectML 的 RITnet full-class 只保留一条正式生产路径：

```text
CORE_VERSION = fullclass-final-core-v8-interface-safe-plain-csv
EYE_METRICS_SCHEMA_VERSION = 6
FRAME_COVERAGE_SCHEMA_VERSION = 2
```

正式流程固定为：

```text
历史 formal run（已完成）
→ 复用历史 eyes.csv 中的 YOLO bbox；绝不重跑 YOLO
→ 从原始 NIR AVI 读取对应帧
→ 固定 1.6 宽高比 ROI，必要时 replicate padding
→ resize 到 640×400
→ RITnet FP32 / fixed batch 16 / DirectML
→ 保留四分类 hard label 与四类 soft fraction
→ 仅计算 pupil 几何 + 必要 QC / uncertainty / temporal scalar
→ SQLite 临时 checkpoint 防止长任务中断丢失
→ plain CSV + bounded QC
→ summary / manifest / completion 完整性验收
```

`completion.json` 是唯一的“该被试最终完成”标志。没有 `completion.json` 的目录只能视为中间态或失败态。

---

## 2. 科学变量与明确废弃项

RITnet 四分类语义保持不变：

```text
0 background
1 sclera
2 iris
3 pupil
```

正式 cohort 保留：

- 四类 hard pixel count / fraction；
- `iris_outer = iris + pupil`、`ocular = non-background` 两个廉价 union 的 count / fraction；
- 四类 soft class fraction；
- pupil connected-component / fragmentation QC；
- pupil ellipse、center、axis、area、equivalent diameter、geometric-mean diameter；
- source-backed valid-domain / padding QC；
- 三个 ocular uncertainty mean：max probability、top1-top2 margin、entropy；
- gap-safe temporal delta / jump QC；
- historical YOLO bbox/provenance；
- frame coverage；
- bounded QC images + sparse pixel evidence。

以下内容已经从正式科学输出中废弃，不得重新引入：

- iris / `iris_outer` ellipse geometry；
- pupil-to-iris ratio（PIR）；
- ocular aperture ratio（OAR）；
- 用不可信 iris mask 制造的 iris diameter / area / center relation；
- cohort 全量 percentile uncertainty；
- cohort boundary-band uncertainty；
- cohort low-probability threshold fields；
- 全量 400×640 hard-label store。

iris 仍是四分类中的一个类别，但仅作为 segmentation class 保留，不再被解释为可靠的几何标尺。

---

## 3. ROI 与 padding 的正式规则

最终 ROI contract：

```text
target_width  = 640
target_height = 400
aspect_ratio  = 1.6
expand_horizontal_each_side = 0.30
expand_vertical_each_side   = 0.45
padding_mode = replicate
```

ROI 先按 1.6 的目标宽高比扩展；超出原视频边界时允许 replicate padding。

**所有正式 hard/soft/uncertainty 科学统计只使用原视频真实支持的 valid-source domain。人工 padding 不进入正式分母。**

padding 本身仍保留为 QC provenance，包括有效像素比例、padding 尺寸、结构是否预测到 padding、是否接触真实有效域边界等。

---

## 4. 正式推理输出与 CPU 后处理

正式 cohort 的 DirectML 推理只请求：

```text
labels
class_probability
```

不为每个 eye 从 DirectML 搬回完整的 max-probability / margin / entropy map。

三项正式 ocular uncertainty mean 由四分类 probability 在 CPU 上直接派生；production fast path 不构造 percentile、boundary-band 或 threshold 字段，也不在 SQLite 中保存这些字段的 null 占位。

RITnet runtime 保留周期性的完整输出合法性检查，用于发现模型输出异常。这属于质量保障，不作为“无用 CPU”删除。

---

## 5. checkpoint：SQLite 只是恢复工作区

每个被试的临时恢复数据库位于：

```text
<historical-output-root>/.ritnet-fullclass-work/<subject>.sqlite
```

它不是最终科学数据文件，不进入最终 manifest。

SQLite 保存已经完成的逐眼原子 scalar row，使用：

```text
WAL
synchronous=NORMAL
checkpoint_rows = 128
```

目的只有一个：正式长任务如果中断，不丢失已经完成的 RITnet 数值结果。

完整 checkpoint 恢复时：

```text
validate source prefix / scientific identity
→ 读取 SQLite rows
→ 重新计算 deterministic temporal facts
→ 写 final CSV / coverage / QC / completion
```

**完整 checkpoint 恢复不得初始化 DirectML runtime，也不得重新跑全部 RITnet。**

### v7 → v8 的已知恢复规则

历史 v7 checkpoint 可以迁移到 v8，但只有在以下条件全部满足时：

- source identity 相同；
- RITnet ONNX 与 external-data SHA256 相同；
- 640×400 / FP32 / b16 相同；
- class mapping 相同；
- ROI algorithm / valid-source-mask contract 相同；
- soft-class、temporal、schema contract 相同；
- SQLite stored rows 是当前 `eyes.csv` 的严格连续 prefix；
- 成功 row 的实际 payload 版本与 v8 科学 contract 一致。

Git commit、branch、整个 config 文件 SHA、summary worker 数等执行/provenance 信息不作为这次显式 v7→v8 科学兼容迁移的阻断条件。

迁移 metadata 只有在 prefix 与真实 payload 都验证通过后才提交；失败不会先修改旧 checkpoint identity。

---

## 6. 最终文件：plain CSV，不再 gzip

正式数据表固定为：

```text
ritnet-fullclass-final/<subject>/
├── data/
│   ├── eye_metrics.csv
│   └── frame_coverage.csv
├── qc/
│   ├── images/
│   ├── qc_index.csv
│   └── qc_pixel_evidence.npz
├── summary.json
├── manifest.json
└── completion.json
```

不再写：

```text
eye_metrics.csv.gz
frame_coverage.csv.gz
```

原因：当前 lean scalar schema 在 ≤1 GiB/subject 的硬限制内无需依赖 gzip；去掉 gzip 可以减少 CPU 压缩和人工检查复杂度。

旧 `.csv.gz` 只会被 runner 当作“旧失败/旧架构产物”检测并阻止自动混用。它们不会被当前正式管线读取或继续执行。

---

## 7. frame coverage 的正式意义

`frame_coverage.csv` 以历史 `frames.csv` 为基准，因此不只记录 RITnet 成功帧，还保留：

- YOLO 无眼；
- 单眼；
- 双眼；
- source video read failure；
- final video decode failure；
- ROI invalid；
- RITnet 无成功结果。

这样 QC 与后续统计不会因为只看成功 eye row 而把缺失帧静默丢掉。

coverage API 固定使用：

```text
source_frames
source_eye_rows
final_eye_rows
fixed_anchor_keys
```

不得再次引入 `frames_rows` / `eye_metric_rows` 等不存在的旧错误接口。

---

## 8. bounded QC

QC 是最终正式输出的一部分，但必须有界：

```text
qc_interval_sec = 30
qc_anomaly_max_per_reason = 5
qc_image_max_count = 80
qc_pixel_evidence_max_eyes = 16
qc_artifact_budget_bytes = 256 MiB
```

QC composite overlay 使用低强度 alpha（约 0.25），正式几何只画 pupil ellipse，不画 iris ellipse。

QC 会对少量选中帧重新读取原视频并执行 bounded RITnet：

- composite 主要使用 labels-only；
- sparse pixel evidence 最多 16 eyes，保留 hard labels、entropy、valid-source mask；
- `qc_pixel_evidence.npz` 使用普通 `np.savez`，不做额外压缩；
- QC PNG 采用低压缩级别以减少 CPU 编码开销。

这一小段重复 RITnet 是用于可审计的 QC evidence，不等于重新跑 cohort 全量。

QC 直接接收 numeric core 已在内存中的 final rows，不再把刚写完的 8 万行 CSV 重新 parse 一遍。

---

## 9. completion 与完整性

`finalize_subject()` 在发布 completion 前执行一次完整最终检查，包括：

- eye CSV exact schema / subject / schema version / row count；
- frame coverage exact schema / subject / schema version / row count；
- QC index、QC image path / size / SHA256；
- sparse pixel evidence shape / dtype / value domain；
- manifest required artifacts；
- required artifact SHA256 / size；
- subject 总输出不超过 `1073741824` bytes。

通过后才写：

```text
summary.json
manifest.json
completion.json
```

runner 不在 completion 刚写完后立刻再把 8 万行 CSV 全扫描第二遍。以后 strict skip / 外部验收仍会调用完整 validator。

---

## 10. CPU / GPU 工作边界

正式 cohort 中以下工作是必要的：

- producer：视频解码、ROI、resize/preprocess；
- DirectML：RITnet；
- summary workers：hard class scalar、pupil geometry、soft fractions、三项 ocular uncertainty mean；
- SQLite checkpoint；
- temporal facts；
- frame coverage；
- bounded QC；
- completion integrity pass。

已经删除的无意义/重复工作包括：

- final CSV gzip；
- sparse NPZ 压缩；
- high-level PNG compression；
- cohort percentile/boundary/threshold uncertainty；
- SQLite 中对应的 null 占位字段；
- complete-checkpoint 时提前加载 DirectML runtime；
- QC 前重新 parse 完整 final CSV；
- completion 后立即第二次完整表扫描；
- dead gzip I/O helpers。

`summary_workers=2`、`max_pending_summaries=2`、单 producer 仍保留，因为它们用于把必要 CPU 工作与 DirectML 推理重叠，不是重复科学计算。

---

## 11. 唯一正式入口

单被试：

```powershell
python run_ritnet_fullclass_extension.py `
  --run-dir "<historical-formal-run>" `
  --config ".\config.yaml" `
  --device 0
```

批量：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --device 0
```

指定被试：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-034" `
  --device 0
```

正式运行前始终要求 clean Git worktree。历史 formal source 必须已经通过原 formal completion validator。

不存在当前正式参数：

```text
--chunk-rows
--allow-model-mismatch
--postprocess-workers
```

不要根据历史文档重新添加这些参数。

---

## 12. 当前实机状态与下一步

`sub-034` 已经完成 80,479 个 eye row 的昂贵 RITnet 数值推理并写入 v7 SQLite checkpoint；失败发生在旧 v7 最终序列化/coverage 接口，而不是模型推理。

在 v8 修复经过 CI 与本机 preflight 验证后，应优先复用该 checkpoint 完成：

```text
v7 checkpoint migration
→ temporal facts
→ plain CSV
→ coverage
→ bounded QC
→ completion
```

如果恢复过程中出现从 0 开始的全量 `[FULLCLASS] ... batch=...` 推理日志，应立即停止并检查 identity/prefix；完整 80,479-row checkpoint 不应重新执行 cohort RITnet。
