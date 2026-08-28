# NIR Formal Runtime（AMD / DirectML）

这是 `Attention-Analysis` 的 AMD 正式 NIR runtime。历史 formal producer 已经完成 YOLO 眼睛检测；当前最终 RITnet full-class 管线**严格复用历史 `eyes.csv` 的 YOLO bbox，不重跑 YOLO**，从原始 NIR AVI 重建固定 1.6 ROI 后执行 RITnet。

当前正式核心：

```text
fullclass-final-core-v8-interface-safe-plain-csv
EYE_METRICS_SCHEMA_VERSION = 6
FRAME_COVERAGE_SCHEMA_VERSION = 2
```

最终目标：每被试新增 full-class 输出 ≤ 1 GiB；只保留后续科研分析、QC 与敏感性分析真正需要的 scalar/小型证据，不全量落盘 400×640 segmentation/probability map。

详细科学与恢复契约见 `RITNET_FULLCLASS_EXTENSION.md`。

---

## 1. 每次打开新终端

```powershell
cd "D:\AAAWORK\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

git switch amd-DirectML
git pull --ff-only
git status --short --branch

conda activate "D:\CondaEnvs\nir-amd"
cd runtime\nir-formal
```

正式 runner 强制 clean Git worktree。不要为了通过检查使用 `git reset --hard`；先确认本地修改来源。

---

## 2. 最终数据流

```text
历史 formal completion / frames.csv / eyes.csv
        ↓ 严格验证
历史 YOLO bbox + 原始 NIR AVI
        ↓ 不重跑 YOLO
固定 1.6 ROI + 必要 replicate padding
        ↓
640×400
        ↓
RITnet FP32 / fixed b16 / DirectML
        ↓
hard 4-class + 临时 class_probability
        ↓ 只统计真实 source-backed pixels
pupil-only geometry
+ four soft fractions
+ 3 ocular uncertainty means
+ padding/QC facts
        ↓
SQLite interruption checkpoint
        ↓
temporal facts + frame coverage
        ↓
plain CSV + bounded QC
        ↓
summary + manifest + completion + ≤1 GiB
```

人工 padding 可以作为网络输入上下文，但 padding 像素本身不进入正式 hard/soft/uncertainty 科学分母。

---

## 3. 当前正式科学输出

保留：

- hard background / sclera / iris / pupil count 与 fraction；
- cheap `iris_outer` / `ocular` union count 与 fraction；
- four-class soft fractions；
- pupil connected components / fragmentation；
- pupil ellipse / center / axes / area / diameter；
- valid-source / padding QC；
- ocular max-probability mean；
- ocular top1-top2 margin mean；
- ocular entropy mean；
- temporal delta / jump QC；
- historical YOLO provenance；
- frame coverage；
- bounded QC evidence。

不再作为正式输出：

```text
iris ellipse
iris_outer ellipse
PIR / pupil-to-iris ratio
OAR / ocular aperture ratio
cohort percentile uncertainty
cohort boundary-band uncertainty
cohort low-probability threshold fields
full hard-label store
full probability-map store
```

iris 仍保留为四分类类别，但不再作为几何归一化标尺。

---

## 4. 当前 ONNX / runtime 输出

正式 cohort 推理只请求：

```text
labels              uint8   [16,400,640]
class_probability   float32 [16,4,400,640]
```

三项 ocular uncertainty mean 从 `class_probability` 在 CPU 上直接派生。production fast path 不创建完整 max/margin/entropy map，也不计算 percentile/boundary/threshold 统计。

完整多输出只用于模型 qualification / bounded sparse QC。

### 重新导出模型（仅需要时）

```powershell
python fetch_ritnet_upstream_weights.py

python export_ritnet_batch_variants.py `
  --final-uncertainty `
  --batches 16 `
  --force

python validate_ritnet_fullclass_final_model.py --device 0
```

正式模型：

```text
models/ritnet-b16-fp32-uncertainty.onnx
models/ritnet-b16-fp32-uncertainty.onnx.data
```

---

## 5. 代码测试

AMD runtime 全套测试：

```powershell
python -m pytest tests -q
```

正式 CI 还会运行仓库基础 NIR tests。

当前回归测试明确覆盖：

- fixed 1.6 ROI / padding；
- padding 不进入科学统计；
- pupil-only geometry；
- four soft fractions / three ocular means；
- full-domain 与 padded-domain 的 cohort uncertainty 公式 parity；
- padded-domain pupil / iris_outer / ocular 边界 QC 公式 parity；
- pupil connected-component 定义 parity；
- production checkpoint 不保存 QC-only null 字段；
- temporal facts；
- frame coverage；
- rows-only bounded QC API；
- plain CSV I/O；
- complete checkpoint 不初始化 DirectML runtime；
- complete checkpoint → CSV → coverage 的真实调用链；
- v7 → v8 checkpoint migration；
- v8 → v8 仅按 numeric scientific identity 恢复；
- checkpoint prefix/payload 不一致时绝不先改写 identity；
- engine/workstore v8 core version 同步；
- completion/manifest/QC integrity。

测试通过不替代真实 AMD DirectML smoke，但它必须在实跑之前通过。

---

## 6. 运行入口

历史 formal 输出根：

```text
D:\_AttentionData\Beijing-NIR\amd-directml
```

### 只检查历史 source 选择

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-034" `
  --device 0 `
  --dry-run
```

### 指定被试

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-034" `
  --device 0
```

### 单个历史 formal run

```powershell
python run_ritnet_fullclass_extension.py `
  --run-dir "<对应历史 formal run 目录>" `
  --config config.yaml `
  --device 0
```

不存在当前正式参数：

```text
--chunk-rows
--compression
--postprocess-workers
--validate-pupil
--allow-model-mismatch
```

---

## 7. 最终输出结构

```text
D:\_AttentionData\Beijing-NIR\amd-directml\ritnet-fullclass-final\sub-XXX\
├── data\
│   ├── eye_metrics.csv
│   └── frame_coverage.csv
├── qc\
│   ├── images\
│   │   └── *.png
│   ├── qc_index.csv
│   └── qc_pixel_evidence.npz
├── summary.json
├── manifest.json
└── completion.json
```

只有 `completion.json` 严格验证通过才算完成。

不再生成：

```text
eye_metrics.csv.gz
frame_coverage.csv.gz
```

旧 `.csv.gz` 如果残留在未完成 subject 目录，会被 preflight 视为旧失败产物并阻止自动混用；应先人工确认并归档到 subject 目录之外，不自动删除。

---

## 8. SQLite checkpoint

临时恢复数据库：

```text
D:\_AttentionData\Beijing-NIR\amd-directml\.ritnet-fullclass-work\sub-XXX.sqlite
```

它不是最终科研数据。

完整 checkpoint 时：

```text
validate numeric scientific identity + source prefix + stored payload
→ SQLite rows
→ temporal facts
→ final CSV / coverage
→ bounded QC
→ completion
```

**不会初始化 DirectML，也不会重跑全量 RITnet。**

`sub-034` 已有 80,479-row v7 checkpoint；v8 对该迁移有显式 fail-closed 检查。迁移前不会删除或重写 numeric rows。

v8 partial checkpoint 同样按 numeric scientific identity 判断能否复用。Git commit/branch、整份 config SHA 或 scheduling/QC 等非 numeric provenance 发生变化时，只要 source/model/ROI/schema/数值算法版本完全一致，仍必须先通过 source prefix 与真实 payload 检查，随后才允许把 checkpoint identity 迁移到当前 run。模型、source、ROI、class mapping、analysis/uncertainty/temporal/schema 任一科学身份变化则拒绝恢复。

final `completion.json` 的规则更严格：它代表完整 final artifact contract，不会因为 checkpoint 可复用就自动把不同 final config 的旧结果当作当前完成结果。

---

## 9. bounded QC

当前配置：

```text
qc_interval_sec = 30
qc_anomaly_max_per_reason = 5
qc_image_max_count = 80
qc_pixel_evidence_max_eyes = 16
qc_artifact_budget_bytes = 268435456
final_output_limit_bytes = 1073741824
```

QC composite 仅画 pupil ellipse，不画 iris ellipse；overlay alpha 约 0.25。

QC 对少量选中帧执行 bounded RITnet，这是为了生成可复核 evidence，不是重新跑 cohort：

- composite 主要 labels-only；
- sparse pixel evidence 最多 16 eyes；
- NPZ 不额外压缩；
- PNG 使用低压缩级别；
- QC 直接使用 numeric core 内存 rows，不重新 parse 整个 final CSV。

---

## 10. completion / integrity

发布 `completion.json` 前必须验证：

1. plain eye CSV exact schema / subject / schema version / row count；
2. plain frame CSV exact schema / subject / schema version / row count；
3. QC index / images / sparse pixel evidence；
4. required artifact SHA256 与 size；
5. source selection / work identity；
6. 总目录大小 ≤1 GiB。

`finalize_subject()` 做一次完整预发布 integrity pass。runner 不在刚完成后立即再把 8 万行表完整扫描第二遍；之后再次 strict-skip / validate 时仍会执行完整 validator。

---

## 11. 性能边界

当前必要 CPU 工作：

- 视频 decode / ROI / preprocess；
- pupil hard metrics / geometry；
- four soft fractions；
- three ocular uncertainty means；
- SQLite checkpoint；
- temporal facts；
- frame coverage；
- bounded QC；
- final integrity pass。

已经删除或合并的重复/无用工作：

- final CSV gzip；
- sparse NPZ compression；
- 高 PNG compression；
- cohort percentile/boundary/threshold calculations；
- 对应 SQLite null placeholders；
- complete-checkpoint DirectML initialization；
- QC 前完整 CSV readback；
- completion 后立即第二次全表 validation；
- dead gzip helper code；
- padded-domain pupil / iris_outer / ocular 三次重复 boundary dilation（现在一次计算复用）。

`summary_workers=2` 与单 producer 用于将必要 CPU 工作与 DirectML 重叠；实测当前瓶颈是 DirectML，不是 summary，所以不要为了“CPU 看起来更少”把有效 overlap 拆掉。

### 隔离的 DirectML output-transfer benchmark

当前正式 b16 cohort 每次会返回：

```text
class_probability = [16,4,400,640] float32
```

仅该张量约 62.5 MiB/call。它当前用于在 CPU 端计算 four soft fractions 与 three ocular uncertainty means。

仓库提供隔离 benchmark：

```powershell
python benchmark_ritnet_final_output_transfer.py `
  --run-dir "<任一严格完成的 historical formal run>" `
  --config config.yaml `
  --device 0
```

它只使用同一个当前 ONNX/session 和一张真实 b16 tensor，交错比较：

```text
labels only
labels + class_probability   # 当前 cohort contract
all five outputs
```

计时前会检查 labels parity，以及两种 probability 请求方式的逐像素一致性。该 benchmark 不写正式科研输出、不修改 checkpoint、也不会改变正式 runner。

只有 AMD 实机结果证明 output transfer 是显著瓶颈后，才考虑开发独立 scalar-output ONNX candidate；candidate 必须先完成 DirectML parity 与 benchmark，不能直接替换正式模型。padded ROI 仍必须保留当前 source-valid 科学定义和可靠 fallback。
