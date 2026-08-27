# RITnet Full-Class：唯一正式完整证据版

## 1. 当前唯一正式口径

当前 RITnet full-class 只保留一条正式生产路径：

```text
ritnet-fullclass-v2-native640
schema_version = 2
```

这里的 `v2-native640` 是当前正式数据 Schema 的版本标识，不代表还有一套并行生产版。历史上已经生成的旧 full-class 文件可以留作 provenance（来源追踪）和历史比较，但**不再作为可执行的当前分析路径，也不再写成“v1.2/v2 双轨”**。

当前正式流程固定为：

```text
既有 formal eyes.csv 中保存的 frame_idx + YOLO ROI
→ 从原视频按 ROI 重新裁眼睛，不重跑 YOLO
→ 官方 RITnet 网络语义 + 冻结权重
→ 项目 DirectML/ONNX 适配
→ 保存每个眼 ROI 的 uint8 400×640 四分类 hard label
→ 同一张 hard label 派生 pupil / iris / ocular 几何与原子 QC 事实
→ 概率摘要随 label chunk 同步 checkpoint
→ 稀疏 QC
→ manifest + completion + SHA256 完整性证明
```

因此，正式数据层固定为：

```text
原始可重算证据：400×640 hard-label store
派生数值：full-class CSV
人工检查：QC PNG + qc_index.csv
完整性证明：store manifest + run manifest + completion + SHA256
```

CSV 不是不可逆的“原始分割数据”；真正能让后续修改几何算法而不重跑 RITnet 的是完整 hard-label store。

---

## 2. 哪些是 RITnet 官方内容，哪些是本项目后来增加的

`native_*` 的 `native` 只表示**在 640×400 RITnet hard-label 坐标系中保存或测量**，绝不表示“RITnet 官方自带字段”。来源必须严格区分：

| 来源层 | 内容 | 性质 |
|---|---|---|
| RITnet 上游官方 | `DenseNet2D` 网络定义、`best_model.pkl` 冻结权重、四分类语义 background/sclera/iris/pupil、网络 logits、gamma=0.8、CLAHE(1.5, 8×8)、Normalize([0.5],[0.5])；官方测试时也会对 logits 做 argmax | 官方方法/权重/任务语义 |
| Attention-Analysis 确定性运行适配 | 复用 YOLO ROI、ROI resize 到 640×400、fixed-b16 FP32 ONNX、DirectML、在 logits 后加 `ArgMax` 输出 `labels_u8`、`Softmax/Gather` 提取 class-3 `pupil_probability` | 项目自己的运行接口，不是官方 ONNX 原生接口 |
| Attention-Analysis 派生记录 | pixel count/fraction、连通分量、最大主体比例、edge、ellipse、PIR、中心偏移、OAR、概率摘要、`gate_*`、`diagnostic_*`、label store、QC、manifest/completion/hash | 全部是本项目后处理与审计记录 |

必须使用的准确表述是：

> 四分类分割的网络结构、权重、类别语义和官方预处理来自 RITnet；`labels_u8` 与 `pupil_probability` 是 Attention-Analysis 为 DirectML 推理增加的确定性 ONNX 接口；所有几何、PIR、OAR、QC 和完整性记录均为项目派生。

上游权重记录中的 `f0864e...` 是 Git blob SHA-1，因此字段名固定为 `official_weights_git_blob_sha1`。实际运行 `.onnx` 与 `.onnx.data` 分别记录文件内容 SHA256。

---

## 3. 400×640 hard label 是正式证据源

每一个 eye row 都必须保存：

```text
labels.shape == (400, 640)
labels.dtype == uint8
unique(labels) ⊆ {0,1,2,3}
```

类别固定为：

```text
0 background
1 sclera
2 iris
3 pupil
```

同一张 label 构造：

```text
pupil      = labels == 3
iris       = labels == 2
iris_outer = iris | pupil
ocular     = sclera | iris | pupil
```

pupil 与 `iris_outer` 分别从各自最大外轮廓执行 `cv2.fitEllipse`。正式 PIR（pupil-to-iris ratio，瞳孔-虹膜直径比）为：

```text
sqrt(pupil_axis_a * pupil_axis_b)
---------------------------------
sqrt(iris_axis_a * iris_axis_b)
```

四条轴必须来自**同一张 400×640 label**。既有 `eyes.csv` 中过去的 pupil 几何仍可作为 source provenance 保存，但不得参与当前 PIR 计算。

---

## 4. 640×400 是模型坐标，不是无畸变源坐标

YOLO ROI 会通过 `cv2.INTER_LINEAR` resize 到固定 640×400。源 ROI 宽高比不一致时会产生非等比例缩放，因此每一行必须同时记录：

```text
source_frame_width / source_frame_height
source_roi_width / source_roi_height
ritnet_input_width / ritnet_input_height
roi_to_ritnet_scale_x / roi_to_ritnet_scale_y
roi_to_ritnet_aspect_scale_ratio
geometry_coordinate_system = ritnet_native_label
```

所以 `native_pupil_to_iris_diameter_ratio` 是**模型坐标系几何量**，不能未经验证就声称与 source-pixel（源像素）几何完全等价。因为完整 label 和缩放关系已经落盘，后续可以把 mask/轮廓映回 source ROI 再重算，不需要再次执行 RITnet。

---

## 5. 全量 label store、概率 checkpoint 与严格恢复

每个被试产生独立 store：

```text
sub-031_ritnet_fullclass_v2-native640_labels/
├── chunks/
│   ├── chunk-000000.npz
│   └── ...
├── label_index.csv
├── chunk_manifest.csv
└── store_manifest.json
```

每个 chunk 无损保存：

```text
labels                         uint8 [N,400,640]
row_ordinal                    int64 [N]
frame_idx                      int64 [N]
eye_code                       uint8 [N]
pupil_probability_available    uint8 [N]
pupil_probability_stats        float32 [N,6]
```

概率摘要必须随 chunk checkpoint（检查点保存），因为 hard label 无法反推出 class-3 Softmax 概率；如果只把概率摘要放在最终 CSV，中断后仍会迫使已经完成的 ROI 重跑 RITnet。

chunk 提交流程固定为：

```text
shape/dtype/value-domain 校验
→ 临时 NPZ
→ flush + fsync
→ 重新打开做结构校验
→ SHA256
→ os.replace 原子提交
→ 原子更新 label_index / chunk_manifest / store_manifest
```

**已提交 NPZ chunk 是恢复事实源。** 如果断电发生在 chunk rename 成功之后、CSV metadata 提交之前，恢复器根据 chunk 内的 `row_ordinal/frame_idx/eye_code` 重建索引，不删除、不重新推理该 chunk。真实缺块、SHA256 错误、shape/dtype/value-domain 错误则拒绝继续。

已经 finalized（完成）的 store 在内容没有变化时重新打开必须是 byte-stable（字节稳定）：不得把 `complete` 改回 `running`，不得改变 `store_manifest.json` 的 SHA256。

当前默认 `chunk_rows=128` 只影响存储打包，不改变科学数值；正式全量前仍要用 AMD 的 `sub-031` 实测压缩率和吞吐。如需调整，只能在正式 cohort 开始前冻结。

---

## 6. CSV 事务、字段和可重算性

每次运行先创建新的：

```text
.<final-name>.<uuid>.partial
```

恢复时，已经提交的 label chunks 会重新派生一份 partial CSV，然后继续未完成 ordinal。只有同时满足 label store、行数、主键和 CSV↔index 一致性后才 `os.replace` 成正式 CSV。

正式 CSV 至少包含：

- `native_label_*`：label 定位与 Schema；
- source frame/ROI 与 ROI→RITnet 坐标映射；
- background/sclera/iris/pupil/iris_outer/ocular pixel count 与 fraction；
- pupil、iris_outer、ocular 连通分量及最大主体比例；
- whole-mask edge 与 largest-contour edge 分离记录；
- pupil 与 iris_outer 的 ellipse 几何；
- diameter/ellipse-area/contour-area PIR；
- pupil–iris 中心偏移与空间关系；
- OAR（ocular aperture ratio，眼球可见开口比例）median/p90；
- class-3 pupil probability 在 argmax pupil mask 内的 mean/median/p05/p95/min/max；
- `gate_*` 与 `diagnostic_*` 原子事实；
- model/runtime/preprocessing/geometry/version provenance。

如果 hard label 中没有 pupil 像素，probability map 仍可记录 `available=True`，但“pupil mask 内统计量”必须为 null/NaN，不能写 0.0 混淆“没有条件像素”和“概率等于零”。

---

## 7. OAR、QC 与 gate 的正式解释

OAR 基于 `sclera | iris | pupil` 的可见眼球 mask，只是项目派生的几何开口量。它**不是** EAR（Eye Aspect Ratio，基于眼睑关键点的眼睛纵横比），也不是已经验证的 blink/closed/PERCLOS 标签。目前不设置未经验证的 blink cutoff。

正式 full-class 也不定义新的总 `primary_valid`。只保存可以事后重建筛选规则的原子事实，例如：

```text
gate_pupil_fit_valid
gate_iris_outer_fit_valid
gate_pupil_center_in_iris_outer
gate_iris_larger_than_pupil
gate_pir_finite

diagnostic_pupil_whole_mask_edge
diagnostic_pupil_largest_contour_edge
diagnostic_iris_whole_mask_edge
diagnostic_iris_largest_contour_edge
diagnostic_pupil_fragmented
diagnostic_iris_fragmented
diagnostic_ocular_fragmented
```

当前正式 ONNX 提供 hard label 与 class-3 pupil probability。四分类平均 confidence、entropy 等没有正式可验证接口时不伪造；如以后新增 all-class probability 输出，必须先单独完成 ONNX/DirectML parity（等价性）与性能验证，再升级 Schema。

---

## 8. manifest、completion 与强 provenance

正式入口强制：

```text
Git working tree 必须干净
source video 必须计算内容 SHA256
禁止 --allow-model-mismatch
```

manifest 记录 Git commit/branch、config snapshot+SHA256、source eyes SHA256、video identity、ONNX SHA256、`.onnx.data` SHA256、上游仓库/commit/权重 Git blob SHA-1、预处理来源、label store、几何定义和置信度定义。

completion 只有在以下全部通过后才允许写 `status=complete`：

```text
processed_rows == expected_rows
stored_label_rows == expected_rows
label index ordinal 连续且 frame/eye 唯一
CSV key 与 label index 顺序完全一致
所有 committed chunks 存在且 SHA256 正确
所有 label 均为 uint8 400×640 且值域仅 0/1/2/3
store_manifest 仍为 complete 且内嵌 index/manifest SHA256 正确
CSV / label_index / chunk_manifest / store_manifest / summary / manifest / qc_index
    的 SHA256 在 store 恢复检查之后重新核验通过
```

completion 自己不对自己做 hash；它是最后写入的顶层证明文件。

---

## 9. 唯一正式运行入口

用户只使用下面两个入口：

```text
run_ritnet_fullclass_extension.py      # 单个 formal run
run_ritnet_fullclass_batch.py          # 批量
```

文件名中带 `native` 的模块只是当前实现内部文件，不是第二套用户生产路径。历史 fast/320×160 runner 已从正式入口移除。

每次开始：

```powershell
conda activate "D:\CondaEnvs\nir-amd"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML\runtime\nir-formal"

git status --short --branch
git pull --ff-only
python -m pytest tests -q
python run_pipeline.py check-env
```

先对 AMD 本机 `sub-031` 做选择检查：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0 `
  --chunk-rows 128 `
  --dry-run
```

确认选择正确后，先跑 `sub-031`，检查：DirectML provider、完整行数、label store、completion、恢复、磁盘占用、压缩率、吞吐和抽样 QC。只有这些通过后才冻结 chunk 参数并开始 AMD 当前 cohort。

正式入口会自动启用 source-video SHA256，并拒绝 dirty Git worktree 或 model mismatch，因此不要再添加旧版 `--postprocess-workers`、`--validate-pupil` 或 `--allow-model-mismatch` 参数。

---

## 10. 当前尚不能伪称已经验证的内容

代码完整性与 CPU 单元测试可以在仓库中定义，但以下结论必须由 AMD 本机实际运行后才能确认：

- DirectML 端到端执行；
- 当前 ONNX 与 `.onnx.data` 的实机加载；
- `sub-031` 全量结果与 QC；
- `chunk_rows=128` 的实际压缩率/吞吐；
- 中断恢复的本机文件系统行为；
- AMD 当前 cohort 全量完成。

这些是**运行验收项**，不是另一套版本。正式方法本身只有本文件描述的一套。
