# RITnet Full-Class Extension：v1.2 历史路径与 v2-native640 原生证据链

## 当前原则

`ritnet-fullclass-v1.2-fast-qc` 已经产生过历史结果，因此保持原文件名、旧 320×160 指标逻辑和旧 runner，不覆盖、不删除。新的正式补全路径独立命名为：

```text
ritnet-fullclass-v2-native640
schema_version = 2
```

v2 的目标不是单纯把 `analysis_size` 改成 `(640,400)`，而是把 RITnet 四分类 hard label（硬分类标签）变成可恢复的原始证据层，再从同一张 640×400 label 派生 pupil/iris 几何、PIR、OAR 与原子 QC 事实。数据层级固定为：

```text
原始证据：v2 native label store
派生数值：v2 CSV
人工查看：稀疏 QC PNG + qc_index.csv
```

因此不能再把 CSV 称为“完整、可逆的分割数据”。真正可逆的是 label store。

## 上游官方内容与 Attention-Analysis 自己增加的内容

“native”只表示 **640×400 RITnet hard-label 坐标系**，不表示“RITnet 官方自带变量”。来源必须分成三层：

| 层级 | 当前内容 | 性质 |
|---|---|---|
| RITnet 上游官方 | `DenseNet2D` 网络定义；`best_model.pkl` 冻结权重；四分类语义 background/sclera/iris/pupil；gamma=0.8、CLAHE(1.5, 8×8)、Normalize([0.5],[0.5])；网络 logits | 上游方法/权重 |
| 本项目确定性运行适配 | 复用保存的 YOLO ROI；ROI 拉伸到 640×400；fixed-b16 FP32 ONNX；DirectML；logits 后 `ArgMax` 生成 `labels_u8`；`Softmax` 后取 class-3 生成 `pupil_probability` | 不改权重，但属于项目自己的 runtime/ONNX 输出接口 |
| 本项目派生记录 | pixel count/fraction、轮廓、ellipse、PIR、center offset、component、edge、OAR、概率摘要、`gate_*`/`diagnostic_*`、label store、QC、manifest/completion/hash | 全部是项目后处理/审计记录，不是上游 RITnet 原生字段 |

上游 RITnet 的测试代码本身也会对网络输出做 `argmax` 得到 hard prediction 并保存 `.npy`，因此四分类 hard segmentation 与官方任务语义一致；但当前 AMD ONNX 的 `labels_u8`/`pupil_probability` 两个命名输出仍是本项目为了 DirectML 增加的确定性接口，不能写成“官方 ONNX 原生输出”。

## v2 原生几何

v2 不调用 `summarize_fullclass_from_source(...)`，也不把 label 缩到 320×160。每一行严格要求：

```text
labels.shape == (400, 640)
labels.dtype == uint8
unique(labels) ⊆ {0,1,2,3}
```

同一张 label 同时建立：

```text
background = labels == 0
sclera     = labels == 1
iris       = labels == 2
pupil      = labels == 3
iris_outer = iris | pupil
ocular     = sclera | iris | pupil
```

然后 pupil 与 `iris_outer` 分别取最大外轮廓并用 `cv2.fitEllipse` 拟合。新的 native PIR 为：

```text
sqrt(pupil_axis_a * pupil_axis_b)
---------------------------------
sqrt(iris_axis_a * iris_axis_b)
```

四条轴均来自同一张 640×400 label。旧 `eyes.csv` 中的 pupil 几何仍原样保留在 source 字段中，只作为历史 reference，不参与 native PIR。

### 640×400 不是“无畸变源坐标”

源 YOLO ROI 会通过 `cv2.INTER_LINEAR` 调整到固定 640×400。如果源 ROI 的宽高比不同，`scale_x != scale_y`，几何会发生非等比例拉伸。因此 v2 每行保存：

```text
source_roi_width / source_roi_height
roi_to_ritnet_scale_x / roi_to_ritnet_scale_y
roi_to_ritnet_aspect_scale_ratio
geometry_coordinate_system = ritnet_native_label
```

native PIR 明确是 **模型坐标系测量**，不能在没有验证的情况下宣称对源像素几何完全不变。由于完整 hard label 和变换比例都已保存，后续若要把轮廓映回 source ROI 坐标，可以直接下游重算，不需要再次跑 RITnet。

## 全量 label store 与恢复

每个被试独立产生：

```text
sub-031_ritnet_fullclass_v2-native640_labels/
├── chunks/
│   ├── chunk-000000.npz
│   └── ...
├── label_index.csv
├── chunk_manifest.csv
└── store_manifest.json
```

初始 `chunk_rows=128`，但目前标记为 **provisional**，必须在 `sub-031` 上完成压缩率、磁盘吞吐和总速度实测后才能冻结。每个 chunk 无损保存：

```text
labels                         uint8 [N,400,640]
row_ordinal                    int64 [N]
frame_idx                      int64 [N]
eye_code                       uint8 [N]
pupil_probability_available    uint8 [N]
pupil_probability_stats        float32 [N,6]
```

概率摘要也随 chunk 保存，是因为如果程序在最终 CSV 生成前中断，仅凭 hard label 无法重新得到已经传回 CPU 的 pupil probability 摘要。这样恢复时已提交 chunk 不需要再次执行 RITnet。

chunk 流程为：

```text
校验 shape/dtype/value domain
→ 在目标 chunks 目录写临时文件
→ flush + fsync
→ 重新打开临时 NPZ 做结构校验
→ SHA256
→ os.replace 原子 rename
→ 原子更新 index / chunk manifest / store manifest
```

如果断电恰好发生在 chunk rename 之后、metadata 更新之前，恢复器会根据已提交 chunk 内的 ordinal/frame/eye_code 重建索引并接续；不会删除这个 chunk。若发现真实缺块、哈希错误、shape/dtype/value-domain 错误，则拒绝拼接。

## CSV 事务与字段

CSV 每次运行先写新的：

```text
.<final-name>.<uuid>.partial
```

恢复时把已提交 label chunks 重新派生为一份新的 partial CSV，再继续处理未提交 ordinal；最后只有在 label store、行数、主键和 CSV↔index 完全一致后才 `os.replace` 为正式 v2 CSV。异常留下的 partial 只是诊断物，不作为正式结果。

v2 CSV 包含以下类别：

- label 定位：`native_label_*`；
- 源视频/ROI 与坐标映射：`source_frame_*`、`source_roi_*`、`roi_to_ritnet_*`、`geometry_*`；
- 四分类像素数/比例与 `iris_outer`、`ocular`；
- pupil / iris_outer / ocular 连通分量；
- whole-mask edge 与 largest-contour edge 分开记录；
- pupil 与 iris_outer 的 native ellipse 几何；
- diameter/ellipse-area/contour-area PIR、中心偏移与空间关系；
- OAR（ocular aperture ratio，眼球可见开口比例）median/p90；
- class-3 pupil probability 在 argmax pupil mask 内的 mean/median/p05/p95/min/max；
- `gate_*` 与 `diagnostic_*` 原子事实；
- provenance/version/runtime 字段。

没有 pupil argmax 像素时，probability map 仍可标记 `available=True`，但“在 pupil mask 内的均值”等条件统计量为 null；不能再用 `0.0` 混淆“没有 pupil mask”和“模型置信度为 0”。

## gate、OAR 与置信度解释

v2 不生成未经批准的 `native_primary_valid`，也不设置新的 confidence cutoff。只保存可重建的原子事实，例如：

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

`legacy_v1_strict_valid` 只用于敏感性/reference：它把旧逻辑应用到 native640 几何上，**不是**旧 v1.2 320×160 输出的逐行 bit-identical（逐位完全相同）复现。

OAR 是基于 `sclera | iris | pupil` 可见眼球 mask 的几何开口指标，不是 EAR（Eye Aspect Ratio，眼睛纵横比 landmark 指标），也不是已经验证的 blink/closed/PERCLOS 标签。完整 label 已保存，因此以后如需改用最大主体而不是 whole mask 计算 OAR，可以直接派生，不必重跑模型。

当前 production ONNX 只提供 hard label 与 class-3 pupil probability。四分类平均置信度、top-1 mean、entropy mean 不会伪造。`export_ritnet_batch_variants.py --evidence-summary` 可以生成独立 `*-evidence.onnx` 候选接口，但在 DirectML parity 和性能验证通过前，正式 v2 CSV 固定记录：

```text
native_allclass_confidence_available = false
native_allclass_confidence_unavailable_reason = ...
```

## manifest / completion

manifest 记录 Git commit/branch、config snapshot+SHA256、source eyes SHA256、video identity、ONNX SHA256、`.onnx.data` SHA256、上游仓库/commit/权重 Git blob SHA1、预处理每一步来源、label store、几何和置信度数学定义。

注意：上游 `best_model.pkl` 当前记录的 `f086...` 是 **Git blob SHA-1**，不是 SHA256，因此字段名固定为 `official_weights_git_blob_sha1`，不再使用含糊的 `official_weights_blob_sha`。

completion 只有在以下同时满足时才写 `status=complete`：

```text
processed_rows == expected_rows
stored_label_rows == expected_rows
label index ordinal 连续且 frame/eye 唯一
CSV key 与 label index 顺序完全一致
所有 chunk 存在且 SHA256 正确
所有 label 均为 uint8 400×640 且值域仅 0/1/2/3
CSV / index / chunk-manifest / store-manifest / summary / manifest 哈希重新核验通过
```

completion 自己不对自己做 hash；它作为最后写入的顶层证明文件，保存其他最终 artifact 的 SHA256。

## 当前执行入口与验收顺序

旧入口继续保留：

```text
run_ritnet_fullclass_extension.py
run_ritnet_fullclass_batch.py
```

v2 使用独立入口：

```text
run_ritnet_fullclass_native_extension.py
run_ritnet_fullclass_native_batch.py
```

正式 44 人重跑前的顺序固定为：

```powershell
python -m pytest tests -q

python run_ritnet_fullclass_native_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0 `
  --chunk-rows 128 `
  --dry-run
```

然后只做 `sub-031` bounded smoke / 压缩率 / throughput / 中断恢复 / DirectML parity 检查。通过后再冻结 chunk 参数并决定是否开始剩余 44 人；当前代码和文档都不应自动启动全量重跑。
