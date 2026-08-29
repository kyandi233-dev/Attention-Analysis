# NIR fullclass-final 瞳孔-only下游适配器

状态：`nir-pupil-only-adapter-v1.0.0` 的工程契约与字段字典。它只用于正式统计前的小样本、只读输入验证，不表示44场正式数据已经运行或心理结果已经产生。设计依据固定引用正式证据库 `FocusWave-Formal-Analysis` commit `171b081f3a3f9d06496c7b8d36915eebd4e2a3bb`。

## 一、权威落点与边界

适配入口位于 `scripts/nir_pupil_only_adapter.py`，可复用实现位于 `src/attention_pipeline/nir_pupil_only/`，配置模板位于 `configs/nir_pupil_only_adapter.v1.yaml`。它从 `analysis/multimodal-integration` 建立，因为这里是现有 NIR—Behavior 下游连接与刺激视觉属性的权威代码位置；`amd-DirectML` 的 schema v6 生产接口和 `amd-DirectML-geometry-validation` 的 schema v7 topology接口只作为源字段契约依据，不在本适配器中复制或改写生产 runtime。

当前 fullclass-final 没有独立虹膜椭圆或直径。`hard_iris_fraction` 与 `soft_iris_fraction` 只是四分类分割中的 iris 类别比例；适配器原样保留它们作为分割与质量候选，但明确拒绝计算 PIR（瞳孔—虹膜直径比）和 OAR（眼球开口比例）。任何需要 PIR/OAR 的旧 `nir-behavior-v1.2 / schema 2` 入口均不能通过补空列或重命名 iris fraction 来兼容。

## 二、输入、主键与来源

每个场次必须显式提供一个来源 manifest，至少包含 `subject`、`source_schema_version`、`source_path` 与 `source_kind`；推荐同时提供生产分支和提交。可移植模板为 `configs/nir_pupil_only_source_manifest.v1.json`。适配器读取 schema v6/v7 的 `eye_metrics.csv`，按字段名投影，不按列位置拼表。

输出主键为 `(subject, phase, phase_segment, frame_idx, eye)`。源 `eye` 的 `frame_left` / `frame_right` 仅在返回的适配表中标准化为 `left` / `right`，原值保存在 `eye_raw`；源 CSV 不回写。重复主键、schema 与 manifest 不一致、无法识别的 eye 值、组内 `unix_ms` 倒退均直接失败。

来源追溯字段为：

| 字段 | 含义 |
|---|---|
| `source_schema_version` | manifest声明的源 eye schema，当前只允许6或7 |
| `source_path` | manifest声明的生产或验证源路径 |
| `source_manifest_path` | 本次读取的来源manifest路径 |
| `source_kind` | topology、032/033特殊来源、历史YOLO b8等来源类别 |
| `source_branch` / `source_commit` | 可用时保留的生产代码版本 |
| `adapter_version` / `output_schema_version` | 适配器实现和输出字段契约版本 |

## 三、schema v6/v7字段映射

生产 schema v6 与 topology schema v7 的核心字段名相同；v7另外包含验证方法字段。下游使用下列 canonical（统一）名称，并保留源 schema：

| canonical字段 | v6/v7实际源字段优先 | 解释 |
|---|---|---|
| `pupil_axis_a` | `pupil_short_axis`，兼容 `pupil_axis_a` | 瞳孔椭圆短轴 |
| `pupil_axis_b` | `pupil_long_axis`，兼容 `pupil_axis_b` | 瞳孔椭圆长轴 |
| `pupil_equivalent_diameter` | `pupil_equiv_diameter`，兼容全写名称 | 轮廓面积等效直径 |
| `prev_frame_idx` | `temporal_prev_frame_idx` | 同一时序轨上一帧索引 |
| `frame_gap` / `time_gap_ms` | `temporal_frame_gap` / `temporal_time_gap_ms` | 帧与真实时间间隔 |
| `temporal_reset` | `temporal_reset_reason` | 重置原因；不是插值标志 |
| `center_jump` | `delta_pupil_center_distance_px` | 瞳孔中心跳变量 |
| `touches_roi_edge` | `pupil_touches_valid_domain_edge` | 瞳孔是否触及有效域边界 |
| `soft_max_probability` | `ocular_max_probability_mean` | ocular域平均最大类别概率 |
| `soft_margin` | `ocular_top1_top2_margin_mean` | ocular域平均分类间隔 |
| `soft_entropy` | `ocular_entropy_mean` | ocular域平均熵 |

`validation_topology_*` 等v7验证字段不伪装成所有场次共有字段；当前适配只把已冻结的生产核心字段投影为共同 pupil-only 契约。

## 四、质量分轨

质量状态同时以独立布尔字段和一个保守的 `quality_track` 表达，避免把多种失败压成“有效/无效”二值。`source_observed`、`source_missing`、`ritnet_missing`、`roi_clipped`、`geometry_invalid`、`temporal_flagged` 与 `interpolation_only` 均保留；独占标签的优先顺序为插值、来源缺失、RITnet缺失、ROI/有效域受限、几何无效、时序异常、观察轨。

`ritnet_status=success` 只说明推理接口完成，不会覆盖 `pupil_fit_valid`、必要几何缺失、边缘/填充污染或时序异常。适配器不定义由结果反推的数值QC阈值；当前 ROI轨只使用生产的原子状态与边界/填充事实。插值即使以后加入，也只能写入 `interpolation_only` 副轨，不能替代观察值。

## 五、行为与视觉连接字段

眼记录使用 `unix_ms` 与行为 `absolute_onset_time` 的明确 trial 区间连接。输出保留 `behavior_match_delta_ms`、phase、phase segment、匹配状态和失败原因；不以 `frame_idx/30` 伪造时间。当前与上一刺激分别使用 `current_` 和 `previous_` 前缀，保留刺激名称、code、大小、trial onset、下一trial onset、上一trial onset，以及视觉资产中所有 relative luminance（相对亮度）与 RMS contrast（均方根对比度）字段。

视觉键为 `stimulus_name + stimulus_size_pct`。每个block首trial的上一刺激保持空值，状态为 `not_applicable`，原因为 `block_first_trial`；缺失视觉键、行为组不存在、trial区间外分别保留失败原因。相对亮度的固定语义是 linear-sRGB（线性标准红绿蓝色彩空间）数字相对亮度，不是光度计测得的物理 `cd/m²`。

## 六、最小验证与本地迁移

仓库中的去身份化合成夹具覆盖 sub-031 schema v7 topology、sub-032和sub-033 schema v6特殊来源，以及sub-035历史YOLO b8 / RITnet b16 schema v6来源。它们不是正式数据，也不包含真实时间或心理结果。

安装当前包后运行：

```powershell
python -m pytest tests/test_nir_pupil_only_adapter.py -q
```

本地迁移时复制配置模板，设置 `ATTENTION_NIR_ROOT` 与 `ATTENTION_BEHAVIOR_ROOT`，为每个经过人工裁决的场次建立来源manifest，然后先只放入031、032、033和一个历史YOLO b8来源。运行入口为：

```powershell
python scripts/nir_pupil_only_adapter.py --config <本地配置副本>
```

仍需本地验证的项目包括：真实四个小样本的表头与来源manifest一致性、completion与文件hash、032/033特殊来源的协议说明、真实 behavior phase/phase_segment字段、当前/上一刺激join覆盖、Junction解析后的实际路径、输出目录隔离，以及正式QC阈值。以上通过前不得解除到44场的安全门，也不得运行正式统计或写心理结论。
