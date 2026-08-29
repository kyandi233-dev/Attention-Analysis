# 配置

`configs/` 同时保存当前正式下游 science config 与需要长期保留的历史/兼容配置。**当前正式执行入口与历史配置必须分开理解；机器特定路径只能放在 `configs/paths.local.yaml`，不能再把 D:/E:/F:/J:/ 等盘符写入新的 science config。**

| 文件 | 用途 | 当前定位 |
|---|---|---|
| `behavior_formal_v2.yaml` | FocusWave v3.1.3/v3.1.4 science-equivalent Behavior science-v3 下游 | **当前正式 Behavior 配置**；`scripts/sart_formal_analysis.py` 默认入口 |
| `behavior_formal.yaml` | 早期 BB/formal-v1 行为配置，含历史机器路径 | **历史/兼容配置**；当前正式 Behavior/NIR 不再依赖它 |
| `formal_multimodal_v2.yaml` | 当前单模态公共合同 + 历史 adapter/deferred fusion 边界 | **公共治理合同，不是一个全模态生产 runner**；其中 legacy `nir-adapt` 与 `merge-audit` 明确不能授权正式统计/融合 |
| `nir_analysis_ready.yaml` | staged NIR pupil-only materialization + producer OAR QC 保留 | **当前 NIR staged 配置**；JSON source manifest 使用 `nir_analysis_ready_source_manifest_json` |
| `nir_formal_analysis.yaml` | NIR trial/probe/time-on-task 等分析表 | **当前 NIR staged 配置**；Behavior trial 复用 `behavior_formal_v2.yaml` runtime/path registry |
| `rgb_formal.yaml` | preserved RGB 输出的轻量 Motion/Pose/Blink downstream | **当前正式 RGB 轻量配置**；PERCLOS/AU/emotion/rPPG/复杂预测/fusion 默认 deferred |
| `sart_bbb_v3_0.yaml` | 2026-08-16、sub-011~030、BBB SART 分析 | 历史可执行配置，不是当前口径 |
| `preexperiment.yaml` | 预实验 v2 路径、窗口、审批门等 | 历史兼容配置；保留原机器路径用于 provenance，不作为 current CLI 默认入口 |
| `formal.yaml` | 08-16 阶段 NIR ROI / PuReST 候选链 | 历史兼容配置；不作为 current NIR 配置 |
| `../runtime/nir-formal/config.yaml` | YOLO26n + RITnet 正式 NIR 生产提取 | 当前生产 runtime 配置；与本目录 downstream science config 分层 |

## 当前权威执行链

```text
Behavior
scripts/sart_formal_analysis.py
  -> configs/behavior_formal_v2.yaml

NIR downstream
scripts/nir_formal_pipeline.py
  -> configs/nir_analysis_ready.yaml
  -> configs/nir_formal_analysis.yaml
  -> configs/nir_pipeline_validation.yaml

RGB downstream
scripts/rgb_formal_downstream.py
  -> configs/rgb_formal.yaml
```

`scripts/formal_multimodal_analysis.py nir-adapt` 仅保留旧 CSV adapter 的历史复现能力；它不是 `10_analysis_ready` 的权威入口。`merge-audit` 仅是 deferred merge-contract audit；当前不允许据此声称正式多模态融合已经生产可用。

## 本机路径规则

正式下游统一通过 `configs/paths.local.yaml` 或环境变量 `ATTENTION_ANALYSIS_PATHS_CONFIG` 解析本机路径。仓库只提交 `configs/paths.example.yaml` 作为字段模板；`paths.local.yaml` 必须 gitignored。当前 path-registry loader 支持版本 1/2/3；未知未来版本 fail closed。

历史配置中仍存在的 D:/E:/F: 等绝对路径只代表历史运行环境，不得复制到新的正式配置。历史测试需要这些路径/旧原始数据时必须显式标记/skip，不能让干净环境的全量 pytest 产生无法解释的假红。

## 身份和 cohort 配置边界

- `session_id`/`subid` 是一次实验/采集场次，不是 participant。
- `participant_key` 是问卷/重复登记中的已核验匿名参与者来源字段。
- `participant_group_id` 是 Behavior、NIR、RGB、正式推断、聚类重抽样和 participant-disjoint prediction 的唯一 canonical 内部统计键。
- 旧 `repeat_participant_id` 仅作为 cohort manifest 的 legacy input/provenance 和旧 Behavior 函数边界的兼容别名；进入正式推断前必须验证与 `participant_group_id` 一致。
- staged NIR 的 `analysis_group_token` 只允许作为存储兼容别名；必须通过一一 partition parity 审计证明与 `participant_group_id` 等价。
- 仅靠 legacy repeat group、没有问卷 participant_key/crosswalk 的身份，必须通过 allow-listed `identity_status` 治理状态；否则 session 保留，但参与者级推断为 `not_estimable`。
- 身份无法解析时绝不能用 `session_id` 回退成 participant。
- questionnaire 始终 LEFT JOIN；缺问卷不能缩减 governed cohort。

## NIR ocular aperture 边界

`fullclass_ocular_aperture_ratio_median` / `p90` 是生产 RITnet 可见 ocular mask（sclera + iris + pupil）的眼睛开合 **QC 候选**，staged NIR 必须保留，不得与 PIR 一起静默丢弃。它们不是 pupil metric、不是 iris diameter、不是 MediaPipe EAR，也不能直接叫 blink/PERCLOS。正式 PIR/iris-geometry 派生仍然禁止。

详细身份/管线连续性审计见 `docs/060-formal-analysis/007-身份键与正式管线连续性联合审计_20260830.md`；最终修复后的复审记录应以本轮最新代码与 CI 为准。
