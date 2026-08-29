# 配置

`configs/` 同时保存当前正式下游 science config 与需要长期保留的历史/兼容配置。**当前正式执行入口与历史配置必须分开理解；机器特定路径只能放在 `configs/paths.local.yaml`，不能再把 D:/E:/F:/J:/ 等盘符写入新的 science config。**

| 文件 | 用途 | 当前定位 |
|---|---|---|
| `behavior_formal_v2.yaml` | FocusWave v3.1.3 Behavior science-v3 正式下游 | **当前正式 Behavior 配置**；`scripts/sart_formal_analysis.py` 默认入口 |
| `behavior_formal.yaml` | 早期 BB/formal-v1 行为配置，含历史机器路径 | **历史/兼容配置，不是当前正式 science-v3 权威配置**；当前 NIR 仍引用它属于待修复的管线连续性缺陷，见 `docs/060-formal-analysis/007-身份键与正式管线连续性联合审计_20260830.md` |
| `formal_multimodal_v2.yaml` | Behavior/NIR/RGB/mmWave 接入与 deferred fusion 的总合同 | 当前总合同；具体 RGB 范围以 `rgb_formal.yaml` 为更窄权威 |
| `nir_analysis_ready.yaml` | staged NIR pupil-only analysis-ready materialization | 当前 NIR staged 配置之一；source manifest 权威格式仍需按 007 收口 |
| `nir_formal_analysis.yaml` | NIR trial/probe/time-on-task 等正式分析表与模型配置 | 当前 NIR staged 配置；其 Behavior loader 仍需改为复用正式 Behavior runtime preparation |
| `rgb_formal.yaml` | preserved RGB 输出的轻量 Motion/Pose/Blink downstream | **当前正式 RGB 轻量配置**；PERCLOS/AU/emotion/rPPG/复杂预测/fusion 默认 deferred |
| `sart_bbb_v3_0.yaml` | 2026-08-16、sub-011~030、BBB SART 分析 | 历史可执行配置，不是当前口径 |
| `preexperiment.yaml` | 预实验 v2 路径、窗口、审批门等 | 历史兼容配置；不作为 current CLI 默认入口 |
| `formal.yaml` | 08-16 阶段 NIR ROI / PuReST 候选链 | 历史兼容配置；不作为 current NIR 配置 |
| `../runtime/nir-formal/config.yaml` | YOLO26n + RITnet 正式 NIR 生产提取 | 当前生产 runtime 配置；与本目录的 downstream science config 分层 |

## 本机路径规则

正式下游统一通过 `configs/paths.local.yaml` 或环境变量 `ATTENTION_ANALYSIS_PATHS_CONFIG` 解析本机路径。仓库只提交 `configs/paths.example.yaml` 作为字段模板；`paths.local.yaml` 必须 gitignored。

历史配置中仍存在的 D:/E:/F: 等绝对路径只代表历史运行环境，不得复制到新的正式配置。当前代码若仍从这些历史配置读取正式输入，应视为待修复的 active-path 合同问题，而不是继续扩大硬编码路径。

## 身份和 cohort 配置边界

- `session_id`/`subid` 是一次实验/采集场次，不是 participant。
- `participant_key` 是问卷/重复登记中的已核验匿名参与者来源字段。
- `participant_group_id` 是正式推断、聚类重抽样和 participant-disjoint prediction 的统一内部统计键目标。
- 旧 `repeat_participant_id` 继续作为 cohort manifest 的历史输入/来源追踪字段；Behavior 当前还有兼容列名使用。
- staged NIR 仍保留 `analysis_group_token` 兼容接口；它不得形成第四套独立身份语义。
- 身份无法解析的 session 仍可进入单模态 QC，但不得用 `session_id` 回退为 participant；参与者级分析必须输出 `not_estimable + reason`。

详细当前实现、已确认缺陷和验收顺序见 `docs/060-formal-analysis/007-身份键与正式管线连续性联合审计_20260830.md`。
