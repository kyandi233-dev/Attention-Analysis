# AGENTS.md｜Attention-Analysis 仓库规则

> 更新：2026-08-29（Asia/Shanghai）  
> 本文件是仓库级长期工作约束。涉及正式下游分析时，`analysis/multimodal-integration` 以 `configs/formal_multimodal_v2.yaml` 和 `docs/060-formal-analysis/001-正式多模态V2路径与分析契约.md` 为当前入口；历史 PIR/BBB/validation 资产继续保留，但不得覆盖当前正式口径。

## 当前事实与权威边界

- 正式实验协议源头：FocusWave v3.1.3，正式任务为两个 B block。
- 正式 NIR producer 位于对应 NIR runtime/生产分支；下游分析不得为了统计方便重新运行 YOLO 或改写生产 CSV。
- 当前正式 NIR 是 `fullclass-final` pupil（瞳孔）几何与 QC 链。031 主几何已经冻结为 topology（拓扑规则）版本；其他几何只作紧凑方法证据。
- 当前 `fullclass-final` 没有独立虹膜椭圆，因此 PIR（瞳孔/虹膜直径比）和 `iris_outer` 归一化不再是正式主合同。历史 PIR 脚本、旧 analysis-ready 表和旧结果只作 provenance，不删除。
- `fullclass_ocular_aperture_ratio_median/p90` 若由 producer 明确提供，可作为 NIR eye-opening candidate/QC（眼球开口候选/质量控制）保留；它不是 pupil 指标、不是 EAR、不是 blink event、不是 PERCLOS，也不得从 iris fraction 反推。
- 当前正式样本边界、重复参与者和各模态可用性必须来自外部 cohort/source manifest；不得把 44、38、39、72 等现场数字写死进程序。
- session（场次）是采集与时间轴单位，`repeat_participant_id` 才是正式统计推断、重采样和分组交叉验证的身份单位。同一参与者的所有 session 必须同折。
- 旧 BBB 行为分析是历史版本；当前 BB 行为正式分析必须从当前原始试次重建。旧 BBB 图、p 值和三 Block 结论不得搬成当前正式结果。
- 毫米波 producer 由其权威仓库维护；本仓库只接收通过字段/QC 门的 merge-ready 表。缺失场次保持 missing，不补零。
- RGB producer 先冻结主脸、有效观测、眼睑/眨眼与运动 QC；远程 PPG 已退出当前正式路线。
- 截至当前证据基线，没有正式 44 场行为推断、没有正式多模态融合结果、没有新版正式报告。不得把工程 smoke、算法 success 或历史分析结果描述成正式科学结论。

当前科学证据仓库：`kyandi233-dev/FocusWave-Formal-Analysis`  
当前正式分析代码改动必须至少对照证据提交：`171b081f3a3f9d06496c7b8d36915eebd4e2a3bb`

## 必读入口

涉及正式分析时按以下顺序读取：

1. 根 `README.md`
2. 本 `AGENTS.md`
3. `docs/060-formal-analysis/001-正式多模态V2路径与分析契约.md`
4. `configs/formal_multimodal_v2.yaml`
5. 对应模态当前 README / runtime 说明
6. FocusWave-Formal-Analysis 的当前资产导航、分析设计和对应运行记录
7. 需要追溯时再读取历史工作记录、旧配置和旧输出

不得用旧对话、目录名、旧 README 数字或历史输出代替当前代码、manifest 和实际产物核验。

## 路径与配置规则

正式下游分析采用三层结构：

1. **科学配置**：提交 Git。保存变量定义、窗口、QC、分析单位、模型规则和逻辑 path key，不保存本机盘符。
2. **机器路径注册表**：不提交 Git。使用 `configs/paths.local.yaml` 或由 `ATTENTION_ANALYSIS_PATHS_CONFIG` 指定其他文件。
3. **cohort/source manifest**：数据合同。保存本轮纳入 session、重复分析组、来源状态和权威 producer 输出路径。

`configs/paths.local.yaml` 与 `configs/paths.*.local.yaml` 必须保持 gitignored。

代码不得把 `D:`、`E:`、`F:`、用户目录或某台工作站仓库路径当唯一运行条件。换电脑、换磁盘或追加数据时，应只修改 path registry / manifest；不进入 Python 脚本改路径。

允许同一个逻辑 key 对应多个原始数据根。若同一 session 在多个有效根中出现重复正式数据，必须 fail closed 并报告冲突，不得按搜索顺序静默选一份。

分析结果继续写到仓库外。Git 只维护代码、科学配置、模板、测试、schema、运行说明和聚合安全证据。

## cohort、身份与合并规则

正式 cohort manifest 至少包含 `session_id`、`include`、`repeat_participant_id`。所有纳入正式推断的 session 必须有非空 `repeat_participant_id`。追加后续数据时更新 manifest，并重新生成全样本 participant-disjoint folds（参与者互斥分折）。

跨模态 merge-ready 主键：

- trial：`repeat_participant_id, session_id, block_id, trial_id`
- probe：`repeat_participant_id, session_id, block_id, probe_id, window_name`
- block：`repeat_participant_id, session_id, block_id`
- session：`repeat_participant_id, session_id`

缺列、空主键、重复主键、同一 repeat participant 跨折或 schema 不一致时停止。不得静默补列、静默改单位、静默填充身份。

后续样本与当前样本按同 schema 追加 row-level 数据后重新拟合；不得平均两批 p 值、AUC、系数或各自模型性能来伪装成全样本结果。

## Behavior 当前规则

当前路径无关入口：`scripts/sart_formal_analysis_v2.py`

旧 `scripts/sart_formal_analysis.py` 和 `configs/behavior_formal.yaml` 保留为既有实现/历史兼容，不再作为跨电脑正式 V2 的唯一入口。

V2 可以在 session 级做原始试次提取与结构验证、Block/cycle/probe 工程指标落盘、repeat group 字段附加以及覆盖/分母/schema 审计。

旧 `behavior_formal/stats.py` 仍以 session/subject 为独立单位，当前 V2 默认阻断其正式推断。不得把这个阻断通过开关绕过后直接写正式报告。正式统计必须另外实现 repeat-participant-safe 的层级/聚类模型。

既有 12 项 probe-window 特征不是完整结局体系。trial、probe、block、session 四尺度的 RT 水平/离散/CV/slope、commission、omission、d′、c、β、Q1/Q2 必须按机会数和覆盖门分别处理。

## NIR 当前规则

当前路径无关正式下游入口：`scripts/formal_multimodal_analysis.py`

NIR source manifest 每个 session 只能选择一个 current authoritative `fullclass-final` 来源，并记录 schema/source commit/reason。v6/v7 必须按字段名适配，不能按列位置拼接。

正式跨模态时间键优先使用真实 `unix_ms`。禁止用 `frame_idx / 30`、固定 FPS 或文件顺序伪造时间。

raw eye 值如 `frame_left/frame_right` 只在内存适配层标准化为 `left/right`，同时保留 `eye_raw`。

正式 pupil-only 候选包括现有 producer 明确提供的 pupil diameter/area/axes、hard/soft pupil fraction 及其经过声明的 baseline/within-person/dynamic 派生。禁止：

- 用 hard/soft iris fraction 冒充虹膜直径；
- 从当前分类比例恢复 PIR；
- 为通过旧 alignment 而补伪 PIR/OAR 列；
- 把 RITnet success 当 pupil geometry valid；
- 把人工 padding 纳入科学分母；
- 因结果显著与否反向选择 QC 阈值或窗口。

QC 至少区分 source、geometry、analysis-domain、uncertainty、temporal，并保留 missing/rejected/interpolation 的独立状态。

NIR 的 current/previous stimulus luminance、contrast、size、identity 与 time-on-task 是 pupil 专属前置混淆审计。它们不因在 NIR 中重要就默认进入毫米波/RGB 科学模型。

## RGB、毫米波与融合

RGB 与毫米波在进入本仓库融合前必须先在各自 producer 侧产出版本化 merge-ready 表和运行 manifest。本仓库不复制第二份 producer 逻辑。

RGB：先主脸/有效分钟/eye openness/blink，再头动/pose/global motion/AU；blink/PERCLOS 必须有事件验证；NIR ocular aperture 与 RGB EAR/blink 只能交叉验证，不能互相改名。

毫米波：HR 与 RR/BR 分轨；没有匹配 ECG/RSP 时不得标记外部生理验证通过；IBI/HRV 未过 beat/reference gate 前不作核心生理结论；`breath_rate/breathing_rate`、quality 位置、NaN/Inf candidate 等 producer contract 问题必须先修复；缺失/无效原始记录不补零。

融合顺序固定为设计基线 → 行为上下文基线 → 单模态 → 有理论依据的两两 → 全模态。解释模型与预测模型分轨；缩放、插补、特征选择和调参都必须在训练 participant 内完成。

## 文件与历史保护规则

- `docs/工作记录/` 中的日期型工作记录属于科研 provenance，默认永久保留；不得擅自删除、合并、压缩或追溯改写。
- 旧 BBB、旧 PIR、旧 RGB/NIR validation 脚本只要属于历史复现资产，默认保留；“不再正式使用”不等于“删除”。
- 删除、覆盖、重命名历史资产或清理含用户未提交修改的工作树，必须获得针对性授权。
- 当前方法只保留一个 canonical 说明位置；其他文档需要时链接，不复制第二份当前方法。
- 不为了目录整齐而修改科学参数、样本规则、窗口或结果。

## 测试与放行

任何正式扩展至少经过：

1. config/path preflight；
2. cohort/session uniqueness 与 repeat-group 覆盖；
3. schema 和主键测试；
4. 时间键/对齐测试；
5. participant-disjoint fold 泄漏测试；
6. 代表场次真实输入 smoke；
7. 输出 manifest/hash；
8. 才能扩大到当前 cohort；
9. 后续追加 cohort 使用相同 schema 重新跑合并与拟合。

开发失败、不可评估、当前受阻和不适合报告的结果保留；不得为了“全绿”复制假数据、补零、删除失败证据或降低科学门。
