# 08-24-01｜scripts与models清理及行为版本纠偏工作记录

> 文档性质：仓库整理 / 当前状态纠偏  
> 日期：08-24  
> 当天序号：01  
> 当前维护主线：`main`  
> 状态：已执行

## 一、本轮目的

本轮完成三件事：

1. 按用户明确授权，删除已经退出当前正式路线的历史 scripts；
2. 删除只剩说明文件的根 `models/` 空壳目录；
3. 纠正行为分析版本：旧 v3.0 BBB 分析不能继续作为最终正式实验行为分析。

工作记录本身未删除、未合并、未追溯改写。

## 二、scripts 删除

用户已逐项审查并批准删除以下 17 个旧入口：

```text
scripts/00-目录与映射.md
scripts/build_formal_experiment_recommendation.py
scripts/compare_nir_history_review.py
scripts/deepvog_pupil.py
scripts/gate1_contract_check.py
scripts/iris_landmark.py
scripts/nir_detect_batch.py
scripts/nir_sequence_detect.py
scripts/roi_check.py
scripts/roi_common.py
scripts/roi_compare.py
scripts/roi_faceparts.py
scripts/roi_mediapipe.py
scripts/roi_yolo.py
scripts/roi_yunet.py
scripts/run_all_backends.py
scripts/run_roi_selection.py
```

删除原因统一为：这些脚本属于 08 月中旬 ROI 选型、多算法 benchmark、PuRe/PuReST、DeepVOG、MediaPipe Iris、Gate1 或一次性历史报告生成阶段，均不参与当前正式 NIR 主链；其中多项还直接依赖已经删除的历史模型与 artifacts。继续留在 `main` 会造成“看起来可运行、实际上已经缺失依赖或已经退出研究路线”的误导。

历史实现仍可通过 Git 历史与冻结的 `history/tracking-era-2026-08` 追溯。

## 三、scripts 当前剩余入口

清理后 `scripts/` 只保留：

```text
scripts/
├── README.md
├── evaluate_yolo_eye_test.py
├── extract_eye_dataset.py
└── sart_formal_analysis.py
```

其中前两项分别承担当前 YOLO 眼框评估和训练数据 provenance。

`sart_formal_analysis.py` 暂时保留，但本轮已明确：它当前仍基于旧 v3.0 BBB 行为分析实现，不能继续标记为最终正式行为分析入口。后续需要按最终 v3.1.3 BB 版本重建后，再决定是否原位替换或重新命名。

## 四、models 根目录删除

上一轮已经删除：

```text
models/external/
models/historical/
```

当时为了记录历史留下了：

```text
models/README.md
```

用户指出该目录已经没有当前职责，不应为了说明“为什么删了”而长期留下空壳目录。本轮因此进一步删除 `models/README.md`，使根 `models/` 彻底消失。

删除历史、上游来源和淘汰原因统一由：

```text
docs/工作记录/
docs/050-decisions/
Git history
```

承担。

当前正式模型的 canonical 位置不变：

```text
training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt
runtime/nir-formal/models/nir-eye-yolo26n-best.pt
runtime/nir-formal/models/ritnet-best_model.pkl
runtime/nir-formal/ritnet/
```

## 五、行为分析版本纠偏

本轮核对发现旧行为分析配置与最终正式实验存在实质版本差异。

旧：

```text
configs/sart_formal.yaml
FocusWave / SART v3.0
subjects: sub-011..030
block_order: [B, B, B]
3 个正式 B block
```

旧 `docs/030-behavior/sart-formal/000-正式SART行为分析计划.md` 同样明确按 BBB、B1↔B3、20×1296 试次设计和解释。

而当前正式 runtime 已冻结最终版本：

```text
FocusWave v3.1.3
min_subject_number: 31
expected_formal_blocks: 2
formal phases:
- block1
- block2
```

因此结论是：

> `docs/030-behavior/sart-formal/` 是 v3.0 BBB 的历史分析包；`configs/sart_formal.yaml`、`scripts/sart_formal_analysis.py` 和 `src/attention_pipeline/behavior_formal/` 目前仍带有旧 BBB 假设。它们不能继续冒充最终正式实验的当前行为分析。

旧报告不追溯改写；当前行为分析需要按最终 BB 数据重新建立。

## 六、下一步行为分析要求

后续重建正式 behavior 分析前，需要先用最终正式数据核实：

1. 实际正式被试集合（以 v3.1.3、编号 ≥31 的最终数据为准）；
2. 每名被试是否均为两个 B block；
3. 每 block 的实际 trial 数、No-Go 数、probe 数与 probe 位置；
4. 最终行为 CSV 字段与时间戳语义；
5. 异常/排除被试；
6. 两 block 设计下应采用的统计比较，删除旧 B1↔B3、三水平 Friedman/AnovaRM 等仅适用于 BBB 的分析；
7. 与 NIR phase/block 对齐口径。

不能仅把 `BBB` 文本替换成 `BB`，而必须重新审计分析契约和统计设计。

## 七、本轮同步更新的当前说明

同步更新：

```text
README.md
AGENTS.md
scripts/README.md
configs/README.md
docs/010-overview/013-仓库资产与复现关系.md
docs/030-behavior/README.md
```

这些文件现在统一表达：

- 根 `models/`、`artifacts/` 不再存在于当前 main；
- 当前正式 NIR 资产只认 training/runtime 的 canonical 链；
- 当前最终实验版本为 FocusWave v3.1.3、两个 B block；
- 旧 BBB SART 包保留为历史，不冒充当前最终 behavior 分析。
