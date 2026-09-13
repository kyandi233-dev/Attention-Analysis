"""Summarise the 1.16.26 Ocular head-motion sensitivity.

方法权威：`分析设计/1.16.26-瞳孔头动协变量敏感性预注册_20260913.md`

两个模型跑在**同一个分析集合**上（1,779 probes / 60 participants，与"仅 Ocular 特征
有限"的集合逐行相同，丢 0 行），因此配对增量在冻结合同下合法。配对增量复用**冻结的**
`evaluation.fixed_oof_participant_bootstrap`（1000 / 20260830 / 0.95 / 不重训），
不另写重采样实现。

增量定义与二分类线一致：`baseline_log_loss_minus_added_log_loss`。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from attention_pipeline.supervised_learning.evaluation import fixed_oof_participant_bootstrap

RUN_DIR = Path(
    r"D:\Project\厚粲杯\11_数据\_FormalAnalysis\SupervisedRunsOcularHeadmotionSensitivityV1"
    r"\AS.ocular_headmotion_sensitivity__included_missing_aware"
)
OUT_ROOT = RUN_DIR.parent
BASELINE_MODEL = "ocular_reference"
ADDED_MODEL = "ocular_plus_headmotion"


def _find(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def main() -> int:
    evaluation = pd.read_csv(RUN_DIR / "model_evaluation.csv")
    participant = pd.read_csv(RUN_DIR / "participant_log_loss.csv")
    bootstrap = json.loads((RUN_DIR / "participant_bootstrap.json").read_text(encoding="utf-8"))

    models = {
        str(row.model_id): {
            "n_participants": int(row.n_participants),
            "n_probes": int(row.n_probes),
            "participant_equal_log_loss": float(row.participant_equal_log_loss),
            "pooled_probe_log_loss_descriptive": float(row.pooled_probe_log_loss_descriptive),
        }
        for row in evaluation.itertuples()
    }
    for name, entry in models.items():
        entry["ci_lower_95"] = None
        entry["ci_upper_95"] = None
    # participant_bootstrap.json is a LIST of per-model records keyed by model_id.
    per_model_ci: dict[str, dict[str, float]] = {}
    if isinstance(bootstrap, list):
        for block in bootstrap:
            if isinstance(block, dict) and "model_id" in block:
                per_model_ci[str(block["model_id"])] = {
                    "ci_lower_95": float(block["ci_lower"]),
                    "ci_upper_95": float(block["ci_upper"]),
                    "bootstrap_point_estimate": float(block["point_estimate"]),
                    "bootstrap_replicates": int(block["replicates"]),
                    "bootstrap_seed": int(block["seed"]),
                    "bootstrap_confidence_level": float(block["confidence_level"]),
                }
    elif isinstance(bootstrap, dict):
        for name, block in bootstrap.items():
            low = _find(block, "ci_lower")
            high = _find(block, "ci_upper")
            if low is not None and high is not None:
                per_model_ci[str(name)] = {
                    "ci_lower_95": float(low),
                    "ci_upper_95": float(high),
                }
    if set(per_model_ci) != set(models):
        raise AssertionError(
            f"bootstrap records {sorted(per_model_ci)} do not match models {sorted(models)}"
        )
    for name, entry in models.items():
        entry.update(per_model_ci[name])

    wide = participant.pivot(index="participant_group_id", columns="model_id", values="mean_log_loss")
    if BASELINE_MODEL not in wide.columns or ADDED_MODEL not in wide.columns:
        raise AssertionError(f"expected both models in the participant table; got {list(wide.columns)}")
    paired = wide[[BASELINE_MODEL, ADDED_MODEL]].dropna()
    if len(paired) != models[BASELINE_MODEL]["n_participants"]:
        raise AssertionError(
            f"paired participant count {len(paired)} != {models[BASELINE_MODEL]['n_participants']}"
        )
    increments = (paired[BASELINE_MODEL] - paired[ADDED_MODEL]).to_numpy(dtype=float)
    increment_bootstrap = fixed_oof_participant_bootstrap(increments)
    increment_bootstrap["paired_model_resampling"] = True
    increment_bootstrap["increment_definition"] = "baseline_log_loss_minus_added_log_loss"
    increment_bootstrap["baseline_model_id"] = BASELINE_MODEL
    increment_bootstrap["added_model_id"] = ADDED_MODEL

    point = float(increments.mean())
    ci_low = float(increment_bootstrap["ci_lower"])
    ci_high = float(increment_bootstrap["ci_upper"])
    excludes_zero = bool((ci_low > 0.0) or (ci_high < 0.0))

    report = {
        "method_authority": "分析设计/1.16.26-瞳孔头动协变量敏感性预注册_20260913.md",
        "analysis_set_id": "AS.ocular_headmotion_sensitivity",
        "sample": {
            "n_probes": models[BASELINE_MODEL]["n_probes"],
            "n_participants": models[BASELINE_MODEL]["n_participants"],
            "note": (
                "both models run on the identical rows and participant folds; restricting to "
                "ocular-finite rows already guarantees both covariates are finite, so zero rows "
                "are lost relative to the ocular-only sample"
            ),
        },
        "models": models,
        "paired_increment": {
            "definition": "ocular_reference minus ocular_plus_headmotion",
            "point_estimate": point,
            "ci_lower_95": ci_low,
            "ci_upper_95": ci_high,
            "excludes_zero": excludes_zero,
            "bootstrap": increment_bootstrap,
        },
        "decision": (
            "head motion is a material confound: report direction and magnitude and stop using "
            "the unadjusted ocular statement"
            if excludes_zero
            else "the ocular first-round conclusion is robust to head-motion adjustment"
        ),
        "scope_boundary": (
            "head-motion half only; the scale half is NOT_EXECUTABLE_WITHOUT_PRODUCER_RECOMPUTE "
            "because no iris-pixel / face-size proxy exists in the frozen outputs"
        ),
    }

    (OUT_ROOT / "headmotion_sensitivity_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# Ocular 头动协变量敏感性结果（1.16.26）",
        "",
        "权威方法文件：`分析设计/1.16.26-瞳孔头动协变量敏感性预注册_20260913.md`",
        "",
        "## 1. 样本",
        "",
        f"- 分析集合 `AS.ocular_headmotion_sensitivity`：**{models[BASELINE_MODEL]['n_probes']}** probes / "
        f"**{models[BASELINE_MODEL]['n_participants']}** participants",
        "- 两个模型跑在**逐行相同**的样本与**相同参与者折**上；限制到 Ocular 特征有限已自动蕴含两个协变量存在，"
        "因此相对「仅 Ocular」样本**丢 0 行**。",
        "",
        "## 2. 结果（主指标 = 参与者宏平均对数损失）",
        "",
        "| 模型 | 预测列 | probes | 参与者 | 参与者宏平均 log loss | 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name in (BASELINE_MODEL, ADDED_MODEL):
        entry = models[name]
        low = entry.get("ci_lower_95")
        high = entry.get("ci_upper_95")
        ci = f"[{low:.6f}, {high:.6f}]" if low is not None and high is not None else "n/a"
        n_columns = 5 if name == BASELINE_MODEL else 7
        lines.append(
            f"| `{name}` | {n_columns} 列 | {entry['n_probes']} | {entry['n_participants']} | "
            f"**{entry['participant_equal_log_loss']:.6f}** | {ci} |"
        )
    lines += [
        "",
        "## 3. 配对增量（同分析集合，配对合法）",
        "",
        f"- 定义：`ocular_reference − ocular_plus_headmotion`（正 = 加入协变量后损失更低）",
        f"- 点估计：**{point:+.6f}**",
        f"- 95% 置信区间：**[{ci_low:+.6f}, {ci_high:+.6f}]**",
        f"- 区间是否排除 0：**{excludes_zero}**",
        f"- 判定：{report['decision']}",
        "",
        "## 4. 边界",
        "",
        "- 头动臂是**被协变量调整后的敏感性臂**，**不是**科学模态模型；不得写成「Movement 有贡献」、"
        "模态比较结果或特征重要性证据。",
        "- 两个姿势列在生产端的角色固定为 `sensitivity_auxiliary`，**不得**晋级到正式 feature registry。",
        f"- {report['scope_boundary']}",
        "- 本敏感性走兼容 `feature_schemes` 接口，**未运行** registry 的 time-legality 闸门；协变量合法性由"
        "预注册 §3 逐字引用的生产端 `time_legality_evidence` 保证。",
    ]
    (OUT_ROOT / "HEADMOTION_SENSITIVITY_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(
        {
            "models": {k: {kk: vv for kk, vv in v.items() if kk != "bootstrap"} for k, v in models.items()},
            "paired_increment": {
                "point_estimate": point,
                "ci_lower_95": ci_low,
                "ci_upper_95": ci_high,
                "excludes_zero": excludes_zero,
            },
            "decision": report["decision"],
            "summary_json": str(OUT_ROOT / "headmotion_sensitivity_summary.json"),
            "report_md": str(OUT_ROOT / "HEADMOTION_SENSITIVITY_REPORT.md"),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
