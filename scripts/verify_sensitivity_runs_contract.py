"""Verify that the 1.16.25 / 1.16.26 sensitivity runs honoured the frozen contract.

方法权威：
- `分析设计/1.16.25-首轮主窗口长度敏感性预注册_20260913.md`
- `分析设计/1.16.26-瞳孔头动协变量敏感性预注册_20260913.md`

这两项走的是**冻结的二分类管线**（`configs/supervised_learning_v1.yaml` 或其等价敏感性配置）。
"复用同一配置"**不等于**"这些新运行真的按合同跑了"，因此必须逐折核验它们**自己**的折审计，
与原始 19 个运行同等对待。逐运行检查：

1. `task = q1_equals_1_vs_2_3_4`、`inner_splits = 5`、`C ∈ {0.01,0.1,1.0,10.0}`、
   `outer_method = leave_one_participant_out`、`zero_individual_calibration = true`。
2. 外层：训练/测试参与者互斥，测试折恰一名参与者。
3. 内层：每个内层折训练/验证参与者互斥（二分类线保留逐内层折审计，可核验）。
4. 预处理只在外层训练折拟合：`fit_group_ids` 等于外层训练组，`n_fit_rows` 等于外层训练行数。
5. `participant_specific_transform = false`。
6. bootstrap 1,000 / seed `20260830` / 0.95 / 不重训。
7. 成功折 `p_q1_equals_1` 有限且落在 [0,1]；失败折该列为空。

用法：
    python verify_sensitivity_runs_contract.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis")
RUNS = {
    "1.16.25 window 10s": BASE / "SupervisedRunsWindowSensitivityV1"
    / "AS.behavior_reference__window10s__included_missing_aware",
    "1.16.25 window 20s": BASE / "SupervisedRunsWindowSensitivityV1"
    / "AS.behavior_reference__window20s__included_missing_aware",
    "1.16.26 ocular head-motion": BASE / "SupervisedRunsOcularHeadmotionSensitivityV1"
    / "AS.ocular_headmotion_sensitivity__included_missing_aware",
}
EXPECTED_TASK = "q1_equals_1_vs_2_3_4"
EXPECTED_C = [0.01, 0.1, 1.0, 10.0]
PROBABILITY_COLUMN = "p_q1_equals_1"


def _check(condition: bool, label: str, detail: str, failures: list[str]) -> None:
    if not condition:
        failures.append(f"{label}: {detail}")


def verify(run_dir: Path) -> dict[str, object]:
    failures: list[str] = []
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    audits = json.loads((run_dir / "fold_audits.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run_dir / "probe_predictions.csv", low_memory=False)

    _check(manifest.get("task") == EXPECTED_TASK, "task", str(manifest.get("task")), failures)
    _check(int(manifest.get("inner_splits", -1)) == 5,
           "inner_splits", str(manifest.get("inner_splits")), failures)
    _check([float(v) for v in manifest.get("c_candidates", [])] == EXPECTED_C,
           "c_candidates", str(manifest.get("c_candidates")), failures)
    _check(manifest.get("outer_method") == "leave_one_participant_out",
           "outer_method", str(manifest.get("outer_method")), failures)
    _check(bool(manifest.get("zero_individual_calibration")) is True,
           "zero_individual_calibration", str(manifest.get("zero_individual_calibration")), failures)
    outer_eval = manifest.get("outer_evaluation", {})
    _check(int(outer_eval.get("bootstrap_replicates", -1)) == 1000,
           "bootstrap_replicates", str(outer_eval.get("bootstrap_replicates")), failures)
    _check(int(outer_eval.get("bootstrap_seed", -1)) == 20260830,
           "bootstrap_seed", str(outer_eval.get("bootstrap_seed")), failures)
    _check(float(outer_eval.get("bootstrap_confidence_level", -1)) == 0.95,
           "bootstrap_confidence_level", str(outer_eval.get("bootstrap_confidence_level")), failures)
    _check(outer_eval.get("bootstrap_retrain_models") is False,
           "bootstrap_retrain_models", str(outer_eval.get("bootstrap_retrain_models")), failures)

    n_outer = n_success = n_inner = n_failed = 0
    for audit in audits:
        n_outer += 1
        label = f"fold={audit.get('outer_fold_group')}"
        train_groups = set(map(str, audit.get("outer_train_group_ids") or []))
        test_groups = set(map(str, audit.get("outer_test_group_ids") or []))
        _check(not (train_groups & test_groups), f"{label} outer overlap",
               str(sorted(train_groups & test_groups)), failures)
        _check(len(test_groups) == 1, f"{label} outer test size", str(sorted(test_groups)), failures)
        if audit.get("failed"):
            n_failed += 1
            continue
        n_success += 1
        final_refit = audit.get("final_refit") or {}
        preprocessing = final_refit.get("preprocessing") or {}
        fit_groups = set(map(str, preprocessing.get("fit_group_ids") or []))
        _check(fit_groups == train_groups, f"{label} preprocessing fit groups",
               f"fit={len(fit_groups)} train={len(train_groups)}", failures)
        _check(int(preprocessing.get("n_fit_rows", -1)) == int(audit.get("n_outer_train_rows", -2)),
               f"{label} preprocessing fit rows",
               f"n_fit_rows={preprocessing.get('n_fit_rows')} train_rows={audit.get('n_outer_train_rows')}",
               failures)
        _check(preprocessing.get("participant_specific_transform") is False,
               f"{label} participant_specific_transform",
               str(preprocessing.get("participant_specific_transform")), failures)
        selection = audit.get("selection") or {}
        inner_folds = selection.get("inner_fold_audits") or []
        _check(len(inner_folds) > 0, f"{label} inner fold audits", str(len(inner_folds)), failures)
        for inner in inner_folds:
            n_inner += 1
            inner_train = set(map(str, inner.get("train_group_ids") or []))
            inner_valid = set(map(str, inner.get("validation_group_ids") or []))
            _check(not (inner_train & inner_valid),
                   f"{label} inner fold {inner.get('inner_fold')} overlap",
                   str(sorted(inner_train & inner_valid)), failures)

    _check(PROBABILITY_COLUMN in predictions.columns, "probability column present",
           PROBABILITY_COLUMN, failures)
    if PROBABILITY_COLUMN in predictions.columns:
        failed_mask = predictions["model_failed"].astype(bool)
        successful = pd.to_numeric(
            predictions.loc[~failed_mask, PROBABILITY_COLUMN], errors="coerce"
        ).to_numpy(dtype=float)
        _check(bool(np.isfinite(successful).all()) and bool(((successful >= 0) & (successful <= 1)).all()),
               "successful-fold probabilities in [0,1]", "out-of-range or non-finite value", failures)
        if failed_mask.any():
            failed_values = pd.to_numeric(
                predictions.loc[failed_mask, PROBABILITY_COLUMN], errors="coerce"
            )
            _check(bool(failed_values.isna().all()), "failed-fold probabilities all null",
                   "failed fold produced a probability", failures)

    return {
        "run_dir": str(run_dir),
        "ok": not failures,
        "failures": failures,
        "n_outer_folds": n_outer,
        "n_successful_folds": n_success,
        "n_failed_folds": n_failed,
        "n_inner_fold_records": n_inner,
        "n_prediction_rows": int(len(predictions)),
        "code_sha": manifest.get("provenance", {}).get("code_sha"),
        "config_digest": manifest.get("provenance", {}).get("config_digest"),
    }


def main() -> int:
    reports = []
    for name, run_dir in RUNS.items():
        if not run_dir.is_dir():
            reports.append({"name": name, "run_dir": str(run_dir), "ok": False,
                            "failures": ["run directory missing"]})
            continue
        report = verify(run_dir)
        report["name"] = name
        reports.append(report)

    failed = [r for r in reports if not r["ok"]]
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "n_runs": len(reports),
        "n_pass": len(reports) - len(failed),
        "totals": {
            "outer_folds": sum(int(r.get("n_outer_folds", 0)) for r in reports),
            "successful_folds": sum(int(r.get("n_successful_folds", 0)) for r in reports),
            "failed_folds": sum(int(r.get("n_failed_folds", 0)) for r in reports),
            "inner_fold_records": sum(int(r.get("n_inner_fold_records", 0)) for r in reports),
        },
        "runs": reports,
    }
    out = BASE / "sensitivity_runs_contract_verification.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False, indent=2))
    for report in reports:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"\n{status}  {report['name']}")
        print(f"   code_sha={report.get('code_sha')}  config_digest={report.get('config_digest')}")
        print(f"   外层折={report.get('n_outer_folds')} 成功={report.get('n_successful_folds')} "
              f"失败={report.get('n_failed_folds')} 内层折记录={report.get('n_inner_fold_records')} "
              f"预测行={report.get('n_prediction_rows')}")
        for failure in report["failures"][:10]:
            print(f"   - {failure}")
    print(f"\n报告写入: {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
