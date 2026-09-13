"""Verify that the Q1 four-class runs actually honoured the frozen contract.

方法权威：`分析设计/1.16.24-Q1四分类正式分析与两条公平对照路线_20260913.md` §3/§6

逐运行机械检查以下条款，任何一条不成立即 FAIL 并打印证据：

1. 任务与折结构：`task = q1_nominal_4class_multiclass`、`inner_splits = 5`、
   `C ∈ {0.01, 0.1, 1.0, 10.0}`、`outer_method = leave_one_participant_out`。
2. 外层参与者互斥：每折 `outer_train_group_ids ∩ outer_test_group_ids = ∅`，
   且 `outer_test_group_ids` 恰好一名参与者。
3. 内层参与者互斥：折审计里每个内层折的 `train_group_ids ∩ validation_group_ids = ∅`。
4. 预处理只在训练折拟合：`final_refit.preprocessing.fit_group_ids` 必须等于该折外层训练组，
   且 `n_fit_rows` 等于外层训练行数。
5. 零个体校准：`final_refit.preprocessing.participant_specific_transform = false`。
6. 不确定性：bootstrap `1000` 次 / seed `20260830` / `0.95` / **不重训**。
7. 概率完整性：成功折的四列概率必须有限且逐行和为 1；失败折四列概率必须全为空。
8. 主指标与基线：`participant_macro_multiclass_log_loss` 与逐折类别先验基线都可读，
   且基线是**逐折**重算而不是常数。

用法：
    python verify_four_class_contract.py [--root <运行根>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis\SupervisedRuns4ClassV1")
EXPECTED_TASK = "q1_nominal_4class_multiclass"
EXPECTED_INNER_SPLITS = 5
EXPECTED_C = [0.01, 0.1, 1.0, 10.0]
EXPECTED_ROUTE_A_SUFFIX = "__included_missing_aware__4classA"
EXPECTED_ROUTE_B_SUFFIX = "__included_missing_aware__4classB"
PROBABILITY_COLUMNS = (
    "p_q1_multiclass_1",
    "p_q1_multiclass_2",
    "p_q1_multiclass_3",
    "p_q1_multiclass_4",
)


def _check(condition: bool, label: str, detail: str, failures: list[str]) -> None:
    if not condition:
        failures.append(f"{label}: {detail}")


def _verify_run(run_dir: Path) -> dict[str, object]:
    failures: list[str] = []
    manifest_path = run_dir / "run_manifest.json"
    predictions_path = run_dir / "probe_predictions.csv"
    audits_path = run_dir / "fold_audits.json"
    evaluation_path = run_dir / "multiclass_evaluation.json"
    for path in (manifest_path, predictions_path, audits_path, evaluation_path):
        if not path.is_file():
            return {"run_dir": str(run_dir), "ok": False, "failures": [f"missing {path.name}"]}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audits = json.loads(audits_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))["evaluation"]
    predictions = pd.read_csv(predictions_path, low_memory=False)

    # --- 1. 任务与折结构 -------------------------------------------------------------
    _check(manifest.get("task") == EXPECTED_TASK, "task", str(manifest.get("task")), failures)
    _check(int(manifest.get("inner_splits", -1)) == EXPECTED_INNER_SPLITS,
           "inner_splits", str(manifest.get("inner_splits")), failures)
    _check([float(v) for v in manifest.get("c_candidates", [])] == EXPECTED_C,
           "c_candidates", str(manifest.get("c_candidates")), failures)
    _check(manifest.get("outer_method") == "leave_one_participant_out",
           "outer_method", str(manifest.get("outer_method")), failures)
    _check(bool(manifest.get("zero_individual_calibration")) is True,
           "zero_individual_calibration", str(manifest.get("zero_individual_calibration")), failures)
    _check(bool(manifest.get("outer_test_outcomes_passed_to_model")) is False,
           "outer_test_outcomes_passed_to_model",
           str(manifest.get("outer_test_outcomes_passed_to_model")), failures)

    # --- 6. 不确定性 -----------------------------------------------------------------
    outer_eval = manifest.get("outer_evaluation", {})
    _check(int(outer_eval.get("bootstrap_replicates", -1)) == 1000,
           "bootstrap_replicates", str(outer_eval.get("bootstrap_replicates")), failures)
    _check(int(outer_eval.get("bootstrap_seed", -1)) == 20260830,
           "bootstrap_seed", str(outer_eval.get("bootstrap_seed")), failures)
    _check(float(outer_eval.get("bootstrap_confidence_level", -1)) == 0.95,
           "bootstrap_confidence_level", str(outer_eval.get("bootstrap_confidence_level")), failures)
    _check(outer_eval.get("bootstrap_retrain_models") is False,
           "bootstrap_retrain_models", str(outer_eval.get("bootstrap_retrain_models")), failures)

    # --- 2/3/4/5. 折级检查 -----------------------------------------------------------
    n_outer = 0
    n_missing_class_folds = 0
    n_inner_fold_records = 0
    n_success = 0
    for audit in audits:
        n_outer += 1
        train_groups = set(map(str, audit.get("outer_train_group_ids") or []))
        test_groups = set(map(str, audit.get("outer_test_group_ids") or []))
        label = f"outer_fold={audit.get('outer_fold_group')}"
        _check(not (train_groups & test_groups), f"{label} outer overlap",
               str(sorted(train_groups & test_groups)), failures)
        _check(len(test_groups) == 1, f"{label} outer test size", str(sorted(test_groups)), failures)

        if not audit.get("outer_train_class_complete", False):
            n_missing_class_folds += 1
            _check(audit.get("failed") is True, f"{label} missing-class fold not marked failed",
                   str(audit.get("failed")), failures)
            _check("not estimable" in str(audit.get("reason", "")),
                   f"{label} missing-class reason", str(audit.get("reason")), failures)
            continue
        if audit.get("failed"):
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
            n_inner_fold_records += 1
            inner_train = set(map(str, inner.get("train_group_ids") or []))
            inner_valid = set(map(str, inner.get("validation_group_ids") or []))
            _check(not (inner_train & inner_valid),
                   f"{label} inner fold {inner.get('inner_fold')} overlap",
                   str(sorted(inner_train & inner_valid)), failures)

    # --- 7. 概率完整性 ---------------------------------------------------------------
    _check(all(column in predictions.columns for column in PROBABILITY_COLUMNS),
           "probability columns present",
           str([c for c in PROBABILITY_COLUMNS if c not in predictions.columns]), failures)
    if all(column in predictions.columns for column in PROBABILITY_COLUMNS):
        failed_mask = predictions["model_failed"].astype(bool)
        probabilities = predictions.loc[~failed_mask, list(PROBABILITY_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        )
        _check(bool(np.isfinite(probabilities.to_numpy(dtype=float)).all()),
               "successful-fold probabilities finite", "non-finite probability found", failures)
        row_sums = probabilities.sum(axis=1).to_numpy(dtype=float)
        _check(bool(np.allclose(row_sums, 1.0, atol=1e-6)),
               "successful-fold probabilities sum to 1",
               f"max|sum-1|={float(np.max(np.abs(row_sums - 1.0))) if len(row_sums) else 'n/a'}", failures)
        if failed_mask.any():
            failed_probs = predictions.loc[failed_mask, list(PROBABILITY_COLUMNS)].apply(
                pd.to_numeric, errors="coerce"
            )
            _check(bool(failed_probs.isna().all().all()),
                   "failed-fold probabilities all null", "failed fold produced a probability", failures)

    # --- 8. 主指标与逐折基线 ---------------------------------------------------------
    for model_id, block in evaluation["per_model"].items():
        summary = block.get("summary", {})
        _check(np.isfinite(float(summary.get("participant_macro_multiclass_log_loss", np.nan))),
               f"model {model_id} primary metric", str(summary), failures)
        prior = evaluation["prior_baseline"]["per_model"].get(model_id, {})
        per_fold = prior.get("per_fold") or []
        # 逐折基线的键名是 `participant_prior_baseline_multiclass_log_loss`；同时检查
        # 逐折类别先验本身也随折变化——这比只查基线值更强，能证明基线是**逐折重算**的。
        baseline_values = [
            entry.get("participant_prior_baseline_multiclass_log_loss") for entry in per_fold
        ]
        prior_vectors = [
            tuple(sorted((entry.get("class_priors") or {}).items())) for entry in per_fold
        ]
        _check(all(value is not None for value in baseline_values),
               f"model {model_id} per-fold baseline present",
               f"null entries={sum(1 for v in baseline_values if v is None)}", failures)
        _check(len({round(float(v), 9) for v in baseline_values if v is not None}) > 1,
               f"model {model_id} prior baseline is per-fold",
               f"distinct per-fold baselines={len({round(float(v), 9) for v in baseline_values if v is not None})}",
               failures)
        _check(len(set(prior_vectors)) > 1,
               f"model {model_id} class priors are per-fold",
               f"distinct prior vectors={len(set(prior_vectors))}", failures)

    return {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "route": manifest.get("route", "A" if run_dir.name.endswith(EXPECTED_ROUTE_A_SUFFIX) else "B"),
        "ok": not failures,
        "failures": failures,
        "n_outer_folds": n_outer,
        "n_successful_folds": n_success,
        "n_missing_class_folds": n_missing_class_folds,
        "n_inner_fold_records": n_inner_fold_records,
        "n_prediction_rows": int(len(predictions)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)

    run_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and (
            path.name.endswith(EXPECTED_ROUTE_A_SUFFIX) or path.name.endswith(EXPECTED_ROUTE_B_SUFFIX)
        )
    )
    if not run_dirs:
        print(json.dumps({"status": "NO_RUNS", "root": str(root)}, ensure_ascii=False, indent=2))
        return 1

    reports = [_verify_run(path) for path in run_dirs]
    failed = [report for report in reports if not report["ok"]]
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "root": str(root),
        "n_runs": len(reports),
        "n_pass": len(reports) - len(failed),
        "n_fail": len(failed),
        "totals": {
            "outer_folds": sum(int(r["n_outer_folds"]) for r in reports),
            "successful_folds": sum(int(r["n_successful_folds"]) for r in reports),
            "missing_class_folds": sum(int(r["n_missing_class_folds"]) for r in reports),
            "inner_fold_records": sum(int(r["n_inner_fold_records"]) for r in reports),
            "prediction_rows": sum(int(r["n_prediction_rows"]) for r in reports),
        },
        "runs": reports,
    }
    out_path = root / "four_class_contract_verification.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, ensure_ascii=False, indent=2))
    for report in failed:
        print(f"\nFAIL {report['run_id']}")
        for failure in report["failures"][:20]:
            print(f"   - {failure}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
