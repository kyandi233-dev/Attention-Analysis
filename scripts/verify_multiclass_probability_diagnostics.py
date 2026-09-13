"""机械核验多类折外概率诊断层的产物（独立复算，不复用被测模块）。

文件：verify_multiclass_probability_diagnostics.py
版本：1.0.0
功能：对 ``multiclass_probability_diagnostics`` 目录做**独立**核验。所有复算都用
      pandas/numpy 现场重写一遍口径，**不调用**
      ``probability_diagnostics_multiclass`` 的任何函数，因此该脚本能发现
      被测模块自身的口径错误，而不只是复述它的输出。
用法：
    $env:PYTHONPATH = "<worktree>\\src"
    python scripts/verify_multiclass_probability_diagnostics.py `
      --diagnostics-root <...\\multiclass_probability_diagnostics> `
      --run-root <...\\SupervisedRuns4ClassV1> `
      --frozen-summary <...\\SupervisedRuns4ClassV1\\four_class_summary.csv> `
      --output <verification.json>

检查项（任一失败即 exit 1，并逐条打印）：
    A 诊断表身份键唯一
    B 先验重建与冻结基线的对账计数闭合且 0 处不一致
    C 重新计算的主指标与冻结汇总逐行一致
    D 逐 probe 计数闭合（诊断表 n_probes == 归档成功行数）
    E 参与者宏平均多类布里尔分数独立复算一致
    F 逐折类别先验常数预测器的布里尔分数与对数损失独立复算一致
    G 可靠性分箱对探针的计数闭合
    H 期望校准误差可由分箱表还原
    I 每个区间包住其点估计
    J 治理标志（非选择性 / 不重训预测模型 / 不报告参与者级 AUROC）
    K 校准斜率可解读性分级计数闭合
    L 产物 SHA-256 记录
依赖：pandas、numpy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "q1_nominal_4class"
PROB_COLUMNS = [f"p_q1_multiclass_{k}" for k in (1, 2, 3, 4)]
CLASSES = (1, 2, 3, 4)
IDENTITY = ["run_id", "analysis_set_id", "model_id", "membership_type"]
PRIOR_FLOOR = 1e-12


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    """独立重写一遍归档清洗口径。"""
    data = frame.copy()
    if "model_failed" in data.columns:
        failed = data["model_failed"]
        if failed.dtype == object:
            failed = failed.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        data = data[~failed.astype(bool)]
    data[LABEL] = pd.to_numeric(data[LABEL], errors="coerce")
    for column in PROB_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[LABEL, "participant_group_id", "outer_fold_group", *PROB_COLUMNS])
    data = data[data[LABEL].isin(list(CLASSES))]
    probability = data.loc[:, PROB_COLUMNS].to_numpy(dtype=float)
    data = data[
        np.isfinite(probability).all(axis=1)
        & (probability >= 0.0).all(axis=1)
        & (probability <= 1.0).all(axis=1)
    ]
    data[LABEL] = data[LABEL].astype(int)
    data["participant_group_id"] = data["participant_group_id"].astype(str)
    data["outer_fold_group"] = data["outer_fold_group"].astype(str)
    return data.reset_index(drop=True)


def _brier(frame: pd.DataFrame, probability: np.ndarray | None = None) -> np.ndarray:
    labels = frame[LABEL].to_numpy(dtype=int)
    proba = frame.loc[:, PROB_COLUMNS].to_numpy(dtype=float) if probability is None else probability
    onehot = np.zeros_like(proba)
    for index, label in enumerate(CLASSES):
        onehot[:, index] = (labels == label).astype(float)
    return ((proba - onehot) ** 2).sum(axis=1)


def _log_loss(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    labels = frame[LABEL].to_numpy(dtype=int)
    picked = np.array(
        [probability[i, list(CLASSES).index(int(label))] for i, label in enumerate(labels.tolist())],
        dtype=float,
    )
    return -np.log(np.clip(picked, PRIOR_FLOOR, None))


def _participant_macro(frame: pd.DataFrame, values: np.ndarray) -> float:
    working = frame[["participant_group_id"]].copy()
    working["__v"] = np.asarray(values, dtype=float)
    return float(working.groupby("participant_group_id", sort=True)["__v"].mean().mean())


def _prior_matrix(frame: pd.DataFrame) -> np.ndarray:
    labels = frame[LABEL].to_numpy(dtype=int)
    folds = frame["outer_fold_group"].to_numpy()
    rows = []
    for fold in folds.tolist():
        train = labels[folds != fold]
        vector = np.array([float((train == k).sum()) for k in CLASSES], dtype=float)
        vector = vector / vector.sum()
        vector = np.maximum(vector, PRIOR_FLOOR)
        rows.append(vector / vector.sum())
    return np.vstack(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--frozen-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    root = Path(args.diagnostics_root)
    diagnostics = _read_csv(root / "multiclass_probability_diagnostics.csv")
    bins = _read_csv(root / "multiclass_calibration_bins.csv")
    audit = json.loads((root / "multiclass_probability_diagnostics_audit.json").read_text("utf-8"))
    frozen = _read_csv(Path(args.frozen_summary))

    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # A 身份键唯一
    duplicated = int(diagnostics.duplicated(subset=IDENTITY).sum())
    record("A_identity_unique", duplicated == 0, f"duplicated identity rows = {duplicated}")

    # B 先验对账
    mismatch = int(audit["n_prior_rows_mismatching_frozen"])
    checked = int(audit["n_prior_rows_checked_against_frozen"])
    record(
        "B_prior_reconstruction_matches_frozen",
        mismatch == 0 and checked == len(diagnostics),
        f"checked={checked} mismatching={mismatch} diagnostic_rows={len(diagnostics)}",
    )

    # C 主指标与冻结汇总一致
    frozen_key = frozen.set_index(["analysis_set_id", "model_id"])
    failures: list[str] = []
    for _, row in diagnostics.iterrows():
        key = (row["analysis_set_id"], row["model_id"])
        if key not in frozen_key.index:
            failures.append(f"missing frozen row for {key}")
            continue
        expected = float(frozen_key.loc[key, "participant_macro_multiclass_log_loss"])
        actual = float(row["participant_macro_multiclass_log_loss_recomputed"])
        if abs(actual - expected) > args.tolerance:
            failures.append(f"{key}: recomputed={actual} frozen={expected}")
    record(
        "C_primary_metric_matches_frozen",
        not failures,
        "; ".join(failures[:5]) if failures else f"all {len(diagnostics)} rows agree",
    )

    # D/E/F 逐 run 独立复算
    cache: dict[str, pd.DataFrame] = {}
    probe_failures: list[str] = []
    brier_failures: list[str] = []
    prior_brier_failures: list[str] = []
    prior_ll_failures: list[str] = []
    for _, row in diagnostics.iterrows():
        run_id = str(row["run_id"])
        if run_id not in cache:
            cache[run_id] = _valid(_read_csv(Path(args.run_root) / run_id / "probe_predictions.csv"))
        frame = cache[run_id]
        group = frame[frame["model_id"].astype(str) == str(row["model_id"])].reset_index(drop=True)
        if len(group) != int(row["n_probes"]):
            probe_failures.append(f"{run_id}/{row['model_id']}: {len(group)} != {row['n_probes']}")
            continue
        probability = group.loc[:, PROB_COLUMNS].to_numpy(dtype=float)
        model_brier = _participant_macro(group, _brier(group, probability))
        if abs(model_brier - float(row["participant_macro_multiclass_brier"])) > args.tolerance:
            brier_failures.append(
                f"{run_id}/{row['model_id']}: {model_brier} != {row['participant_macro_multiclass_brier']}"
            )
        prior = _prior_matrix(group)
        prior_brier = _participant_macro(group, _brier(group, prior))
        if abs(prior_brier - float(row["prior_baseline_participant_macro_multiclass_brier"])) > args.tolerance:
            prior_brier_failures.append(
                f"{run_id}/{row['model_id']}: {prior_brier} != "
                f"{row['prior_baseline_participant_macro_multiclass_brier']}"
            )
        prior_ll = _participant_macro(group, _log_loss(group, prior))
        if abs(prior_ll - float(row["prior_baseline_participant_macro_multiclass_log_loss_recomputed"])) > args.tolerance:
            prior_ll_failures.append(
                f"{run_id}/{row['model_id']}: {prior_ll} != "
                f"{row['prior_baseline_participant_macro_multiclass_log_loss_recomputed']}"
            )
    record("D_probe_counts_close", not probe_failures, "; ".join(probe_failures[:5]) or "all rows close")
    record(
        "E_participant_macro_brier_recomputes",
        not brier_failures,
        "; ".join(brier_failures[:5]) or "all rows agree",
    )
    record(
        "F_prior_constant_brier_and_logloss_recompute",
        not prior_brier_failures and not prior_ll_failures,
        "; ".join((prior_brier_failures + prior_ll_failures)[:5]) or "all rows agree",
    )

    # G 分箱计数闭合
    bin_totals = bins.groupby(IDENTITY, sort=True)["n_probes"].sum()
    diag_totals = diagnostics.set_index(IDENTITY)["n_probes"]
    bin_failures = [
        f"{key}: bins={int(bin_totals.get(key, -1))} diagnostics={int(diag_totals.loc[key])}"
        for key in diag_totals.index
        if int(bin_totals.get(key, -1)) != int(diag_totals.loc[key])
    ]
    record("G_calibration_bins_close", not bin_failures, "; ".join(bin_failures[:5]) or "all models close")

    # H 期望校准误差可由分箱表还原
    ece_failures: list[str] = []
    indexed = diagnostics.set_index(IDENTITY)
    for key, block in bins.groupby(IDENTITY, sort=True):
        total = float(block["n_probes"].sum())
        recomputed = float(((block["n_probes"] / total) * block["abs_gap"]).sum())
        if key not in indexed.index:
            ece_failures.append(f"{key}: no diagnostic row")
            continue
        reported = float(indexed.loc[key, "expected_calibration_error"])
        if abs(recomputed - reported) > 1e-9:
            ece_failures.append(f"{key}: {recomputed} != {reported}")
    record("H_expected_calibration_error_recomputes", not ece_failures, "; ".join(ece_failures[:5]) or "all models agree")

    # I 区间包住点估计
    interval_failures: list[str] = []
    pairs = [
        ("participant_macro_multiclass_brier", "participant_macro_multiclass_brier_ci_lower", "participant_macro_multiclass_brier_ci_upper"),
        ("prior_baseline_participant_macro_multiclass_brier", "prior_baseline_participant_macro_multiclass_brier_ci_lower", "prior_baseline_participant_macro_multiclass_brier_ci_upper"),
        ("participant_macro_brier_minus_prior", "participant_macro_brier_minus_prior_ci_lower", "participant_macro_brier_minus_prior_ci_upper"),
        ("participant_macro_top_label_calibration_in_the_large", "participant_macro_top_label_calibration_in_the_large_ci_lower", "participant_macro_top_label_calibration_in_the_large_ci_upper"),
        ("pooled_probe_ovr_macro_auroc", "pooled_probe_ovr_macro_auroc_ci_lower", "pooled_probe_ovr_macro_auroc_ci_upper"),
    ]
    for _, row in diagnostics.iterrows():
        for point, low, high in pairs:
            values = [row[point], row[low], row[high]]
            if not all(math.isfinite(float(v)) for v in values):
                continue
            if not (float(row[low]) - 1e-12 <= float(row[point]) <= float(row[high]) + 1e-12):
                interval_failures.append(f"{row['run_id']}/{row['model_id']}/{point}")
    record("I_intervals_cover_point_estimates", not interval_failures, "; ".join(interval_failures[:5]) or "all covered")

    # J 治理标志
    governance_failures: list[str] = []
    if bool(diagnostics["used_for_model_or_c_selection"].any()):
        governance_failures.append("used_for_model_or_c_selection is not all False")
    if bool(diagnostics["used_for_feature_selection"].any()):
        governance_failures.append("used_for_feature_selection is not all False")
    if bool(diagnostics["prediction_model_retrained_in_bootstrap"].any()):
        governance_failures.append("prediction_model_retrained_in_bootstrap is not all False")
    if not bool(diagnostics["calibration_model_refitted_in_bootstrap"].all()):
        governance_failures.append("calibration_model_refitted_in_bootstrap is not all True")
    if any("participant_macro_auroc" == c for c in diagnostics.columns):
        governance_failures.append("participant-level AUROC column present")
    record("J_governance_flags", not governance_failures, "; ".join(governance_failures) or "all flags correct")

    # K 可解读性分级计数闭合
    statuses = diagnostics["calibration_slope_reporting"].value_counts().to_dict()
    total_graded = int(sum(statuses.values()))
    record(
        "K_calibration_slope_grading_closes",
        total_graded == len(diagnostics),
        f"graded={total_graded} rows={len(diagnostics)} distribution={statuses}",
    )

    # L 产物 SHA-256
    hashes = {
        name: _sha256(root / name)
        for name in (
            "multiclass_probability_diagnostics.csv",
            "multiclass_calibration_bins.csv",
            "multiclass_probability_diagnostics_audit.json",
        )
    }

    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "verifier": "verify_multiclass_probability_diagnostics",
        "diagnostics_root": str(root),
        "run_root": str(Path(args.run_root)),
        "n_diagnostic_rows": int(len(diagnostics)),
        "n_calibration_bin_rows": int(len(bins)),
        "checks": checks,
        "all_passed": passed,
        "artifact_sha256": hashes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for check in checks:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['check']}: {check['detail']}")
    print(f"ALL_PASSED={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
