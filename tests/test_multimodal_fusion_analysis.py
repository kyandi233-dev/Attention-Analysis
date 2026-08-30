"""Contract tests for the multimodal fusion formal analysis (synthetic data only).

File: test_multimodal_fusion_analysis.py
Version: multimodal-fusion-v1.0.0
Purpose:
    用小型合成数据逐条断言六条融合契约（与 1.10 规格第 8 节一致）：

    1. participant_disjoint - 同一参与者的全部观测同折，训练/测试参与者互斥；
    2. preprocessing_inside_training_folds - 标准化/插补/筛选/组均值分解只由
       训练折拟合，测试折只应用；
    3. common_cohort_paired_comparison - 增量比较只在共同子集同一折集合上
       逐折配对，bootstrap 按参与者重抽样；
    4. report_coverage_alongside_performance - 逐折性能表与覆盖率同报；
    5. no_missingness_identity_leakage - 缺失插补不引用参与者身份，缺失
       模式不推断身份；
    6. explanation_and_prediction_separate - 边际贡献（解释）与 LOSO 性能
       （预测）分轨输出。

    Synthetic tables mirror the real input contract: behavior is authoritative
    for Q1/Q2 labels; NIR availability defines the common subset; mmWave
    carries an R-code identity bridge with one split-flag pair.

Dependencies:
    pytest, numpy, pandas, PyYAML
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from attention_pipeline.multimodal_formal.preprocessing import (
    apply_preprocessing,
    fit_preprocessing,
)
from attention_pipeline.multimodal_formal.runner import (
    GROUP_COLUMN,
    loso_fold_split,
    run_multimodal_fusion,
)

# ---- 合成数据规模（保持测试 <30s） ----
N_GROUPS = 12
SESSIONS_PER_GROUP = 2
BLOCKS = ("b1", "b2")
PROBES_PER_BLOCK = 6
# NIR 缺失场次（模拟 109/116 口径：共同子集由 NIR 决定）
NIR_MISSING_SESSIONS = {"s-10-0", "s-11-0"}
# 毫米波结构缺失场次（特征全 NaN，行保留）
MMWAVE_MISSING_SESSIONS = {"s-09-1"}
# 毫米波跨批分裂条目（R029=R096 同一 P）
SPLIT_SESSIONS = {"s-00-0", "s-00-1"}

BEHAVIOR_FEATURES = [
    "go_correct_rt_median_ms", "go_correct_rt_cv", "omission_rate",
    "raw_go_omission_rate", "timing_ambiguous_go_omission_rate",
    "commission_rate", "dprime_loglinear",
]
NIR_METRICS = ["pupil_geom_mean_diameter", "hard_pupil_fraction"]
MMWAVE_FEATURES = [
    "mmwave_hr_fused_bpm_median", "mmwave_breath_rate_breaths_per_min_median",
    "mmwave_motion_proxy_median", "mmwave_hr_usable_window_fraction",
    "mmwave_hr_mean_confidence", "mmwave_phase_stability_median",
    "mmwave_timestamp_coverage_fraction",
]
RGB_FEATURES = [
    "body_motion_energy_median", "exposure_change_abs_median",
    "pose_lateral_right_per_sec_median", "pose_vertical_up_per_sec_median",
    "pose_radial_proximity_direction_score_median", "blink_event_rate_per_min",
    "blink_frame_ratio", "win_valid_frame_ratio",
]


def _synthetic_data_root(tmp_path: Path, rng: np.random.Generator) -> Path:
    """构造合成数据根：四模态 probe 表 + 毫米波身份桥表。

    Parameters
    ----------
    tmp_path : pytest 临时目录。
    rng : 随机数生成器。

    Returns
    -------
    数据根路径（内部目录结构与真实 _FormalAnalysis 一致）。
    """
    groups = [f"P-{i:03d}" for i in range(N_GROUPS)]
    sessions = [f"s-{g:02d}-{v}" for g in range(N_GROUPS) for v in range(SESSIONS_PER_GROUP)]
    session_group = {s: groups[int(s.split("-")[1])] for s in sessions}

    rows = []
    for session in sessions:
        group = session_group[session]
        group_index = int(session.split("-")[1])
        for block in BLOCKS:
            for probe in range(1, PROBES_PER_BLOCK + 1):
                # 标签与特征弱关联；轮转保证每场（每测试折）四类齐全
                rt_cv = 0.10 + 0.05 * group_index + rng.normal(0, 0.02)
                q1 = ((probe - 1 + group_index) % 4) + 1
                q2 = ((probe + group_index) % 4) + 1
                rows.append({
                    "session_id": session,
                    "block_id": block.upper(),
                    "probe_order_in_block": probe,
                    "probe_event_id": f"{session}|{block}|probe|{probe}",
                    "probe_time_ms": 1_700_000_000_000 + group_index * 86_400_000 + probe * 90_000,
                    "q1_nominal_4class": q1,
                    "q2_ordinal_4level": q2,
                    "go_correct_rt_median_ms": 380 + 30 * group_index + 40 * (q2 - 1) + rng.normal(0, 20),
                    "go_correct_rt_cv": rt_cv + 0.02 * (q1 >= 3) + rng.normal(0, 0.01),
                    "omission_rate": np.clip(0.02 + 0.03 * (q1 == 3) + rng.normal(0, 0.02), 0, 1),
                    "raw_go_omission_rate": np.clip(0.01 + 0.02 * (q1 == 3) + rng.normal(0, 0.01), 0, 1),
                    "timing_ambiguous_go_omission_rate": np.clip(rng.normal(0.01, 0.01), 0, 0.2),
                    "commission_rate": np.clip(0.02 + 0.03 * (q2 == 1) + rng.normal(0, 0.02), 0, 1),
                    "dprime_loglinear": 2.0 - 0.3 * (q1 >= 3) + rng.normal(0, 0.3),
                })
    behavior = pd.DataFrame(rows)
    # 注入 QC 语义缺失：cv 在少数行缺失（模拟 rt_cv_min_n 门槛）
    cv_missing = rng.choice(len(behavior), size=int(len(behavior) * 0.05), replace=False)
    behavior.loc[cv_missing, "go_correct_rt_cv"] = np.nan
    behavior["participant_group_id"] = behavior["session_id"].map(session_group)
    behavior["repeat_participant_id"] = behavior["participant_group_id"]

    # NIR：部分场次不可用（共同子集约束）
    nir_rows = []
    for _, row in behavior.iterrows():
        if row["session_id"] in NIR_MISSING_SESSIONS:
            continue
        group_index = int(row["session_id"].split("-")[1])
        nir_rows.append({
            "session_id": row["session_id"],
            "analysis_group_token": row["participant_group_id"],
            "block_num": int(row["block_id"][-1]),
            "probe_index_in_block": row["probe_order_in_block"],
            "probe_index_global": row["probe_order_in_block"],
            "probe_onset_ms": row["probe_time_ms"],
            "window_name": "pre_30s",
            "track": "binocular_primary",
            "probe_response": row["q1_nominal_4class"],
            "probe_vigilance": row["q2_ordinal_4level"],
            "pupil_geom_mean_diameter": 33.0 + 1.5 * group_index + 0.5 * (row["q2_ordinal_4level"] - 1) + rng.normal(0, 1.0),
            "hard_pupil_fraction": 0.8 + 0.01 * group_index + 0.02 * (row["q1_nominal_4class"] == 1) + rng.normal(0, 0.02),
        })
    nir = pd.DataFrame(nir_rows)
    nir_missing_mask = rng.choice(len(nir), size=int(len(nir) * 0.05), replace=False)
    nir.loc[nir_missing_mask, NIR_METRICS] = np.nan

    # 毫米波：全部场次（含 1 场 STRUCTURAL_MISSING 全 NaN）
    # 跨批分裂模拟：s-00-1 使用不同 R 编码（R999）但同属 P-000（与桥表一致）
    mm_rows = []
    for _, row in behavior.iterrows():
        missing = row["session_id"] in MMWAVE_MISSING_SESSIONS
        if row["session_id"] == "s-00-1":
            r_code = "R999"
        else:
            r_code = f"R{int(row['session_id'].split('-')[1]):03d}"
        mm_rows.append({
            "session_id": row["session_id"],
            "repeat_participant_id": r_code,
            "block_id": f"block-{row['block_id'][-1]}",
            "probe_index_in_block": row["probe_order_in_block"],
            "probe_id": f"probe-{row['probe_order_in_block']:02d}",
            "window_name": "pre_30s",
            "probe_onset_unix_ms": row["probe_time_ms"],
            "label_probe_response": row["q1_nominal_4class"],
            "label_probe_vigilance": row["q2_ordinal_4level"],
            "mmwave_observed": not missing,
            "mmwave_hr_fused_bpm_median": np.nan if missing else 68.0 + rng.normal(0, 5),
            "mmwave_breath_rate_breaths_per_min_median": np.nan if missing else 16.0 + rng.normal(0, 1.5),
            "mmwave_motion_proxy_median": np.nan if missing else 0.5 + rng.normal(0, 0.2),
            "mmwave_hr_usable_window_fraction": np.nan if missing else 0.9 + rng.normal(0, 0.05),
            "mmwave_hr_mean_confidence": np.nan if missing else 0.8 + rng.normal(0, 0.1),
            "mmwave_phase_stability_median": np.nan if missing else 0.7 + rng.normal(0, 0.1),
            "mmwave_timestamp_coverage_fraction": np.nan if missing else 0.95 + rng.normal(0, 0.03),
        })
    mmwave = pd.DataFrame(mm_rows)
    # 模拟两批：前 6 组写 J 文件、后 6 组写 E 文件（session 零重叠）
    j_sessions = sessions[: N_GROUPS]
    e_sessions = sessions[N_GROUPS:]
    mmwave_j = mmwave[mmwave["session_id"].isin(j_sessions)]
    mmwave_e = mmwave[mmwave["session_id"].isin(e_sessions)]

    # RGB：全部场次（blink 特征部分缺失，模拟 partial_no_blink）
    rgb_rows = []
    for _, row in behavior.iterrows():
        blink_missing = rng.random() < 0.05
        rgb_rows.append({
            "session_id": row["session_id"],
            "participant_group_id": row["participant_group_id"],
            "block_id": row["block_id"],
            "probe_order_in_block": row["probe_order_in_block"],
            "probe_event_id": row["probe_event_id"],
            "probe_time_ms": row["probe_time_ms"],
            "q1_nominal_4class": row["q1_nominal_4class"],
            "q2_ordinal_4level": row["q2_ordinal_4level"],
            "rgb_source_status": "partial_no_blink" if blink_missing else "ok",
            "win_n_frames": 900,
            "win_coverage_sec": 29.9,
            "win_valid_frame_ratio": 1.0,
            "body_motion_energy_median": 0.05 + rng.normal(0, 0.02),
            "exposure_change_abs_median": 0.004 + rng.normal(0, 0.001),
            "pose_lateral_right_per_sec_median": rng.normal(0, 0.001),
            "pose_vertical_up_per_sec_median": rng.normal(0, 0.001),
            "pose_radial_proximity_direction_score_median": rng.normal(0, 0.3),
            "blink_event_rate_per_min": np.nan if blink_missing else 3.0 + rng.normal(0, 1.0),
            "blink_frame_ratio": np.nan if blink_missing else 0.05 + rng.normal(0, 0.01),
        })
    rgb = pd.DataFrame(rgb_rows)

    # 身份桥表：R 编码 -> P 编码（含 1 例跨批分裂标注）
    bridge_rows = []
    for session in sessions:
        group = session_group[session]
        if session == "s-00-1":
            r_code = "R999"  # 分裂：同一 P-000 在另一批使用不同 R
        else:
            r_code = f"R{int(session.split('-')[1]):03d}"
        bridge_rows.append({
            "session_id": session,
            "repeat_participant_id": r_code,
            "participant_group_id": group,
            "include": True,
            "identity_status": "verified",
            "r_code_split_flag": session in SPLIT_SESSIONS,
        })
    bridge = pd.DataFrame(bridge_rows)

    # 落盘目录结构（与真实 _FormalAnalysis 一致）
    behavior_dir = tmp_path / "Behavior" / "formal_v3"
    nir_dir = tmp_path / "NIR" / "11_analysis_tables" / "probe_pupil_models"
    mm_dir = tmp_path / "mmWave"
    rgb_dir = tmp_path / "RGB" / "11_analysis_tables_116cohort"
    map_dir = tmp_path / "mapping"
    for directory in (behavior_dir, nir_dir, mm_dir, rgb_dir, map_dir):
        directory.mkdir(parents=True, exist_ok=True)
    behavior.to_csv(behavior_dir / "probe_primary_30s.csv", index=False, encoding="utf-8-sig")
    nir.to_csv(nir_dir / "probe_pupil_model_table.csv", index=False, encoding="utf-8-sig")
    mmwave_j.to_csv(mm_dir / "mmwave_probe_merge_ready.csv", index=False, encoding="utf-8-sig")
    mmwave_e.to_csv(mm_dir / "mmwave_probe_merge_ready_E.csv", index=False, encoding="utf-8-sig")
    rgb.to_csv(rgb_dir / "rgb_probe_pre30s_features.csv", index=False, encoding="utf-8-sig")
    bridge.to_csv(map_dir / "mmwave_identity_bridge.csv", index=False, encoding="utf-8-sig")
    return tmp_path


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """模块级共享的端到端合成跑：4 折、8 组合、Q1 结局、两模型、RF 小森林。

    Returns
    -------
    (run_dir, manifest)：契约 3/4/6 共享，避免重复跑端到端。
    """
    tmp_path = tmp_path_factory.mktemp("fusion-contract")
    rng = np.random.default_rng(7)
    data_root = _synthetic_data_root(tmp_path, rng)
    paths_yaml = tmp_path / "paths.local.yaml"
    paths_yaml.write_text(
        f"version: 3\npaths:\n"
        f"  multimodal_fusion_data_root: \"{data_root.as_posix()}\"\n"
        f"  multimodal_fusion_output_root: \"{(tmp_path / 'MultiModal').as_posix()}\"\n",
        encoding="utf-8",
    )
    # 临时科学配置：RF 降为 30 棵树（契约测试只验证逻辑，不等价于全量参数）
    config_path = Path(__file__).resolve().parents[1] / "configs" / "multimodal_fusion.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data["models"]["random_forest"]["n_estimators"] = 30
    test_config = tmp_path / "multimodal_fusion_contract.yaml"
    test_config.write_text(yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest = run_multimodal_fusion(
        test_config,
        paths_config=paths_yaml,
        run_id="contract-test",
        jobs=1,
        fold_limit=4,
        combinations=None,
        outcomes=["q1"],
        models=None,
    )
    return tmp_path / "MultiModal" / "contract-test", manifest


def _synthetic_frame(rng: np.random.Generator) -> pd.DataFrame:
    """构造带两场次与离群参与者的特征矩阵（预处理契约测试用）。"""
    rows = []
    for group_index in range(8):
        group = f"P-{group_index:03d}"
        for visit in range(2):
            session = f"s-{group_index:02d}-{visit}"
            for probe in range(6):
                # P-000 为离群参与者（特征水平远高于他人）
                shift = 50.0 if group_index == 0 else 0.0
                rows.append({
                    "session_id": session,
                    "participant_group_id": group,
                    "probe_index_in_block": probe + 1,
                    "feat_a": 10.0 + shift + rng.normal(0, 1),
                    "feat_b": 0.5 + rng.normal(0, 0.1),
                    "nir_metric": 30.0 + shift + rng.normal(0, 2),
                })
    return pd.DataFrame(rows)


# ---- 契约 1：participant_disjoint ----
def test_contract_participant_disjoint() -> None:
    """同一参与者的全部观测同折；训练与测试参与者集合互斥。"""
    rng = np.random.default_rng(0)
    frame = _synthetic_frame(rng)
    train, test = loso_fold_split(frame, "P-000")
    train_groups = set(train[GROUP_COLUMN].astype(str))
    test_groups = set(test[GROUP_COLUMN].astype(str))
    assert not (train_groups & test_groups), "training and test participants must be disjoint"
    assert test_groups == {"P-000"}
    # 同一参与者的两个 session 必须全部落在测试折
    assert set(test["session_id"].unique()) == {"s-00-0", "s-00-1"}
    assert not train["session_id"].astype(str).str.startswith("s-00-").any()
    # 对每个可能的留出组都成立
    for group in sorted(set(frame[GROUP_COLUMN].astype(str))):
        tr, te = loso_fold_split(frame, group)
        assert not (set(tr[GROUP_COLUMN]) & set(te[GROUP_COLUMN]))


# ---- 契约 2：preprocessing_inside_training_folds ----
def test_contract_preprocessing_inside_training_folds() -> None:
    """标准化/插补统计量只由训练折拟合；测试折只应用。"""
    rng = np.random.default_rng(1)
    frame = _synthetic_frame(rng)
    columns = ["feat_a", "feat_b", "nir_metric"]
    train, test = loso_fold_split(frame, "P-000")
    state = fit_preprocessing(
        train[[GROUP_COLUMN] + columns], columns=columns,
        group_col=GROUP_COLUMN, nir_metrics=("nir_metric",),
    )
    # 离群参与者 P-000 只在测试折：训练统计必须不含其极端值
    assert state.standardization_mean["feat_a"] < 15.0, "train statistics must exclude held-out outlier"
    assert state.standardization_mean["feat_a"] == pytest.approx(
        float(pd.to_numeric(train["feat_a"], errors="coerce").mean()), abs=1e-9
    )
    assert state.standardization_std["feat_a"] == pytest.approx(
        float(pd.to_numeric(train["feat_a"], errors="coerce").std(ddof=0)), abs=1e-9
    )
    # 测试折应用后数值只由训练统计量决定
    x_test = apply_preprocessing(test, state, group_col=GROUP_COLUMN, nir_metrics=("nir_metric",))
    expected = (pd.to_numeric(test["feat_a"], errors="coerce") - state.standardization_mean["feat_a"]) / state.standardization_std["feat_a"]
    assert np.allclose(x_test["feat_a"].to_numpy(), expected.to_numpy())
    # NIR within/between：测试折 between 等于训练组均值先验，不使用测试折自身统计量
    assert x_test["nir_metric_between"].nunique() == 1
    assert np.allclose(
        x_test["nir_metric_between"].to_numpy()[0],
        (state.nir_group_mean_priors["nir_metric"] - state.standardization_mean["nir_metric_between"])
        / state.standardization_std["nir_metric_between"],
    )


# ---- 契约 3：common_cohort_paired_comparison ----
def test_contract_common_cohort_paired_comparison(synthetic_run: tuple[Path, dict]) -> None:
    """增量比较只在共同子集同一折集合上逐折配对；bootstrap 按参与者重抽样。"""
    run_dir, manifest = synthetic_run
    comparison = pd.read_csv(run_dir / "comparison" / "incremental_vs_M0.csv", encoding="utf-8-sig")
    assert not comparison.empty
    assert (comparison["paired_by"] == "fold").all()
    assert (comparison["bootstrap_resample_unit"] == "participant").all()
    n_folds = manifest["n_fold_groups"]
    assert (comparison["n_folds_total"] == n_folds).all(), "paired comparison must cover the same fold set"
    # 共同子集规模：由 NIR 缺失场次决定（12 组 x 2 场 - 2 场 = 22 场）
    assert manifest["common_subset"]["session_n"] == 22
    assert manifest["common_subset"]["participant_group_n"] == N_GROUPS
    # 所有比较组合都来自共同子集内的折
    assert set(comparison["comparison"].unique()).issubset(
        {f"M{i}_vs_M0" for i in range(1, 8)}
    )


# ---- 契约 4：report_coverage_alongside_performance ----
def test_contract_report_coverage_alongside_performance(synthetic_run: tuple[Path, dict]) -> None:
    """逐折性能表与覆盖率同报。"""
    run_dir, _ = synthetic_run
    by_fold = pd.read_csv(run_dir / "performance" / "performance_by_fold.csv", encoding="utf-8-sig")
    assert not by_fold.empty
    for column in ("n_train_rows", "n_test_rows", "test_categories", "model_failed"):
        assert column in by_fold.columns
    assert (by_fold["n_test_rows"] > 0).all()
    assert by_fold[["roc_auc_ovr_macro", "balanced_accuracy", "macro_f1", "log_loss"]].notna().any().any()
    # 共同子集清单与特征覆盖表存在且非空
    manifest = pd.read_csv(run_dir / "common_cohort" / "common_cohort_manifest.csv", encoding="utf-8-sig")
    coverage = pd.read_csv(run_dir / "common_cohort" / "feature_coverage.csv", encoding="utf-8-sig")
    assert len(manifest) > 0 and len(coverage) > 0
    assert (coverage["coverage"] >= 0).all() and (coverage["coverage"] <= 1).all()


# ---- 契约 5：no_missingness_identity_leakage ----
def test_contract_no_missingness_identity_leakage() -> None:
    """缺失插补不引用参与者身份；缺失模式不推断身份、不因缺失换折。"""
    rng = np.random.default_rng(2)
    frame = _synthetic_frame(rng)
    # 注入：P-007 全部观测的 feat_b 缺失（缺失与身份相关）
    frame.loc[frame[GROUP_COLUMN].eq("P-007"), "feat_b"] = np.nan
    columns = ["feat_a", "feat_b"]
    train, test = loso_fold_split(frame, "P-007")
    state = fit_preprocessing(
        train[[GROUP_COLUMN] + columns], columns=columns, group_col=GROUP_COLUMN, nir_metrics=()
    )
    # 插补统计量 = 训练折中位数（训练折不含 P-007 的任何行）
    assert state.imputation_medians["feat_b"] == pytest.approx(
        float(pd.to_numeric(train["feat_b"], errors="coerce").median()), abs=1e-9
    )
    # 预处理状态不含任何 per-participant 插补项
    assert not hasattr(state, "per_participant_imputation")
    x_test = apply_preprocessing(test, state, group_col=GROUP_COLUMN, nir_metrics=())
    imputed_rows = test["feat_b"].isna()
    for value in x_test.loc[imputed_rows, "feat_b"]:
        assert np.isclose(
            value,
            (state.imputation_medians["feat_b"] - state.standardization_mean["feat_b"])
            / state.standardization_std["feat_b"],
        ), "imputed test values must equal the training-fold median, not any identity-derived value"
    # 缺失行保留在同一参与者折内，不因缺失改变分组
    assert test[GROUP_COLUMN].eq("P-007").all()
    assert len(test) == int(frame[GROUP_COLUMN].eq("P-007").sum())


# ---- 契约 6：explanation_and_prediction_separate ----
def test_contract_explanation_and_prediction_separate(synthetic_run: tuple[Path, dict]) -> None:
    """边际贡献（解释产物）与 LOSO 性能（预测产物）分轨输出。"""
    run_dir, manifest = synthetic_run
    assert bool(manifest["explanation_and_prediction_separate"]) is True
    marginal_path = run_dir / "marginal" / "marginal_contribution.csv"
    performance_path = run_dir / "performance" / "performance_summary.csv"
    assert marginal_path.is_file() and performance_path.is_file()
    marginal = pd.read_csv(marginal_path, encoding="utf-8-sig")
    performance = pd.read_csv(performance_path, encoding="utf-8-sig")
    # 边际贡献表不含任何折级预测列（不含逐折样本）
    assert not any("fold" in str(c).lower() for c in marginal.columns)
    assert not any(c in marginal.columns for c in ("n_test_rows", "n_train_rows"))
    # 边际贡献只覆盖三传感模态，基值为 M0，全值为 M7
    assert set(marginal["modality"]) == {"nir", "mmwave", "rgb"}
    assert (marginal["base_value_M0"].notna()).any()
    # 预测产物与解释产物物理分轨（不同目录、不同文件）
    assert marginal_path.parent != performance_path.parent
    # 解释产物由预测管线性能派生：M7/M0 汇总值必须存在于性能表
    m0_auc = performance.loc[
        (performance["combination"].eq("M0")) & (performance["outcome"].eq("q1"))
        & (performance["model"].eq("logistic")), "roc_auc_ovr_macro_mean"
    ].iloc[0]
    assert np.isfinite(m0_auc)
    assert np.allclose(
        marginal.loc[
            (marginal["outcome"].eq("q1")) & (marginal["model"].eq("logistic"))
            & (marginal["metric"].eq("roc_auc_ovr_macro")), "base_value_M0"
        ].to_numpy(), m0_auc,
    )
