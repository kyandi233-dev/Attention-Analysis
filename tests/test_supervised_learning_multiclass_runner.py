"""Four-class runner and entrypoint tests: LOSO integrity, archive contract reuse,
missing-class fold recording, and the prior-baseline comparator at fold level."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from attention_pipeline.supervised_learning.entrypoint_multiclass import (
    _multiclass_evaluation_block,
    _multiclass_predictions_to_archive_contract,
    _require_frozen_runtime_contract_multiclass,
    run_multiclass_from_config,
)
from attention_pipeline.supervised_learning.feature_schemes import FeatureScheme
from attention_pipeline.supervised_learning.runner_multiclass import (
    run_nested_loso_multiclass,
)
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError
from attention_pipeline.supervised_learning.task_multiclass import Q1_MULTICLASS_SPEC

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_4CLASS = REPO_ROOT / "configs" / "supervised_learning_4class_v1.yaml"
CONFIG_BINARY = REPO_ROOT / "configs" / "supervised_learning_v1.yaml"

FEATURE_COLUMNS = (
    "go_correct_rt_median_ms",
    "go_correct_rt_cv",
    "ocular__rseg_hard__rgb_nir_qc__level_mean__2s",
)


def test_feature_columns_are_real_registered_predictors() -> None:
    """守护测试：下文的合成帧列名必须真的属于冻结注册表，否则契约测试会失去意义。"""
    config = yaml.safe_load(CONFIG_4CLASS.read_text(encoding="utf-8"))
    registered = {
        column
        for feature in config["feature_registry"]["features"]
        for column in feature["columns"]
    }
    assert set(FEATURE_COLUMNS).issubset(registered)


def _analysis_frame(
    *, n_groups: int = 8, rows_per_group: int = 8, seed: int = 5
) -> pd.DataFrame:
    """构造带完整 probe 定位列的四分类分析帧；四类在每名参与者内轮流出现。"""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group_index in range(n_groups):
        participant = f"P-{group_index:02d}"
        for row_index in range(rows_per_group):
            y = int(row_index % 4) + 1
            rows.append(
                {
                    "session_id": f"{participant}-S1",
                    "block_id": "B1",
                    "probe_event_id": f"{participant}-E{row_index:02d}",
                    "probe_index_in_block": row_index,
                    "probe_time_ms": 1000 * row_index,
                    "participant_group_id": participant,
                    "analysis_set_id": "AS.test",
                    "membership_type": "included_missing_aware",
                    "q1_nominal_4class": y,
                    "q2_ordinal_4level": (row_index % 4) + 1,
                    "go_correct_rt_median_ms": 300.0 + 60.0 * (y - 1) + rng.normal(0, 3),
                    "go_correct_rt_cv": 0.1 * y + rng.normal(0, 0.01),
                    "ocular__rseg_hard__rgb_nir_qc__level_mean__2s": rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows)


def _full_model_contract_columns() -> tuple[str, ...]:
    """返回冻结注册表中 ``full`` 方案的预测列（按注册顺序）。"""
    from attention_pipeline.supervised_learning.feature_registry import (
        build_feature_comparison_plan,
        load_registered_features,
    )

    config = yaml.safe_load(CONFIG_4CLASS.read_text(encoding="utf-8"))
    plan = build_feature_comparison_plan(
        load_registered_features(config["feature_registry"])
    )
    return tuple(plan.model_map()["full"].columns)


def _with_full_model_sample_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """给合成帧补上 ``comparison_models`` / ``required_features`` 两列 B 层样本契约。

    ``required_features`` 必须是「按科学模态分组的 full 方案预测列」JSON；entrypoint 会
    检查它与所声明模型的预测列并集完全一致，任何多余的样本过滤列都会被拒绝。
    """
    from attention_pipeline.supervised_learning.feature_registry import (
        build_feature_comparison_plan,
        load_registered_features,
    )

    config = yaml.safe_load(CONFIG_4CLASS.read_text(encoding="utf-8"))
    registry = load_registered_features(config["feature_registry"])
    plan = build_feature_comparison_plan(registry)
    full_columns = set(plan.model_map()["full"].columns)
    modalities: dict[str, list[str]] = {}
    for feature in registry:
        for column in feature.columns:
            if column in full_columns:
                modalities.setdefault(feature.modality, []).append(column)

    out = frame.copy()
    # 合成帧必须真的带齐 full 方案的全部 11 个预测列，否则运行前范围检查会失败。
    for feature in registry:
        for column in feature.columns:
            if column not in out.columns:
                out[column] = 0.0
    out["comparison_models"] = json.dumps(["full"])
    out["required_features"] = json.dumps(modalities, sort_keys=True)
    out["required_outcomes"] = json.dumps([Q1_MULTICLASS_SPEC.source_column])
    return out


def _synthetic_frame(n_groups: int = 6, *, seed: int = 5) -> pd.DataFrame:
    """runner 单元测试用的帧。

    runner 本身不做 B 层样本成员资格判断（那是 entrypoint 的职责），因此这里只需要
    probe 定位列与预测列；B 层契约列由 ``_with_full_model_sample_contract`` 追加。
    """
    return _analysis_frame(n_groups=n_groups, seed=seed)


def _schemes() -> dict[str, list[FeatureScheme]]:
    return {
        "full": [FeatureScheme("full", tuple(FEATURE_COLUMNS))],
    }


def test_route_a_runs_and_preserves_the_declared_prediction_columns() -> None:
    frame = _analysis_frame()
    result = run_nested_loso_multiclass(
        frame,
        model_feature_schemes=_schemes(),
        route="A",
        c_candidates=[0.1, 1.0],
        inner_splits=4,
        run_id="test-route-a",
        analysis_set_id="AS.test",
        membership_type="included_missing_aware",
    )
    expected_columns = {
        "session_id",
        "block_id",
        "probe_event_id",
        "run_id",
        "analysis_set_id",
        "membership_type",
        "model_id",
        "outer_fold_group",
        "q1_nominal_4class",
        "predicted_q1_multiclass",
        "feature_set_id",
        "selected_c",
        "model_failed",
        "failure_reason",
        "q2_ordinal_4level",
        *Q1_MULTICLASS_SPEC.probability_columns,
    }
    assert expected_columns.issubset(set(result.predictions.columns))
    assert len(result.predictions) == len(frame)
    assert result.failures.empty
    # 每条折外预测行的外层折必须等于被留出的参与者本人。
    assert (
        result.predictions["outer_fold_group"].astype(str)
        == result.predictions["participant_group_id"].astype(str)
    ).all()
    # 原始四类整数值原样保留。
    assert sorted(result.predictions["q1_nominal_4class"].unique().tolist()) == [1, 2, 3, 4]
    probabilities = result.predictions[list(Q1_MULTICLASS_SPEC.probability_columns)].to_numpy(float)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(result.predictions["predicted_q1_multiclass"].unique()).issubset({1, 2, 3, 4})
    assert len(result.fold_audits) == 8
    for audit in result.fold_audits:
        assert audit["failed"] is False
        assert audit["outer_train_class_complete"] is True
        assert audit["outer_train_missing_classes"] == []
        assert len(audit["outer_train_group_ids"]) == 7
        assert audit["outer_test_group_ids"] == [audit["outer_fold_group"]]
        # 折审计不得携带原始概率/标签数组。
        assert "class_probabilities" not in audit["final_refit"]
        assert "predicted_class" not in audit["final_refit"]
        assert audit["final_refit"]["model_classes"] == [1, 2, 3, 4]


def test_route_b_records_the_forward_selection_audit_per_outer_fold() -> None:
    frame = _analysis_frame(n_groups=6)
    pool = {
        "behavior.rt_level.median.v1": "go_correct_rt_median_ms",
        "behavior.rt_variability.cv.v1": "go_correct_rt_cv",
        "movement.body_motion_energy.median.pre30s.v1": "ocular__rseg_hard__rgb_nir_qc__level_mean__2s",
    }
    result = run_nested_loso_multiclass(
        frame,
        registered_features=pool,
        route="B",
        c_candidates=[0.1, 1.0],
        inner_splits=3,
        run_id="test-route-b",
        analysis_set_id="AS.test",
        membership_type="included_missing_aware",
    )
    assert result.metadata["route"] == "B"
    assert result.metadata["model_ids"] == ["forward_selected_4class"]
    assert len(result.fold_audits) == 6
    for audit in result.fold_audits:
        selection = audit["selection"]
        assert selection["route"] == "B_forward_selection_within_outer_training_fold"
        assert "selected_feature_set" in selection
        assert isinstance(selection["selected_feature_set"], list)
        assert selection["candidate_feature_pool"] == pool
        # 逐内层折的大数组已从折审计中精简掉，只留计数。
        for step in selection["steps"]:
            for evaluation in step["candidate_evaluations"]:
                assert "inner_fold_audits" not in evaluation
                assert "n_inner_fold_audits" in evaluation


def test_route_b_requires_the_registered_pool() -> None:
    frame = _analysis_frame(n_groups=6)
    with pytest.raises(SupervisedLearningContractError, match="registered feature pool"):
        run_nested_loso_multiclass(
            frame,
            route="B",
            inner_splits=3,
            analysis_set_id="AS.test",
            membership_type="included_missing_aware",
        )


def test_route_a_requires_feature_schemes() -> None:
    frame = _analysis_frame(n_groups=6)
    with pytest.raises(SupervisedLearningContractError, match="route A requires"):
        run_nested_loso_multiclass(
            frame,
            route="A",
            inner_splits=3,
            analysis_set_id="AS.test",
            membership_type="included_missing_aware",
        )


def _frame_with_class_3_in_a_single_group(
    *, n_groups: int = 8, rows_per_group: int = 8, seed: int = 5
) -> tuple[pd.DataFrame, str]:
    """构造「类别 3 只出现在一个参与者中」的分析帧。

    返回 (帧, 承载类别 3 的参与者)。外层训练折缺类别 3 当且仅当**所有**承载类别 3 的
    参与者都被留出；因此只有当承载者恰好只有一个时，才存在「留出它 → 训练折缺类」的折。
    这正是要检验的边界：承载者多于一个时，留出任何一个之后训练折仍含类别 3，外层缺类
    检查永远不会触发（旧版构造把类别 3 留给前 3 组、从后 5 组删除，因此断言必然落空）。
    注意：其余 7 个外层折的**内层**训练折仍可能缺类别 3 而失败，那是内层失败路径，与
    本测试要隔离的外层前置检查是两回事，因此断言不要求失败集合恰等于 {承载者}。
    """
    frame = _analysis_frame(n_groups=n_groups, rows_per_group=rows_per_group, seed=seed)
    group_order = sorted(frame["participant_group_id"].unique().tolist())
    carrier = group_order[0]
    keep = ~(
        frame["participant_group_id"].ne(carrier) & frame["q1_nominal_4class"].eq(3)
    )
    return frame.loc[keep].reset_index(drop=True), carrier


def test_missing_class_in_an_outer_training_fold_is_recorded_not_coerced() -> None:
    """外层训练折缺类时该折不可估，必须写成显式失败记录，绝不静默补类。"""
    frame, carrier = _frame_with_class_3_in_a_single_group()
    result = run_nested_loso_multiclass(
        frame,
        model_feature_schemes=_schemes(),
        route="A",
        c_candidates=[1.0],
        inner_splits=3,
        run_id="test-missing-class",
        analysis_set_id="AS.test",
        membership_type="included_missing_aware",
    )
    audits = {audit["outer_fold_group"]: audit for audit in result.fold_audits}

    # 外层缺类的折：恰好是留出唯一承载类别 3 的参与者那一折。
    outer_incomplete = {
        group for group, audit in audits.items() if not audit["outer_train_class_complete"]
    }
    assert outer_incomplete == {carrier}
    for group, audit in audits.items():
        if group not in outer_incomplete:
            assert audit["outer_train_missing_classes"] == []

    assert not result.failures.empty, "expected at least one not-estimable outer fold"
    assert outer_incomplete <= set(result.failures["outer_fold_group"].astype(str))

    audit = audits[carrier]
    assert audit["failed"] is True
    assert "outer training fold is not estimable" in audit["reason"]
    assert "missing declared classes" in audit["reason"]
    assert 3 in audit["outer_train_missing_classes"]
    assert audit["outer_train_class_complete"] is False

    assert len(result.predictions) == len(frame)
    failed_rows = result.predictions.loc[result.predictions["model_failed"].astype(bool)]
    # 被外层前置检查拦下的折必须显式失败，且不得产出任何类别概率。
    assert outer_incomplete <= set(failed_rows["outer_fold_group"].astype(str))
    carrier_rows = failed_rows.loc[failed_rows["outer_fold_group"].eq(carrier)]
    assert not carrier_rows.empty
    assert carrier_rows[list(Q1_MULTICLASS_SPEC.probability_columns)].isna().all().all()

    # 外层检查必须是**有选择性**的：只有承载者那一折带外层缺类原因，其余折若失败，
    # 原因必须是内层路径而不是外层前置检查。
    #
    # 本构造下其余折确实也会失败，且这是预注册要求的 fail-closed 行为：类别 3 只剩一个
    # 承载者时，内层分组 CV 会把该承载者放进某一个内层验证折，使该内层训练折缺类别 3，
    # 候选因此不可估。也就是说「同一帧里既有外层缺类折、又有可估折」在单承载者构造下
    # 不可达；可估折的正常产出由 test_route_a_runs_and_preserves_the_declared_prediction_columns
    # （四类在每名参与者内齐全）覆盖。
    for group, audit in audits.items():
        if group == carrier:
            continue
        assert "outer training fold is not estimable" not in audit["reason"]


def test_the_outer_training_fold_class_check_runs_before_any_inner_cv(monkeypatch) -> None:
    """四类不全的外层训练折必须在任何内层 CV 之前就被判定不可估。

    用记录调用的替身包住内层 CV，断言内层 CV 的**调用次数恰好等于外层四类齐全的折数**：
    被外层缺类检查拦下的折一次都不能进入内层 CV。
    """
    import attention_pipeline.supervised_learning.models_multiclass as multiclass_models

    frame, carrier = _frame_with_class_3_in_a_single_group()
    calls: list[list[str]] = []
    original = multiclass_models._inner_grouped_cv_participant_losses

    def _recording_inner(outer_train: pd.DataFrame, *args: object, **kwargs: object):
        calls.append(sorted(outer_train["participant_group_id"].astype(str).unique()))
        return original(outer_train, *args, **kwargs)

    monkeypatch.setattr(
        multiclass_models, "_inner_grouped_cv_participant_losses", _recording_inner
    )
    result = run_nested_loso_multiclass(
        frame,
        model_feature_schemes=_schemes(),
        route="A",
        c_candidates=[1.0],
        inner_splits=3,
        run_id="test-precheck-order",
        analysis_set_id="AS.test",
        membership_type="included_missing_aware",
    )
    audits = {a["outer_fold_group"]: a for a in result.fold_audits}
    outer_incomplete = {
        group for group, audit in audits.items() if not audit["outer_train_class_complete"]
    }
    assert outer_incomplete == {carrier}

    assert calls, "expected the estimable folds to run inner CV"
    n_outer_complete = sum(1 for a in audits.values() if a["outer_train_class_complete"])
    assert len(calls) == n_outer_complete, (
        "inner CV must be invoked exactly once per outer-complete fold, never for a "
        f"fold rejected by the outer class check: {len(calls)} calls vs "
        f"{n_outer_complete} outer-complete folds"
    )

    audit = audits[carrier]
    assert audit["failed"] is True
    assert audit["selection"] is None
    assert audit["final_refit"] is None
    assert "outer training fold is not estimable" in audit["reason"]


def test_a_class_missing_from_the_whole_frame_never_produces_a_three_class_model() -> None:
    """整帧缺类时外层每个折都缺类，所有折都必须失败，绝不产生 3 类模型。"""
    frame = _analysis_frame(n_groups=8)
    depleted = frame.loc[frame["q1_nominal_4class"].ne(3)].reset_index(drop=True)
    result = run_nested_loso_multiclass(
        depleted,
        model_feature_schemes=_schemes(),
        route="A",
        c_candidates=[1.0],
        inner_splits=3,
        run_id="test-missing-class-all-folds",
        analysis_set_id="AS.test",
        membership_type="included_missing_aware",
    )
    assert len(result.failures) == result.metadata["n_participant_groups"]
    for audit in result.fold_audits:
        assert audit["failed"] is True
        assert audit["outer_train_class_complete"] is False
        assert 3 in audit["outer_train_missing_classes"]
        # 缺类原因必须点名「外层训练折」而不是笼统的内层聚合失败。
        assert "outer training fold is not estimable" in audit["reason"]
    assert result.predictions["model_failed"].astype(bool).all()
    # 失败折的概率列必须是 NaN，而不是某个静默的三类概率。
    assert (
        result.predictions[list(Q1_MULTICLASS_SPEC.probability_columns)].isna().all().all()
    )
    assert result.predictions["predicted_q1_multiclass"].isna().all()


def test_unexpected_route_is_rejected() -> None:
    frame = _analysis_frame(n_groups=6)
    with pytest.raises(SupervisedLearningContractError, match="unsupported multiclass route"):
        run_nested_loso_multiclass(
            frame,
            route="C",
            inner_splits=3,
            analysis_set_id="AS.test",
            membership_type="included_missing_aware",
        )


def test_duplicate_probe_locators_are_rejected() -> None:
    frame = _analysis_frame(n_groups=6)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(SupervisedLearningContractError, match="duplicate probe locator rows"):
        run_nested_loso_multiclass(
            duplicated,
            model_feature_schemes=_schemes(),
            route="A",
            inner_splits=3,
            analysis_set_id="AS.test",
            membership_type="included_missing_aware",
        )


# --------------------------------------------------------------------------- archive adapter
def test_archive_adapter_maps_four_class_output_to_the_binary_reporting_contract() -> None:
    frame = _analysis_frame(n_groups=6)
    result = run_nested_loso_multiclass(
        frame,
        model_feature_schemes=_schemes(),
        route="A",
        c_candidates=[1.0],
        inner_splits=3,
        run_id="test-adapter",
        analysis_set_id="AS.test",
        membership_type="included_missing_aware",
    )
    adapted = _multiclass_predictions_to_archive_contract(
        result.predictions, spec=Q1_MULTICLASS_SPEC, expected_rows=len(frame)
    )
    for column in ("q1_binary", "p_q1_equals_1", "predicted_q1_binary"):
        assert column in adapted.columns
    # 适配列必须由原始四类值现场导出，而不是重新定义标签含义。
    assert (adapted["q1_binary"] == adapted["q1_nominal_4class"].eq(1).astype(int)).all()
    assert np.allclose(
        adapted["p_q1_equals_1"].to_numpy(float),
        adapted["p_q1_multiclass_1"].to_numpy(float),
    )
    assert (
        adapted["predicted_q1_binary"].to_numpy(float)
        == adapted["predicted_q1_multiclass"].eq(1).astype(float).to_numpy()
    ).all()
    # 四分类列原样保留。
    assert sorted(adapted["q1_nominal_4class"].unique().tolist()) == [1, 2, 3, 4]


def test_archive_adapter_rejects_a_row_count_mismatch() -> None:
    frame = _analysis_frame(n_groups=6)
    result = run_nested_loso_multiclass(
        frame,
        model_feature_schemes=_schemes(),
        route="A",
        c_candidates=[1.0],
        inner_splits=3,
        run_id="test-adapter-2",
        analysis_set_id="AS.test",
        membership_type="included_missing_aware",
    )
    with pytest.raises(SupervisedLearningContractError, match="row count mismatch"):
        _multiclass_predictions_to_archive_contract(
            result.predictions, spec=Q1_MULTICLASS_SPEC, expected_rows=len(frame) + 1
        )


# --------------------------------------------------------------------------- evaluation block
def test_evaluation_block_computes_the_prior_baseline_per_outer_fold() -> None:
    frame = _analysis_frame(n_groups=6)
    result = run_nested_loso_multiclass(
        frame,
        model_feature_schemes=_schemes(),
        route="A",
        c_candidates=[1.0],
        inner_splits=3,
        run_id="test-eval-block",
        analysis_set_id="AS.test",
        membership_type="included_missing_aware",
    )
    block = _multiclass_evaluation_block(
        frame, result.predictions, group_col="participant_group_id", spec=Q1_MULTICLASS_SPEC
    )
    baseline = block["prior_baseline"]["per_model"]["full"]
    assert baseline["n_folds"] == 6
    for fold in baseline["per_fold"]:
        assert fold["n_outer_train_rows"] == len(frame) - 8
        assert fold["n_outer_test_rows"] == 8
        assert fold["n_test_participants"] == 1
        assert sum(fold["class_priors"].values()) == pytest.approx(1.0)
    assert baseline["overall_prior_baseline_participant_macro_multiclass_log_loss"] is not None
    model_block = block["per_model"]["full"]
    assert model_block["summary"]["metric"] == "multiclass_log_loss"
    assert "participant_macro_multiclass_log_loss" in model_block["summary"]
    assert model_block["bootstrap"]["seed"] == 20260830
    assert model_block["discrimination"]["confusion_matrix_labels"] == [1, 2, 3, 4]
    assert block["participant_level_auroc_reported"] is False
    assert isinstance(model_block["exceeds_no_information_baseline"], bool)


# --------------------------------------------------------------------------- config contract
def test_four_class_config_reuses_every_frozen_binary_value() -> None:
    new = yaml.safe_load(CONFIG_4CLASS.read_text(encoding="utf-8"))
    old = yaml.safe_load(CONFIG_BINARY.read_text(encoding="utf-8"))
    assert new["feature_registry"] == old["feature_registry"]
    assert new["validation"] == old["validation"]
    assert new["preprocessing"] == old["preprocessing"]
    assert new["uncertainty"] == old["uncertainty"]
    assert new["models"]["primary"]["C_candidates"] == old["models"]["primary"]["C_candidates"]
    assert new["models"]["primary"]["max_iter"] == old["models"]["primary"]["max_iter"]
    assert new["pipeline"]["random_seed"] == old["pipeline"]["random_seed"]
    assert new["task"]["name"] == "q1_nominal_4class_multiclass"
    assert new["task"]["classes"] == [1, 2, 3, 4]
    assert new["multiclass_route"] in {"A", "B"}
    assert len(new["feature_registry"]["features"]) == 11
    assert _require_frozen_runtime_contract_multiclass(new) == new["multiclass_route"]


def test_four_class_config_contract_rejects_drift() -> None:
    new = yaml.safe_load(CONFIG_4CLASS.read_text(encoding="utf-8"))
    drifted_c = json.loads(json.dumps(new))
    drifted_c["models"]["primary"]["C_candidates"] = [0.01, 0.1, 1.0]
    with pytest.raises(SupervisedLearningContractError, match="regularisation candidates"):
        _require_frozen_runtime_contract_multiclass(drifted_c)

    drifted_seed = json.loads(json.dumps(new))
    drifted_seed["pipeline"]["random_seed"] = 1
    with pytest.raises(SupervisedLearningContractError, match="random_seed"):
        _require_frozen_runtime_contract_multiclass(drifted_seed)

    drifted_classes = json.loads(json.dumps(new))
    drifted_classes["task"]["classes"] = [0, 1, 2, 3]
    with pytest.raises(SupervisedLearningContractError, match="frozen four-class value"):
        _require_frozen_runtime_contract_multiclass(drifted_classes)

    drifted_route = json.loads(json.dumps(new))
    drifted_route["multiclass_route"] = "C"
    with pytest.raises(SupervisedLearningContractError, match="multiclass_route"):
        _require_frozen_runtime_contract_multiclass(drifted_route)


def test_end_to_end_run_writes_a_run_directory_with_four_class_outputs(tmp_path: Path) -> None:
    """端到端：运行落在 <output_root>/<run_id>，并写出四分类专属评估块。

    这里使用**真实的冻结注册表与真实的 full 方案**，只在内存里缩小参与者数量；
    这样被验证的是生产路径本身，而不是一个为测试特制的旁路。
    """
    frame = _with_full_model_sample_contract(_analysis_frame(n_groups=6))
    assert frame["comparison_models"].nunique() == 1
    input_path = tmp_path / "AS.test.csv"
    frame.to_csv(input_path, index=False, encoding="utf-8-sig")

    output_root = tmp_path / "out"
    manifest = run_multiclass_from_config(
        CONFIG_4CLASS,
        input_table=input_path,
        output_root=output_root,
        route="A",
        run_id="e2e-route-a",
    )
    run_root = output_root / "e2e-route-a"
    assert run_root.is_dir()
    for filename in (
        "probe_predictions.csv",
        "fold_audits.json",
        "failures.csv",
        "multiclass_evaluation.json",
        "run_manifest.json",
    ):
        assert (run_root / filename).is_file(), filename

    predictions = pd.read_csv(run_root / "probe_predictions.csv", encoding="utf-8-sig")
    for column in Q1_MULTICLASS_SPEC.probability_columns:
        assert column in predictions.columns
    assert "predicted_q1_multiclass" in predictions.columns
    assert "route" in predictions.columns
    assert sorted(predictions["q1_nominal_4class"].unique().tolist()) == [1, 2, 3, 4]

    evaluation = json.loads((run_root / "multiclass_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["task"] == "q1_nominal_4class_multiclass"
    assert evaluation["route"] == "A"
    assert evaluation["class_semantics"] == "nominal_unordered"
    per_model = evaluation["evaluation"]["per_model"]["full"]
    assert per_model["summary"]["n_participants"] == 6

    written_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert written_manifest["four_class_task"] == "q1_nominal_4class_multiclass"
    assert written_manifest["binary_reporting_adapter"]["binary_line_modified"] is False
    assert written_manifest["multiclass_evaluation"]["primary_metric"] == (
        "participant_macro_multiclass_log_loss"
    )
    assert manifest["route"] == "A"
    assert manifest["time_legality_runtime_verified"] is True
