from __future__ import annotations

import copy
from dataclasses import replace

import pandas as pd

from ..config import Config
from .cohort import attach_repeat_groups, included_cohort, load_cohort_manifest
from .identity_contract import (
    DEFAULT_ALLOWED_LEGACY_IDENTITY_STATUSES,
    assert_participant_group_contract,
    reconcile_formal_identity,
)
from .identity_questionnaire import attach_identity_metadata, load_repeat_registry


CURRENT_BEHAVIOR_PIPELINE_NAME = "attention-analysis-behavior-formal-v3"
RT_CV_MATHEMATICAL_MIN_N = 2
RT_CV_MATHEMATICAL_BASIS = "sample_sd_mathematical_requirement_not_empirical_gate"


def assert_current_behavior_rt_cv_contract(config: Config) -> None:
    """Fail closed if the declared current formal Behavior config drifts on RT-CV.

    The value 2 is not a tunable empirical threshold: it only records the
    mathematical requirement for a sample standard deviation. Generic adapter
    fixtures and historical/unrelated pipeline configs that do not declare the
    current formal-v3 pipeline remain outside this current-science guard.
    """
    pipeline = config.data.get("pipeline")
    if not isinstance(pipeline, dict):
        return
    if str(pipeline.get("name", "")).strip() != CURRENT_BEHAVIOR_PIPELINE_NAME:
        return
    behavior = config.section("behavior")
    if "rt_cv_min_n" not in behavior:
        raise ValueError(
            "current formal Behavior config must explicitly set behavior.rt_cv_min_n=2; "
            "historical n>=20 fallback is forbidden"
        )
    try:
        minimum = int(behavior["rt_cv_min_n"])
    except (TypeError, ValueError) as exc:
        raise ValueError("behavior.rt_cv_min_n must be the integer 2 for the current formal pipeline") from exc
    if minimum != RT_CV_MATHEMATICAL_MIN_N:
        raise ValueError(
            "current formal Behavior RT-CV contract requires rt_cv_min_n=2 as a mathematical sample-SD condition; "
            f"got {minimum}"
        )
    basis = str(behavior.get("rt_cv_min_n_basis", "")).strip()
    if basis != RT_CV_MATHEMATICAL_BASIS:
        raise ValueError(
            "current formal Behavior RT-CV contract requires "
            f"rt_cv_min_n_basis={RT_CV_MATHEMATICAL_BASIS!r}; got {basis!r}"
        )


def _legacy_identity_policy(config: Config) -> tuple[str, tuple[str, ...]]:
    cohort_cfg = config.section("cohort")
    status_column = str(cohort_cfg.get("legacy_identity_status_column", "identity_status"))
    raw_allowed = cohort_cfg.get(
        "allowed_legacy_identity_statuses",
        sorted(DEFAULT_ALLOWED_LEGACY_IDENTITY_STATUSES),
    )
    if not isinstance(raw_allowed, list) or not raw_allowed:
        raise ValueError("cohort.allowed_legacy_identity_statuses must be a non-empty list")
    return status_column, tuple(str(value) for value in raw_allowed)


def prepare_behavior_runtime_config(config: Config) -> tuple[Config, pd.DataFrame]:
    """Materialize machine paths, governed cohort membership and participant identity in memory.

    The cohort manifest remains authoritative for which sessions are in the
    formal queue. The questionnaire-derived repeat registry may overlay the
    anonymous participant identity, but it never creates/deletes cohort rows.
    Participant identity is reconciled before any group-required gate so a
    questionnaire-verified session is not rejected merely because the legacy
    cohort grouping column is empty.
    """
    assert_current_behavior_rt_cv_contract(config)

    data = copy.deepcopy(config.data)
    data_cfg = data.setdefault("data", {})
    root_key = str(data_cfg.get("roots_path_key", "formal_raw_roots"))
    data_cfg["roots"] = [str(path) for path in config.registry_paths(root_key)]

    cohort_cfg = config.section("cohort")
    cohort = load_cohort_manifest(
        config,
        path_key=str(cohort_cfg.get("manifest_path_key", "cohort_manifest")),
        session_column=str(cohort_cfg.get("session_column", "session_id")),
        include_column=str(cohort_cfg.get("include_column", "include")),
        group_column=str(cohort_cfg.get("repeat_group_column", "repeat_participant_id")),
    )
    identity_registry_key = cohort_cfg.get("identity_registry_path_key")
    if identity_registry_key not in (None, ""):
        registry = load_repeat_registry(config, path_key=str(identity_registry_key))
        status_column, allowed_statuses = _legacy_identity_policy(config)
        cohort = reconcile_formal_identity(
            cohort,
            registry,
            legacy_status_column=status_column,
            allowed_legacy_statuses=allowed_statuses,
        )

    # Cohort membership and participant-level estimability are separate concerns.
    # Keep every governed included session in the runtime queue; unresolved
    # participant identity is blocked only when participant-level inference is requested.
    included = included_cohort(cohort, require_groups=False)
    data_cfg["include"] = included["session_id"].tolist()
    data_cfg["exclude"] = []
    data_cfg["min_subject_number"] = 0
    return replace(config, data=data), cohort


def attach_behavior_groups(
    trials: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    grouped = attach_repeat_groups(
        trials, cohort, session_column="subject", require_all=require_all
    )
    out = attach_identity_metadata(grouped, cohort)
    # science-v3 still consumes repeat_participant_id at a legacy module boundary;
    # assert that the alias is exactly the canonical participant_group_id.
    assert_participant_group_contract(out, require_resolved=require_all)
    return out


def assert_behavior_inference_allowed(config: Config, trials: pd.DataFrame) -> None:
    policy = config.section("analysis_policy")
    require_groups = bool(
        policy.get(
            "require_participant_group_id_for_inference",
            policy.get("require_repeat_participant_id_for_inference", True),
        )
    )
    if require_groups:
        assert_participant_group_contract(trials, require_resolved=True)
    if bool(policy.get("require_participant_group_identity_source", True)):
        if "participant_identity_source" not in trials.columns:
            raise RuntimeError("正式推断缺少 participant_identity_source 审计字段")
        if trials["participant_identity_source"].astype(str).eq("unresolved").any():
            raise RuntimeError("正式推断存在 participant identity unresolved 的场次")

    if not bool(policy.get("allow_legacy_session_level_stats", False)):
        raise RuntimeError(
            "正式 v2 已阻断旧 session-level stats.py。请使用按参与者聚类/分层的正式统计实现；"
            "不得把场次级 Wilcoxon/AnovaRM/MixedLM 伪装成参与者级推断。"
        )
