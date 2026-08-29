from __future__ import annotations

import copy
from dataclasses import replace

import pandas as pd

from ..config import Config
from .cohort import attach_repeat_groups, included_cohort, load_cohort_manifest
from .identity_questionnaire import (
    attach_identity_metadata,
    load_repeat_registry,
    reconcile_cohort_identity,
)


def prepare_behavior_runtime_config(config: Config) -> tuple[Config, pd.DataFrame]:
    """Materialize machine paths, governed cohort membership and participant identity in memory.

    The cohort manifest remains authoritative for which sessions are in the
    formal queue.  The questionnaire-derived repeat registry may overlay the
    anonymous participant identity, but it never creates/deletes cohort rows.
    """
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
        cohort = reconcile_cohort_identity(cohort, registry)

    included = included_cohort(cohort, require_groups=True)
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
    return attach_identity_metadata(grouped, cohort)


def assert_behavior_inference_allowed(config: Config, trials: pd.DataFrame) -> None:
    policy = config.section("analysis_policy")
    if bool(policy.get("require_repeat_participant_id_for_inference", True)):
        if "repeat_participant_id" not in trials.columns:
            raise RuntimeError("正式推断缺少参与者聚类键 repeat_participant_id")
        if trials["repeat_participant_id"].isna().any():
            raise RuntimeError("正式推断存在无法解析参与者聚类键的场次")
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
