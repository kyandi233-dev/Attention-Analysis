from __future__ import annotations

import copy
from dataclasses import replace

import pandas as pd

from ..config import Config
from .cohort import attach_repeat_groups, included_cohort, load_cohort_manifest


def prepare_behavior_runtime_config(config: Config) -> tuple[Config, pd.DataFrame]:
    """Materialize machine paths and cohort membership only in memory."""
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
    return attach_repeat_groups(
        trials, cohort, session_column="subject", require_all=require_all
    )


def assert_behavior_inference_allowed(config: Config, trials: pd.DataFrame) -> None:
    policy = config.section("analysis_policy")
    if bool(policy.get("require_repeat_participant_id_for_inference", True)):
        if "repeat_participant_id" not in trials.columns:
            raise RuntimeError("正式推断缺少 repeat_participant_id")
        if trials["repeat_participant_id"].isna().any():
            raise RuntimeError("正式推断存在缺失 repeat_participant_id 的场次")

    if not bool(policy.get("allow_legacy_session_level_stats", False)):
        raise RuntimeError(
            "正式 v2 已阻断旧 session-level stats.py。请先使用按 repeat_participant_id "
            "聚类/分层的正式统计实现；不得把场次级 Wilcoxon/AnovaRM/MixedLM "
            "伪装成参与者级推断。"
        )
