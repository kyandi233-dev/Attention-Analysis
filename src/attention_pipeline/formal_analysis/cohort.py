from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import Config
from ..path_registry import PathRegistry

_SESSION_RE = re.compile(r"^sub-(\d+)_?$", re.IGNORECASE)
_TRUE = {"1", "true", "yes", "y", "include", "included"}


def canonical_session_id(value: object) -> str:
    text = str(value).strip()
    match = _SESSION_RE.match(text)
    if not match:
        raise ValueError(f"非法场次标识: {value!r}; 期望 sub-XXX")
    return f"sub-{int(match.group(1)):03d}"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in _TRUE


@dataclass(frozen=True)
class CohortSummary:
    sessions: int
    groups: int
    repeated_groups: int
    repeated_sessions: int


def load_cohort_manifest(
    source: Config | PathRegistry | str | Path,
    *,
    path_key: str = "cohort_manifest",
    session_column: str = "session_id",
    include_column: str = "include",
    group_column: str = "repeat_participant_id",
) -> pd.DataFrame:
    if isinstance(source, Config):
        path = source.registry_path(path_key)
    elif isinstance(source, PathRegistry):
        path = source.path_value(path_key)
    else:
        path = Path(source).resolve()

    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {session_column, include_column, group_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"cohort manifest 缺少列: {sorted(missing)}")

    result = table.copy()
    result["session_id"] = result[session_column].map(canonical_session_id)
    result["include"] = result[include_column].map(_as_bool)
    result["repeat_participant_id"] = result[group_column].astype("string").str.strip()
    result.loc[result["repeat_participant_id"].isin(["", "nan", "<NA>"]), "repeat_participant_id"] = pd.NA

    if result["session_id"].duplicated().any():
        dup = result.loc[result["session_id"].duplicated(keep=False), "session_id"].tolist()
        raise ValueError(f"cohort manifest 存在重复 session_id: {sorted(set(dup))}")

    return result


def included_cohort(table: pd.DataFrame, *, require_groups: bool = False) -> pd.DataFrame:
    result = table.loc[table["include"].eq(True)].copy()
    if result.empty:
        raise ValueError("cohort manifest 没有 include=true 的场次")
    if require_groups and result["repeat_participant_id"].isna().any():
        missing = result.loc[result["repeat_participant_id"].isna(), "session_id"].tolist()
        raise ValueError(
            "正式推断要求每个纳入场次都有 repeat_participant_id；缺失: "
            + ", ".join(missing)
        )
    return result


def summarize_cohort(table: pd.DataFrame) -> CohortSummary:
    included = included_cohort(table, require_groups=True)
    counts = included.groupby("repeat_participant_id", dropna=False)["session_id"].nunique()
    repeated = counts[counts > 1]
    return CohortSummary(
        sessions=int(len(included)),
        groups=int(counts.size),
        repeated_groups=int(repeated.size),
        repeated_sessions=int(repeated.sum()),
    )


def attach_repeat_groups(
    frame: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    session_column: str = "subject",
    require_all: bool = True,
) -> pd.DataFrame:
    if session_column not in frame.columns:
        raise ValueError(f"数据表缺少场次列: {session_column}")
    mapping = included_cohort(cohort, require_groups=False)[
        ["session_id", "repeat_participant_id"]
    ].copy()
    data = frame.copy()
    data["session_id"] = data[session_column].map(canonical_session_id)
    merged = data.merge(mapping, on="session_id", how="left", validate="many_to_one")
    if require_all and merged["repeat_participant_id"].isna().any():
        missing = sorted(merged.loc[merged["repeat_participant_id"].isna(), "session_id"].unique())
        raise ValueError(
            "数据含未进入 cohort manifest 或缺少重复分组的场次: " + ", ".join(missing)
        )
    return merged


def validate_participant_disjoint_folds(
    table: pd.DataFrame,
    *,
    group_column: str = "repeat_participant_id",
    fold_column: str = "fold",
) -> None:
    missing = {group_column, fold_column} - set(table.columns)
    if missing:
        raise ValueError(f"fold 审计缺少列: {sorted(missing)}")
    grouped = table.dropna(subset=[group_column, fold_column]).groupby(group_column)[fold_column].nunique()
    leaking = grouped[grouped > 1]
    if len(leaking):
        raise ValueError(
            "同一 repeat_participant_id 被分到多个折: "
            + ", ".join(map(str, leaking.index.tolist()))
        )
