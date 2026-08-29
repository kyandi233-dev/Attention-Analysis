from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..path_registry import PathRegistry
from .cohort import canonical_session_id

IDENTITY_CORE_COLUMNS = (
    "participant_key",
    "experiment_id",
    "visit_order",
    "prior_visit_count",
    "total_visit_count",
    "is_repeat_participant",
    "is_first_visit",
    "is_pre_experiment_experienced",
    "is_cross_stage_repeat",
    "is_within_stage_repeat",
    "prior_same_stage_count",
    "repeat_group",
    "identity_match_basis",
    "identity_conflict_flag",
    "identity_conflict_fields",
    "analysis_group",
    "stage",
    "location",
    "questionnaire_version",
)

_VISIT_INTEGER_COLUMNS = (
    "visit_order",
    "prior_visit_count",
    "total_visit_count",
    "prior_same_stage_count",
)


def _source_path(source: Config | PathRegistry | str | Path, path_key: str) -> Path:
    if isinstance(source, Config):
        return source.registry_path(path_key)
    if isinstance(source, PathRegistry):
        return source.path_value(path_key)
    return Path(source).expanduser().resolve()


def _clean_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "<NA>"]))


def _normalize_session_table(table: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
    if "subid" not in table.columns:
        raise ValueError(f"{source_label} 缺少 subid")
    out = table.copy()
    out["subid_raw"] = _clean_string(out["subid"])
    if out["subid_raw"].isna().any():
        raise ValueError(f"{source_label} 存在空 subid")
    out["session_id"] = out["subid_raw"].map(canonical_session_id)
    if out["session_id"].duplicated().any():
        dup = sorted(out.loc[out["session_id"].duplicated(keep=False), "session_id"].unique())
        raise ValueError(f"{source_label} 存在重复 session: {dup}")
    if "participant_key" in out.columns:
        out["participant_key"] = _clean_string(out["participant_key"])
    for column in _VISIT_INTEGER_COLUMNS:
        if column in out.columns:
            numeric = pd.to_numeric(out[column], errors="coerce")
            finite = numeric.dropna()
            if len(finite) and not np.allclose(finite, np.round(finite), rtol=0.0, atol=1e-9):
                raise ValueError(f"{source_label}.{column} 存在非整数值")
            out[column] = numeric.round().astype("Int64")
    return out


def load_repeat_registry(
    source: Config | PathRegistry | str | Path,
    *,
    path_key: str = "repeat_registry",
) -> pd.DataFrame:
    """Load the questionnaire-derived repeat registry without using any PII.

    The registry is one row per questionnaire/session, not one row per natural
    person.  ``participant_key`` is the verified anonymous repeated-participant
    identifier and ``subid`` is canonicalized only for cross-modal joins.
    """
    path = _source_path(source, path_key)
    table = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"subid", "participant_key", "visit_order", "prior_visit_count", "total_visit_count"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"repeat registry 缺少列: {sorted(missing)}")
    out = _normalize_session_table(table, source_label="repeat registry")
    if out["participant_key"].isna().any():
        missing_sessions = out.loc[out["participant_key"].isna(), "session_id"].tolist()
        raise ValueError("repeat registry 的实际问卷行缺少 participant_key: " + ", ".join(missing_sessions))

    for participant, current in out.groupby("participant_key", sort=False):
        order = sorted(pd.to_numeric(current["visit_order"], errors="coerce").dropna().astype(int).tolist())
        expected = list(range(1, len(current) + 1))
        if order != expected:
            raise ValueError(f"participant_key={participant} 的 visit_order 非连续 1..N: {order}")
        prior = pd.to_numeric(current["prior_visit_count"], errors="coerce")
        visit = pd.to_numeric(current["visit_order"], errors="coerce")
        if ((prior != visit - 1) & prior.notna() & visit.notna()).any():
            raise ValueError(f"participant_key={participant} 的 prior_visit_count 与 visit_order 不一致")
        total = pd.to_numeric(current["total_visit_count"], errors="coerce")
        if total.isna().any() or not total.eq(len(current)).all():
            raise ValueError(f"participant_key={participant} 的 total_visit_count 与 registry 行数不一致")
        if "is_first_visit" in current.columns:
            first = pd.to_numeric(current["is_first_visit"], errors="coerce")
            expected_first = visit.eq(1).astype(float)
            if ((first != expected_first) & first.notna()).any():
                raise ValueError(f"participant_key={participant} 的 is_first_visit 与 visit_order 不一致")
    return out


def load_questionnaire_data(
    source: Config | PathRegistry | str | Path,
    *,
    path_key: str = "questionnaire_derived_data",
) -> pd.DataFrame:
    """Load the derived questionnaire table; never reconstruct identity from PII."""
    path = _source_path(source, path_key)
    table = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"subid", "experiment_id", "participant_key"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"questionnaire derived data 缺少列: {sorted(missing)}")
    return _normalize_session_table(table, source_label="questionnaire derived data")


def validate_questionnaire_registry_consistency(
    questionnaire: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Fail closed when the two generated questionnaire products disagree."""
    q_sessions = set(questionnaire["session_id"].astype(str))
    r_sessions = set(registry["session_id"].astype(str))
    if q_sessions != r_sessions:
        raise ValueError(
            "questionnaire 与 repeat registry 的 session 集合不一致: "
            f"questionnaire_only={sorted(q_sessions-r_sessions)}, registry_only={sorted(r_sessions-q_sessions)}"
        )
    compare = [c for c in IDENTITY_CORE_COLUMNS if c in questionnaire.columns and c in registry.columns]
    q = questionnaire[["session_id", *compare]].copy()
    r = registry[["session_id", *compare]].copy()
    merged = q.merge(r, on="session_id", how="outer", validate="one_to_one", suffixes=("__questionnaire", "__registry"))
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for row in merged.itertuples(index=False):
        record = {"session_id": str(row.session_id), "status": "consistent", "mismatch_fields": ""}
        bad: list[str] = []
        for column in compare:
            a = getattr(row, f"{column}__questionnaire")
            b = getattr(row, f"{column}__registry")
            same = (pd.isna(a) and pd.isna(b)) or str(a).strip() == str(b).strip()
            if not same:
                bad.append(column)
        if bad:
            record["status"] = "mismatch"
            record["mismatch_fields"] = ";".join(bad)
            mismatches.append(f"{row.session_id}:{','.join(bad)}")
        rows.append(record)
    audit = pd.DataFrame(rows)
    if mismatches:
        raise ValueError("questionnaire / registry 核心身份字段不一致: " + "; ".join(mismatches))
    return audit


def reconcile_cohort_identity(cohort: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Overlay participant_key on the governed cohort without deriving the cohort from questionnaire.

    ``participant_key`` remains missing when a session has no questionnaire row.
    ``participant_group_id`` is the inference/prediction grouping interface.  It
    uses participant_key when present.  For a session missing from the registry,
    a governed legacy cohort group may be retained or bridged only when its
    overlap with verified participant_key values is internally unambiguous.
    """
    required = {"session_id", "include", "repeat_participant_id"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort identity reconciliation 缺少列: {sorted(missing)}")
    base = cohort.copy()
    base["session_id"] = base["session_id"].map(canonical_session_id)
    base["legacy_repeat_participant_id"] = _clean_string(base["repeat_participant_id"])
    reg_cols = ["session_id", *[c for c in IDENTITY_CORE_COLUMNS if c in registry.columns]]
    merged = base.merge(registry[reg_cols], on="session_id", how="left", validate="one_to_one")

    overlap = merged.dropna(subset=["legacy_repeat_participant_id", "participant_key"])
    legacy_map_counts = overlap.groupby("legacy_repeat_participant_id")["participant_key"].nunique()
    conflict = legacy_map_counts[legacy_map_counts > 1]
    if len(conflict):
        raise ValueError(
            "既有 cohort repeat group 与新 participant_key 发生一对多冲突: "
            + ", ".join(map(str, conflict.index.tolist()))
        )
    legacy_to_key = (
        overlap.drop_duplicates(["legacy_repeat_participant_id", "participant_key"])
        .set_index("legacy_repeat_participant_id")["participant_key"]
        .to_dict()
    )

    group_ids: list[object] = []
    sources: list[str] = []
    for row in merged.itertuples(index=False):
        participant = getattr(row, "participant_key", pd.NA)
        legacy = getattr(row, "legacy_repeat_participant_id", pd.NA)
        if pd.notna(participant):
            group_ids.append(str(participant))
            sources.append("questionnaire_repeat_registry")
        elif pd.notna(legacy) and str(legacy) in legacy_to_key:
            group_ids.append(str(legacy_to_key[str(legacy)]))
            sources.append("governed_cohort_crosswalk_for_missing_questionnaire")
        elif pd.notna(legacy):
            group_ids.append(f"legacy:{str(legacy)}")
            sources.append("governed_cohort_fallback_no_questionnaire_identity")
        else:
            group_ids.append(pd.NA)
            sources.append("unresolved")
    merged["participant_group_id"] = pd.Series(group_ids, index=merged.index, dtype="string")
    merged["participant_identity_source"] = pd.Series(sources, index=merged.index, dtype="string")
    merged["repeat_participant_id"] = merged["participant_group_id"]
    merged["participant_identity_resolved_for_clustering"] = merged["participant_group_id"].notna()
    return merged


def attach_identity_metadata(frame: pd.DataFrame, cohort_identity: pd.DataFrame) -> pd.DataFrame:
    if "session_id" not in frame.columns:
        raise ValueError("identity metadata join requires session_id")
    metadata_cols = [
        "session_id", "participant_key", "participant_group_id", "participant_identity_source",
        "participant_identity_resolved_for_clustering", "visit_order", "prior_visit_count",
        "total_visit_count", "is_first_visit", "identity_conflict_flag", "identity_conflict_fields",
        "legacy_repeat_participant_id",
    ]
    available = [c for c in metadata_cols if c in cohort_identity.columns]
    metadata = cohort_identity[available].drop_duplicates("session_id")
    out = frame.copy()
    drop_existing = [c for c in available if c != "session_id" and c in out.columns]
    if drop_existing:
        out = out.drop(columns=drop_existing)
    return out.merge(metadata, on="session_id", how="left", validate="many_to_one")


def left_join_questionnaire(session_table: pd.DataFrame, questionnaire: pd.DataFrame) -> pd.DataFrame:
    """Preserve every session and prefix questionnaire payload fields to prevent semantic collisions."""
    if "session_id" not in session_table.columns:
        raise ValueError("questionnaire join requires session_id")
    payload = questionnaire.copy()
    rename = {
        c: f"questionnaire__{c}"
        for c in payload.columns
        if c not in {"session_id", "subid_raw"}
    }
    payload = payload.rename(columns=rename)
    out = session_table.merge(payload, on="session_id", how="left", validate="many_to_one")
    presence_col = "questionnaire__experiment_id"
    out["questionnaire_present"] = out[presence_col].notna().astype(int) if presence_col in out else 0
    out["questionnaire_missing_reason"] = np.where(
        out["questionnaire_present"].eq(1), "", "no_derived_questionnaire_row_for_session"
    )
    return out


def build_identity_audit(cohort_identity: pd.DataFrame, questionnaire: pd.DataFrame) -> pd.DataFrame:
    q_sessions = set(questionnaire["session_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for row in cohort_identity.itertuples(index=False):
        session = str(row.session_id)
        rows.append({
            "session_id": session,
            "include": bool(row.include),
            "questionnaire_present": int(session in q_sessions),
            "participant_key_present": int(pd.notna(getattr(row, "participant_key", pd.NA))),
            "participant_group_id": getattr(row, "participant_group_id", pd.NA),
            "participant_identity_source": getattr(row, "participant_identity_source", pd.NA),
            "legacy_repeat_participant_id": getattr(row, "legacy_repeat_participant_id", pd.NA),
            "visit_order": getattr(row, "visit_order", pd.NA),
            "prior_visit_count": getattr(row, "prior_visit_count", pd.NA),
            "identity_conflict_flag": getattr(row, "identity_conflict_flag", pd.NA),
            "identity_conflict_fields": getattr(row, "identity_conflict_fields", pd.NA),
            "inference_group_resolved": int(bool(getattr(row, "participant_identity_resolved_for_clustering", False))),
        })
    return pd.DataFrame(rows)


def build_questionnaire_variable_audit(questionnaire: pd.DataFrame) -> pd.DataFrame:
    """Inventory fields without automatically assigning them to a statistical model."""
    structural = {"session_id", "subid", "subid_raw", *IDENTITY_CORE_COLUMNS,
                  "source_file", "source_row", "submission_time", "backup_source", "备注"}
    rows: list[dict[str, Any]] = []
    n = len(questionnaire)
    for column in questionnaire.columns:
        values = questionnaire[column]
        nonmissing = int(values.notna().sum())
        rows.append({
            "column": column,
            "dtype": str(values.dtype),
            "n_rows": int(n),
            "n_nonmissing": nonmissing,
            "missing_fraction": float(1 - nonmissing / n) if n else np.nan,
            "unique_nonmissing_n": int(values.dropna().nunique()),
            "role_status": "structural_identity_or_provenance" if column in structural else "unfrozen_questionnaire_variable",
            "automatic_model_use_allowed": False,
            "decision_contract": "variable role must be prespecified/documented; never select by outcome significance",
        })
    return pd.DataFrame(rows)
