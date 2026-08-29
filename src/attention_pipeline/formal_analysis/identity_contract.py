from __future__ import annotations

from typing import Iterable

import pandas as pd

from .identity_questionnaire import reconcile_cohort_identity

CANONICAL_PARTICIPANT_GROUP_COLUMN = "participant_group_id"
LEGACY_PARTICIPANT_ALIAS = "repeat_participant_id"
DEFAULT_ALLOWED_LEGACY_IDENTITY_STATUSES = frozenset(
    {"temporary_confirmed", "confirmed", "verified", "validated", "governed"}
)


def _clean(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def reconcile_formal_identity(
    cohort: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    legacy_status_column: str = "identity_status",
    allowed_legacy_statuses: Iterable[str] = DEFAULT_ALLOWED_LEGACY_IDENTITY_STATUSES,
) -> pd.DataFrame:
    """Apply the formal participant-identity gate on top of the canonical reconciler.

    The underlying reconciler remains the single source of identity crosswalk logic.
    This wrapper adds the missing governance rule for *legacy-only* fallback: a
    questionnaire-missing session may use a legacy cohort group without a verified
    participant_key crosswalk only when the cohort explicitly marks that legacy
    identity with an allow-listed governance status. Otherwise the session remains
    in the cohort but participant-level inference is unresolved.
    """
    out = reconcile_cohort_identity(cohort, registry)
    if "legacy_repeat_participant_id" not in out.columns:
        raise ValueError("reconciled identity missing legacy_repeat_participant_id")

    if legacy_status_column in cohort.columns:
        statuses = cohort[["session_id", legacy_status_column]].copy()
        statuses["session_id"] = statuses["session_id"].astype(str).str.strip().str.rstrip("_")
        statuses["session_id"] = statuses["session_id"].str.replace(
            r"^sub-(\d+)$", lambda m: f"sub-{int(m.group(1)):03d}", regex=True
        )
        statuses = statuses.rename(columns={legacy_status_column: "legacy_identity_status"})
        out = out.drop(columns=["legacy_identity_status"], errors="ignore").merge(
            statuses, on="session_id", how="left", validate="one_to_one"
        )
    else:
        out["legacy_identity_status"] = pd.NA

    allowed = {str(value).strip().lower() for value in allowed_legacy_statuses if _clean(value)}
    legacy_only = out["participant_identity_source"].astype("string").eq(
        "governed_cohort_fallback_no_questionnaire_identity"
    )
    approved = out["legacy_identity_status"].map(
        lambda value: (_clean(value) or "").lower() in allowed
    )
    rejected = legacy_only & ~approved
    if rejected.any():
        out.loc[rejected, CANONICAL_PARTICIPANT_GROUP_COLUMN] = pd.NA
        out.loc[rejected, LEGACY_PARTICIPANT_ALIAS] = pd.NA
        out.loc[rejected, "participant_identity_source"] = "unresolved"
    out["participant_identity_resolution_reason"] = ""
    out.loc[legacy_only & approved, "participant_identity_resolution_reason"] = (
        "legacy-only participant group accepted by explicit governance status"
    )
    out.loc[rejected, "participant_identity_resolution_reason"] = (
        "legacy group exists but identity governance status is missing or not allow-listed"
    )
    direct = out["participant_identity_source"].astype("string").eq("questionnaire_repeat_registry")
    crosswalk = out["participant_identity_source"].astype("string").eq(
        "governed_cohort_crosswalk_for_missing_questionnaire"
    )
    out.loc[direct, "participant_identity_resolution_reason"] = "verified participant_key available"
    out.loc[crosswalk, "participant_identity_resolution_reason"] = (
        "legacy group has unambiguous overlap with verified participant_key"
    )
    unresolved = out[CANONICAL_PARTICIPANT_GROUP_COLUMN].isna()
    out.loc[unresolved & out["participant_identity_resolution_reason"].eq(""),
            "participant_identity_resolution_reason"] = (
        "no verified participant_key or governance-approved legacy identity"
    )
    out["participant_identity_resolved_for_clustering"] = ~unresolved
    return out


def assert_participant_group_contract(
    frame: pd.DataFrame,
    *,
    require_resolved: bool = True,
    compatibility_alias: str = LEGACY_PARTICIPANT_ALIAS,
) -> None:
    """Fail closed on participant-key drift or session-as-participant fallback."""
    if CANONICAL_PARTICIPANT_GROUP_COLUMN not in frame.columns:
        raise ValueError("participant_group_id missing from formal analysis table")
    groups = frame[CANONICAL_PARTICIPANT_GROUP_COLUMN].astype("string").str.strip()
    groups = groups.mask(groups.isin(["", "nan", "<NA>"]))
    if require_resolved and groups.isna().any():
        raise ValueError("participant_group_id contains unresolved sessions")

    if "session_id" in frame.columns:
        sessions = frame["session_id"].astype("string").str.strip().str.rstrip("_")
        sessions = sessions.str.replace(
            r"^sub-(\d+)$", lambda m: f"sub-{int(m.group(1)):03d}", regex=True
        )
        if (groups.notna() & groups.eq(sessions)).any():
            raise ValueError("session_id must never be used as participant_group_id")

    if compatibility_alias in frame.columns:
        alias = frame[compatibility_alias].astype("string").str.strip()
        alias = alias.mask(alias.isin(["", "nan", "<NA>"]))
        mismatch = groups.notna() & alias.notna() & groups.ne(alias)
        if mismatch.any():
            raise ValueError(
                f"{compatibility_alias} drifted from participant_group_id; compatibility alias is not a second identity"
            )
