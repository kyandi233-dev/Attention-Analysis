"""Behavior-authoritative probe-key mapping for cross-modal supervised tables.

The mapping is intentionally explicit: a global probe index is never renamed,
modulo-transformed, or otherwise guessed into a within-block index. Modality rows
are linked to the Behavior authority only through fields whose semantics are
explicitly probe-timed, together with session and block identity.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


CANONICAL_KEY_COLUMNS = ("session_id", "block_id", "probe_index_in_block")
BEHAVIOR_REQUIRED_COLUMNS = (
    "session_id",
    "block_id",
    "probe_order_in_block",
    "probe_event_id",
    "probe_time_ms",
)
# These names are permitted only because their declared semantics are the probe
# onset / probe-locked window end. Formal-experiment ``absolute_onset_time`` is
# the SART trial stimulus onset and is deliberately excluded.
DEFAULT_MODALITY_TIME_CANDIDATES = (
    "probe_time_ms",
    "probe_onset_time",
    "probe_onset_ms",
    "probe_onset_unix_ms",
    "window_end_ms",
    "window_end_unix_ms",
)
_ALLOWED_BLOCKS = frozenset({"b1", "b2"})


class ProbeKeyAuthorityError(ValueError):
    """Raised when probe identity cannot be mapped to Behavior without guessing."""


def normalize_block(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower().str.replace("-", "", regex=False)
    text = text.str.replace("block", "", regex=False)
    normalized = text.where(text.str.startswith("b"), "b" + text)
    invalid = normalized.isna() | ~normalized.isin(_ALLOWED_BLOCKS)
    if invalid.any():
        bad = sorted(values.loc[invalid].astype("string").fillna("<NA>").unique().tolist())
        raise ProbeKeyAuthorityError(f"illegal formal block identifiers: {bad}")
    return normalized.astype(str)


def _integral_ms(values: pd.Series, *, context: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ProbeKeyAuthorityError(f"{context} contains missing or non-finite probe times")
    rounded = np.rint(numeric.to_numpy(dtype=float))
    if not np.allclose(numeric.to_numpy(dtype=float), rounded, rtol=0.0, atol=1e-6):
        raise ProbeKeyAuthorityError(f"{context} must use integral Unix milliseconds")
    return pd.Series(rounded.astype("int64"), index=values.index, dtype="int64")


def _positive_integral_index(values: pd.Series, *, context: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    as_float = numeric.to_numpy(dtype=float)
    if numeric.isna().any() or not np.isfinite(as_float).all():
        raise ProbeKeyAuthorityError(f"{context} contains missing or non-finite values")
    rounded = np.rint(as_float)
    if not np.allclose(as_float, rounded, rtol=0.0, atol=1e-12) or np.any(rounded < 1):
        raise ProbeKeyAuthorityError(f"{context} must contain positive integers")
    return pd.Series(rounded.astype("int64"), index=values.index, dtype="Int64")


def _prepare_behavior_authority(behavior: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(BEHAVIOR_REQUIRED_COLUMNS) - set(behavior.columns))
    if missing:
        raise ProbeKeyAuthorityError(f"Behavior probe authority missing columns: {missing}")
    authority = behavior.copy()
    if authority.empty:
        raise ProbeKeyAuthorityError("Behavior probe authority is empty")
    authority["session_id"] = authority["session_id"].astype("string").str.strip()
    if authority["session_id"].isna().any() or authority["session_id"].eq("").any():
        raise ProbeKeyAuthorityError("Behavior session_id contains missing or blank values")
    authority["session_id"] = authority["session_id"].astype(str)
    authority["block_id"] = normalize_block(authority["block_id"])
    authority["probe_index_in_block"] = _positive_integral_index(
        authority["probe_order_in_block"], context="Behavior probe_order_in_block"
    )
    authority["_authority_probe_time_ms"] = _integral_ms(
        authority["probe_time_ms"], context="Behavior probe_time_ms"
    )
    event = authority["probe_event_id"].astype("string").str.strip()
    if event.isna().any() or event.eq("").any():
        raise ProbeKeyAuthorityError("Behavior probe_event_id contains missing or blank values")
    authority["probe_event_id"] = event.astype(str)
    if authority.duplicated(["session_id", "block_id", "_authority_probe_time_ms"]).any():
        raise ProbeKeyAuthorityError("Behavior authority has duplicate session/block/probe_time_ms keys")
    if authority.duplicated(list(CANONICAL_KEY_COLUMNS)).any():
        raise ProbeKeyAuthorityError("Behavior authority has duplicate canonical probe keys")
    if authority["probe_event_id"].duplicated().any():
        raise ProbeKeyAuthorityError("Behavior authority has duplicate probe_event_id values")
    return authority


def _available_time_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def _resolve_modality_probe_time(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> tuple[str | None, pd.Series | None]:
    available = _available_time_columns(frame, candidates)
    if not available:
        return None, None
    resolved: dict[str, pd.Series] = {
        column: _integral_ms(frame[column], context=f"modality {column}")
        for column in available
    }
    reference_name = available[0]
    reference = resolved[reference_name]
    for column in available[1:]:
        if not reference.equals(resolved[column]):
            raise ProbeKeyAuthorityError(
                "multiple probe-time fields disagree: "
                f"{reference_name} vs {column}; mapping requires one unambiguous probe time"
            )
    return reference_name, reference


def _check_existing_authoritative_value(
    merged: pd.DataFrame,
    *,
    existing_column: str,
    authority_column: str,
    context: str,
    numeric: bool = False,
) -> None:
    if existing_column not in merged.columns or authority_column not in merged.columns:
        return
    existing = merged[existing_column]
    authority = merged[authority_column]
    present = existing.notna()
    if not numeric:
        existing_text = existing.astype("string").str.strip()
        present &= existing_text.ne("")
        mismatch = present & existing_text.ne(authority.astype("string").str.strip())
    else:
        existing_num = pd.to_numeric(existing, errors="coerce")
        authority_num = pd.to_numeric(authority, errors="coerce")
        malformed = present & existing_num.isna()
        if malformed.any():
            raise ProbeKeyAuthorityError(f"existing {existing_column} contains non-numeric values")
        mismatch = present & existing_num.ne(authority_num)
    if mismatch.any():
        raise ProbeKeyAuthorityError(f"existing {context} disagrees with Behavior authority")


def map_probe_keys_to_behavior(
    modality_frame: pd.DataFrame,
    behavior_probe_authority: pd.DataFrame,
    *,
    time_candidates: Sequence[str] = DEFAULT_MODALITY_TIME_CANDIDATES,
) -> pd.DataFrame:
    """Attach canonical probe identity using the Behavior probe table.

    Preferred mapping key is ``session_id + block_id + explicit probe time``. If
    multiple recognized probe-time aliases exist they must agree exactly. If no
    probe-time field exists, an already-existing within-block probe index may be
    verified against Behavior. ``probe_index_global`` alone is deliberately
    insufficient because its semantics differ from a within-block index.
    """
    if modality_frame.empty:
        return modality_frame.copy()
    if "session_id" not in modality_frame.columns:
        raise ProbeKeyAuthorityError("modality probe table missing session_id")
    block_source = (
        "block_id"
        if "block_id" in modality_frame.columns
        else "block_num"
        if "block_num" in modality_frame.columns
        else None
    )
    if block_source is None:
        raise ProbeKeyAuthorityError("modality probe table missing block_id/block_num")

    authority = _prepare_behavior_authority(behavior_probe_authority)
    frame = modality_frame.copy()
    frame["session_id"] = frame["session_id"].astype("string").str.strip()
    if frame["session_id"].isna().any() or frame["session_id"].eq("").any():
        raise ProbeKeyAuthorityError("modality session_id contains missing or blank values")
    frame["session_id"] = frame["session_id"].astype(str)
    frame["block_id"] = normalize_block(frame[block_source])

    authority_keep = [
        "session_id",
        "block_id",
        "probe_index_in_block",
        "probe_order_in_block",
        "probe_event_id",
        "probe_time_ms",
        "_authority_probe_time_ms",
    ]
    for optional in ("participant_group_id", "q1_nominal_4class", "q2_ordinal_4level"):
        if optional in authority.columns:
            authority_keep.append(optional)
    authority_sub = authority[authority_keep].rename(
        columns={
            "probe_index_in_block": "_authority_probe_index_in_block",
            "probe_order_in_block": "_authority_probe_order_in_block",
            "probe_event_id": "_authority_probe_event_id",
            "probe_time_ms": "_authority_probe_time_raw",
            "participant_group_id": "_authority_participant_group_id",
            "q1_nominal_4class": "_authority_q1_nominal_4class",
            "q2_ordinal_4level": "_authority_q2_ordinal_4level",
        }
    )

    time_column, modality_probe_time = _resolve_modality_probe_time(frame, time_candidates)
    if time_column is not None and modality_probe_time is not None:
        frame["_modality_probe_time_ms"] = modality_probe_time
        merged = frame.merge(
            authority_sub,
            left_on=["session_id", "block_id", "_modality_probe_time_ms"],
            right_on=["session_id", "block_id", "_authority_probe_time_ms"],
            how="left",
            validate="many_to_one",
        )
    elif "probe_index_in_block" in frame.columns:
        frame["_modality_probe_index_in_block"] = _positive_integral_index(
            frame["probe_index_in_block"], context="existing probe_index_in_block"
        )
        merged = frame.merge(
            authority_sub,
            left_on=["session_id", "block_id", "_modality_probe_index_in_block"],
            right_on=["session_id", "block_id", "_authority_probe_index_in_block"],
            how="left",
            validate="many_to_one",
        )
    else:
        raise ProbeKeyAuthorityError(
            "cannot map probe identity: need an explicit probe-time field or an existing "
            "probe_index_in_block; probe_index_global and trial absolute_onset_time are never "
            "treated as within-block/probe-time substitutes"
        )

    if merged["_authority_probe_index_in_block"].isna().any():
        unmatched = int(merged["_authority_probe_index_in_block"].isna().sum())
        raise ProbeKeyAuthorityError(
            f"{unmatched} modality probe rows could not be mapped to Behavior authority"
        )

    if "probe_index_in_block" in frame.columns:
        existing = _positive_integral_index(
            merged["probe_index_in_block"], context="existing probe_index_in_block"
        )
        mismatch = existing.ne(merged["_authority_probe_index_in_block"].astype("Int64"))
        if mismatch.any():
            raise ProbeKeyAuthorityError("existing probe_index_in_block disagrees with Behavior authority")

    _check_existing_authoritative_value(
        merged,
        existing_column="probe_event_id",
        authority_column="_authority_probe_event_id",
        context="probe_event_id",
    )
    _check_existing_authoritative_value(
        merged,
        existing_column="q1_nominal_4class",
        authority_column="_authority_q1_nominal_4class",
        context="q1_nominal_4class",
        numeric=True,
    )
    _check_existing_authoritative_value(
        merged,
        existing_column="q2_ordinal_4level",
        authority_column="_authority_q2_ordinal_4level",
        context="q2_ordinal_4level",
        numeric=True,
    )

    if "participant_group_id" in frame.columns and "_authority_participant_group_id" in merged.columns:
        existing_group = merged["participant_group_id"].astype("string").str.strip()
        authority_group = merged["_authority_participant_group_id"].astype("string").str.strip()
        present = existing_group.notna() & existing_group.ne("")
        mismatch = present & authority_group.notna() & existing_group.ne(authority_group)
        if mismatch.any():
            raise ProbeKeyAuthorityError("participant_group_id disagrees with Behavior authority")

    merged["probe_index_in_block"] = merged["_authority_probe_index_in_block"].astype("Int64")
    merged["probe_order_in_block"] = merged["_authority_probe_order_in_block"].astype("Int64")
    merged["probe_event_id"] = merged["_authority_probe_event_id"].astype(str)
    merged["probe_time_ms"] = merged["_authority_probe_time_ms"].astype("int64")
    if "_authority_participant_group_id" in merged.columns:
        authority_group = merged["_authority_participant_group_id"].astype("string").str.strip()
        if "participant_group_id" not in merged.columns:
            merged["participant_group_id"] = authority_group
        else:
            existing_group = merged["participant_group_id"].astype("string").str.strip()
            missing_group = existing_group.isna() | existing_group.eq("")
            merged["participant_group_id"] = existing_group.where(~missing_group, authority_group)
    if "_authority_q1_nominal_4class" in merged.columns:
        merged["q1_nominal_4class"] = merged["_authority_q1_nominal_4class"]
    if "_authority_q2_ordinal_4level" in merged.columns:
        merged["q2_ordinal_4level"] = merged["_authority_q2_ordinal_4level"]

    drop_columns = [
        column
        for column in merged.columns
        if column.startswith("_authority_") or column.startswith("_modality_")
    ]
    return merged.drop(columns=drop_columns)
