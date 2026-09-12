"""Behavior-authoritative probe-key mapping for cross-modal supervised tables.

The mapping is intentionally explicit: a global probe index is never renamed,
modulo-transformed, or otherwise guessed into a within-block index. Current NIR
Task-D rows can instead be linked to the Behavior authority through the shared
absolute probe time together with session and block identity.
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
DEFAULT_MODALITY_TIME_CANDIDATES = (
    "probe_time_ms",
    "probe_onset_ms",
    "window_end_ms",
    "absolute_onset_time",
)


class ProbeKeyAuthorityError(ValueError):
    """Raised when probe identity cannot be mapped to Behavior without guessing."""


def normalize_block(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower().str.replace("-", "", regex=False)
    text = text.str.replace("block", "", regex=False)
    return text.where(text.str.startswith("b"), "b" + text)


def _integral_ms(values: pd.Series, *, context: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ProbeKeyAuthorityError(f"{context} contains missing or non-finite probe times")
    rounded = np.rint(numeric.to_numpy(dtype=float))
    if not np.allclose(numeric.to_numpy(dtype=float), rounded, rtol=0.0, atol=1e-6):
        raise ProbeKeyAuthorityError(f"{context} must use integral Unix milliseconds")
    return pd.Series(rounded.astype("int64"), index=values.index, dtype="int64")


def _prepare_behavior_authority(behavior: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(BEHAVIOR_REQUIRED_COLUMNS) - set(behavior.columns))
    if missing:
        raise ProbeKeyAuthorityError(f"Behavior probe authority missing columns: {missing}")
    authority = behavior.copy()
    if authority.empty:
        raise ProbeKeyAuthorityError("Behavior probe authority is empty")
    authority["session_id"] = authority["session_id"].astype(str).str.strip()
    authority["block_id"] = normalize_block(authority["block_id"])
    authority["probe_index_in_block"] = pd.to_numeric(
        authority["probe_order_in_block"], errors="coerce"
    ).astype("Int64")
    if authority["probe_index_in_block"].isna().any() or (authority["probe_index_in_block"] < 1).any():
        raise ProbeKeyAuthorityError("Behavior probe_order_in_block must be positive integers")
    authority["_authority_probe_time_ms"] = _integral_ms(
        authority["probe_time_ms"], context="Behavior probe_time_ms"
    )
    if authority["probe_event_id"].isna().any() or authority["probe_event_id"].astype(str).str.strip().eq("").any():
        raise ProbeKeyAuthorityError("Behavior probe_event_id contains missing or blank values")
    if authority.duplicated(["session_id", "block_id", "_authority_probe_time_ms"]).any():
        raise ProbeKeyAuthorityError("Behavior authority has duplicate session/block/probe_time_ms keys")
    if authority.duplicated(list(CANONICAL_KEY_COLUMNS)).any():
        raise ProbeKeyAuthorityError("Behavior authority has duplicate canonical probe keys")
    return authority


def _first_time_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def map_probe_keys_to_behavior(
    modality_frame: pd.DataFrame,
    behavior_probe_authority: pd.DataFrame,
    *,
    time_candidates: Sequence[str] = DEFAULT_MODALITY_TIME_CANDIDATES,
) -> pd.DataFrame:
    """Attach canonical probe identity using the Behavior probe table.

    Preferred mapping key is ``session_id + block_id + absolute probe time``.
    If no time field exists, an already-existing within-block probe index may be
    verified against Behavior. ``probe_index_global`` alone is deliberately
    insufficient because its semantics differ from a within-block index.
    """
    if modality_frame.empty:
        return modality_frame.copy()
    if "session_id" not in modality_frame.columns:
        raise ProbeKeyAuthorityError("modality probe table missing session_id")
    block_source = "block_id" if "block_id" in modality_frame.columns else "block_num" if "block_num" in modality_frame.columns else None
    if block_source is None:
        raise ProbeKeyAuthorityError("modality probe table missing block_id/block_num")

    authority = _prepare_behavior_authority(behavior_probe_authority)
    frame = modality_frame.copy()
    frame["session_id"] = frame["session_id"].astype(str).str.strip()
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

    time_column = _first_time_column(frame, time_candidates)
    if time_column is not None:
        frame["_modality_probe_time_ms"] = _integral_ms(
            frame[time_column], context=f"modality {time_column}"
        )
        merged = frame.merge(
            authority_sub,
            left_on=["session_id", "block_id", "_modality_probe_time_ms"],
            right_on=["session_id", "block_id", "_authority_probe_time_ms"],
            how="left",
            validate="many_to_one",
        )
    elif "probe_index_in_block" in frame.columns:
        existing_index = pd.to_numeric(frame["probe_index_in_block"], errors="coerce").astype("Int64")
        if existing_index.isna().any():
            raise ProbeKeyAuthorityError("existing probe_index_in_block contains missing/non-numeric values")
        frame["_modality_probe_index_in_block"] = existing_index
        merged = frame.merge(
            authority_sub,
            left_on=["session_id", "block_id", "_modality_probe_index_in_block"],
            right_on=["session_id", "block_id", "_authority_probe_index_in_block"],
            how="left",
            validate="many_to_one",
        )
    else:
        raise ProbeKeyAuthorityError(
            "cannot map probe identity: need an absolute probe-time field or an existing "
            "probe_index_in_block; probe_index_global is never treated as a within-block index"
        )

    if merged["_authority_probe_index_in_block"].isna().any():
        unmatched = int(merged["_authority_probe_index_in_block"].isna().sum())
        raise ProbeKeyAuthorityError(
            f"{unmatched} modality probe rows could not be mapped to Behavior authority"
        )

    if "probe_index_in_block" in frame.columns:
        existing = pd.to_numeric(merged["probe_index_in_block"], errors="coerce").astype("Int64")
        mismatch = existing.notna() & existing.ne(merged["_authority_probe_index_in_block"].astype("Int64"))
        if mismatch.any():
            raise ProbeKeyAuthorityError("existing probe_index_in_block disagrees with Behavior authority")

    if "probe_event_id" in frame.columns:
        existing_event = merged["probe_event_id"].astype("string")
        authority_event = merged["_authority_probe_event_id"].astype("string")
        mismatch = existing_event.notna() & existing_event.str.strip().ne("") & existing_event.ne(authority_event)
        if mismatch.any():
            raise ProbeKeyAuthorityError("existing probe_event_id disagrees with Behavior authority")

    if "participant_group_id" in frame.columns and "_authority_participant_group_id" in merged.columns:
        existing_group = merged["participant_group_id"].astype("string").str.strip()
        authority_group = merged["_authority_participant_group_id"].astype("string").str.strip()
        mismatch = existing_group.notna() & existing_group.ne("") & authority_group.notna() & existing_group.ne(authority_group)
        if mismatch.any():
            raise ProbeKeyAuthorityError("participant_group_id disagrees with Behavior authority")

    merged["probe_index_in_block"] = merged["_authority_probe_index_in_block"].astype("Int64")
    merged["probe_order_in_block"] = merged["_authority_probe_order_in_block"].astype("Int64")
    merged["probe_event_id"] = merged["_authority_probe_event_id"].astype(str)
    merged["probe_time_ms"] = merged["_authority_probe_time_ms"].astype("int64")
    if "_authority_participant_group_id" in merged.columns:
        if "participant_group_id" not in merged.columns:
            merged["participant_group_id"] = merged["_authority_participant_group_id"]
        else:
            merged["participant_group_id"] = merged["participant_group_id"].where(
                merged["participant_group_id"].notna(), merged["_authority_participant_group_id"]
            )
    if "_authority_q1_nominal_4class" in merged.columns:
        merged["q1_nominal_4class"] = merged["_authority_q1_nominal_4class"]
    if "_authority_q2_ordinal_4level" in merged.columns:
        merged["q2_ordinal_4level"] = merged["_authority_q2_ordinal_4level"]

    drop_columns = [column for column in merged.columns if column.startswith("_authority_") or column.startswith("_modality_")]
    return merged.drop(columns=drop_columns)
