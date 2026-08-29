from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

UNIT_KEYS: dict[str, list[str]] = {
    "trial": ["repeat_participant_id", "session_id", "block_id", "trial_id"],
    "probe": ["repeat_participant_id", "session_id", "block_id", "probe_id", "window_name"],
    "block": ["repeat_participant_id", "session_id", "block_id"],
    "session": ["repeat_participant_id", "session_id"],
}


def keys_for_unit(unit: str) -> list[str]:
    if unit not in UNIT_KEYS:
        raise ValueError(f"未知分析单位: {unit!r}; 可用: {sorted(UNIT_KEYS)}")
    return list(UNIT_KEYS[unit])


def validate_merge_ready(
    table: pd.DataFrame,
    *,
    unit: str,
    modality: str,
    require_group: bool = True,
) -> dict[str, object]:
    keys = keys_for_unit(unit)
    required = set(keys)
    if not require_group:
        required.discard("repeat_participant_id")
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{modality} {unit} 表缺少合并键: {sorted(missing)}")

    active_keys = [key for key in keys if key in table.columns]
    if table[active_keys].isna().any().any():
        bad = table[active_keys].isna().any(axis=1)
        raise ValueError(f"{modality} {unit} 表有 {int(bad.sum())} 行合并键缺失，禁止静默填补")
    if table.duplicated(active_keys).any():
        dup = int(table.duplicated(active_keys, keep=False).sum())
        raise ValueError(f"{modality} {unit} 表主键重复: {dup} rows")

    return {
        "modality": modality,
        "unit": unit,
        "rows": int(len(table)),
        "sessions": int(table["session_id"].nunique()) if "session_id" in table else None,
        "groups": int(table["repeat_participant_id"].nunique()) if "repeat_participant_id" in table else None,
        "keys": active_keys,
    }


def merge_modalities(
    tables: Mapping[str, pd.DataFrame],
    *,
    unit: str,
    how: str = "inner",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge standardized row-level modality tables; never average model results."""
    if not tables:
        raise ValueError("至少需要一个 modality table")
    if how not in {"inner", "outer"}:
        raise ValueError("how 仅支持 inner 或 outer")

    keys = keys_for_unit(unit)
    audits = []
    prepared: list[tuple[str, pd.DataFrame]] = []
    for modality, table in tables.items():
        audits.append(validate_merge_ready(table, unit=unit, modality=modality))
        nonkeys = [column for column in table.columns if column not in keys]
        renamed = table.rename(columns={column: f"{modality}__{column}" for column in nonkeys})
        prepared.append((modality, renamed))

    _, merged = prepared[0]
    merged = merged.copy()
    for _, table in prepared[1:]:
        merged = merged.merge(table, on=keys, how=how, validate="one_to_one")
    return merged, pd.DataFrame(audits)
