from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from attention_pipeline.config import Config
from attention_pipeline.nir_formal_analysis.pupil_blink_measurement import (
    DEFAULT_BLINK_BUFFERS,
    GEOMETRY_SIGNAL,
    RSEG_HARD_SIGNAL,
    RSEG_SOFT_SIGNAL,
    BlinkBuffer,
)


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    raise ValueError(f"unsupported table format for {source}; use CSV or Parquet")


def resolve_source(config: Config, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config.path.parent.parent / path).resolve()


def load_audit_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("measurement-audit config root must be a mapping")
    return payload


def blink_buffers(config: Mapping[str, Any]) -> tuple[BlinkBuffer, ...]:
    rows = config.get("blink_buffers")
    if not rows:
        return DEFAULT_BLINK_BUFFERS
    result: list[BlinkBuffer] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("blink_buffers entries must be mappings")
        result.append(BlinkBuffer(str(row["id"]), float(row["pre_ms"]), float(row["post_ms"])))
    return tuple(result)


def fixed_bin_widths(config: Mapping[str, Any]) -> tuple[float, ...]:
    values = config.get("fixed_bin_width_sec_candidates", [])
    if not isinstance(values, list) or not values:
        raise ValueError("fixed_bin_width_sec_candidates must be a non-empty audit-only list")
    result = tuple(float(x) for x in values)
    if any(x <= 0 for x in result):
        raise ValueError("fixed bin widths must be positive")
    return result


def cleaning_tracks(config: Mapping[str, Any]) -> tuple[str, ...]:
    values = config.get(
        "cleaning_tracks",
        ["original_nir", "rgb_blink_only", "nir_qc_only", "rgb_plus_nir_qc"],
    )
    allowed = {"original_nir", "rgb_blink_only", "nir_qc_only", "rgb_plus_nir_qc"}
    result = tuple(str(x) for x in values)
    invalid = sorted(set(result) - allowed)
    if invalid:
        raise ValueError(f"unsupported cleaning tracks: {invalid}")
    return result


def probe_onset_ms(row: pd.Series) -> float:
    # ``probe_time_ms`` is the authoritative current Behavior formal-v3 field.
    # The remaining names are retained for compatible historical/audit tables.
    for name in ("probe_time_ms", "probe_onset_ms", "window_end_ms", "absolute_onset_time"):
        if name in row.index:
            value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    raise ValueError("probe row missing finite probe onset time")


def probe_block(row: pd.Series) -> int:
    for name in ("block_num", "block"):
        if name in row.index:
            value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return int(value)
    if "block_id" in row.index:
        text = str(row["block_id"]).strip()
        if text.upper().startswith("B"):
            text = text[1:]
        value = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
        if np.isfinite(value):
            return int(value)
    raise ValueError("probe row missing block_num/block/block_id")


def selected_records(records: list[dict[str, Any]], subjects: Iterable[str] | None) -> list[dict[str, Any]]:
    if subjects is None:
        return records
    wanted = {str(x).strip() for x in subjects if str(x).strip()}
    selected = [row for row in records if str(row["session_id"]) in wanted]
    missing = sorted(wanted - {str(row["session_id"]) for row in selected})
    if missing:
        raise ValueError(f"requested sessions absent from NIR source manifest: {missing}")
    return selected


def audit_signals(timepoints: pd.DataFrame, include_soft: bool) -> tuple[str, ...]:
    values = [GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL]
    if include_soft and RSEG_SOFT_SIGNAL in timepoints.columns:
        values.append(RSEG_SOFT_SIGNAL)
    return tuple(values)
