from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def subject_paths(raw_root: Path, subject: str) -> dict[str, Path]:
    root = raw_root / f"{subject}_"
    return {
        "root": root,
        "beh_dir": root / "beh",
        "master_timeline": root / "beh" / "master_timeline.csv",
        "nir_video": root / "nir" / f"{subject}_nir.avi",
        "nir_timestamps": root / "nir" / f"{subject}_nir_timestamps.csv",
        "rgb_video": root / "rgb" / f"{subject}_rgb.avi",
        "rgb_timestamps": root / "rgb" / f"{subject}_rgb_timestamps.csv",
    }


def load_timestamps(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 2:
        raise ValueError(f"时间戳文件少于两列: {path}")
    columns = ["capture_frame_idx", "unix_ms", "status"] + [f"extra_{i}" for i in range(raw.shape[1] - 3)]
    raw.columns = columns[: raw.shape[1]]
    if "status" not in raw:
        raw["status"] = ""
    raw["capture_frame_idx"] = pd.to_numeric(raw["capture_frame_idx"], errors="coerce")
    raw["unix_ms"] = pd.to_numeric(raw["unix_ms"], errors="coerce")
    raw["status"] = raw["status"].fillna("").astype(str).str.strip().str.lower()
    raw["is_dropped"] = raw["status"].eq("dropped")
    raw["avi_frame_idx"] = np.nan
    normal = raw.index[~raw["is_dropped"] & raw["unix_ms"].notna()]
    raw.loc[normal, "avi_frame_idx"] = np.arange(len(normal), dtype=int)
    return raw


def block_windows(master_timeline: Path) -> list[dict]:
    """Parse block_start/block_stop windows for both historical and final protocols.

    The historical pre-experiment timeline can contain six A/B/C blocks, while
    the final FocusWave v3.1.3 formal experiment contains two B blocks.  The
    parser therefore validates the timeline itself instead of hard-coding a
    block count.
    """
    timeline = pd.read_csv(master_timeline)
    timeline["unix_ms"] = pd.to_numeric(timeline["unix_ms"], errors="coerce")
    starts = timeline.loc[timeline["event"].eq("block_start")].reset_index(drop=True)
    stops = timeline.loc[timeline["event"].eq("block_stop")].reset_index(drop=True)
    if starts.empty or len(starts) != len(stops):
        raise ValueError(
            f"block 起止数量必须非零且一致，实际 {len(starts)}/{len(stops)}"
        )

    result = []
    for index in range(len(starts)):
        detail = str(starts.loc[index, "detail"])
        match = re.fullmatch(r"Block(\d+)_([ABC])", detail)
        if not match:
            raise ValueError(f"无法解释 block detail: {detail}")
        start_ms = float(starts.loc[index, "unix_ms"])
        end_ms = float(stops.loc[index, "unix_ms"])
        if not np.isfinite(start_ms) or not np.isfinite(end_ms) or end_ms <= start_ms:
            raise ValueError(f"{detail} block 起止时间无效: {start_ms}/{end_ms}")
        result.append({
            "block_num": int(match.group(1)),
            "condition": match.group(2),
            "start_ms": start_ms,
            "end_ms": end_ms,
        })
    return result


def nearest_written_frame(timestamp_rows: pd.DataFrame, target_ms: float) -> dict:
    valid = timestamp_rows.loc[~timestamp_rows["is_dropped"] & timestamp_rows["unix_ms"].notna()].copy()
    if valid.empty:
        raise ValueError("没有可映射到 AVI 的时间戳")
    position = int(np.argmin(np.abs(valid["unix_ms"].to_numpy(dtype=float) - target_ms)))
    row = valid.iloc[position]
    return {
        "capture_frame_idx": int(row["capture_frame_idx"]),
        "avi_frame_idx": int(row["avi_frame_idx"]),
        "unix_ms": float(row["unix_ms"]),
        "target_error_ms": float(row["unix_ms"] - target_ms),
    }
