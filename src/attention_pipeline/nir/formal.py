"""正式实验NIR路径与行为/NIR时间轴对齐。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_subject(subject: str) -> tuple[str, str]:
    """返回(目录名, 文件stem)，同时接受sub-011和sub-011_。"""
    value = str(subject).strip()
    stem = value.rstrip("_")
    return f"{stem}_", stem


def formal_subject_paths(root: str | Path, subject: str) -> dict[str, Path]:
    directory, stem = normalize_subject(subject)
    base = Path(root) / directory
    return {
        "base": base,
        "nir_video": base / "nir" / f"{stem}_nir.avi",
        "nir_timestamps": base / "nir" / f"{stem}_nir_timestamps.csv",
        "master_timeline": base / "beh" / "master_timeline.csv",
    }


def load_nir_timestamps(path: str | Path) -> pd.DataFrame:
    """读取无表头frame_idx,unix_ms,status；保证有限、严格递增且帧号唯一。"""
    frame = pd.read_csv(path, header=None, names=["frame_idx", "unix_ms", "status"])
    frame["frame_idx"] = pd.to_numeric(frame["frame_idx"], errors="coerce")
    frame["unix_ms"] = pd.to_numeric(frame["unix_ms"], errors="coerce")
    if frame[["frame_idx", "unix_ms"]].isna().any().any():
        raise ValueError(f"invalid NIR timestamp rows: {path}")
    frame[["frame_idx", "unix_ms"]] = frame[["frame_idx", "unix_ms"]].astype("int64")
    if frame["frame_idx"].duplicated().any() or not frame["frame_idx"].is_monotonic_increasing:
        raise ValueError("NIR frame_idx must be unique and monotonic")
    if not frame["unix_ms"].is_monotonic_increasing or (np.diff(frame["unix_ms"].to_numpy()) <= 0).any():
        raise ValueError("NIR unix_ms must be strictly increasing")
    return frame


def block_window(master_timeline: str | Path, block: int) -> tuple[int, int]:
    timeline = pd.read_csv(master_timeline)
    label = f"Block{int(block)}_"
    starts = timeline[(timeline["event"] == "block_start") & timeline["detail"].astype(str).str.startswith(label)]
    stops = timeline[(timeline["event"] == "block_stop") & timeline["detail"].astype(str).str.startswith(label)]
    if len(starts) != 1 or len(stops) != 1:
        raise ValueError(f"expected one start/stop for Block{block}; got {len(starts)}/{len(stops)}")
    start_ms = int(starts.iloc[0]["unix_ms"])
    stop_ms = int(stops.iloc[0]["unix_ms"])
    if stop_ms <= start_ms:
        raise ValueError(f"invalid Block{block} window")
    return start_ms, stop_ms


def locate_continuous_window(
    timestamps: pd.DataFrame,
    start_unix_ms: int,
    duration_ms: int,
    stop_unix_ms: int | None = None,
) -> dict:
    """定位首个>=行为起点的NIR帧，并取不越过duration/Block终点的连续窗口。"""
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    starts = timestamps.index[timestamps["unix_ms"] >= int(start_unix_ms)]
    if len(starts) == 0:
        raise ValueError("no NIR frame at or after requested start")
    start_pos = int(starts[0])
    start_row = timestamps.loc[start_pos]
    target_end = int(start_row["unix_ms"]) + int(duration_ms)
    if stop_unix_ms is not None:
        target_end = min(target_end, int(stop_unix_ms))
    eligible = timestamps.index[(timestamps.index >= start_pos) & (timestamps["unix_ms"] < target_end)]
    if len(eligible) == 0:
        raise ValueError("empty NIR window")
    end_pos = int(eligible[-1])
    selected = timestamps.loc[start_pos:end_pos]
    frame_ids = selected["frame_idx"].to_numpy()
    if len(frame_ids) > 1 and not np.all(np.diff(frame_ids) == 1):
        raise ValueError("requested window crosses a real frame-index gap")
    return {
        "start_frame_idx": int(selected.iloc[0]["frame_idx"]),
        "end_frame_idx": int(selected.iloc[-1]["frame_idx"]),
        "start_unix_ms": int(selected.iloc[0]["unix_ms"]),
        "end_unix_ms": int(selected.iloc[-1]["unix_ms"]),
        "n_frames": int(len(selected)),
        "start_offset_ms": int(selected.iloc[0]["unix_ms"] - start_unix_ms),
    }


def locate_block_segment(root: str | Path, subject: str, block: int, duration_sec: float) -> tuple[dict, pd.DataFrame, dict[str, Path]]:
    paths = formal_subject_paths(root, subject)
    for name in ("nir_video", "nir_timestamps", "master_timeline"):
        if not paths[name].exists():
            raise FileNotFoundError(paths[name])
    timestamps = load_nir_timestamps(paths["nir_timestamps"])
    start_ms, stop_ms = block_window(paths["master_timeline"], block)
    window = locate_continuous_window(timestamps, start_ms, round(float(duration_sec) * 1000), stop_ms)
    return window, timestamps, paths
