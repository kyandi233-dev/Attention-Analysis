from __future__ import annotations

import numpy as np


def rolling_perclos(ear, timestamps_ms, threshold: float, window_sec: float = 60.0):
    values = np.asarray(ear, dtype=float)
    times = np.asarray(timestamps_ms, dtype=float)
    result = np.full(len(values), np.nan)
    for index, now in enumerate(times):
        start = int(np.searchsorted(times, now - window_sec * 1000, side="left"))
        valid = values[start : index + 1]
        valid = valid[np.isfinite(valid)]
        if len(valid):
            result[index] = float(np.mean(valid <= threshold))
    return result

