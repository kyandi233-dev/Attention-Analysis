"""Data-admission checks for the post-freeze paired Behavior window validation."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from behavior_window_paired_2030 import FEATURES, build_inputs  # noqa: E402


def _row(participant: str, window: int, q1: float) -> dict[str, object]:
    row: dict[str, object] = {
        "participant_group_id": participant,
        "session_id": f"sub-{participant}",
        "block_id": "B1",
        "probe_event_id": f"sub-{participant}|B1|probe|1",
        "q1_nominal_4class": q1,
        "q2_ordinal_4level": 2.0,
        "window_seconds_nominal": window,
    }
    row.update({feature: 0.2 + index + window / 100 for index, feature in enumerate(FEATURES)})
    return row


def _files(tmp_path: Path, *, mismatched_label: bool = False, duplicate: bool = False) -> tuple[Path, Path, Path]:
    source = pd.DataFrame([
        _row("001", 20, 1.0), _row("002", 20, 2.0),
        _row("001", 30, 1.0), _row("002", 30, 3.0 if mismatched_label else 2.0),
        _row("003", 30, 1.0),
    ])
    twenty = source.loc[source.window_seconds_nominal.eq(20)].copy()
    thirty = source.loc[source.window_seconds_nominal.eq(30)].copy()
    if duplicate:
        twenty = pd.concat([twenty, twenty.iloc[[0]]], ignore_index=True)
    twenty["block_id"] = "b1"
    thirty["block_id"] = "b1"
    paths = (tmp_path / "20.csv", tmp_path / "30.csv", tmp_path / "source.csv")
    for path, frame in zip(paths, (twenty, thirty, source), strict=True):
        frame.to_csv(path, index=False)
    return paths


def _build(paths: tuple[Path, Path, Path]):
    return build_inputs(*paths, expected_probes=2, expected_participants=2, expected_sessions=2)


def test_exact_common_sample_excludes_only_extra_30s_probe(tmp_path: Path) -> None:
    frames, audit = _build(_files(tmp_path))
    assert len(frames[20]) == len(frames[30]) == 2
    assert frames[20]["probe_event_id"].equals(frames[30]["probe_event_id"])
    assert audit["window30_excluded_from_pair"] == 1
    assert all(frame["analysis_set_id"].nunique() == 1 for frame in frames.values())


def test_rejects_cross_window_label_change(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paired labels/context differ"):
        _build(_files(tmp_path, mismatched_label=True))


def test_rejects_duplicate_input_probe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate .*probe key"):
        _build(_files(tmp_path, duplicate=True))
