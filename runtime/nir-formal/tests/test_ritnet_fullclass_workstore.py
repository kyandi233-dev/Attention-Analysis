from __future__ import annotations

import pytest

from ritnet_fullclass_workstore import FullClassWorkStore


def source_row(frame, eye):
    return {"phase": "block1", "phase_segment": 1, "frame_idx": frame, "eye": eye}


def payload(frame, eye, value):
    return {**source_row(frame, eye), "hard_pupil_fraction": value}


def test_workstore_commits_and_reopens_exact_prefix(tmp_path):
    path = tmp_path / "work.sqlite"
    identity = {"subject": "sub-031", "source": "abc"}
    source = [source_row(10, "frame_left"), source_row(10, "frame_right"), source_row(11, "frame_left")]

    with FullClassWorkStore(path, identity=identity) as store:
        store.append_rows([(0, payload(10, "frame_left", 0.1)), (1, payload(10, "frame_right", 0.2))])
        assert store.validate_prefix(source) == 2

    with FullClassWorkStore(path, identity=identity) as reopened:
        assert reopened.validate_prefix(source) == 2
        rows = list(reopened.iter_rows())
        assert [row["hard_pupil_fraction"] for row in rows] == [0.1, 0.2]


def test_workstore_rejects_different_resume_identity(tmp_path):
    path = tmp_path / "work.sqlite"
    with FullClassWorkStore(path, identity={"subject": "sub-031"}):
        pass
    with pytest.raises(RuntimeError, match="identity digest"):
        FullClassWorkStore(path, identity={"subject": "sub-032"})


def test_workstore_prefix_rejects_source_key_change(tmp_path):
    path = tmp_path / "work.sqlite"
    identity = {"subject": "sub-031"}
    with FullClassWorkStore(path, identity=identity) as store:
        store.append_rows([(0, payload(10, "frame_left", 0.1))])
        with pytest.raises(RuntimeError, match="key mismatch"):
            store.validate_prefix([source_row(10, "frame_right")])


def test_workstore_unique_eye_key_prevents_duplicate_frame_eye(tmp_path):
    path = tmp_path / "work.sqlite"
    with FullClassWorkStore(path, identity={"subject": "sub-031"}) as store:
        store.append_rows([(0, payload(10, "frame_left", 0.1))])
        with pytest.raises(Exception):
            store.append_rows([(1, payload(10, "frame_left", 0.2))])
