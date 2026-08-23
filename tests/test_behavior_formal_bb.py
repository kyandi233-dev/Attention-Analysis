from pathlib import Path

import pandas as pd
import pytest

from attention_pipeline.config import Config
from attention_pipeline.behavior_formal import stats as fstat
from attention_pipeline.behavior_formal.extract import subject_behavior_dir


def _config(tmp_path: Path, roots: list[Path] | None = None) -> Config:
    data = {
        "stats": {
            "seed": 1,
            "bootstrap_iterations": 200,
            "main_metrics": ["commission_rate", "go_rt_median_ms"],
        }
    }
    if roots is not None:
        data["data"] = {
            "roots": [str(root) for root in roots],
            "behavior_dir": "beh",
        }
    return Config(path=tmp_path / "configs" / "behavior_formal.yaml", data=data, digest="test")


def test_paired_block_effects_uses_only_b1_b2(tmp_path):
    config = _config(tmp_path)
    rows = []
    for subject, offset in [("sub-031", 0.0), ("sub-032", 0.01), ("sub-033", -0.01)]:
        rows.extend([
            {"subject": subject, "block_num": 1, "commission_rate": 0.10 + offset, "go_rt_median_ms": 400 + offset},
            {"subject": subject, "block_num": 2, "commission_rate": 0.20 + offset, "go_rt_median_ms": 420 + offset},
        ])
    result = fstat.paired_block_effects(config, pd.DataFrame(rows))
    assert set(result["metric"]) == {"commission_rate", "go_rt_median_ms"}
    assert (result["B2_minus_B1_mean"] > 0).all()
    assert "B3_mean" not in result.columns


def test_behavior_root_discovery_ignores_missing_candidates(tmp_path):
    missing_a = tmp_path / "missing-a"
    active = tmp_path / "active"
    missing_b = tmp_path / "missing-b"
    beh = active / "sub-031_" / "beh"
    beh.mkdir(parents=True)

    config = _config(tmp_path, [missing_a, active, missing_b])

    assert subject_behavior_dir(config, "sub-031") == beh.resolve()


def test_behavior_root_discovery_rejects_duplicate_subject_data(tmp_path):
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    (root_a / "sub-031_" / "beh").mkdir(parents=True)
    (root_b / "sub-031" / "beh").mkdir(parents=True)

    config = _config(tmp_path, [root_a, root_b])

    with pytest.raises(RuntimeError, match="duplicate behavior directories"):
        subject_behavior_dir(config, "sub-031")
