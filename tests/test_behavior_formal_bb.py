from pathlib import Path

import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.behavior_formal import stats as fstat


def _config(tmp_path: Path) -> Config:
    data = {"stats": {"seed": 1, "bootstrap_iterations": 200, "main_metrics": ["commission_rate", "go_rt_median_ms"]}}
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
