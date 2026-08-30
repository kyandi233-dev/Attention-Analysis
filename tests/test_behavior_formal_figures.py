import pandas as pd

from attention_pipeline.behavior_formal.science_v3_figures import (
    BEHAVIOR_FIGURE_CONTRACT,
    formal_figure_contract_is_english,
)
from attention_pipeline.behavior_formal.science_v3_figures_formal import _participant_first


def test_behavior_formal_figure_contract_is_english():
    assert formal_figure_contract_is_english()
    assert len(BEHAVIOR_FIGURE_CONTRACT) >= 10
    assert "行为图07_候选指标覆盖.png" in BEHAVIOR_FIGURE_CONTRACT
    assert "行为图08_候选指标冗余.png" in BEHAVIOR_FIGURE_CONTRACT
    assert "行为图10_任务时间进程.png" in BEHAVIOR_FIGURE_CONTRACT


def test_probe_descriptive_uncertainty_is_participant_first():
    # p1 contributes three probes and p2 one probe. Participant-first averaging
    # must give (mean(p1)=2 + mean(p2)=10)/2 = 6, not row-weighted 4.
    frame = pd.DataFrame(
        {
            "repeat_participant_id": ["p1", "p1", "p1", "p2"],
            "q1_nominal_4class": [1, 1, 1, 1],
            "go_correct_rt_median_ms": [1.0, 2.0, 3.0, 10.0],
        }
    )
    out = _participant_first(
        frame,
        group_cols=["q1_nominal_4class"],
        value_col="go_correct_rt_median_ms",
    )
    assert len(out) == 1
    assert out.iloc[0]["mean"] == 6.0
    assert out.iloc[0]["participant_group_n"] == 2
