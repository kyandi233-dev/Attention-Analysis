from pathlib import Path

import pandas as pd
import yaml

from attention_pipeline.formal_analysis.merge import UNIT_KEYS, merge_modalities


def test_formal_config_fusion_units_match_merge_contract() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs" / "formal_multimodal_v2.yaml").read_text(encoding="utf-8"))
    configured = set(config["fusion"]["units"])
    assert configured == set(UNIT_KEYS)
    assert config["fusion"]["key_contract"] == UNIT_KEYS


def test_cycle_level_merge_is_one_to_one_and_key_strict() -> None:
    keys = {
        "repeat_participant_id": ["g1", "g2"],
        "session_id": ["sub-031", "sub-033"],
        "block_id": [1, 1],
        "cycle_bin": [2, 3],
    }
    behavior = pd.DataFrame({**keys, "rt_cv": [0.10, 0.20]})
    nir = pd.DataFrame({**keys, "pupil": [30.0, 31.0]})
    merged, audit = merge_modalities({"behavior": behavior, "nir": nir}, unit="cycle")
    assert len(merged) == 2
    assert "behavior__rt_cv" in merged
    assert "nir__pupil" in merged
    assert set(audit["unit"]) == {"cycle"}


def test_participant_group_merge_uses_repeat_identity_only() -> None:
    behavior = pd.DataFrame({"repeat_participant_id": ["g1", "g2"], "score": [1.0, 2.0]})
    rgb = pd.DataFrame({"repeat_participant_id": ["g1", "g2"], "motion": [0.1, 0.2]})
    merged, _ = merge_modalities({"behavior": behavior, "rgb": rgb}, unit="participant_group")
    assert merged["repeat_participant_id"].tolist() == ["g1", "g2"]
    assert "behavior__score" in merged
    assert "rgb__motion" in merged
