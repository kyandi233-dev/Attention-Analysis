from attention_pipeline.contracts import PROBE_LABELS
from attention_pipeline.protocol import validate_protocol


def test_formal_protocol_matches_authoritative_program(config):
    result = validate_protocol(config)
    assert result["ok"], result["checks"]
    assert result["sequence_rows"] == {"B1": 432, "B2": 432, "B3": 432}


def test_historical_protocol_unchanged_for_behavior(config):
    historical = config.section("protocol")
    assert historical["block_order"] == ["A", "B", "C", "C", "B", "A"]
    assert historical["probe_after_trials"] == [30, 82, 137, 191]
    assert set(historical["probe_labels"]) == {1, 2, 3, 4}


def test_formal_protocol_matches_current_bbb_dual_probe_program(config):
    formal = config.section("formal_protocol")
    assert formal["block_order"] == ["B", "B", "B"]
    assert formal["trials_per_cycle"] == 18 and formal["cycles_per_block"] == 24
    assert formal["nominal_trial_ms"] == 1150
    assert formal["schedule_version"] == "sched-v1.0-4cat-vig-20260814"
    assert formal["probe_positions"] == {
        1: [31, 73, 117, 162, 204, 251, 293, 338, 380, 424],
        2: [32, 75, 118, 162, 205, 247, 295, 339, 381, 424],
        3: [32, 76, 119, 161, 204, 248, 292, 335, 382, 425],
    }


def test_probe_states_are_nominal_not_numeric_score(config):
    assert PROBE_LABELS == {
        1: "完全专注",
        2: "关注实验但未聚焦任务",
        3: "任务无关思维",
        4: "大脑空白",
    }
    assert config.section("stages")["attention_score"] is False


def test_current_nir_stage_flags_match_completed_history_and_production_stop(config):
    stages = config.section("stages")
    assert stages["nir_build_review_preview"] is True
    assert stages["nir_build_review_full"] is False
    assert stages["nir_benchmark"] is True and stages["nir_evaluate"] is True
    assert stages["nir_sequence_build"] is True and stages["nir_sequence_detect"] is True
    assert stages["nir_extract"] is False and stages["nir_report"] is False



