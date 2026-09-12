import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.probe_key_authority import (
    ProbeKeyAuthorityError,
    map_probe_keys_to_behavior,
)


def _behavior_authority() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s1"],
            "participant_group_id": ["p1"] * 4,
            "block_id": ["B1", "B1", "B2", "B2"],
            "probe_order_in_block": [1, 2, 1, 2],
            "probe_event_id": ["e1", "e2", "e3", "e4"],
            "probe_time_ms": [1000, 2000, 3000, 4000],
            "q1_nominal_4class": [1, 2, 3, 1],
            "q2_ordinal_4level": [4, 3, 2, 4],
        }
    )


def test_current_task_d_global_index_maps_by_authoritative_time_not_by_renaming():
    nir = pd.DataFrame(
        {
            "participant_group_id": ["p1", "p1", "p1", "p1"],
            "session_id": ["s1"] * 4,
            "block_num": [1, 1, 2, 2],
            "probe_index_global": [1, 2, 3, 4],
            "probe_onset_ms": [1000, 2000, 3000, 4000],
            "pupil_mean": [3.0, 3.1, 3.2, 3.3],
        }
    )
    mapped = map_probe_keys_to_behavior(nir, _behavior_authority())

    assert mapped["block_id"].tolist() == ["b1", "b1", "b2", "b2"]
    assert mapped["probe_index_in_block"].tolist() == [1, 2, 1, 2]
    assert mapped["probe_event_id"].tolist() == ["e1", "e2", "e3", "e4"]
    assert mapped["probe_time_ms"].tolist() == [1000, 2000, 3000, 4000]
    assert mapped["q1_nominal_4class"].tolist() == [1, 2, 3, 1]
    assert mapped["probe_index_global"].tolist() == [1, 2, 3, 4]


def test_global_probe_index_alone_is_never_treated_as_within_block_index():
    nir = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_num": [2],
            "probe_index_global": [3],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="probe_index_global"):
        map_probe_keys_to_behavior(nir, _behavior_authority())


def test_formal_trial_absolute_onset_is_not_accepted_as_probe_time():
    # formaltest/sart_task.py defines absolute_onset_time as SART stimulus onset,
    # whereas probe_onset_time is the actual Q1 probe onset. The mapper must not
    # guess that a trial timestamp is a probe timestamp.
    modality = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_num": [1],
            "probe_index_global": [1],
            "absolute_onset_time": [1000],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="trial absolute_onset_time"):
        map_probe_keys_to_behavior(modality, _behavior_authority())


def test_actual_formal_probe_onset_time_is_accepted():
    modality = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_num": [1],
            "probe_onset_time": [1000],
        }
    )
    mapped = map_probe_keys_to_behavior(modality, _behavior_authority())
    assert mapped.loc[0, "probe_event_id"] == "e1"
    assert mapped.loc[0, "probe_index_in_block"] == 1


def test_multiple_probe_time_aliases_must_agree():
    modality = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_num": [1],
            "probe_onset_ms": [1000],
            "window_end_ms": [1001],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="probe-time fields disagree"):
        map_probe_keys_to_behavior(modality, _behavior_authority())


def test_existing_within_block_index_is_verified_against_authority():
    nir = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_id": ["B2"],
            "probe_index_in_block": [2],
            "probe_onset_ms": [3000],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="probe_index_in_block disagrees"):
        map_probe_keys_to_behavior(nir, _behavior_authority())


def test_participant_identity_mismatch_fails_closed():
    nir = pd.DataFrame(
        {
            "participant_group_id": ["wrong"],
            "session_id": ["s1"],
            "block_num": [1],
            "probe_index_global": [1],
            "probe_onset_ms": [1000],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="participant_group_id disagrees"):
        map_probe_keys_to_behavior(nir, _behavior_authority())


def test_blank_participant_identity_is_filled_from_behavior_authority():
    modality = pd.DataFrame(
        {
            "participant_group_id": [""],
            "session_id": ["s1"],
            "block_num": [1],
            "probe_onset_ms": [1000],
        }
    )
    mapped = map_probe_keys_to_behavior(modality, _behavior_authority())
    assert mapped.loc[0, "participant_group_id"] == "p1"


def test_existing_q1_or_q2_disagreement_fails_closed():
    q1_bad = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_num": [1],
            "probe_onset_ms": [1000],
            "q1_nominal_4class": [4],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="q1_nominal_4class disagrees"):
        map_probe_keys_to_behavior(q1_bad, _behavior_authority())

    q2_bad = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_num": [1],
            "probe_onset_ms": [1000],
            "q2_ordinal_4level": [1],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="q2_ordinal_4level disagrees"):
        map_probe_keys_to_behavior(q2_bad, _behavior_authority())


def test_unmatched_time_fails_instead_of_guessing_from_global_order():
    nir = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_num": [1],
            "probe_index_global": [1],
            "probe_onset_ms": [1500],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="could not be mapped"):
        map_probe_keys_to_behavior(nir, _behavior_authority())


def test_illegal_block_or_duplicate_behavior_event_fails_closed():
    illegal = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_num": [3],
            "probe_onset_ms": [1000],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="illegal formal block"):
        map_probe_keys_to_behavior(illegal, _behavior_authority())

    authority = _behavior_authority()
    authority.loc[1, "probe_event_id"] = "e1"
    modality = pd.DataFrame(
        {
            "session_id": ["s1"],
            "block_num": [1],
            "probe_onset_ms": [1000],
        }
    )
    with pytest.raises(ProbeKeyAuthorityError, match="duplicate probe_event_id"):
        map_probe_keys_to_behavior(modality, authority)
