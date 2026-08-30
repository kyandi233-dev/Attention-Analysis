from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd
import pytest
import yaml

from attention_pipeline.formal_analysis.join_keys import normalize_known_join_dtypes
from attention_pipeline.formal_analysis.merge import merge_modalities, validate_merge_ready
from attention_pipeline.formal_analysis.nir_adapter import adapt_nir_frame_table
from attention_pipeline.formal_analysis.provenance import collect_runtime_provenance
from attention_pipeline.nir_pupil_only import attach_behavior_and_visual


def _init_git_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Synthetic Test"], check=True)
    (path / "tracked.txt").write_text("synthetic\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "synthetic fixture"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_runtime_provenance_resolves_real_clean_code_and_evidence_heads(tmp_path: Path) -> None:
    code_repo = tmp_path / "code"
    evidence_repo = tmp_path / "evidence"
    code_sha = _init_git_repo(code_repo)
    evidence_sha = _init_git_repo(evidence_repo)

    payload = collect_runtime_provenance(
        code_repo=code_repo,
        evidence_repo=evidence_repo,
        evidence_repository="example/evidence",
    )

    assert payload["code"]["commit_sha"] == code_sha
    assert payload["evidence"]["commit_sha"] == evidence_sha
    assert payload["code"]["worktree_clean"] is True
    assert payload["evidence"]["worktree_clean"] is True
    assert payload["policy"]["fixed_commit_fallback_allowed"] is False
    assert payload["policy"]["unresolved_commit_behavior"] == "fail_closed"


def test_runtime_provenance_fails_closed_for_missing_or_dirty_checkout(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    dirty = tmp_path / "dirty"
    _init_git_repo(clean)
    _init_git_repo(dirty)
    (dirty / "tracked.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="未提交改动"):
        collect_runtime_provenance(code_repo=dirty, evidence_repo=clean)
    with pytest.raises(RuntimeError, match="不存在"):
        collect_runtime_provenance(code_repo=clean, evidence_repo=tmp_path / "missing")


def test_no_fixed_evidence_commit_fallback_is_committed() -> None:
    root = Path(__file__).parents[1]
    config_text = (root / "configs" / "formal_multimodal_v2.yaml").read_text(encoding="utf-8")
    script_text = (root / "scripts" / "formal_multimodal_analysis.py").read_text(encoding="utf-8")
    assert "171b081" not in config_text
    assert "171b081" not in script_text
    config = yaml.safe_load(config_text)
    assert "evidence_commit" not in config["pipeline"]
    assert config["provenance"]["fixed_commit_fallback_allowed"] is False
    assert config["provenance"]["unresolved_commit_behavior"] == "fail_closed"


def test_join_key_normalizer_handles_mixed_dtype_and_explicit_seconds() -> None:
    frame = pd.DataFrame(
        {
            "phase_segment": [1, 2],
            "frame_idx": ["1", 2.0],
            "unix_ms": ["1500", 2500.0],
            "absolute_onset_time": [1.5, 2.5],
        }
    )
    out = normalize_known_join_dtypes(
        frame,
        required_non_null=["phase_segment", "frame_idx", "unix_ms", "absolute_onset_time"],
        time_units={"absolute_onset_time": "s"},
    )
    assert out["phase_segment"].tolist() == ["1", "2"]
    assert out["frame_idx"].tolist() == [1, 2]
    assert out["unix_ms"].tolist() == [1500, 2500]
    assert out["absolute_onset_time"].tolist() == [1500, 2500]
    assert str(out["frame_idx"].dtype) == "int64"
    assert str(out["unix_ms"].dtype) == "int64"


def test_time_key_timezone_contract_normalizes_to_utc_epoch_ms_and_rejects_naive() -> None:
    aware = pd.DataFrame(
        {
            "unix_ms": pd.to_datetime(["2026-08-29T06:00:00Z"]),
            "absolute_onset_time": ["2026-08-29T14:00:00+08:00"],
        }
    )
    out = normalize_known_join_dtypes(
        aware,
        required_non_null=["unix_ms", "absolute_onset_time"],
    )
    assert out.loc[0, "unix_ms"] == out.loc[0, "absolute_onset_time"]

    naive = pd.DataFrame({"unix_ms": ["2026-08-29 06:00:00"]})
    with pytest.raises(ValueError, match="没有时区"):
        normalize_known_join_dtypes(naive, required_non_null=["unix_ms"])
    explicit = normalize_known_join_dtypes(
        naive,
        required_non_null=["unix_ms"],
        naive_timezone="UTC",
    )
    assert int(explicit.loc[0, "unix_ms"]) == int(out.loc[0, "unix_ms"])


def test_time_key_mixed_numeric_and_datetime_fails_closed() -> None:
    frame = pd.DataFrame(
        {"absolute_onset_time": ["1000", "2026-08-29T06:00:01Z"]}
    )
    with pytest.raises(ValueError, match="混合"):
        normalize_known_join_dtypes(
            frame, required_non_null=["absolute_onset_time"]
        )


def test_real_modality_merge_normalizes_key_dtypes_and_detects_post_normalization_duplicates() -> None:
    behavior = pd.DataFrame(
        {
            "repeat_participant_id": ["g1"],
            "session_id": [" sub-001 "],
            "block_id": [1],
            "rt_cv": [0.1],
        }
    )
    nir = pd.DataFrame(
        {
            "repeat_participant_id": pd.Series(["g1"], dtype="string"),
            "session_id": ["sub-001"],
            "block_id": ["1"],
            "pupil": [30.0],
        }
    )
    merged, audit = merge_modalities(
        {"behavior": behavior, "nir": nir}, unit="block", how="inner"
    )
    assert len(merged) == 1
    assert merged.loc[0, "block_id"] == "1"
    assert set(audit["key_normalization"]) == {
        "canonical-string-integer-utc-epoch-ms-v1"
    }

    duplicate_after_strip = pd.DataFrame(
        {
            "repeat_participant_id": ["g1", " g1 "],
            "session_id": ["sub-001", "sub-001"],
            "block_id": [1, "1"],
        }
    )
    with pytest.raises(ValueError, match="主键重复"):
        validate_merge_ready(
            duplicate_after_strip, unit="block", modality="synthetic"
        )


def test_missing_merge_key_fails_closed_after_normalization() -> None:
    table = pd.DataFrame(
        {
            "repeat_participant_id": ["g1"],
            "session_id": ["sub-001"],
            "block_id": [None],
        }
    )
    with pytest.raises(ValueError, match="缺失"):
        validate_merge_ready(table, unit="block", modality="nir")


def test_nir_adapter_emits_canonical_key_dtypes_without_restoring_pir() -> None:
    frame = pd.DataFrame(
        {
            "subject": ["sub-001"],
            "phase": ["block1"],
            "phase_segment": [1],
            "frame_idx": ["1"],
            "eye": ["frame_left"],
            "unix_ms": ["1000"],
            "pupil_found": [1],
            "pupil_fit_valid": [1],
            "pupil_equivalent_diameter": [30.0],
            "hard_pupil_fraction": [0.1],
            "soft_pupil_fraction": [0.11],
            "hard_iris_fraction": [0.2],
            "soft_iris_fraction": [0.21],
        }
    )
    out = adapt_nir_frame_table(frame, schema_version=6)
    assert out.loc[0, "phase_segment"] == "1"
    assert out.loc[0, "frame_idx"] == 1
    assert out.loc[0, "unix_ms"] == 1000
    assert "hard_iris_fraction" in out.columns
    assert "fullclass_pupil_to_iris_diameter_ratio" not in out.columns
    assert out.loc[0, "ocular_aperture_role"] == "unavailable_not_reconstructed"


def test_pupil_behavior_join_normalizes_phase_dtype_and_timezone_before_merge_asof() -> None:
    pupil = pd.DataFrame(
        {
            "subject": ["sub-001"],
            "phase": ["block1"],
            "phase_segment": [1],
            "frame_idx": [1],
            "eye": ["left"],
            "unix_ms": pd.to_datetime(["2026-08-29T06:00:01Z"]),
        }
    )
    behavior = pd.DataFrame(
        {
            "subject": ["sub-001"],
            "phase": ["block1"],
            "phase_segment": ["1"],
            "absolute_onset_time": ["2026-08-29T14:00:00+08:00"],
            "next_trial_onset_time": ["2026-08-29T14:00:02+08:00"],
            "stimulus_name": ["apple.png"],
            "stimulus_size": [100],
        }
    )
    visual = pd.DataFrame(
        {
            "stimulus_name": ["apple.png"],
            "stimulus_code": ["apple"],
            "stimulus_size_pct": [100],
            "screen_rel_lum_mean": [0.4],
        }
    )
    linked = attach_behavior_and_visual(pupil, behavior, visual)
    assert linked.loc[0, "behavior_match_status"] == "matched"
    assert linked.loc[0, "behavior_match_delta_ms"] == 1000
    assert linked.loc[0, "phase_segment"] == "1"
    assert linked.loc[0, "current_stimulus_code"] == "apple"


def test_pupil_behavior_join_rejects_missing_phase_segment_and_naive_time() -> None:
    base_pupil = pd.DataFrame(
        {
            "subject": ["sub-001"],
            "phase": ["block1"],
            "phase_segment": [None],
            "frame_idx": [1],
            "eye": ["left"],
            "unix_ms": [1000],
        }
    )
    behavior = pd.DataFrame(
        {
            "subject": ["sub-001"],
            "phase": ["block1"],
            "phase_segment": ["block1"],
            "absolute_onset_time": [900],
            "stimulus_name": ["apple.png"],
            "stimulus_size": [100],
        }
    )
    visual = pd.DataFrame(
        {
            "stimulus_name": ["apple.png"],
            "stimulus_code": ["apple"],
            "stimulus_size_pct": [100],
        }
    )
    with pytest.raises(ValueError, match="phase_segment"):
        attach_behavior_and_visual(base_pupil, behavior, visual)

    good_pupil = base_pupil.copy()
    good_pupil["phase_segment"] = "block1"
    naive_behavior = behavior.copy()
    naive_behavior["absolute_onset_time"] = "2026-08-29 06:00:00"
    with pytest.raises(ValueError, match="没有时区"):
        attach_behavior_and_visual(good_pupil, naive_behavior, visual)


def test_science_boundary_config_keeps_oar_qc_only_and_mmwave_external_validation_blocked() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (root / "configs" / "formal_multimodal_v2.yaml").read_text(encoding="utf-8")
    )
    oar = config["nir"]["ocular_aperture_policy"]
    assert oar["role"] == "nir_eye_opening_candidate_qc"
    assert oar["not_blink_event"] is True
    assert oar["not_perclos"] is True
    assert oar["never_reconstruct_from_iris_fraction"] is True
    assert "fullclass_pupil_to_iris_diameter_ratio" in config["nir"]["forbidden_formal_metrics"]
    assert config["mmwave"]["no_external_ecg_rsp_validation_claim"] is True
    assert config["cohort"]["expected_session_count"] == 116
    assert config["cohort"]["expected_group_count"] == 61
    assert config["cohort"]["participant_group_size_policy"] == "any_positive_integer"
