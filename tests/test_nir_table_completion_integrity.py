from __future__ import annotations

import json

from attention_pipeline.nir_formal_analysis import pupil_tables_hardened as hardened


def _fake_paths(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    paths = {key: root / f"{key}.dat" for key in hardened.OUTPUT_KEYS}
    paths.update(
        {
            "root": root,
            "completion": root / "completion.json",
        }
    )
    return paths


def test_completion_requires_all_output_digests_and_detects_tampering(tmp_path, monkeypatch) -> None:
    session_id = "sub-001"
    paths = _fake_paths(tmp_path)
    analysis_ready = tmp_path / "analysis_ready.csv"
    analysis_ready.write_text("session_id,value\nsub-001,1\n", encoding="utf-8")
    for key in hardened.OUTPUT_KEYS:
        paths[key].write_text(f"{key}\n", encoding="utf-8")

    monkeypatch.setattr(hardened.base, "_session_paths", lambda config, sid: paths)
    monkeypatch.setattr(
        hardened.base, "_session_frame_path", lambda config, sid: analysis_ready
    )

    hardened._seal_completion(object(), session_id)
    marker = json.loads(paths["completion"].read_text(encoding="utf-8"))
    assert marker["completion_contract_version"] == hardened.COMPLETION_CONTRACT_VERSION
    assert set(marker["outputs_sha256"]) == set(hardened.OUTPUT_KEYS)
    assert hardened._completion_is_valid(object(), session_id)

    paths["trial_windows"].write_text("tampered\n", encoding="utf-8")
    assert not hardened._completion_is_valid(object(), session_id)


def test_legacy_completion_without_output_digests_is_not_reused(tmp_path, monkeypatch) -> None:
    session_id = "sub-001"
    paths = _fake_paths(tmp_path)
    analysis_ready = tmp_path / "analysis_ready.csv"
    analysis_ready.write_text("session_id,value\nsub-001,1\n", encoding="utf-8")
    for key in hardened.OUTPUT_KEYS:
        paths[key].write_text(f"{key}\n", encoding="utf-8")

    monkeypatch.setattr(hardened.base, "_session_paths", lambda config, sid: paths)
    monkeypatch.setattr(
        hardened.base, "_session_frame_path", lambda config, sid: analysis_ready
    )
    paths["completion"].write_text(
        json.dumps(
            {
                "status": "complete",
                "pipeline_version": hardened.base.PIPELINE_VERSION,
                "schema_version": hardened.base.SCHEMA_VERSION,
                "session_id": session_id,
                "analysis_ready_sha256": hardened.base._digest_file(analysis_ready),
            }
        ),
        encoding="utf-8",
    )

    assert not hardened._completion_is_valid(object(), session_id)
