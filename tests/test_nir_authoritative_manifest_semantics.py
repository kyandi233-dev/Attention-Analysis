from __future__ import annotations

import json

import attention_pipeline.nir_analysis_ready as nir_ready


def test_authoritative_materialization_splits_pir_from_ocular_aperture_qc(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "analysis_ready_manifest.json"
    legacy_summary = {
        "signal_semantics": "pupil_geometry_only",
        "iris_geometry_used": False,
        "pir_oar_allowed": False,
    }
    legacy_manifest = {
        "signal_semantics": "pupil_geometry_only",
        "pir_oar_refused": True,
    }
    summary_path.write_text(json.dumps(legacy_summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")

    def fake_materializer(*args, **kwargs):
        return {
            "summary": dict(legacy_summary),
            "summary_path": str(summary_path),
            "manifest_path": str(manifest_path),
        }

    fake_contract = {
        "canonical_cohort_topology": {
            "n_sessions": 4,
            "n_analysis_groups": 2,
            "n_repeated_participant_groups": 1,
            "max_sessions_per_participant": 3,
            "group_size_distribution": {"1": 1, "3": 1},
        },
        "nir_availability": {
            "n_available_sessions": 3,
            "n_unavailable_sessions": 1,
            "n_canonical_sessions": 4,
            "complete_accounting": True,
            "availability_does_not_redefine_cohort": True,
        },
    }

    monkeypatch.setattr(nir_ready, "_run_materialization", fake_materializer)
    monkeypatch.setattr(nir_ready, "_full_contract", lambda *_args, **_kwargs: fake_contract)
    result = nir_ready.run_materialization("unused-config.yaml")

    assert "pir_oar_allowed" not in result["summary"]
    assert result["summary"]["pir_iris_geometry_refused"] is True
    assert result["summary"]["ocular_aperture_qc_preserved"] is True
    assert result["summary"]["ocular_aperture_formal_endpoint"] is False

    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for payload in (persisted_summary, persisted_manifest):
        assert "pir_oar_allowed" not in payload
        assert "pir_oar_refused" not in payload
        assert payload["pir_iris_geometry_refused"] is True
        assert payload["ocular_aperture_qc_preserved"] is True
        assert payload["ocular_aperture_formal_endpoint"] is False
        assert payload["ocular_aperture_interpretation"] == "producer_qc_not_ear_not_blink_not_perclos"
