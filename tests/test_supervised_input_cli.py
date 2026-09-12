from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_file_level_materialization_cli_writes_task_a_input_and_refuses_overwrite(tmp_path: Path):
    analysis_sets = pd.DataFrame(
        [
            {
                "session_id": "s1",
                "participant_group_id": "p1",
                "block_id": "b1",
                "probe_index_in_block": 1,
                "probe_event_id": "s1|B1|probe|1",
                "q1_nominal_4class": 1,
                "analysis_set_id": "behavior_vs_x",
                "included_complete": True,
                "included_missing_aware": True,
                "comparison_models": '["B", "B+x"]',
                "required_features": '{"behavior": ["b"], "sensor": ["x"]}',
            },
            {
                "session_id": "s2",
                "participant_group_id": "p2",
                "block_id": "b1",
                "probe_index_in_block": 1,
                "probe_event_id": "s2|B1|probe|1",
                "q1_nominal_4class": 2,
                "analysis_set_id": "behavior_vs_x",
                "included_complete": True,
                "included_missing_aware": True,
                "comparison_models": '["B", "B+x"]',
                "required_features": '{"behavior": ["b"], "sensor": ["x"]}',
            },
        ]
    )
    status_rows = []
    for session, participant, b, x in (("s1", "p1", 1.0, 0.5), ("s2", "p2", 2.0, 0.2)):
        for modality, feature, value in (("behavior", "b", b), ("sensor", "x", x)):
            status_rows.append(
                {
                    "session_id": session,
                    "participant_group_id": participant,
                    "block_id": "b1",
                    "probe_index_in_block": 1,
                    "modality": modality,
                    "feature": feature,
                    "value": value,
                    "feature_computable": True,
                    "eligible_for_missing_strategy": True,
                    "missing_kind": "available",
                }
            )
    status = pd.DataFrame(status_rows)

    analysis_path = tmp_path / "analysis_sets.csv"
    status_path = tmp_path / "probe_feature_status.csv"
    output_path = tmp_path / "task_a_input.csv"
    analysis_sets.to_csv(analysis_path, index=False, encoding="utf-8-sig")
    status.to_csv(status_path, index=False, encoding="utf-8-sig")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "materialize_supervised_input.py"),
        "--analysis-sets",
        str(analysis_path),
        "--probe-feature-status",
        str(status_path),
        "--analysis-set-id",
        "behavior_vs_x",
        "--membership-type",
        "included_complete",
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()

    out = pd.read_csv(output_path, encoding="utf-8-sig")
    assert len(out) == 2
    assert set(("b", "x", "analysis_set_id", "membership_type")) <= set(out.columns)
    assert out["analysis_set_id"].eq("behavior_vs_x").all()
    assert out["membership_type"].eq("included_complete").all()

    repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
