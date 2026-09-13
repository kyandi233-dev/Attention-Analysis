from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_prediction_archive_cli_validates_native_task_a_file(tmp_path: Path):
    analysis_sets = pd.DataFrame(
        [
            {
                "session_id": "s1",
                "participant_group_id": "p1",
                "block_id": "b1",
                "probe_index_in_block": 1,
                "analysis_set_id": "behavior_vs_x",
                "included_complete": True,
                "included_missing_aware": True,
                "comparison_models": '["B", "B+x"]',
                "required_outcomes": '["q1_nominal_4class"]',
                "q1_nominal_4class": 1,
            },
            {
                "session_id": "s2",
                "participant_group_id": "p2",
                "block_id": "b1",
                "probe_index_in_block": 1,
                "analysis_set_id": "behavior_vs_x",
                "included_complete": True,
                "included_missing_aware": True,
                "comparison_models": '["B", "B+x"]',
                "required_outcomes": '["q1_nominal_4class"]',
                "q1_nominal_4class": 2,
            },
        ]
    )

    prediction_rows = []
    for model_id, feature_set_id in (("B", "behavior-v1"), ("B+x", "behavior-plus-x-v1")):
        for session, participant, q1, probability in (
            ("s1", "p1", 1, 0.8),
            ("s2", "p2", 2, 0.2),
        ):
            prediction_rows.append(
                {
                    "run_id": "smoke-run",
                    "feature_set_id": feature_set_id,
                    "membership_type": "included_complete",
                    "session_id": session,
                    "participant_group_id": participant,
                    "block_id": "b1",
                    "probe_event_id": f"{session}|B1|probe|1",
                    "analysis_set_id": "behavior_vs_x",
                    "outer_fold_group": participant,
                    "model_id": model_id,
                    "q1_binary": 1 if q1 == 1 else 0,
                    "predicted_q1_binary": 1 if probability >= 0.5 else 0,
                    "p_q1_equals_1": probability,
                    "model_failed": False,
                    "failure_reason": "",
                }
            )

    predictions_path = tmp_path / "probe_predictions.csv"
    analysis_sets_path = tmp_path / "analysis_sets.csv"
    archive_path = tmp_path / "validated_archive.csv"
    audit_path = tmp_path / "archive_audit.json"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False, encoding="utf-8-sig")
    analysis_sets.to_csv(analysis_sets_path, index=False, encoding="utf-8-sig")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "validate_supervised_prediction_archive.py"),
        "--predictions",
        str(predictions_path),
        "--analysis-sets",
        str(analysis_sets_path),
        "--analysis-set-id",
        "behavior_vs_x",
        "--membership-type",
        "included_complete",
        "--archive-output",
        str(archive_path),
        "--audit-output",
        str(audit_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert archive_path.is_file()
    assert audit_path.is_file()
    archive = pd.read_csv(archive_path, encoding="utf-8-sig")
    assert len(archive) == 4
    assert set(archive["outcome"]) == {"q1_equals_1_vs_2_3_4"}
    assert '"status": "PASS_PREDICTION_ARCHIVE"' in audit_path.read_text(encoding="utf-8")
