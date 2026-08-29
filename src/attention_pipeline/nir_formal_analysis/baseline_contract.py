from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from attention_pipeline.config import load_config

BASELINE_CONTRACT_VERSION = "nir-baseline-semantics-v1"


def _resolve_output(config) -> Path:
    raw = config.section("paths").get("output_root")
    if raw in (None, ""):
        raise KeyError("formal pupil config missing paths.output_root")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def baseline_contract_rows() -> list[dict[str, Any]]:
    return [
        {
            "reference_name": "session_eye_centering_reference",
            "status": "implemented",
            "implementation": "nir_analysis_ready.candidate_metrics.compute_candidate_baselines",
            "scope": "session × eye × candidate metric × quality track; formal block1+block2 rows within the session",
            "window": "all available formal-phase rows in the current session for that eye/metric",
            "estimator": "median; MAD; robust_sigma=1.4826×MAD",
            "purpose": "within-session/eye robust centering and robust-z scaling",
            "resting_physiological_baseline": False,
            "cross_session_shared": False,
            "interpretation_zh": "场次-眼-指标稳健中心化参考值，不是静息生理基线。重复场次不共享参考值。",
        },
        {
            "reference_name": "pre_event_local_baseline",
            "status": "implemented",
            "implementation": "nir_formal_analysis.event_response.event_response_features",
            "scope": "one trial event",
            "window": "-200 ms <= time relative to trial onset < 0 ms",
            "estimator": "median and MAD with valid-N/gap gates",
            "purpose": "phasic pupil event-response amplitude, latency and recovery reference",
            "resting_physiological_baseline": False,
            "cross_session_shared": False,
            "interpretation_zh": "单试次刺激前局部参考，用于瞬时瞳孔响应；不等于整场静息基线。",
        },
        {
            "reference_name": "resting_or_task_start_baseline",
            "status": "not_defined_without_protocol_evidence",
            "implementation": "none",
            "scope": "not admitted",
            "window": "not defined",
            "estimator": "not defined",
            "purpose": "would require a protocol-verified stable resting/task-start segment",
            "resting_physiological_baseline": True,
            "cross_session_shared": False,
            "interpretation_zh": "当前实验协议证据不足，不从视频开头任意截取时间段冒充静息基线。",
        },
    ]


def run_baseline_contract(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _resolve_output(config)
    root.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(baseline_contract_rows())
    csv_path = root / "nir_baseline_semantics_contract.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_version": BASELINE_CONTRACT_VERSION,
        "status": "complete",
        "session_centering_must_not_be_reported_as_resting_baseline": True,
        "pre_event_baseline_is_local_not_resting": True,
        "resting_baseline_admitted": False,
        "resting_baseline_reason": "no protocol-verified stable resting/task-start segment has been frozen",
        "output": str(csv_path),
    }
    (root / "nir_baseline_semantics_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
