from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from attention_pipeline.config import load_config

BASELINE_CONTRACT_VERSION = "nir-baseline-semantics-v2"


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
            "reference_name": "resting_task_start_interval",
            "status": "protocol_interval_verified_reference_pending_observability_gate",
            "implementation": "nir_formal_analysis.resting_observability.run_resting_observability",
            "scope": "per session; formal program timeline interval only",
            "window": "master_timeline baseline_start <= unix_ms < baseline_stop (nominally ~180 s)",
            "estimator": "observability/QC audit first; exploratory pupil median/MAD only after prefrozen observability gates",
            "purpose": "exploratory task-start resting reference when the eye is measurably observable",
            "resting_physiological_baseline": True,
            "cross_session_shared": False,
            "interpretation_zh": (
                "正式程序已确认约3分钟静息区间，但并非所有场次都一定有可用瞳孔。"
                "必须先按时间轴事件和眼部可观测性审计；阈值未预先冻结前不授权静息瞳孔参考，"
                "不可探测也不得自动解释为闭眼。"
            ),
        },
    ]


def _resting_gate_state(config) -> dict[str, Any]:
    resting = config.section("resting_observability")
    min_fraction = resting.get("min_primary_valid_fraction")
    min_run = resting.get("min_longest_primary_valid_sec")
    frozen = min_fraction not in (None, "") and min_run not in (None, "")
    return {
        "thresholds_frozen": bool(frozen),
        "min_primary_valid_fraction": None if min_fraction in (None, "") else float(min_fraction),
        "min_longest_primary_valid_sec": None if min_run in (None, "") else float(min_run),
    }


def run_baseline_contract(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _resolve_output(config)
    root.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(baseline_contract_rows())
    csv_path = root / "nir_baseline_semantics_contract.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    gate = _resting_gate_state(config)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_version": BASELINE_CONTRACT_VERSION,
        "status": "complete",
        "session_centering_must_not_be_reported_as_resting_baseline": True,
        "pre_event_baseline_is_local_not_resting": True,
        "resting_interval_protocol_verified": True,
        "resting_interval_source": "master_timeline baseline_start -> baseline_stop",
        "fixed_video_head_180s_substitution_allowed": False,
        "resting_observability_thresholds_frozen": gate["thresholds_frozen"],
        "resting_baseline_admitted": False,
        "resting_baseline_reason": (
            "protocol interval is verified, but a resting pupil reference remains exploratory and "
            "is not authorized as a main-model baseline; observability thresholds must be prefrozen"
            if not gate["thresholds_frozen"]
            else "observability thresholds are configured, but resting reference remains exploratory only"
        ),
        "resting_not_observed_may_be_called_eyes_closed": False,
        "output": str(csv_path),
    }
    (root / "nir_baseline_semantics_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
