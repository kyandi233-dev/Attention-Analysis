from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.nir_analysis_ready.pupil_only import load_source_manifest
from attention_pipeline.nir_pupil_only import adapt_session_rows


RESTING_OBSERVABILITY_VERSION = "nir-resting-observability-v1"


@dataclass(frozen=True)
class RestingObservabilityConfig:
    start_event: str = "baseline_start"
    stop_event: str = "baseline_stop"
    expected_duration_sec: float = 180.0
    duration_tolerance_sec: float = 30.0
    max_contiguous_gap_sec: float = 0.25
    min_primary_valid_fraction: float | None = None
    min_longest_primary_valid_sec: float | None = None

    @property
    def thresholds_frozen(self) -> bool:
        return (
            self.min_primary_valid_fraction is not None
            and self.min_longest_primary_valid_sec is not None
        )


def _resolve(config: Config, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config.path.parent.parent / path).resolve()


def _selected_records(
    records: list[dict[str, Any]], subjects: Iterable[str] | None
) -> list[dict[str, Any]]:
    if not subjects:
        return list(records)
    wanted = {str(x).strip() for x in subjects if str(x).strip()}
    selected = [row for row in records if str(row["session_id"]) in wanted]
    missing = sorted(wanted - {str(row["session_id"]) for row in selected})
    if missing:
        raise ValueError(f"requested sessions absent from NIR source manifest: {missing}")
    return selected


def _timeline_candidates(root: Path, session_id: str) -> list[Path]:
    variants = [session_id, f"{session_id}_" if not session_id.endswith("_") else session_id.rstrip("_")]
    return [root / variant / "beh" / "master_timeline.csv" for variant in dict.fromkeys(variants)]


def find_master_timeline(config: Config, session_id: str) -> Path:
    roots = config.registry_paths("formal_raw_roots")
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(_timeline_candidates(Path(root), session_id))
    hits = [path for path in candidates if path.is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{session_id}: expected exactly one beh/master_timeline.csv across formal_raw_roots; "
            f"found {len(hits)}"
        )
    return hits[0]


def baseline_interval_from_timeline(
    timeline: pd.DataFrame,
    *,
    start_event: str = "baseline_start",
    stop_event: str = "baseline_stop",
) -> dict[str, Any]:
    required = {"event", "unix_ms"}
    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"master timeline missing {sorted(missing)}")
    event = timeline["event"].astype(str).str.strip()
    start = pd.to_numeric(timeline.loc[event.eq(start_event), "unix_ms"], errors="coerce").dropna()
    stop = pd.to_numeric(timeline.loc[event.eq(stop_event), "unix_ms"], errors="coerce").dropna()
    if len(start) != 1 or len(stop) != 1:
        raise ValueError(
            f"expected exactly one {start_event}/{stop_event}; got {len(start)}/{len(stop)}"
        )
    start_ms = float(start.iloc[0])
    stop_ms = float(stop.iloc[0])
    if not np.isfinite(start_ms) or not np.isfinite(stop_ms) or stop_ms <= start_ms:
        raise ValueError("invalid baseline_start/baseline_stop interval")
    return {
        "baseline_start_ms": start_ms,
        "baseline_stop_ms": stop_ms,
        "baseline_duration_sec": (stop_ms - start_ms) / 1000.0,
    }


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def _longest_valid_run_sec(
    times_ms: pd.Series,
    valid: pd.Series,
    *,
    max_gap_sec: float,
) -> float:
    t = pd.to_numeric(times_ms, errors="coerce").to_numpy(dtype=float)
    v = _as_bool(valid).to_numpy(dtype=bool)
    mask = np.isfinite(t)
    t = t[mask]
    v = v[mask]
    if len(t) == 0:
        return 0.0
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    max_gap_ms = float(max_gap_sec) * 1000.0
    best = 0.0
    run_start: float | None = None
    previous: float | None = None
    for current, is_valid in zip(t, v, strict=False):
        if not is_valid:
            if run_start is not None and previous is not None:
                best = max(best, (previous - run_start) / 1000.0)
            run_start = None
            previous = None
            continue
        if run_start is None:
            run_start = float(current)
            previous = float(current)
            continue
        assert previous is not None
        if float(current) - previous > max_gap_ms:
            best = max(best, (previous - run_start) / 1000.0)
            run_start = float(current)
        previous = float(current)
    if run_start is not None and previous is not None:
        best = max(best, (previous - run_start) / 1000.0)
    return float(best)


def summarize_resting_rows(
    adapted: pd.DataFrame,
    *,
    start_ms: float,
    stop_ms: float,
    cfg: RestingObservabilityConfig,
) -> dict[str, Any]:
    unix = pd.to_numeric(adapted.get("unix_ms"), errors="coerce")
    resting = adapted.loc[unix.ge(start_ms) & unix.lt(stop_ms)].copy()
    result: dict[str, Any] = {
        "resting_observability_version": RESTING_OBSERVABILITY_VERSION,
        "resting_source_row_n": int(len(resting)),
        "resting_interval_start_ms": float(start_ms),
        "resting_interval_stop_ms": float(stop_ms),
        "resting_interval_duration_sec": float((stop_ms - start_ms) / 1000.0),
        "observability_semantics": "not-observed is a data/measurement state and is never automatically labeled eyes-closed",
    }
    if resting.empty:
        result.update({
            "resting_observability_status": "not_estimable_no_nir_rows_in_timeline_interval",
            "resting_reference_status": "not_estimable",
            "resting_timepoint_n": 0,
        })
        return result

    # Build one timepoint per source frame so the two eye rows do not double the
    # denominator. Any-eye validity is used for an exploratory pupil reference;
    # left/right-specific fractions are retained separately below.
    frame_key = "frame_idx" if "frame_idx" in resting.columns else "unix_ms"
    grouped_rows: list[dict[str, Any]] = []
    for _, group in resting.groupby(frame_key, sort=True, dropna=False):
        grouped_rows.append({
            "unix_ms": float(pd.to_numeric(group["unix_ms"], errors="coerce").median()),
            "source_observed_any": bool(_as_bool(group["source_observed"]).any()),
            "primary_valid_any": bool(_as_bool(group["pupil_valid_primary"]).any()),
            "strict_valid_any": bool(_as_bool(group["pupil_valid_strict"]).any()),
        })
    timepoints = pd.DataFrame(grouped_rows).sort_values("unix_ms", kind="stable")
    result["resting_timepoint_n"] = int(len(timepoints))
    result["source_observed_any_fraction"] = float(timepoints["source_observed_any"].mean())
    result["primary_valid_any_fraction"] = float(timepoints["primary_valid_any"].mean())
    result["strict_valid_any_fraction"] = float(timepoints["strict_valid_any"].mean())
    result["longest_primary_valid_run_sec"] = _longest_valid_run_sec(
        timepoints["unix_ms"],
        timepoints["primary_valid_any"],
        max_gap_sec=cfg.max_contiguous_gap_sec,
    )

    for eye in ("left", "right"):
        current = resting[resting["eye"].astype(str).eq(eye)] if "eye" in resting else resting.iloc[0:0]
        result[f"{eye}_row_n"] = int(len(current))
        result[f"{eye}_source_observed_fraction"] = float(_as_bool(current["source_observed"]).mean()) if len(current) else np.nan
        result[f"{eye}_primary_valid_fraction"] = float(_as_bool(current["pupil_valid_primary"]).mean()) if len(current) else np.nan
        result[f"{eye}_strict_valid_fraction"] = float(_as_bool(current["pupil_valid_strict"]).mean()) if len(current) else np.nan

    for axis in (
        "source_missing", "ritnet_missing", "roi_clipped", "geometry_invalid",
        "temporal_flagged", "interpolation_only",
    ):
        result[f"{axis}_row_fraction"] = (
            float(_as_bool(resting[axis]).mean()) if axis in resting.columns else np.nan
        )

    primary = _as_bool(resting["pupil_valid_primary"])
    pupil = pd.to_numeric(resting.get("pupil_geom_mean_diameter"), errors="coerce")
    valid_pupil = pupil[primary & np.isfinite(pupil)]
    result["resting_primary_pupil_n"] = int(len(valid_pupil))
    result["resting_pupil_median_candidate"] = float(valid_pupil.median()) if len(valid_pupil) else np.nan
    result["resting_pupil_mad_candidate"] = (
        float(np.median(np.abs(valid_pupil.to_numpy(dtype=float) - float(valid_pupil.median()))))
        if len(valid_pupil) else np.nan
    )

    if not cfg.thresholds_frozen:
        result["resting_observability_status"] = "audit_only_thresholds_not_frozen"
        result["resting_reference_status"] = "not_authorized_thresholds_not_frozen"
        return result

    assert cfg.min_primary_valid_fraction is not None
    assert cfg.min_longest_primary_valid_sec is not None
    reasons: list[str] = []
    if result["primary_valid_any_fraction"] < cfg.min_primary_valid_fraction:
        reasons.append("primary_valid_fraction_below_gate")
    if result["longest_primary_valid_run_sec"] < cfg.min_longest_primary_valid_sec:
        reasons.append("longest_primary_valid_run_below_gate")
    if reasons:
        result["resting_observability_status"] = "not_observable_for_exploratory_reference"
        result["resting_reference_status"] = "not_estimable_observability_gate_failed"
        result["resting_observability_reasons"] = ";".join(reasons)
    else:
        result["resting_observability_status"] = "observable_for_exploratory_reference"
        result["resting_reference_status"] = "exploratory_candidate_estimable"
        result["resting_observability_reasons"] = ""
    return result


def _cfg_from_mapping(value: Mapping[str, Any]) -> RestingObservabilityConfig:
    def _optional_float(name: str) -> float | None:
        raw = value.get(name)
        return None if raw in (None, "") else float(raw)

    return RestingObservabilityConfig(
        start_event=str(value.get("start_event", "baseline_start")),
        stop_event=str(value.get("stop_event", "baseline_stop")),
        expected_duration_sec=float(value.get("expected_duration_sec", 180.0)),
        duration_tolerance_sec=float(value.get("duration_tolerance_sec", 30.0)),
        max_contiguous_gap_sec=float(value.get("max_contiguous_gap_sec", 0.25)),
        min_primary_valid_fraction=_optional_float("min_primary_valid_fraction"),
        min_longest_primary_valid_sec=_optional_float("min_longest_primary_valid_sec"),
    )


def run_resting_observability(
    materialize_config_path: str | Path,
    tables_config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    paths_config: str | Path | None = None,
) -> dict[str, Any]:
    materialize_config = load_config(materialize_config_path, paths_config=paths_config)
    tables_config = load_config(tables_config_path, paths_config=paths_config)
    _, all_records = load_source_manifest(materialize_config)
    records = _selected_records(all_records, subjects)
    cfg = _cfg_from_mapping(tables_config.section("resting_observability"))
    output_root = tables_config.path_value("output_root")
    qc_root = output_root / "resting_observability"
    qc_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        session_id = str(record["session_id"])
        try:
            timeline_path = find_master_timeline(materialize_config, session_id)
            timeline = pd.read_csv(timeline_path, encoding="utf-8-sig", low_memory=False)
            interval = baseline_interval_from_timeline(
                timeline, start_event=cfg.start_event, stop_event=cfg.stop_event
            )
            source_path = _resolve(materialize_config, str(record["source_csv"]))
            source = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)
            adapted = adapt_session_rows(source, record)
            row = {
                "session_id": session_id,
                "analysis_group_token": str(record["analysis_group_token"]),
                "timeline_path_role": "formal_raw_roots/<session>/beh/master_timeline.csv",
                "source_schema_version": int(record["source_schema_version"]),
                **interval,
                **summarize_resting_rows(
                    adapted,
                    start_ms=float(interval["baseline_start_ms"]),
                    stop_ms=float(interval["baseline_stop_ms"]),
                    cfg=cfg,
                ),
            }
            duration_error = abs(float(interval["baseline_duration_sec"]) - cfg.expected_duration_sec)
            row["duration_error_sec"] = duration_error
            row["duration_within_expected_tolerance"] = duration_error <= cfg.duration_tolerance_sec
            rows.append(row)
        except Exception as exc:
            failures.append({
                "session_id": session_id,
                "stage": "resting_observability",
                "status": "not_estimable",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    summary = pd.DataFrame(rows)
    failure_table = pd.DataFrame(
        failures,
        columns=["session_id", "stage", "status", "error_type", "error"],
    )
    summary.to_csv(qc_root / "resting_observability_by_session.csv", index=False, encoding="utf-8-sig")
    failure_table.to_csv(qc_root / "resting_observability_failures.csv", index=False, encoding="utf-8-sig")
    contract = {
        "version": RESTING_OBSERVABILITY_VERSION,
        "interval_source": "master_timeline baseline_start -> baseline_stop Unix-ms events",
        "video_head_180s_allowed": False,
        "not_observed_may_be_labeled_eyes_closed": False,
        "thresholds_frozen": cfg.thresholds_frozen,
        "min_primary_valid_fraction": cfg.min_primary_valid_fraction,
        "min_longest_primary_valid_sec": cfg.min_longest_primary_valid_sec,
        "selection_rule": "observability gates must be frozen without using downstream Behavior outcomes",
        "role": "exploratory_resting_reference_only_not_forced_into_main_models",
        "n_sessions_requested": int(len(records)),
        "n_sessions_summarized": int(len(summary)),
        "n_sessions_failed": int(len(failure_table)),
        "status": (
            "audit_only_thresholds_not_frozen"
            if not cfg.thresholds_frozen
            else "complete" if failure_table.empty else "complete_with_failures"
        ),
    }
    (qc_root / "resting_observability_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return contract
