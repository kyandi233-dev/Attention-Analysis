from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import Config
from .behavior_qc import add_behavior_qc
from .contract import (
    ALIGNMENT_PIPELINE_VERSION,
    ALIGNMENT_SCHEMA_VERSION,
    EYES,
    OAR_COLUMN,
    OAR_P90_COLUMN,
    OPTIONAL_NIR_QC_COLUMNS,
    PIR_COLUMN,
    PIR_VALID_COLUMN,
    REQUIRED_NIR_COLUMNS,
    WindowSpec,
    normalize_subject,
    parse_window_specs,
    subject_output_paths,
)
from .discovery import (
    NirSource,
    alignment_output_root,
    find_nir_source,
    load_behavior_trials,
    sha256,
)
from .features import coerce_bool_series, iqr, mad, summarize_signal


@dataclass(frozen=True)
class NirEyeIndex:
    subject: str
    block_num: int
    eye: str
    frame: pd.DataFrame
    times_ms: np.ndarray

    def slice(self, start_ms: float, end_ms: float) -> pd.DataFrame:
        left = int(np.searchsorted(self.times_ms, start_ms, side="left"))
        right = int(np.searchsorted(self.times_ms, end_ms, side="left"))
        return self.frame.iloc[left:right]


def _normalize_eye(value: Any) -> str:
    text = str(value).strip().lower()
    for prefix in ("frame_", "eye_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text in {"l", "left"}:
        return "left"
    if text in {"r", "right"}:
        return "right"
    return text


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def load_nir_frame(source: NirSource) -> pd.DataFrame:
    df = pd.read_csv(source.csv_path, encoding="utf-8-sig")
    missing = REQUIRED_NIR_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{source.csv_path}: missing full-class columns {sorted(missing)}"
        )

    df = df.copy()
    df["subject"] = df["subject"].map(normalize_subject)
    if set(df["subject"].dropna().unique()) != {source.subject}:
        raise ValueError(f"{source.csv_path}: mixed or unexpected subject identifiers")
    df["eye"] = df["eye"].map(_normalize_eye)
    unexpected_eyes = sorted(set(df["eye"].dropna().unique()) - set(EYES))
    if unexpected_eyes:
        raise ValueError(f"{source.csv_path}: unexpected eye labels {unexpected_eyes}")

    _to_numeric(
        df,
        [
            "phase_segment",
            "frame_idx",
            "video_time_ms",
            "unix_ms",
            PIR_COLUMN,
            OAR_COLUMN,
            OAR_P90_COLUMN,
            *OPTIONAL_NIR_QC_COLUMNS,
        ],
    )
    df[PIR_VALID_COLUMN] = coerce_bool_series(df[PIR_VALID_COLUMN])
    if "roi_clipped" in df.columns:
        df["roi_clipped"] = coerce_bool_series(df["roi_clipped"])
    if "ritnet_found" in df.columns:
        df["ritnet_found"] = coerce_bool_series(df["ritnet_found"])

    df = df[df["phase"].isin(["block1", "block2"])].copy()
    df["block_num"] = df["phase"].map({"block1": 1, "block2": 2}).astype(int)
    df = df[df["unix_ms"].notna()].copy()
    df = df.sort_values(["block_num", "eye", "unix_ms", "frame_idx"]).reset_index(
        drop=True
    )
    return df


def build_nir_indices(
    nir: pd.DataFrame, subject: str
) -> dict[tuple[int, str], NirEyeIndex]:
    result: dict[tuple[int, str], NirEyeIndex] = {}
    for block_num in (1, 2):
        for eye in EYES:
            frame = nir[(nir["block_num"] == block_num) & (nir["eye"] == eye)].copy()
            if frame.empty:
                continue
            frame = frame.sort_values("unix_ms").reset_index(drop=True)
            result[(block_num, eye)] = NirEyeIndex(
                subject=subject,
                block_num=block_num,
                eye=eye,
                frame=frame,
                times_ms=frame["unix_ms"].to_numpy(dtype=float),
            )
    return result


def _window_nir_features(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"n_nir_rows": int(len(frame))}
    if frame.empty:
        for prefix in ("pir", "oar"):
            result.update(summarize_signal(np.array([]), np.array([]), prefix))
        result.update(
            {
                "pir_valid_fraction": None,
                "oar_valid_fraction": None,
                "roi_clipped_fraction": None,
                "ritnet_found_fraction": None,
                "ocular_fragmented_candidate_fraction": None,
            }
        )
        return result

    times = frame["unix_ms"].to_numpy(dtype=float)
    pir_raw = pd.to_numeric(frame[PIR_COLUMN], errors="coerce").to_numpy(dtype=float)
    pir_gate = frame[PIR_VALID_COLUMN].fillna(False).to_numpy(dtype=bool)
    pir_values = np.where(pir_gate, pir_raw, np.nan)
    oar_values = pd.to_numeric(frame[OAR_COLUMN], errors="coerce").to_numpy(dtype=float)

    result.update(summarize_signal(times, pir_values, "pir"))
    result.update(summarize_signal(times, oar_values, "oar"))
    result["pir_valid_fraction"] = float(np.isfinite(pir_values).mean())
    result["oar_valid_fraction"] = float(np.isfinite(oar_values).mean())

    if "roi_clipped" in frame.columns:
        result["roi_clipped_fraction"] = float(
            frame["roi_clipped"].fillna(False).astype(bool).mean()
        )
    else:
        result["roi_clipped_fraction"] = None
    if "ritnet_found" in frame.columns:
        result["ritnet_found_fraction"] = float(
            frame["ritnet_found"].fillna(False).astype(bool).mean()
        )
    else:
        result["ritnet_found_fraction"] = None

    if {
        "fullclass_ocular_component_count",
        "fullclass_ocular_largest_component_fraction",
    }.issubset(frame.columns):
        component_count = pd.to_numeric(
            frame["fullclass_ocular_component_count"], errors="coerce"
        )
        largest_fraction = pd.to_numeric(
            frame["fullclass_ocular_largest_component_fraction"], errors="coerce"
        )
        fragmented = component_count.gt(1) & largest_fraction.lt(0.90)
        result["ocular_fragmented_candidate_fraction"] = float(
            fragmented.fillna(False).mean()
        )
    else:
        result["ocular_fragmented_candidate_fraction"] = None
    return result


def _add_trial_context(trials: pd.DataFrame) -> pd.DataFrame:
    df = trials.copy().sort_values(["subject", "block_num", "trial_num"]).reset_index(
        drop=True
    )
    df["global_trial_index"] = df.groupby("subject").cumcount() + 1
    group = df.groupby(["subject", "block_num"], sort=False)
    df["next_trial_onset_time"] = group["absolute_onset_time"].shift(-1)
    df["prev_stimulus_name"] = group["stimulus_name"].shift(1)
    df["prev_stimulus_size"] = group["stimulus_size"].shift(1)
    df["prev_is_no_go"] = group["is_no_go"].shift(1)
    return df


def _probe_times_by_block(trials: pd.DataFrame) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for block_num, frame in trials.groupby("block_num"):
        values = pd.to_numeric(
            frame.loc[
                pd.to_numeric(frame["is_probe"], errors="coerce").eq(1),
                "probe_onset_time",
            ],
            errors="coerce",
        ).dropna()
        result[int(block_num)] = np.sort(values.to_numpy(dtype=float))
    return result


def _last_probe_before(probe_times: np.ndarray, reference_ms: float) -> float | None:
    if probe_times.size == 0:
        return None
    idx = int(np.searchsorted(probe_times, reference_ms, side="left")) - 1
    return float(probe_times[idx]) if idx >= 0 else None


def _count_probes_in_window(
    probe_times: np.ndarray, start_ms: float, end_ms: float
) -> int:
    if probe_times.size == 0:
        return 0
    left = int(np.searchsorted(probe_times, start_ms, side="left"))
    right = int(np.searchsorted(probe_times, end_ms, side="left"))
    return max(0, right - left)


def add_trial_nir_linkage(
    trials: pd.DataFrame, indices: dict[tuple[int, str], NirEyeIndex]
) -> pd.DataFrame:
    df = trials.copy()
    for eye in EYES:
        counts: list[int] = []
        for row in df.itertuples(index=False):
            block = int(row.block_num)
            index = indices.get((block, eye))
            onset = float(row.absolute_onset_time)
            next_onset = getattr(row, "next_trial_onset_time")
            if pd.isna(next_onset):
                next_onset = onset + 1150.0
            if index is None or not np.isfinite(onset):
                counts.append(0)
            else:
                counts.append(len(index.slice(onset, float(next_onset))))
        df[f"nir_rows_trial_{eye}"] = counts
        df[f"nir_has_trial_{eye}"] = pd.Series(counts).gt(0)
    return df


def build_trial_windows(
    trials: pd.DataFrame,
    indices: dict[tuple[int, str], NirEyeIndex],
    specs: list[WindowSpec],
) -> pd.DataFrame:
    probe_times = _probe_times_by_block(trials)
    rows: list[dict[str, Any]] = []
    for trial in trials.itertuples(index=False):
        onset = float(trial.absolute_onset_time)
        block_num = int(trial.block_num)
        block_probes = probe_times.get(block_num, np.array([], dtype=float))
        previous_probe = _last_probe_before(block_probes, onset)
        for spec in specs:
            start_ms = onset + spec.start_offset_ms
            end_ms = onset + spec.end_offset_ms
            n_probes = _count_probes_in_window(block_probes, start_ms, end_ms)
            for eye in EYES:
                index = indices.get((block_num, eye))
                frame = (
                    index.slice(start_ms, end_ms) if index is not None else pd.DataFrame()
                )
                record: dict[str, Any] = {
                    "subject": trial.subject,
                    "block_num": block_num,
                    "trial_num": int(trial.trial_num),
                    "global_trial_index": int(trial.global_trial_index),
                    "trial_onset_ms": onset,
                    "window_family": spec.family,
                    "window_name": spec.name,
                    "window_start_offset_ms": spec.start_offset_ms,
                    "window_end_offset_ms": spec.end_offset_ms,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "eye": eye,
                    "n_probes_in_window": n_probes,
                    "window_crosses_probe": bool(n_probes > 0),
                    "last_probe_before_trial_ms": previous_probe,
                    "time_since_last_probe_sec": (
                        (onset - previous_probe) / 1000.0
                        if previous_probe is not None
                        else None
                    ),
                }
                record.update(_window_nir_features(frame))
                rows.append(record)
    return pd.DataFrame(rows)


def _behavior_window_features(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_trials": int(len(frame)),
        "n_go": int(
            pd.to_numeric(frame.get("is_no_go"), errors="coerce").eq(0).sum()
        )
        if len(frame)
        else 0,
        "n_nogo": int(
            pd.to_numeric(frame.get("is_no_go"), errors="coerce").eq(1).sum()
        )
        if len(frame)
        else 0,
        "n_commission": int(
            pd.to_numeric(frame.get("commission"), errors="coerce").eq(1).sum()
        )
        if len(frame)
        else 0,
        "n_omission": int(
            pd.to_numeric(frame.get("omission"), errors="coerce").eq(1).sum()
        )
        if len(frame)
        else 0,
        "n_prestimulus_press": int(
            frame.get("prestimulus_press_flag", pd.Series(dtype=bool))
            .fillna(False)
            .sum()
        )
        if len(frame)
        else 0,
        "n_ambiguous_omission": int(
            frame.get("ambiguous_omission_flag", pd.Series(dtype=bool))
            .fillna(False)
            .sum()
        )
        if len(frame)
        else 0,
        "n_anticipatory_candidate": int(
            frame.get("anticipatory_candidate_flag", pd.Series(dtype=bool))
            .fillna(False)
            .sum()
        )
        if len(frame)
        else 0,
    }
    if frame.empty:
        result.update(
            {
                "go_rt_median_ms": None,
                "go_rt_mad_ms": None,
                "go_rt_iqr_ms": None,
                "go_rt_cv": None,
            }
        )
        return result

    go_rt = pd.to_numeric(
        frame.loc[
            pd.to_numeric(frame["is_no_go"], errors="coerce").eq(0)
            & pd.to_numeric(frame["correct"], errors="coerce").eq(1),
            "rt",
        ],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    if go_rt.size:
        mean_rt = float(np.mean(go_rt))
        result.update(
            {
                "go_rt_median_ms": float(np.median(go_rt)),
                "go_rt_mad_ms": mad(go_rt),
                "go_rt_iqr_ms": iqr(go_rt),
                "go_rt_cv": (
                    float(np.std(go_rt, ddof=1) / mean_rt)
                    if go_rt.size >= 2 and mean_rt
                    else None
                ),
            }
        )
    else:
        result.update(
            {
                "go_rt_median_ms": None,
                "go_rt_mad_ms": None,
                "go_rt_iqr_ms": None,
                "go_rt_cv": None,
            }
        )
    return result


def build_probe_windows(
    trials: pd.DataFrame,
    indices: dict[tuple[int, str], NirEyeIndex],
    specs: list[WindowSpec],
) -> pd.DataFrame:
    probes = trials[
        pd.to_numeric(trials["is_probe"], errors="coerce").eq(1)
        & pd.to_numeric(trials["probe_onset_time"], errors="coerce").notna()
    ].copy()
    probes = probes.sort_values(["block_num", "probe_onset_time"]).reset_index(drop=True)
    probes["probe_index_global"] = np.arange(1, len(probes) + 1)
    probes["probe_index_in_block"] = probes.groupby("block_num").cumcount() + 1
    probes["previous_probe_onset_ms"] = probes.groupby("block_num")[
        "probe_onset_time"
    ].shift(1)

    rows: list[dict[str, Any]] = []
    for probe in probes.itertuples(index=False):
        probe_onset = float(probe.probe_onset_time)
        block_num = int(probe.block_num)
        previous_probe = getattr(probe, "previous_probe_onset_ms")
        previous_probe = None if pd.isna(previous_probe) else float(previous_probe)
        for spec in specs:
            start_ms = probe_onset + spec.start_offset_ms
            end_ms = probe_onset + spec.end_offset_ms
            behavior_frame = trials[
                (trials["block_num"] == block_num)
                & pd.to_numeric(trials["absolute_onset_time"], errors="coerce").ge(
                    start_ms
                )
                & pd.to_numeric(trials["absolute_onset_time"], errors="coerce").lt(
                    end_ms
                )
            ]
            behavior_features = _behavior_window_features(behavior_frame)
            crosses_previous = bool(
                previous_probe is not None and start_ms <= previous_probe < end_ms
            )
            for eye in EYES:
                index = indices.get((block_num, eye))
                frame = (
                    index.slice(start_ms, end_ms) if index is not None else pd.DataFrame()
                )
                record: dict[str, Any] = {
                    "subject": probe.subject,
                    "block_num": block_num,
                    "probe_index_global": int(probe.probe_index_global),
                    "probe_index_in_block": int(probe.probe_index_in_block),
                    "probe_trial_num": int(probe.trial_num),
                    "probe_onset_ms": probe_onset,
                    "probe_response": probe.probe_response,
                    "probe_rt": probe.probe_rt,
                    "probe_vigilance": probe.probe_vigilance,
                    "probe_vigilance_rt": probe.probe_vigilance_rt,
                    "previous_probe_onset_ms": previous_probe,
                    "seconds_since_previous_probe": (
                        (probe_onset - previous_probe) / 1000.0
                        if previous_probe is not None
                        else None
                    ),
                    "window_family": spec.family,
                    "window_name": spec.name,
                    "window_start_offset_ms": spec.start_offset_ms,
                    "window_end_offset_ms": spec.end_offset_ms,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "window_crosses_previous_probe": crosses_previous,
                    "seconds_of_window_before_previous_probe": (
                        max(0.0, (previous_probe - start_ms) / 1000.0)
                        if crosses_previous and previous_probe is not None
                        else None
                    ),
                    "eye": eye,
                }
                record.update(behavior_features)
                record.update(_window_nir_features(frame))
                rows.append(record)
    return pd.DataFrame(rows)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def _git_commit(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _source_identity(
    config: Config, source: NirSource, trials: pd.DataFrame
) -> dict[str, Any]:
    behavior_files = sorted(
        {Path(value) for value in trials["source_file"].dropna().astype(str)}
    )
    return {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "pipeline_version": ALIGNMENT_PIPELINE_VERSION,
        "config_digest": config.digest,
        "nir_csv": str(source.csv_path),
        "nir_sha256": sha256(source.csv_path),
        "nir_completion": str(source.completion_path),
        "nir_extension_version": source.completion.get("extension_version"),
        "behavior_files": [
            {"path": str(path), "sha256": sha256(path)} for path in behavior_files
        ],
    }


def _completion_matches(path: Path, identity: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return current.get("status") == "complete" and current.get("identity") == identity


def run_subject_alignment(
    config: Config,
    subject: str,
    *,
    force: bool = False,
    make_diagnostics: bool = True,
) -> dict[str, Any]:
    subject = normalize_subject(subject)
    output_root = alignment_output_root(config)
    paths = subject_output_paths(output_root, subject)
    source = find_nir_source(config, subject)
    trials = load_behavior_trials(config, subject)
    trials = add_behavior_qc(
        trials,
        carryover_ms=float(
            config.section("behavior_qc").get("carryover_candidate_ms", 200)
        ),
    )
    trials = _add_trial_context(trials)

    identity = _source_identity(config, source, trials)
    if not force and _completion_matches(paths["completion"], identity):
        return {
            "subject": subject,
            "status": "skipped",
            "reason": "validated_completion",
        }

    nir = load_nir_frame(source)
    indices = build_nir_indices(nir, subject)
    missing_eye_blocks = [
        f"block{block}-{eye}"
        for block in (1, 2)
        for eye in EYES
        if (block, eye) not in indices
    ]

    trial_specs = parse_window_specs(
        config.section("windows").get("trial", []), family="trial"
    )
    probe_specs = parse_window_specs(
        config.section("windows").get("probe", []), family="probe"
    )

    trial_level = add_trial_nir_linkage(trials, indices)
    trial_windows = build_trial_windows(trial_level, indices, trial_specs)
    probe_windows = build_probe_windows(trial_level, indices, probe_specs)

    paths["subject_dir"].mkdir(parents=True, exist_ok=True)
    trial_level.to_csv(paths["trial_level"], index=False, encoding="utf-8-sig")
    trial_windows.to_csv(paths["trial_windows"], index=False, encoding="utf-8-sig")
    probe_windows.to_csv(paths["probe_windows"], index=False, encoding="utf-8-sig")

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "subject": subject,
        "created_at_utc": created_at,
        "identity": identity,
        "git_commit": _git_commit(config.path.parent.parent),
        "nir_source_run_dir": str(source.run_dir),
        "nir_source_alternatives": [str(path) for path in source.alternatives],
        "windows": {
            "trial": [spec.__dict__ for spec in trial_specs],
            "probe": [spec.__dict__ for spec in probe_specs],
        },
        "outputs": {
            "trial_level": str(paths["trial_level"]),
            "trial_windows": str(paths["trial_windows"]),
            "probe_windows": str(paths["probe_windows"]),
            "qc_dir": str(paths["qc_dir"]),
        },
        "principles": {
            "raw_behavior_scoring_overwritten": False,
            "raw_pupil_pixels_used_as_primary": False,
            "pir_primary": PIR_COLUMN,
            "pir_requires_normalization_valid": True,
            "oar_primary": OAR_COLUMN,
            "roi_clipped_is_exclusion_gate": False,
            "left_right_fused": False,
        },
    }
    _atomic_json(paths["manifest"], manifest)

    probe_count = int(
        (
            pd.to_numeric(trial_level["is_probe"], errors="coerce").eq(1)
            & pd.to_numeric(trial_level["probe_onset_time"], errors="coerce").notna()
        ).sum()
    )
    summary = {
        "subject": subject,
        "status": "complete",
        "trial_rows": int(len(trial_level)),
        "trial_window_rows": int(len(trial_windows)),
        "probe_count": probe_count,
        "probe_window_rows": int(len(probe_windows)),
        "nir_rows_block1_block2": int(len(nir)),
        "missing_eye_blocks": missing_eye_blocks,
        "behavior_qc": {
            "multiple_keypress_trials": int(trial_level["multiple_keypress_flag"].sum()),
            "prestimulus_press_trials": int(trial_level["prestimulus_press_flag"].sum()),
            "ambiguous_omission_trials": int(
                trial_level["ambiguous_omission_flag"].sum()
            ),
            "carryover_candidate_trials": int(
                trial_level["carryover_candidate_flag"].sum()
            ),
            "rt_lt_100_trials": int(trial_level["rt_candidate_lt_100_flag"].sum()),
            "rt_lt_150_trials": int(trial_level["rt_candidate_lt_150_flag"].sum()),
            "rt_lt_200_trials": int(trial_level["rt_candidate_lt_200_flag"].sum()),
        },
    }

    if make_diagnostics:
        from .diagnostics import generate_diagnostics

        summary["diagnostics"] = generate_diagnostics(
            subject,
            nir,
            trial_level,
            probe_windows,
            paths,
        )
    _atomic_json(paths["summary"], summary)
    _atomic_json(
        paths["completion"],
        {
            "status": "complete",
            "subject": subject,
            "identity": identity,
            "created_at_utc": created_at,
            "required_artifacts": [
                str(paths["trial_level"]),
                str(paths["trial_windows"]),
                str(paths["probe_windows"]),
                str(paths["manifest"]),
                str(paths["summary"]),
            ],
        },
    )
    return summary
