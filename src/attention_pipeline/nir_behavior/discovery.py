from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..behavior_formal import extract as behavior_extract
from ..config import Config, load_config
from ..formal_analysis.behavior_adapter import prepare_behavior_runtime_config
from .contract import FULLCLASS_EXTENSION_VERSION, normalize_subject, parse_subject_list


@dataclass(frozen=True)
class NirSource:
    subject: str
    csv_path: Path
    completion_path: Path
    run_dir: Path
    completion: dict[str, Any]
    alternatives: tuple[Path, ...]


def repo_root(config: Config) -> Path:
    return config.path.parent.parent


def resolve_repo_path(config: Config, raw: str | Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = (repo_root(config) / path).resolve()
    return path


def alignment_output_root(config: Config) -> Path:
    raw = config.section("paths").get("output_root")
    if raw is None:
        raise KeyError("alignment config missing paths.output_root")
    return resolve_repo_path(config, raw)


def behavior_config(config: Config) -> Config:
    """Load Behavior science config while preserving the caller's path registry.

    NIR must not silently fall back to environment-only path discovery when the
    parent NIR command was given an explicit --paths-config registry.
    """
    raw = config.section("paths").get("behavior_config")
    if raw is None:
        raise KeyError("alignment config missing paths.behavior_config")
    loaded = load_config(resolve_repo_path(config, raw), use_env_paths=False)
    if config.path_registry is not None:
        loaded = replace(loaded, path_registry=config.path_registry)
    return loaded


def nir_source_roots(config: Config) -> list[Path]:
    raw_roots = config.section("paths").get("nir_source_roots", [])
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ValueError("alignment config requires paths.nir_source_roots")
    roots = [resolve_repo_path(config, value) for value in raw_roots]
    return roots


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _completion_candidates(config: Config, subject: str) -> list[tuple[Path, dict[str, Any]]]:
    subject = normalize_subject(subject)
    result: list[tuple[Path, dict[str, Any]]] = []
    pattern = f"{subject}_formal_*/*_ritnet_fullclass_v1-2-fast-qc_completion.json"
    for root in nir_source_roots(config):
        if not root.is_dir():
            continue
        for path in sorted(root.glob(pattern)):
            try:
                marker = load_json(path)
            except Exception:
                continue
            try:
                marker_subject = normalize_subject(marker.get("subject", subject))
            except ValueError:
                continue
            if marker_subject != subject:
                continue
            if marker.get("status") != "complete":
                continue
            if marker.get("extension_version") != FULLCLASS_EXTENSION_VERSION:
                continue
            if bool(marker.get("pupil_validation_mode")):
                continue
            csv_path = path.with_name(path.name.replace("_completion.json", ".csv"))
            if not csv_path.is_file():
                continue
            result.append((path, marker))
    return result


def _rank_candidate(path: Path, marker: dict[str, Any]) -> tuple[int, int, int]:
    run_name = path.parent.name.lower()
    production_yolo_b8 = int("yolo-b8" in run_name or "yolo_b8" in run_name)
    labels_only = int(bool(marker.get("labels_only", True)))
    mtime_ns = path.stat().st_mtime_ns
    return production_yolo_b8, labels_only, mtime_ns


def find_nir_source(config: Config, subject: str) -> NirSource:
    subject = normalize_subject(subject)
    candidates = _completion_candidates(config, subject)
    if not candidates:
        roots = ", ".join(str(path) for path in nir_source_roots(config))
        raise FileNotFoundError(
            f"{subject}: no completed {FULLCLASS_EXTENSION_VERSION} production output found under {roots}"
        )
    ranked = sorted(candidates, key=lambda item: _rank_candidate(*item), reverse=True)
    selected_path, selected_marker = ranked[0]
    csv_path = selected_path.with_name(selected_path.name.replace("_completion.json", ".csv"))
    alternatives = tuple(path.parent for path, _ in ranked[1:])
    return NirSource(
        subject=subject,
        csv_path=csv_path.resolve(),
        completion_path=selected_path.resolve(),
        run_dir=selected_path.parent.resolve(),
        completion=selected_marker,
        alternatives=alternatives,
    )


def discover_nir_subjects(config: Config) -> list[str]:
    subjects: set[str] = set()
    for root in nir_source_roots(config):
        if not root.is_dir():
            continue
        for path in root.glob("sub-*_formal_*/*_ritnet_fullclass_v1-2-fast-qc_completion.json"):
            try:
                marker = load_json(path)
                if marker.get("status") != "complete":
                    continue
                if marker.get("extension_version") != FULLCLASS_EXTENSION_VERSION:
                    continue
                if bool(marker.get("pupil_validation_mode")):
                    continue
                subjects.add(
                    normalize_subject(
                        marker.get("subject", path.parent.name.split("_formal_", 1)[0])
                    )
                )
            except Exception:
                continue
    return sorted(subjects, key=lambda x: int(x.split("-")[1]))


def selected_subjects(config: Config, override: list[str] | None = None) -> list[str]:
    if override:
        return parse_subject_list(override)
    raw = config.section("subjects").get("include", [])
    if raw:
        return parse_subject_list(raw)
    return discover_nir_subjects(config)


def load_behavior_trials(config: Config, subject: str):
    """Read NIR-linked Behavior trials through the authoritative Behavior v2 runtime."""
    bconfig = behavior_config(config)
    runtime, cohort = prepare_behavior_runtime_config(bconfig)
    session = normalize_subject(subject)
    governed = set(cohort.loc[cohort["include"].eq(True), "session_id"].astype(str))
    if session not in governed:
        raise ValueError(f"{session}: NIR-linked Behavior request is outside governed cohort")
    return behavior_extract.extract_formal_trials(runtime, session)
