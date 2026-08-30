"""Build the machine-local canonical cohort and staged NIR source manifest.

The registered-session CSV remains row-complete. ``include`` is derived from
the presence of the verified anonymous repeat-participant key; modality
availability is never used to define the cohort. NIR sources are then attached
as an availability layer for included sessions only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_SCHEMAS = {6, 7}


def _atomic_text(path: Path, text: str, *, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _nested(value: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for keys in paths:
        current: Any = value
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current not in (None, ""):
            return current
    return None


def _canonical_session(value: str) -> str:
    text = str(value).strip().rstrip("_")
    if not text.startswith("sub-"):
        raise ValueError(f"invalid session_id: {value!r}")
    return text


def build_cohort(
    source: Path,
    *,
    expected_sessions: int,
    expected_groups: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"session_id", "include", "repeat_participant_id", "identity_status"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"registered cohort must contain {sorted(required)}")

    seen: set[str] = set()
    governed: list[dict[str, str]] = []
    for raw in rows:
        session = _canonical_session(raw["session_id"])
        if session in seen:
            raise ValueError(f"duplicate session_id: {session}")
        seen.add(session)
        repeat_id = str(raw.get("repeat_participant_id", "")).strip()
        identity_status = str(raw.get("identity_status", "")).strip()
        include = bool(repeat_id)
        if include and not identity_status:
            raise ValueError(f"included session lacks identity_status: {session}")
        governed.append(
            {
                "session_id": session,
                "include": "true" if include else "false",
                "repeat_participant_id": repeat_id,
                "identity_status": identity_status,
            }
        )

    included = [row for row in governed if row["include"] == "true"]
    group_counts = Counter(row["repeat_participant_id"] for row in included)
    if len(included) != expected_sessions:
        raise ValueError(f"included sessions={len(included)} expected={expected_sessions}")
    if len(group_counts) != expected_groups:
        raise ValueError(f"participant groups={len(group_counts)} expected={expected_groups}")
    summary = {
        "registered_sessions": len(governed),
        "included_sessions": len(included),
        "participant_groups": len(group_counts),
        "group_size_distribution": dict(sorted(Counter(group_counts.values()).items())),
        "exactly_two_session_groups": sum(size == 2 for size in group_counts.values()),
        "repeated_participant_groups": sum(size > 1 for size in group_counts.values()),
    }
    return governed, summary


def _cohort_csv(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=["session_id", "include", "repeat_participant_id", "identity_status"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _find_eye_metrics(session_dir: Path) -> Path:
    candidates = [
        session_dir / "data" / "eye_metrics.csv",
        session_dir / "data" / "eye_metrics.csv.gz",
    ]
    existing = [path.resolve() for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"{session_dir}: expected one eye_metrics CSV[.gz], found {len(existing)}")
    return existing[0]


def build_nir_manifest(
    cohort_rows: list[dict[str, str]],
    roots: list[Path],
    *,
    expected_nir_sessions: int,
    cohort_path: Path,
    history_roots: list[Path] | None = None,
) -> dict[str, Any]:
    included = {row["session_id"]: row for row in cohort_rows if row["include"] == "true"}
    group_counts = Counter(row["repeat_participant_id"] for row in included.values())
    discovered: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    incompatible: dict[str, str] = {}
    archived_incompatible: dict[str, str] = {}

    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for session_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            session = _canonical_session(session_dir.name)
            if session not in included:
                continue
            if session in discovered:
                raise ValueError(
                    f"duplicate authoritative NIR source for {session}: "
                    f"{discovered[session][0]} and {session_dir}"
                )
            manifest_path = session_dir / "manifest.json"
            completion_path = session_dir / "completion.json"
            if not manifest_path.is_file() or not completion_path.is_file():
                legacy_completion = session_dir / "completion.json"
                legacy_eye_table = session_dir / "eyes.csv"
                if legacy_completion.is_file() and legacy_eye_table.is_file():
                    incompatible[session] = "legacy_pre_schema6_7_source_requires_reprocessing"
                    continue
                raise FileNotFoundError(f"{session_dir}: missing manifest.json or completion.json")
            completion = _read_json(completion_path)
            if str(completion.get("status", "")).lower() != "complete":
                raise ValueError(f"{session}: completion status is not complete")
            manifest = _read_json(manifest_path)
            discovered[session] = (session_dir.resolve(), _find_eye_metrics(session_dir), manifest)

    if len(discovered) != expected_nir_sessions:
        raise ValueError(f"NIR sessions={len(discovered)} expected={expected_nir_sessions}")

    sessions: list[dict[str, Any]] = []
    for session in sorted(discovered, key=lambda value: int(value.split("-")[1])):
        session_dir, eye_metrics, manifest = discovered[session]
        schema = int(
            _nested(
                manifest,
                ("work_identity", "eye_metrics_schema_version"),
                ("scientific_identity", "eye_metrics_schema_version"),
            )
        )
        if schema not in ALLOWED_SCHEMAS:
            raise ValueError(f"{session}: unsupported eye_metrics schema {schema}")
        branch = _nested(
            manifest,
            ("work_identity", "git_branch"),
            ("scientific_identity", "git_branch"),
            ("provenance_identity", "git_branch"),
        )
        commit = _nested(
            manifest,
            ("work_identity", "git_commit"),
            ("scientific_identity", "git_commit"),
            ("provenance_identity", "git_commit"),
        )
        cohort_row = included[session]
        sessions.append(
            {
                "session_id": session,
                "analysis_group_token": cohort_row["repeat_participant_id"],
                "repeat_group_size": group_counts[cohort_row["repeat_participant_id"]],
                "source_schema_version": schema,
                "source_kind": "ritnet-fullclass-pupil-only",
                "source_csv": str(eye_metrics),
                "source_manifest": str((session_dir / "manifest.json").resolve()),
                "source_completion": str((session_dir / "completion.json").resolve()),
                "source_branch": branch,
                "source_commit": commit,
                "source_selection_reason": "unique_complete_authoritative_source",
            }
        )

    # Keep a reproducible distinction between a source that was never found
    # and a verified legacy source deliberately moved out of the active roots.
    # ``history_roots`` is opt-in so ordinary runs never scan archive trees.
    if history_roots:
        for session in sorted(set(included) - set(discovered) - set(incompatible), key=lambda value: int(value.split("-")[1])):
            for history_root in history_roots:
                history_root = history_root.resolve()
                if not history_root.is_dir():
                    continue
                for candidate in sorted(path for path in history_root.rglob(session) if path.is_dir()):
                    if (candidate / "eyes.csv").is_file() and (candidate / "completion.json").is_file():
                        archived_incompatible[session] = str(candidate)
                        break
                if session in archived_incompatible:
                    break

    unavailable = []
    for session in sorted(set(included) - set(discovered), key=lambda value: int(value.split("-")[1])):
        if session in archived_incompatible:
            unavailable.append(
                {
                    "session_id": session,
                    "status": "incompatible_archived",
                    "reason": "verified_legacy_source_moved_out_of_active_final",
                    "historical_source_dir": archived_incompatible[session],
                }
            )
            continue
        unavailable.append(
            {
                "session_id": session,
                "status": "incompatible" if session in incompatible else "source_missing",
                "reason": incompatible.get(session, "no_authoritative_complete_NIR_source"),
            }
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_manifest": str(cohort_path.resolve()),
        "cohort_rule": "modality-independent identity-resolved canonical cohort",
        "availability_rule": "NIR source missing remains explicit and does not change cohort membership",
        "source_roots": [str(root.resolve()) for root in roots],
        "session_count": len(sessions),
        "unavailable_session_count": len(unavailable),
        "unavailable_sessions": unavailable,
        "sessions": sessions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-cohort", type=Path, required=True)
    parser.add_argument("--cohort-output", type=Path, required=True)
    parser.add_argument("--nir-source-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--nir-history-root",
        type=Path,
        action="append",
        default=[],
        help="optional archive root used only to label verified legacy sources moved out of active roots",
    )
    parser.add_argument("--nir-output", type=Path, required=True)
    parser.add_argument("--expected-sessions", type=int, required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--expected-nir-sessions", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cohort, cohort_summary = build_cohort(
        args.registered_cohort,
        expected_sessions=args.expected_sessions,
        expected_groups=args.expected_groups,
    )
    nir = build_nir_manifest(
        cohort,
        args.nir_source_root,
        expected_nir_sessions=args.expected_nir_sessions,
        cohort_path=args.cohort_output,
        history_roots=args.nir_history_root,
    )
    _atomic_text(args.cohort_output, _cohort_csv(cohort), overwrite=args.overwrite)
    _atomic_text(
        args.nir_output,
        json.dumps(nir, ensure_ascii=False, indent=2) + "\n",
        overwrite=args.overwrite,
    )
    print(json.dumps({"cohort": cohort_summary, "nir_sessions": nir["session_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
