"""Canonical single-subject RITnet full-class runner.

This is the only user-facing single-subject gate. Before the implementation is
allowed to run it strictly validates the historical formal source, resolves the
source AVI even when Windows drive letters have changed, enforces source-video
content hashing, validates subject/eye identity, and requires a clean Git tree.

The implementation behind this gate is being migrated to the final <=1 GiB
contract; this preflight deliberately fails closed rather than silently accepting
weak or ambiguous historical inputs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import run_ritnet_fullclass_native_extension as implementation
from formal_completion import validate_completion
from ritnet_label_store import sha256_file

PACKAGE_ROOT = Path(__file__).resolve().parent
ALLOWED_EYE_LABELS = frozenset({"frame_left", "frame_right"})
REQUIRED_YOLO_SOURCE_FIELDS = frozenset(
    {
        "frame_idx",
        "eye",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "anchor_yolo_confidence",
    }
)


def _arg_path(flag: str, default: Path | None = None) -> Path:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        if default is None:
            raise SystemExit(f"Canonical full-class run requires {flag}")
        return default.resolve()
    if index + 1 >= len(sys.argv) or str(sys.argv[index + 1]).startswith("--"):
        raise SystemExit(f"{flag} requires a path value")
    return Path(sys.argv[index + 1]).resolve()


def _enforce_canonical_provenance() -> None:
    if "--allow-model-mismatch" in sys.argv:
        raise SystemExit(
            "Canonical full-class runs do not permit --allow-model-mismatch. "
            "Use the frozen model/config that matches the source evidence."
        )
    if "--hash-video" not in sys.argv:
        sys.argv.append("--hash-video")
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PACKAGE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        raise SystemExit("Canonical full-class run requires a readable Git worktree") from exc
    if dirty:
        raise SystemExit(
            "Canonical full-class run requires a clean Git worktree so the recorded commit "
            "fully identifies the executed code. Commit/stash local changes first."
        )


def _strict_source_completion(run_dir: Path) -> dict[str, Any]:
    validation = validate_completion(run_dir)
    if not validation.valid or not validation.marker:
        raise SystemExit(
            "Historical formal source failed strict completion validation: "
            f"{run_dir} :: {validation.reason}"
        )
    return dict(validation.marker)


def _candidate_video_paths(
    *,
    config: dict[str, Any],
    subject: str,
    original_video: Path,
) -> list[Path]:
    names = {original_video.name, f"{subject}_nir.avi"}
    candidates: set[Path] = set()
    for root_text in config.get("data", {}).get("roots", []):
        root = Path(str(root_text))
        for name in names:
            candidate = root / f"{subject}_" / "nir" / name
            if candidate.is_file():
                candidates.add(candidate.resolve())
    return sorted(candidates, key=lambda value: str(value).lower())


def _resolve_source_video(
    *,
    marker: dict[str, Any],
    config: dict[str, Any],
    subject: str,
) -> tuple[Path, dict[str, Any]]:
    raw = str(marker.get("video") or "").strip()
    if not raw:
        raise SystemExit("Historical formal completion is missing source video path")
    original = Path(raw)
    if original.is_file():
        resolved = original.resolve()
        digest = sha256_file(resolved)
        return resolved, {
            "original_path": raw,
            "resolved_path": str(resolved),
            "resolution_reason": "completion_path_available",
            "content_sha256": digest,
            "candidate_count": 1,
        }

    candidates = _candidate_video_paths(
        config=config,
        subject=subject,
        original_video=original,
    )
    if not candidates:
        raise SystemExit(
            "Source video path from historical completion is unavailable and no current data-root "
            f"candidate was found for {subject}: original={raw!r}"
        )

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for candidate in candidates:
        by_hash[sha256_file(candidate)].append(candidate)
    if len(by_hash) != 1:
        detail = {
            digest: [str(path) for path in paths]
            for digest, paths in sorted(by_hash.items())
        }
        raise SystemExit(
            "Multiple current source-video candidates have different content; refusing ambiguous "
            f"drive-letter rediscovery: {json.dumps(detail, ensure_ascii=False, sort_keys=True)}"
        )

    digest, identical_paths = next(iter(by_hash.items()))
    resolved = sorted(identical_paths, key=lambda value: str(value).lower())[0]
    return resolved, {
        "original_path": raw,
        "resolved_path": str(resolved),
        "resolution_reason": "rediscovered_by_subject_filename_and_identical_sha256",
        "content_sha256": digest,
        "candidate_count": len(candidates),
        "equivalent_candidates": [str(path) for path in identical_paths],
    }


def _install_completion_video_guard(
    *,
    run_dir: Path,
    validated_marker: dict[str, Any],
    resolved_video: Path,
    resolution: dict[str, Any],
) -> None:
    """Supply a relocated video path in memory without modifying completion.json."""
    original_load_json = implementation.load_json
    completion_path = (run_dir / "completion.json").resolve()

    def guarded_load_json(path: Path):
        path = Path(path).resolve()
        if path == completion_path:
            marker = dict(validated_marker)
            marker["video"] = str(resolved_video)
            marker["fullclass_source_video_resolution"] = dict(resolution)
            return marker
        return original_load_json(path)

    implementation.load_json = guarded_load_json


def _as_float(row: dict[str, Any], key: str, ordinal: int) -> float:
    value = row.get(key)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key} at source eye row {ordinal}: {value!r}") from exc
    if not (result == result and abs(result) != float("inf")):
        raise ValueError(f"non-finite {key} at source eye row {ordinal}: {value!r}")
    return result


def _install_subject_identity_guard() -> None:
    """Validate source YOLO/eye identity without modifying the historical CSV."""
    original = implementation._source_rows

    def source_rows_with_subject(path: Path, subject: str):
        fields, rows = original(path, subject)
        missing = REQUIRED_YOLO_SOURCE_FIELDS - set(fields)
        if missing:
            raise ValueError(
                "historical eyes.csv lacks YOLO source fields required by the new full-class contract: "
                + ", ".join(sorted(missing))
            )
        if "subject" not in fields:
            fields = ["subject", *fields]

        normalized_rows = []
        by_frame: dict[int, dict[str, tuple[float, int]]] = defaultdict(dict)
        for ordinal, row in enumerate(rows):
            copied = dict(row)
            row_subject = implementation.normalize_subject(copied.get("subject") or subject)
            if row_subject != subject:
                raise ValueError(
                    f"mixed subjects in source eyes.csv at row {ordinal}: "
                    f"{row_subject} != {subject}"
                )
            eye = str(copied.get("eye") or "").strip()
            if eye not in ALLOWED_EYE_LABELS:
                raise ValueError(
                    f"unsupported eye label at source row {ordinal}: {eye!r}; "
                    f"allowed={sorted(ALLOWED_EYE_LABELS)}"
                )
            frame_idx = implementation.parse_int(copied.get("frame_idx"))
            x1 = _as_float(copied, "bbox_x1", ordinal)
            y1 = _as_float(copied, "bbox_y1", ordinal)
            x2 = _as_float(copied, "bbox_x2", ordinal)
            y2 = _as_float(copied, "bbox_y2", ordinal)
            confidence = _as_float(copied, "anchor_yolo_confidence", ordinal)
            if not (x1 < x2 and y1 < y2):
                raise ValueError(
                    f"invalid YOLO bbox at source row {ordinal}: {(x1, y1, x2, y2)}"
                )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"YOLO confidence outside [0,1] at source row {ordinal}: {confidence}"
                )
            center_x = (x1 + x2) / 2.0
            if eye in by_frame[frame_idx]:
                raise ValueError(f"duplicate {eye} identity at frame {frame_idx}")
            by_frame[frame_idx][eye] = (center_x, ordinal)
            copied["subject"] = subject
            normalized_rows.append(copied)

        for frame_idx, pair in by_frame.items():
            if set(pair) == ALLOWED_EYE_LABELS:
                left_x, left_ordinal = pair["frame_left"]
                right_x, right_ordinal = pair["frame_right"]
                if not left_x < right_x:
                    raise ValueError(
                        "source eye identity/order violation: frame_left must be left of frame_right "
                        f"in image coordinates; frame={frame_idx}, rows={left_ordinal}/{right_ordinal}, "
                        f"centers={left_x}/{right_x}"
                    )
        return fields, normalized_rows

    implementation._source_rows = source_rows_with_subject


def _install_strict_source_preflight() -> None:
    run_dir = _arg_path("--run-dir")
    config_path = _arg_path("--config", PACKAGE_ROOT / "config.yaml")
    marker = _strict_source_completion(run_dir)
    subject = implementation.normalize_subject(
        marker.get("subject") or run_dir.name.split("_formal_", 1)[0]
    )
    config = implementation.load_config(config_path)
    video, resolution = _resolve_source_video(
        marker=marker,
        config=config,
        subject=subject,
    )
    _install_completion_video_guard(
        run_dir=run_dir,
        validated_marker=marker,
        resolved_video=video,
        resolution=resolution,
    )
    _install_subject_identity_guard()


if __name__ == "__main__":
    _enforce_canonical_provenance()
    _install_strict_source_preflight()
    raise SystemExit(implementation.main())
