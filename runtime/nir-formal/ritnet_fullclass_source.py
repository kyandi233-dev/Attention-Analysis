"""Strict historical-formal source loading for final RITnet full-class analysis."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from formal_completion import validate_completion
from ritnet_fullclass_contract import normalize_subject
from ritnet_fullclass_coverage import load_source_frames
from ritnet_label_store import sha256_file


ALLOWED_EYE_LABELS = frozenset({"frame_left", "frame_right"})
REQUIRED_SOURCE_EYE_FIELDS = frozenset(
    {
        "phase",
        "phase_segment",
        "frame_idx",
        "video_time_ms",
        "unix_ms",
        "phase_time_ms",
        "eye",
        "source",
        "redetect_reason",
        "frame_status",
        "status",
        "anchor_yolo_confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "yolo_batch_size",
    }
)


@dataclass(frozen=True)
class SourceFormalContext:
    run_dir: Path
    subject: str
    completion: dict[str, Any]
    config: dict[str, Any]
    video: Path
    video_resolution: dict[str, Any]
    eye_fields: tuple[str, ...]
    eye_rows: tuple[dict[str, str], ...]
    frame_rows: tuple[dict[str, str], ...]
    source_identity: dict[str, Any]


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _finite_float(value: Any, *, name: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} at source eye row {row_number}: {value!r}") from exc
    if not (number == number and abs(number) != float("inf")):
        raise ValueError(f"non-finite {name} at source eye row {row_number}: {value!r}")
    return number


def _int(value: Any, name: str) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing {name}")
    return int(float(value))


def load_source_eye_rows(path: Path, subject: str) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = REQUIRED_SOURCE_EYE_FIELDS - set(fields)
        if missing:
            raise ValueError(f"eyes.csv missing final-source columns: {sorted(missing)}")
        rows = tuple(dict(row) for row in reader)
    if not rows:
        raise ValueError(f"eyes.csv contains no source eye rows: {path}")

    seen: set[tuple[str, int, int, str]] = set()
    by_frame: dict[tuple[str, int, int], dict[str, tuple[float, int]]] = defaultdict(dict)
    previous_frame = -1
    for ordinal, row in enumerate(rows, start=2):
        row_subject = normalize_subject(row.get("subject") or subject)
        if row_subject != subject:
            raise ValueError(f"mixed subject at eyes.csv line {ordinal}: {row_subject} != {subject}")
        phase = str(row.get("phase") or "")
        segment = _int(row.get("phase_segment"), "phase_segment")
        frame = _int(row.get("frame_idx"), "frame_idx")
        if frame < previous_frame:
            raise ValueError("eyes.csv frame_idx must be globally nondecreasing")
        previous_frame = frame
        eye = str(row.get("eye") or "").strip()
        if eye not in ALLOWED_EYE_LABELS:
            raise ValueError(f"unsupported eye label at eyes.csv line {ordinal}: {eye!r}")
        key = (phase, segment, frame, eye)
        if key in seen:
            raise ValueError(f"duplicate source eye key: {key}")
        seen.add(key)

        x1 = _finite_float(row.get("bbox_x1"), name="bbox_x1", row_number=ordinal)
        y1 = _finite_float(row.get("bbox_y1"), name="bbox_y1", row_number=ordinal)
        x2 = _finite_float(row.get("bbox_x2"), name="bbox_x2", row_number=ordinal)
        y2 = _finite_float(row.get("bbox_y2"), name="bbox_y2", row_number=ordinal)
        confidence = _finite_float(
            row.get("anchor_yolo_confidence"),
            name="anchor_yolo_confidence",
            row_number=ordinal,
        )
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"invalid source YOLO bbox at eyes.csv line {ordinal}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"YOLO confidence outside [0,1] at eyes.csv line {ordinal}")
        frame_key = (phase, segment, frame)
        by_frame[frame_key][eye] = ((x1 + x2) / 2.0, ordinal)

    for frame_key, pair in by_frame.items():
        if set(pair) == ALLOWED_EYE_LABELS:
            left_x, left_line = pair["frame_left"]
            right_x, right_line = pair["frame_right"]
            if not left_x < right_x:
                raise ValueError(
                    "source left/right identity violation: frame_left center must be left of frame_right; "
                    f"frame={frame_key}, lines={left_line}/{right_line}, centers={left_x}/{right_x}"
                )
    return fields, rows


def _video_candidates(config: dict[str, Any], subject: str, original: Path) -> list[Path]:
    names = {original.name, f"{subject}_nir.avi"}
    result: set[Path] = set()
    for root_text in config.get("data", {}).get("roots", []):
        root = Path(str(root_text))
        for name in names:
            candidate = root / f"{subject}_" / "nir" / name
            if candidate.is_file():
                result.add(candidate.resolve())
    return sorted(result, key=lambda path: str(path).lower())


def resolve_source_video(
    *,
    completion: dict[str, Any],
    config: dict[str, Any],
    subject: str,
) -> tuple[Path, dict[str, Any]]:
    original_text = str(completion.get("video") or "").strip()
    if not original_text:
        raise ValueError("source completion has no video path")
    original = Path(original_text)
    if original.is_file():
        path = original.resolve()
        return path, {
            "original_path": original_text,
            "resolved_path": str(path),
            "resolution_reason": "completion_path_available",
            "content_sha256": sha256_file(path),
            "candidate_count": 1,
        }

    candidates = _video_candidates(config, subject, original)
    if not candidates:
        raise FileNotFoundError(
            f"source video unavailable and not rediscovered for {subject}: {original_text}"
        )
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        by_hash[sha256_file(path)].append(path)
    if len(by_hash) != 1:
        detail = {digest: [str(path) for path in paths] for digest, paths in sorted(by_hash.items())}
        raise RuntimeError(
            "ambiguous rediscovered source videos with different SHA256: "
            + json.dumps(detail, ensure_ascii=False, sort_keys=True)
        )
    digest, identical = next(iter(by_hash.items()))
    path = sorted(identical, key=lambda value: str(value).lower())[0]
    return path, {
        "original_path": original_text,
        "resolved_path": str(path),
        "resolution_reason": "rediscovered_identical_content",
        "content_sha256": digest,
        "candidate_count": len(candidates),
        "equivalent_candidates": [str(value) for value in identical],
    }


@lru_cache(maxsize=16)
def _load_source_context_cached(run_dir: Path, config_path: Path) -> SourceFormalContext:
    """Load one immutable formal source identity once per Python process.

    The canonical single-subject entrypoint performs a strict preflight before
    calling the numeric core. Both stages need the same source context, including
    the expensive source-video SHA256. The historical source is immutable during
    one final run, so reusing the validated context avoids hashing a large AVI a
    second time without weakening cross-process integrity checks.
    """
    validation = validate_completion(run_dir)
    if not validation.valid or not validation.marker:
        raise RuntimeError(
            f"source formal completion validation failed: {run_dir} :: {validation.reason}"
        )
    completion = dict(validation.marker)
    config = load_config(config_path)
    subject = normalize_subject(completion.get("subject") or run_dir.name.split("_formal_", 1)[0])

    configured_yolo_batch = int(config.get("yolo", {}).get("batch_size", 8))
    source_yolo_batch = int(completion.get("yolo_batch_size", -1))
    if source_yolo_batch != configured_yolo_batch:
        raise RuntimeError(
            f"source formal YOLO batch {source_yolo_batch} != configured production batch {configured_yolo_batch}"
        )
    if not completion.get("yolo_model_sha256"):
        raise RuntimeError("source completion lacks yolo_model_sha256")

    video, video_resolution = resolve_source_video(
        completion=completion,
        config=config,
        subject=subject,
    )
    eyes_path = run_dir / "eyes.csv"
    frames_path = run_dir / "frames.csv"
    completion_path = run_dir / "completion.json"
    eye_fields, eye_rows = load_source_eye_rows(eyes_path, subject)
    frame_rows = tuple(load_source_frames(frames_path, subject))

    source_identity = {
        "subject": subject,
        "source_run_id": completion.get("run_id"),
        "source_completion_sha256": sha256_file(completion_path),
        "source_eyes_sha256": sha256_file(eyes_path),
        "source_frames_sha256": sha256_file(frames_path),
        "source_video_sha256": video_resolution["content_sha256"],
        "source_yolo_model_sha256": completion.get("yolo_model_sha256"),
        "source_yolo_batch_size": source_yolo_batch,
        "source_focuswave_release": completion.get("focuswave_release"),
        "source_expected_frames": completion.get("expected_frames"),
        "source_phases": completion.get("phases"),
    }
    return SourceFormalContext(
        run_dir=run_dir,
        subject=subject,
        completion=completion,
        config=config,
        video=video,
        video_resolution=video_resolution,
        eye_fields=eye_fields,
        eye_rows=eye_rows,
        frame_rows=frame_rows,
        source_identity=source_identity,
    )


def load_source_context(run_dir: Path, config_path: Path) -> SourceFormalContext:
    run_dir = Path(run_dir).resolve()
    config_path = Path(config_path).resolve()
    return _load_source_context_cached(run_dir, config_path)
