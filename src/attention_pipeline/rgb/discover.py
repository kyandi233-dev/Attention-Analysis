from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from attention_pipeline.config import Config


_SUBJECT_RE = re.compile(r"^sub-(\d+)$")


@dataclass(frozen=True)
class RGBSubjectFiles:
    subject: str
    root: Path
    subject_dir: Path
    video: Path
    timestamps: Path
    behavior_dir: Path
    master_timeline: Path
    block1_behavior: Path | None
    block2_behavior: Path | None


def subject_number(subject: str) -> int:
    match = _SUBJECT_RE.fullmatch(subject.rstrip("_"))
    if not match:
        raise ValueError(f"Invalid numeric subject id: {subject}")
    return int(match.group(1))


def _existing_roots(config: Config) -> list[Path]:
    roots = []
    for raw in config.section("data").get("roots", []):
        path = Path(str(raw))
        if path.exists():
            roots.append(path)
    return roots


def _first_or_none(paths: list[Path]) -> Path | None:
    return paths[0] if paths else None


def discover_rgb_subjects(config: Config) -> tuple[list[RGBSubjectFiles], dict[str, list[Path]]]:
    """Discover formal RGB recordings without silently resolving duplicates.

    Returns one record for subjects found exactly once and a mapping of duplicate
    subject ids to their candidate subject directories.
    """
    data = config.section("data")
    min_subject = int(data.get("min_subject_number", 0))
    video_pattern = str(data.get("rgb_video_pattern", "sub-*_/rgb/*_rgb.avi"))
    timestamp_suffix = str(data.get("rgb_timestamp_suffix", "_rgb_timestamps.csv"))
    behavior_dir_name = str(data.get("behavior_dir", "beh"))
    timeline_name = str(data.get("master_timeline", "master_timeline.csv"))

    candidates: dict[str, list[tuple[Path, Path]]] = {}
    for root in _existing_roots(config):
        for video in sorted(root.glob(video_pattern)):
            try:
                subject_dir = video.parent.parent
                subject = subject_dir.name.rstrip("_")
                if subject_number(subject) < min_subject:
                    continue
            except ValueError:
                continue
            candidates.setdefault(subject, []).append((root, video))

    duplicates = {
        subject: [video.parent.parent for _, video in entries]
        for subject, entries in candidates.items()
        if len(entries) > 1
    }

    records: list[RGBSubjectFiles] = []
    for subject, entries in sorted(candidates.items(), key=lambda item: subject_number(item[0])):
        if len(entries) != 1:
            continue
        root, video = entries[0]
        subject_dir = video.parent.parent
        behavior_dir = subject_dir / behavior_dir_name
        timestamps = video.with_name(f"{video.stem}{timestamp_suffix.removeprefix('_rgb')}")
        # The config suffix is expressed as _rgb_timestamps.csv; for a stem such
        # as sub-031_rgb the direct sibling name is sub-031_rgb_timestamps.csv.
        timestamps = video.with_name(f"{video.stem}_timestamps.csv")
        block1 = sorted(behavior_dir.glob(f"{subject}_Block1_*_beh.csv"))
        block2 = sorted(behavior_dir.glob(f"{subject}_Block2_*_beh.csv"))
        records.append(
            RGBSubjectFiles(
                subject=subject,
                root=root,
                subject_dir=subject_dir,
                video=video,
                timestamps=timestamps,
                behavior_dir=behavior_dir,
                master_timeline=behavior_dir / timeline_name,
                block1_behavior=_first_or_none(block1),
                block2_behavior=_first_or_none(block2),
            )
        )
    return records, duplicates
