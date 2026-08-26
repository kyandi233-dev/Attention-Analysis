from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

from attention_pipeline.config import Config, load_config
from attention_pipeline.rgb.motion import run_motion_test
from attention_pipeline.rgb.paths import RGBOutputLayout
from attention_pipeline.rgb.pose import run_pose_test


def _formal_engine_config(config: Config, subject: str) -> Config:
    """Reuse validated Motion/Pose engines while routing their test names into the subject dir."""
    data = copy.deepcopy(config.data)
    output = data.setdefault("output", {})
    output["test_dir"] = subject
    return Config(path=config.path, data=data, digest=config.digest)


def _complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("completion_status") == "complete"
    except Exception:
        return False


def _archive_existing(paths: list[Path], *, subject_dir: Path, label: str) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = subject_dir / "_superseded" / f"{stamp}-{label}"
    archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = archive / path.name
        suffix = 1
        while destination.exists():
            destination = archive / f"{path.stem}-{suffix}{path.suffix}"
            suffix += 1
        path.replace(destination)


def _guard(paths: list[Path], *, force: bool, label: str, subject_dir: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise RuntimeError(
            f"{label} output already/partly exists. Inspect first or rerun with --force: {existing}"
        )
    if force:
        _archive_existing(paths, subject_dir=subject_dir, label=label.lower().replace(" ", "-"))


def run_motion_formal(config: Config, subject: str, *, force: bool = False) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    final_raw = layout.subject_file(subject, "motion_raw.parquet")
    final_manifest = layout.subject_file(subject, "motion_manifest.json")
    if not force and final_raw.is_file() and _complete(final_manifest):
        return {"status": "skipped_complete", "subject": subject, "raw": str(final_raw)}

    engine_config = _formal_engine_config(config, subject)
    subject_dir = layout.subject_dir(subject)
    temp_raw = subject_dir / f"{subject}_motion-test.parquet"
    temp_manifest = subject_dir / f"{subject}_motion-test_manifest.json"
    _guard(
        [temp_raw, temp_manifest, final_raw, final_manifest],
        force=force,
        label="Motion formal",
        subject_dir=subject_dir,
    )

    run_motion_test(engine_config, subject)
    if not temp_raw.is_file() or not temp_manifest.is_file():
        raise RuntimeError("Validated Motion engine did not produce expected outputs")

    manifest = json.loads(temp_manifest.read_text(encoding="utf-8"))
    temp_raw.replace(final_raw)
    temp_manifest.replace(final_manifest)
    manifest["stage"] = "motion-formal"
    manifest["output_mode"] = "formal"
    manifest["completion_status"] = "complete"
    manifest["engine_origin"] = (
        "validated full-span run_motion_test implementation; formal wrapper changes routing/naming only"
    )
    if isinstance(manifest.get("output"), dict):
        manifest["output"]["parquet"] = str(final_raw)
        manifest["output"]["manifest"] = str(final_manifest)
    else:
        manifest["output"] = {"parquet": str(final_raw), "manifest": str(final_manifest)}
    final_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_pose_formal(config: Config, subject: str, *, force: bool = False) -> dict[str, object]:
    """Extract Pose landmark raw only; deterministic Pose features are deferred downstream."""
    layout = RGBOutputLayout.from_config(config)
    final_raw = layout.subject_file(subject, "pose_landmarks.parquet")
    final_manifest = layout.subject_file(subject, "pose_manifest.json")
    if not force and final_raw.is_file() and _complete(final_manifest):
        return {"status": "skipped_complete", "subject": subject, "landmarks": str(final_raw)}

    engine_config = _formal_engine_config(config, subject)
    subject_dir = layout.subject_dir(subject)
    temp_raw = subject_dir / f"{subject}_pose-test.parquet"
    temp_manifest = subject_dir / f"{subject}_pose-test_manifest.json"
    _guard(
        [temp_raw, temp_manifest, final_raw, final_manifest],
        force=force,
        label="Pose formal",
        subject_dir=subject_dir,
    )

    run_pose_test(engine_config, subject)
    if not temp_raw.is_file() or not temp_manifest.is_file():
        raise RuntimeError("Validated Pose engine did not produce expected outputs")

    manifest = json.loads(temp_manifest.read_text(encoding="utf-8"))
    temp_raw.replace(final_raw)
    temp_manifest.replace(final_manifest)
    manifest["stage"] = "pose-formal"
    manifest["output_mode"] = "formal"
    manifest["completion_status"] = "complete"
    manifest["derived_features_deferred"] = True
    manifest["engine_origin"] = (
        "validated full-span run_pose_test implementation; formal wrapper changes routing/naming only"
    )
    if isinstance(manifest.get("output"), dict):
        manifest["output"]["parquet"] = str(final_raw)
        manifest["output"]["manifest"] = str(final_manifest)
    else:
        manifest["output"] = {"parquet": str(final_raw), "manifest": str(final_manifest)}
    final_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formal raw Motion/Pose runner. Production orchestration launches stages in parallel."
    )
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--stage", choices=["motion", "pose", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    result: dict[str, object] = {"subject": args.subject}
    if args.stage in {"motion", "all"}:
        result["motion"] = run_motion_formal(config, args.subject, force=args.force)
    if args.stage in {"pose", "all"}:
        result["pose"] = run_pose_formal(config, args.subject, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[rgb:formal-motion-pose] complete {args.subject}")


if __name__ == "__main__":
    main()
