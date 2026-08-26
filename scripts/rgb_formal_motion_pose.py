from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from attention_pipeline.config import Config, load_config
from attention_pipeline.rgb.motion import run_motion_test
from attention_pipeline.rgb.paths import RGBOutputLayout
from attention_pipeline.rgb.pose import run_pose_test
from attention_pipeline.rgb.pose_features import run_pose_features


def _formal_engine_config(config: Config, subject: str) -> Config:
    """Reuse validated full-span engines while routing their temporary names into the subject directory."""
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


def _guard(paths: list[Path], *, force: bool, label: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise RuntimeError(
            f"{label} output already/partly exists. Inspect first or rerun with --force: {existing}"
        )
    if force:
        for path in paths:
            if path.is_file():
                path.unlink()


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
    _guard([temp_raw, temp_manifest, final_raw, final_manifest], force=force, label="Motion formal")

    run_motion_test(engine_config, subject)
    if not temp_raw.is_file() or not temp_manifest.is_file():
        raise RuntimeError("Validated Motion engine did not produce expected outputs")

    manifest = json.loads(temp_manifest.read_text(encoding="utf-8"))
    temp_raw.replace(final_raw)
    temp_manifest.replace(final_manifest)
    manifest["stage"] = "motion-formal"
    manifest["output_mode"] = "formal"
    manifest["completion_status"] = "complete"
    manifest["engine_origin"] = "validated full-span run_motion_test implementation; formal wrapper changes routing/naming only"
    if isinstance(manifest.get("output"), dict):
        manifest["output"]["parquet"] = str(final_raw)
        manifest["output"]["manifest"] = str(final_manifest)
    else:
        manifest["output"] = {"parquet": str(final_raw), "manifest": str(final_manifest)}
    final_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_pose_formal(config: Config, subject: str, *, force: bool = False) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    final_raw = layout.subject_file(subject, "pose_landmarks.parquet")
    final_manifest = layout.subject_file(subject, "pose_manifest.json")
    final_features = layout.subject_file(subject, "pose_features.parquet")
    final_features_manifest = layout.subject_file(subject, "pose_features_manifest.json")
    if (
        not force
        and final_raw.is_file()
        and final_features.is_file()
        and _complete(final_manifest)
        and _complete(final_features_manifest)
    ):
        return {
            "status": "skipped_complete",
            "subject": subject,
            "landmarks": str(final_raw),
            "features": str(final_features),
        }

    engine_config = _formal_engine_config(config, subject)
    subject_dir = layout.subject_dir(subject)
    temp_raw = subject_dir / f"{subject}_pose-test.parquet"
    temp_manifest = subject_dir / f"{subject}_pose-test_manifest.json"
    temp_features = subject_dir / f"{subject}_pose-features-test.parquet"
    _guard(
        [
            temp_raw, temp_manifest, temp_features,
            final_raw, final_manifest, final_features, final_features_manifest,
        ],
        force=force,
        label="Pose formal",
    )

    run_pose_test(engine_config, subject)
    feature_summary = run_pose_features(engine_config, subject)
    if not temp_raw.is_file() or not temp_manifest.is_file() or not temp_features.is_file():
        raise RuntimeError("Validated Pose engine did not produce expected outputs")

    manifest = json.loads(temp_manifest.read_text(encoding="utf-8"))
    temp_raw.replace(final_raw)
    temp_manifest.replace(final_manifest)
    temp_features.replace(final_features)

    manifest["stage"] = "pose-formal"
    manifest["output_mode"] = "formal"
    manifest["completion_status"] = "complete"
    manifest["engine_origin"] = "validated full-span run_pose_test implementation; formal wrapper changes routing/naming only"
    if isinstance(manifest.get("output"), dict):
        manifest["output"]["parquet"] = str(final_raw)
        manifest["output"]["manifest"] = str(final_manifest)
    else:
        manifest["output"] = {"parquet": str(final_raw), "manifest": str(final_manifest)}
    final_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    feature_manifest = {
        **feature_summary,
        "stage": "pose-features-formal",
        "output_mode": "formal",
        "completion_status": "complete",
        "subject": subject,
        "source_pose_parquet": str(final_raw),
        "output": str(final_features),
        "engine_origin": "validated derive_pose_features implementation; formal wrapper changes routing/naming only",
    }
    final_features_manifest.write_text(
        json.dumps(feature_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"pose": manifest, "features": feature_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal full-span Motion/Pose runner for one RGB subject")
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
