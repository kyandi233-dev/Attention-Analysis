from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.rgb.paths import RGBOutputLayout

import face_derive_tracking_eyelid_v02 as derive


SCHEMA_VERSION = "rgb-face-formal-derived-v1.0"


def _complete_manifest(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if data.get("completion_status") == "complete" else None


def run_formal_derive(config_path: str, subject: str, *, force: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    layout = RGBOutputLayout.from_config(config)
    raw_path = layout.subject_file(subject, "face_raw.parquet")
    tracks_path = layout.subject_file(subject, "face_tracks.parquet")
    eye_path = layout.subject_file(subject, "eye_features.parquet")
    manifest_path = layout.subject_file(subject, "face_derived_manifest.json")

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Formal Face raw output not found: {raw_path}. Run face_formal_directml.py first."
        )

    if not force:
        existing = _complete_manifest(manifest_path)
        if existing is not None and tracks_path.is_file() and eye_path.is_file():
            print(json.dumps({
                "status": "skipped_complete",
                "subject": subject,
                "tracks": str(tracks_path),
                "eye_features": str(eye_path),
            }, ensure_ascii=False, indent=2))
            return existing
        if tracks_path.exists() or eye_path.exists() or manifest_path.exists():
            raise RuntimeError(
                "Partial formal Face-derived output exists. Inspect first or rerun with --force."
            )

    raw = pd.read_parquet(raw_path)
    detected = raw.copy()
    if "detected" in detected.columns:
        detected = detected[detected["detected"].fillna(False).astype(bool)].copy()
    detected = detected[derive._bbox_mask(detected)]

    max_track_gap_ms = float(
        config.section("face").get("primary_face", {}).get("max_track_gap_ms", 2000.0)
    )
    tracked, tracking_mode = derive.assign_tracks_window_aware(
        detected,
        max_gap_ms=max_track_gap_ms,
    )
    tracked, primary_info = derive.choose_primary_segments(tracked)
    eye, eye_summary = derive.derive_eye_from_primary_segments(tracked)

    tracked.to_parquet(tracks_path, index=False, engine="pyarrow", compression="zstd")
    eye.to_parquet(eye_path, index=False, engine="pyarrow", compression="zstd")

    frame_col = derive._frame_col(tracked)
    multi_face_frames = int((tracked.groupby(frame_col).size() > 1).sum()) if not tracked.empty else 0
    primary_frames = int(
        tracked.loc[tracked["primary_face"].fillna(False), frame_col].nunique()
    ) if not tracked.empty else 0
    total_frames = int(raw[frame_col].nunique()) if frame_col in raw.columns else 0

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "face-formal-derived",
        "output_mode": "formal",
        "completion_status": "complete",
        "subject": subject,
        "raw_input": str(raw_path),
        "tracking": {
            "detected_face_rows": int(len(tracked)),
            "track_count": int(tracked["face_track_id"].nunique()) if not tracked.empty else 0,
            "multi_face_frames": multi_face_frames,
            "max_track_gap_ms": max_track_gap_ms,
            "tracking_mode": tracking_mode,
            "primary_selection": primary_info,
            "primary_frames": primary_frames,
            "total_input_frames": total_frames,
            "primary_frame_coverage": float(primary_frames / total_frames) if total_frames else 0.0,
        },
        "eyelid": eye_summary,
        "outputs": {
            "tracks": str(tracks_path),
            "eye_features": str(eye_path),
        },
        "notes": [
            "Full formal input is temporally continuous, so the v0.2 tracker automatically uses continuous tracking semantics rather than dry-run window segmentation.",
            "Continuous eyelid/closure signals are retained now; blink-event thresholds can be finalized later without rerunning expensive Face inference.",
        ],
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal Face tracking / primary / eyelid derivation")
    parser.add_argument("--config", default="configs/rgb_analysis.yaml")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_formal_derive(args.config, args.subject, force=args.force)


if __name__ == "__main__":
    main()
