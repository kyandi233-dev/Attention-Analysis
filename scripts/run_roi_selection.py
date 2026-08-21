"""正式数据 ROI 选型总控：统一窗口、尺度、EAR 门控、PuReST 与汇总。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import roi_common
from roi_mediapipe import MediaPipeRoi
from roi_yolo import YoloRoi
from roi_yunet import YuNetRoi
from attention_pipeline.config import load_config


def _args(config_path: str, subject: str, duration_sec: float, corner_span: float) -> SimpleNamespace:
    return SimpleNamespace(
        config=config_path, subject=subject, block=None, root=None, out=None,
        n_segments=None, seg_starts="", frames_per_seg=None, duration_sec=duration_sec,
        roi_w=None, roi_h=None, corner_span=corner_span, venv_python=None,
        px_min=None, px_max=None, pupil_min_mm=None, openness_visible=None,
        max_session_gap_ms=None, reset_after_quality_rejects=None,
        face_model=None, yunet_model=None, yolo_model=None,
    )


def _provider(kind: str, args, config):
    if kind == "mediapipe":
        return MediaPipeRoi(config.path_value("face_landmarker_model"), args.roi_w, args.roi_h, args.corner_span)
    if kind == "yunet":
        return YuNetRoi(config.path_value("yunet_model"), args.roi_w, args.roi_h, args.corner_span)
    if kind == "yolo":
        return YoloRoi(config.path_value("yolo_model"), args.roi_w, args.roi_h, args.corner_span)
    raise ValueError(kind)


def _continuity(data: pd.DataFrame) -> tuple[float, float, int]:
    jumps_d, jumps_c, missing_runs = [], [], []
    for _, group in data.groupby(["sequence_id", "eye"], dropna=False):
        group = group.sort_values("frame_offset")
        ok = group["returned"].fillna(0).astype(int).to_numpy() == 1
        run = 0
        for value in ok:
            if value:
                if run:
                    missing_runs.append(run)
                run = 0
            else:
                run += 1
        if run:
            missing_runs.append(run)
        valid = group.loc[ok]
        frame = valid["frame_offset"].to_numpy(dtype=float)
        d = valid["major_diameter"].to_numpy(dtype=float)
        x = valid["center_x"].to_numpy(dtype=float)
        y = valid["center_y"].to_numpy(dtype=float)
        adjacent = np.diff(frame) == 1
        if adjacent.any():
            jumps_d.extend(np.abs(np.log(d[1:][adjacent] / d[:-1][adjacent])))
            jumps_c.extend(np.hypot(np.diff(x)[adjacent], np.diff(y)[adjacent]) / 160.0)
    return (
        float(np.median(jumps_d)) if jumps_d else float("nan"),
        float(np.median(jumps_c)) if jumps_c else float("nan"),
        int(max(missing_runs, default=0)),
    )


def _summarize_candidate(candidate_dir: Path, evaluation_start: int) -> dict:
    manifest = pd.read_csv(candidate_dir / "manifest.csv")
    detections = pd.read_csv(candidate_dir / "detections_sequence.csv")
    eval_manifest = manifest[manifest["frame_offset"] >= evaluation_start]
    purest = detections[
        (detections["algorithm"] == "PuReST")
        & (detections["frame_offset"] >= evaluation_start)
    ].copy()
    diameter_returned = purest[purest["algorithm_returned"] == 1]["major_diameter"]
    d_jump, c_jump, longest_missing = _continuity(purest)
    speed = pd.read_csv(candidate_dir / "speed_roi_only.csv").iloc[0].to_dict()
    detector_speed = pd.read_csv(candidate_dir / "speed_detector.csv").iloc[0].to_dict()
    return {
        "candidate": candidate_dir.name,
        "backend": str(eval_manifest["backend"].dropna().iloc[0]) if len(eval_manifest) else candidate_dir.name,
        "evaluation_eye_rows": len(eval_manifest),
        "roi_valid_rate": float((eval_manifest["roi_status"] == "ok").mean()) if len(eval_manifest) else np.nan,
        "purest_algorithm_return_rate": float((purest["algorithm_returned"] == 1).mean()) if len(purest) else np.nan,
        "purest_accepted_rate": float((purest["returned"] == 1).mean()) if len(purest) else np.nan,
        "diameter_rejected_rate": float((purest["quality_status"] == "diameter_rejected").mean()) if len(purest) else np.nan,
        "diameter_p10_px": float(diameter_returned.quantile(0.10)) if len(diameter_returned) else np.nan,
        "diameter_median_px": float(diameter_returned.median()) if len(diameter_returned) else np.nan,
        "diameter_p90_px": float(diameter_returned.quantile(0.90)) if len(diameter_returned) else np.nan,
        "diameter_log_jump_median": d_jump,
        "center_jump_norm_median": c_jump,
        "longest_unaccepted_run_frames": longest_missing,
        "model_load_ms": speed.get("model_load_ms"),
        "roi_only_mean_ms": speed.get("roi_only_mean_ms"),
        "roi_only_p95_ms": speed.get("roi_only_p95_ms"),
        "end_to_end_mean_ms": speed.get("end_to_end_mean_ms"),
        "end_to_end_p95_ms": speed.get("end_to_end_p95_ms"),
        "roi_peak_rss_mb": speed.get("peak_rss_mb"),
        "detector_wall_seconds": detector_speed.get("wall_seconds"),
        "detector_peak_rss_mb": detector_speed.get("peak_child_rss_mb"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="正式 NIR ROI 选型总控")
    parser.add_argument("--config", default="configs/formal.yaml")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--phase", choices=["calibration", "evaluation"], required=True)
    parser.add_argument("--selected-span", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    cli = parser.parse_args(argv)
    config = load_config(cli.config)
    selection = config.section("nir")["roi_selection"]
    sequence = config.section("nir")["sequence"]
    subject = cli.subject or str(selection["subject"])
    if cli.phase == "calibration":
        duration_sec = float(selection["calibration_sec"])
        spans = [float(v) for v in selection["corner_span_candidates"]]
        evaluation_start = 0
    else:
        if cli.selected_span is None:
            parser.error("evaluation 阶段必须提供 --selected-span（校准冻结后不得再调）")
        duration_sec = float(selection["duration_sec"])
        spans = [float(cli.selected_span)]
        evaluation_start = int(round(float(selection["calibration_sec"]) * float(config.section("nir")["fps_nominal"])))

    artifact_root = config.path_value("roi_selection_artifact_root") / cli.phase
    artifact_root.mkdir(parents=True, exist_ok=True)
    built, shared_gate_manifest = [], None
    model_keys = {"mediapipe": "face_landmarker_model", "yunet": "yunet_model", "yolo": "yolo_model"}
    for span in spans:
        for kind in ("mediapipe", "yunet", "yolo"):
            local = _args(cli.config, subject, duration_sec, span)
            roi_common.resolve_common_args(local, model_keys[kind])
            local.out = str(artifact_root)
            provider = _provider(kind, local, config)
            outdir = artifact_root / subject / provider.name
            manifest = outdir / "manifest.csv"
            if not (cli.skip_existing and manifest.exists()):
                outdir.mkdir(parents=True, exist_ok=True)
                manifest, n_eye_frames = roi_common.build_roi(provider, local, outdir)
            else:
                provider.close()
                n_eye_frames = int((pd.read_csv(manifest)["roi_status"] == "ok").sum())
            if kind == "mediapipe" and shared_gate_manifest is None:
                shared_gate_manifest = manifest
            built.append((kind, local, manifest, outdir, n_eye_frames))

    if shared_gate_manifest is None:
        raise RuntimeError("缺少 MediaPipe manifest，无法建立统一 EAR 门控")
    calibration_frames = int(round(float(selection["calibration_sec"]) * float(config.section("nir")["fps_nominal"])))
    gate_info = roi_common.apply_openness_gate(
        shared_gate_manifest, calibration_frames,
        float(sequence["openness_visible"]), float(sequence["openness_closed"]),
    )
    for kind, local, manifest, outdir, n_eye_frames in built:
        if manifest != shared_gate_manifest:
            roi_common.apply_openness_gate(
                manifest, calibration_frames,
                float(sequence["openness_visible"]), float(sequence["openness_closed"]),
                shared_gate_manifest,
            )
        det_csv = outdir / "detections_sequence.csv"
        if not (cli.skip_existing and det_csv.exists()):
            det_csv = roi_common.run_detect(
                local.venv_python, manifest, outdir, local.px_min, local.px_max,
                local.pupil_min_mm, local.openness_visible,
                local.max_session_gap_ms, local.reset_after_quality_rejects,
            )
        roi_common.compute_stats(local, det_csv, outdir, n_eye_frames)

    summary = pd.DataFrame([_summarize_candidate(outdir, evaluation_start) for _, _, _, outdir, _ in built])
    # YuNet/YOLO do not supply eyelid landmarks. Their production-chain estimate must
    # therefore include the shared MediaPipe EAR inference rather than reporting only
    # their faster/slower ROI locator time.
    mp_rows = summary[summary["candidate"].str.startswith("mediapipe-")]
    mp_speed = pd.read_csv(
        next(outdir for kind, _, _, outdir, _ in built if kind == "mediapipe") / "speed_roi_only.csv"
    ).iloc[0]
    gate_mean = float(mp_speed["inference_mean_ms"])
    gate_p95 = float(mp_speed["inference_p95_ms"])
    summary["requires_mediapipe_ear_gate"] = ~summary["candidate"].str.startswith("mediapipe-")
    summary["complete_chain_roi_gate_mean_ms"] = summary["end_to_end_mean_ms"] + np.where(
        summary["requires_mediapipe_ear_gate"], gate_mean, 0.0
    )
    summary["complete_chain_roi_gate_p95_ms_conservative"] = summary["end_to_end_p95_ms"] + np.where(
        summary["requires_mediapipe_ear_gate"], gate_p95, 0.0
    )
    summary.to_csv(artifact_root / "candidate_summary.csv", index=False)
    (artifact_root / "run_metadata.json").write_text(json.dumps({
        "phase": cli.phase, "subject": subject, "duration_sec": duration_sec,
        "spans": spans, "evaluation_start_frame_offset": evaluation_start,
        "shared_gate": gate_info,
        "warning": "YuNet/YOLO complete production speed must add MediaPipe EAR inference; ROI-only speed is separate.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\n[done] {artifact_root / 'candidate_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

