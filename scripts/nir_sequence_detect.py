"""PuRe/PuReST连续序列适配器（Python 3.10 pupil runtime）。

每个manifest输入行×算法必有输出。PuReST走普通有状态runWithConfidence(image)；
严格像素范围由Python后置门执行，因为当前绑定在首帧、追踪和重检时并不一致地遵守
五参数px重载。程序只报告外层可观察session状态，不猜测内部tracker/fallback分支。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from nir_detect_batch import canonicalize_axes, compute_photometric_contrast

OUTPUT_HEADER = [
    "subject", "block", "sequence_id", "eye", "frame_offset", "frame_idx", "unix_ms",
    "algorithm", "face_status", "roi_status", "image_status", "detector_status",
    "quality_status", "error_code", "session_state", "algorithm_returned", "returned",
    "center_x", "center_y", "raw_axis_w_px", "raw_axis_h_px", "raw_angle_deg",
    "major_diameter", "minor_diameter", "major_angle_deg", "angle_deg",
    "confidence", "outline_confidence", "photometric_contrast", "runtime_ms",
    "px_min_accept", "px_max_accept", "pupil_min_mm_compat",
]


def _number(raw: dict, name: str, default=""):
    value = str(raw.get(name, "")).strip()
    return value if value else default


def load_manifest(path: Path, roi_root: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            roi = str(raw.get("roi_path", "")).strip()
            face_detected = str(raw.get("face_detected", "")).strip()
            rows.append({
                "subject": str(raw.get("subject", "")).strip(),
                "block": _number(raw, "block", _number(raw, "block_num", "")),
                "sequence_id": str(raw["sequence_id"]).strip(),
                "eye": str(raw["eye"]).strip(),
                "frame_offset": int(raw["frame_offset"]),
                "frame_idx": _number(raw, "frame_idx", _number(raw, "avi_frame_idx", "")),
                "unix_ms": _number(raw, "unix_ms", ""),
                "face_status": str(raw.get("face_status", "")).strip()
                    or ("ok" if face_detected in {"1", "True", "true"} else "no_face"),
                "roi_status": str(raw.get("roi_status", "")).strip() or ("ok" if roi else "roi_missing"),
                "openness": _number(raw, "openness", ""),
                "visible_proxy": _number(raw, "visible_proxy", ""),
                "p80_closed_proxy": _number(raw, "p80_closed_proxy", ""),
                "roi_path": str(roi_root / roi) if roi else "",
            })
    return rows


def _base_row(frame: dict, algorithm: str, px_min: int, px_max: int, pupil_min_mm: float) -> dict:
    row = {key: "" for key in OUTPUT_HEADER}
    for key in ("subject", "block", "sequence_id", "eye", "frame_offset", "frame_idx",
                "unix_ms", "face_status", "roi_status"):
        row[key] = frame.get(key, "")
    row.update({
        "algorithm": algorithm,
        "px_min_accept": px_min,
        "px_max_accept": px_max,
        "pupil_min_mm_compat": pupil_min_mm,
        "algorithm_returned": 0,
        "returned": 0,
    })
    return row


def _failed(frame: dict, algorithm: str, px_min: int, px_max: int, pupil_min_mm: float,
            image_status: str, detector_status: str, quality_status: str,
            error_code: str, session_state: str) -> dict:
    row = _base_row(frame, algorithm, px_min, px_max, pupil_min_mm)
    row.update({
        "image_status": image_status,
        "detector_status": detector_status,
        "quality_status": quality_status,
        "error_code": error_code,
        "session_state": session_state,
    })
    return row


def _record(frame: dict, algorithm: str, pupil, image: np.ndarray, elapsed_ms: float,
            px_min: int, px_max: int, pupil_min_mm: float, session_state: str,
            visibility_ok: bool = True) -> tuple[dict, bool]:
    """返回(row, diameter_rejected)。单次直径拒绝不在此处直接reset。"""
    row = _base_row(frame, algorithm, px_min, px_max, pupil_min_mm)
    row.update({
        "image_status": "ok",
        "runtime_ms": elapsed_ms,
        "confidence": float(pupil.confidence),
        "outline_confidence": float(pupil.outline_confidence),
        "session_state": session_state,
    })
    if not pupil.valid(0.0):
        row.update({
            "detector_status": "algorithm_invalid",
            "quality_status": "not_evaluated",
            "error_code": "algorithm_invalid",
        })
        return row, False

    center = pupil.center
    raw_w, raw_h = map(float, pupil.size)
    raw_angle = float(pupil.angle)
    major, minor, major_angle = canonicalize_axes(raw_w, raw_h, raw_angle)
    diameter_ok = px_min <= major <= px_max
    accepted = diameter_ok and visibility_ok
    row.update({
        "algorithm_returned": 1,
        "returned": int(accepted),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "raw_axis_w_px": raw_w,
        "raw_axis_h_px": raw_h,
        "raw_angle_deg": raw_angle,
        "major_diameter": major,
        "minor_diameter": minor,
        "major_angle_deg": major_angle,
        "angle_deg": major_angle,
        "photometric_contrast": compute_photometric_contrast(
            image, center, raw_w, raw_h, raw_angle
        ),
        "detector_status": "returned",
        "quality_status": (
            "accepted" if accepted else ("diameter_rejected" if not diameter_ok else "visibility_rejected")
        ),
        "error_code": (
            "" if accepted else ("diameter_out_of_range" if not diameter_ok else "visibility_gate")
        ),
    })
    return row, not diameter_ok


def _visibility_ok(frame: dict, openness_visible: float) -> bool:
    visible = str(frame.get("visible_proxy", "")).strip()
    closed = str(frame.get("p80_closed_proxy", "")).strip()
    if visible in {"0", "0.0", "False", "false"} or closed in {"1", "1.0", "True", "true"}:
        return False
    try:
        return float(frame.get("openness", "")) >= openness_visible
    except (TypeError, ValueError):
        return True


def _unix_ms(frame: dict) -> float | None:
    try:
        value = float(frame.get("unix_ms", ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NIR continuous PuRe/PuReST adapter")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--roi-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--px-min", type=int, default=6)
    parser.add_argument("--px-max", type=int, default=50)
    parser.add_argument("--pupil-min-mm", type=float, default=0.5,
                        help="Compatibility-only PuReST search setting; not a physical estimate")
    parser.add_argument("--algorithms", default="PuRe,PuReST")
    parser.add_argument("--openness-visible", type=float, default=0.55)
    parser.add_argument("--max-session-gap-ms", type=float, default=200.0)
    parser.add_argument("--reset-after-quality-rejects", type=int, default=3)
    args = parser.parse_args(argv)
    if args.px_min <= 0 or args.px_max <= args.px_min:
        parser.error("require 0 < px-min < px-max")
    if args.reset_after_quality_rejects < 1:
        parser.error("reset-after-quality-rejects must be >= 1")

    algorithms = [name for name in args.algorithms.split(",") if name]
    unsupported = set(algorithms) - {"PuRe", "PuReST"}
    if unsupported:
        parser.error(f"unsupported sequence algorithms: {sorted(unsupported)}")

    rows = load_manifest(Path(args.manifest), Path(args.roi_root))
    import pypupilext

    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["subject"], row["block"], row["sequence_id"], row["eye"])
        groups.setdefault(key, []).append(row)
    for frames in groups.values():
        frames.sort(key=lambda item: item["frame_offset"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        for done, ((_subject, _block, _sequence_id, _eye), frames) in enumerate(groups.items(), start=1):
            pure = pypupilext.PuRe() if "PuRe" in algorithms else None
            purest = pypupilext.PuReST() if "PuReST" in algorithms else None
            if purest is not None:
                purest.minPupilDiameterMM = args.pupil_min_mm
                purest.reset()
            purest_fresh = True
            previous_offset = None
            previous_unix_ms = None
            quality_reject_streak = 0

            for frame in frames:
                current_unix_ms = _unix_ms(frame)
                frame_gap = previous_offset is not None and frame["frame_offset"] != previous_offset + 1
                time_gap = (
                    previous_unix_ms is not None
                    and current_unix_ms is not None
                    and (current_unix_ms <= previous_unix_ms
                         or current_unix_ms - previous_unix_ms > args.max_session_gap_ms)
                )
                previous_offset = frame["frame_offset"]
                if current_unix_ms is not None:
                    previous_unix_ms = current_unix_ms
                if (frame_gap or time_gap) and purest is not None:
                    purest.reset()
                    purest_fresh = True
                    quality_reject_streak = 0

                roi_usable = frame["roi_path"] and frame["face_status"] == "ok" and frame["roi_status"] in {"ok", "ready"}
                if not roi_usable:
                    if purest is not None:
                        purest.reset()
                        purest_fresh = True
                        quality_reject_streak = 0
                    for algorithm in algorithms:
                        writer.writerow(_failed(
                            frame, algorithm, args.px_min, args.px_max, args.pupil_min_mm,
                            "not_available", "not_run", "not_evaluated",
                            frame["roi_status"] or frame["face_status"] or "roi_missing",
                            "session_reset" if algorithm == "PuReST" else "not_stateful",
                        ))
                    continue

                try:
                    image = cv2.imdecode(
                        np.fromfile(frame["roi_path"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                    )
                except Exception as exc:
                    image = None
                    load_error = f"image_read_exception:{type(exc).__name__}:{exc}"
                else:
                    load_error = "image_read_failed" if image is None else ""

                if image is None:
                    if purest is not None:
                        purest.reset()
                        purest_fresh = True
                        quality_reject_streak = 0
                    for algorithm in algorithms:
                        writer.writerow(_failed(
                            frame, algorithm, args.px_min, args.px_max, args.pupil_min_mm,
                            "read_failed", "not_run", "not_evaluated", load_error,
                            "session_reset" if algorithm == "PuReST" else "not_stateful",
                        ))
                    continue

                image = np.ascontiguousarray(image)
                visibility_ok = _visibility_ok(frame, args.openness_visible)
                for algorithm in algorithms:
                    state = "not_stateful"
                    try:
                        start = time.perf_counter()
                        if algorithm == "PuRe":
                            pupil = pypupilext.Pupil()
                            pure.runWithConfidence(
                                image, (0, 0, image.shape[1], image.shape[0]),
                                pupil, args.px_min, args.px_max,
                            )
                        else:
                            state = "first_after_reset" if purest_fresh else "continuing_session"
                            pupil = purest.runWithConfidence(image)
                            purest_fresh = False
                        elapsed = (time.perf_counter() - start) * 1000.0
                        result, diameter_rejected = _record(
                            frame, algorithm, pupil, image, elapsed,
                            args.px_min, args.px_max, args.pupil_min_mm, state, visibility_ok,
                        )
                        if algorithm == "PuReST":
                            if result["quality_status"] == "visibility_rejected":
                                purest.reset()
                                purest_fresh = True
                                quality_reject_streak = 0
                                result["session_state"] = "session_reset"
                            elif diameter_rejected:
                                quality_reject_streak += 1
                                result["session_state"] = "quality_rejected"
                                if quality_reject_streak >= args.reset_after_quality_rejects:
                                    purest.reset()
                                    purest_fresh = True
                                    quality_reject_streak = 0
                                    result["session_state"] = "session_reset"
                            elif result["detector_status"] == "algorithm_invalid":
                                result["session_state"] = "algorithm_invalid"
                            else:
                                quality_reject_streak = 0
                        writer.writerow(result)
                    except Exception as exc:
                        if algorithm == "PuReST":
                            purest.reset()
                            purest_fresh = True
                            quality_reject_streak = 0
                            state = "session_reset"
                        writer.writerow(_failed(
                            frame, algorithm, args.px_min, args.px_max, args.pupil_min_mm,
                            "ok", "exception", "not_evaluated",
                            f"{type(exc).__name__}:{exc}", state,
                        ))
            print(f"[progress] {done}/{len(groups)} sessions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
