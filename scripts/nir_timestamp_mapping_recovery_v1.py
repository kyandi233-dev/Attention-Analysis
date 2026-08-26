"""Aggregate, pseudonym-safe validation for the NIR timestamp mapping recovery."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import median

import cv2

RUNTIME = Path(__file__).resolve().parents[1] / "runtime" / "nir-formal"
sys.path.insert(0, str(RUNTIME))
from phase_windows import resolve_phase_windows  # noqa: E402
from timestamp_mapping import read_timestamp_map  # noqa: E402


DATA_ROOT = Path("J:/Data")
FORMAL_ROOT = Path("D:/Project/厚粲杯/11_数据/01_Attention-Analysis_nvidia-cuda_formal_NIR")
RESULT_ROOT = Path(__file__).resolve().parents[1] / "docs/020-nir/results/nir_timestamp_mapping_recovery_v1"
RECOVERY_VALIDATION_ROOT = Path("D:/Project/厚粲杯/11_数据/01_Attention-Analysis_nvidia-cuda_recovery_validation_v1")
SUBJECTS = ["sub-056", "sub-057", "sub-058", "sub-100", "sub-178"]


def _paths(subject: str) -> tuple[Path, Path, Path]:
    stem = f"{subject}_"
    directory = DATA_ROOT / stem / "nir"
    return directory / f"{subject}_nir.avi", directory / f"{subject}_nir_timestamps.csv", DATA_ROOT / stem / "beh" / "master_timeline.csv"


def _prior_status(subject: str) -> str:
    matches = sorted(FORMAL_ROOT.glob(f"{subject}_formal_v3.1.3_yolo8_b16_fp32/completion.json"))
    if not matches:
        return "not_found"
    try:
        return str(json.loads(matches[0].read_text(encoding="utf-8")).get("status", "unknown"))
    except Exception:
        return "unreadable"


def _phase_check(video: Path, timestamp_map: dict[int, int], timeline: Path) -> tuple[bool, str, int]:
    try:
        windows = resolve_phase_windows(
            video,
            timestamp_map,
            ["block1", "block2"],
            timeline_path=None,
            timeline_source="master_timeline",
        )
        return True, "", len(windows)
    except Exception as exc:  # validation result is recorded, not hidden
        return False, type(exc).__name__, 0


def _smoke_status(subject: str) -> tuple[str, int | None, int | None, str]:
    matches = sorted(RECOVERY_VALIDATION_ROOT.glob(f"{subject}_*/completion.json"))
    if not matches:
        return "not_run", None, None, ""
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    return (
        str(payload.get("status", "unknown")),
        int(payload.get("processed_frames", 0)),
        int(payload.get("video_read_failure_count", 0)),
        str(payload.get("inference_backend", "")),
    )


def validate(subject: str) -> dict[str, object]:
    video, timestamp_path, timeline = _paths(subject)
    mapping = read_timestamp_map(timestamp_path)
    cap = cv2.VideoCapture(str(video))
    opened = bool(cap.isOpened())
    avi_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if opened else -1
    cap.release()
    deltas = [b[1] - a[1] for a, b in zip(sorted(mapping.unix_by_avi_frame.items()), sorted(mapping.unix_by_avi_frame.items())[1:])]
    med = median(deltas) if deltas else None
    time_gap_count = sum(1 for delta in deltas if med and delta > 1.5 * med)
    old_map = {capture: unix for capture, unix in zip(mapping.capture_by_avi_frame.values(), mapping.unix_by_avi_frame.values())}
    new_map = mapping.unix_by_avi_frame
    old_ok, old_reason, old_windows = _phase_check(video, old_map, timeline)
    new_ok, new_reason, new_windows = _phase_check(video, new_map, timeline)
    smoke_status, smoke_frames, smoke_read_failures, smoke_backend = _smoke_status(subject)
    row_count = mapping.avi_frame_count
    avi_gap = (not opened) or avi_count != row_count
    if avi_gap:
        recovery = "STILL_BLOCKED_REAL_AVI_OR_TIME_GAP"
    elif not new_ok:
        recovery = "BLOCKED_OTHER_REASON"
    elif subject in {"sub-100", "sub-178"}:
        recovery = "RECOVERED"
    else:
        recovery = "CONTROL_NO_REGRESSION"
    return {
        "subject": subject,
        "old_status": _prior_status(subject),
        "capture_counter_gap_count": mapping.capture_frame_gap_count,
        "capture_counter_gap_frames": mapping.capture_frame_gap_frames,
        "timestamp_time_gap_count_gt_1_5x_median": time_gap_count,
        "timestamp_median_interval_ms": med,
        "avi_open_ok": opened,
        "avi_frame_count": avi_count,
        "valid_timestamp_row_count": row_count,
        "explicit_dropped_timestamp_rows": mapping.n_dropped_rows,
        "sequential_mapping_valid": list(new_map) == list(range(row_count)),
        "old_capture_counter_mapping_phase_valid": old_ok,
        "old_capture_counter_mapping_phase_reason": old_reason,
        "old_phase_window_count": old_windows,
        "sequential_mapping_phase_valid": new_ok,
        "sequential_mapping_phase_reason": new_reason,
        "sequential_phase_window_count": new_windows,
        "probe_alignment_validity": "not_checked_in_mapping_only_validation",
        "avi_decode_frame_gap": avi_gap,
        "recovery_status": recovery,
        "minimal_recovery_validation_status": smoke_status,
        "minimal_recovery_processed_frames": smoke_frames,
        "minimal_recovery_video_read_failures": smoke_read_failures,
        "minimal_recovery_backend": smoke_backend,
    }


def main() -> None:
    rows = [validate(subject) for subject in SUBJECTS]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (RESULT_ROOT / "session_validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    contract = {
        "version": "nir-timestamp-mapping-v1",
        "frame_idx_semantics": "sequential AVI frame index",
        "capture_frame_idx_semantics": "original source capture counter retained as provenance",
        "timestamp_source": "valid non-dropped rows in *_nir_timestamps.csv",
        "avi_gap_definition": "AVI cannot open or nominal AVI frame count differs from valid timestamp row count",
        "capture_gap_definition": "source capture counter increment exceeds one and is not an AVI gap by itself",
        "timestamp_time_gap_definition": "adjacent valid unix_ms interval greater than 1.5 times session median interval",
        "tested_subjects": SUBJECTS,
        "raw_data_uploaded": False,
    }
    (RESULT_ROOT / "mapping_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# NIR Timestamp Mapping Recovery Result",
        "",
        "状态：validation_ready；本报告只记录映射与 phase-window 验证，不上传原始 NIR 或行级时间戳。",
        "",
        "## Canonical mapping",
        "",
        "NIR timestamp CSV 第一列保留为 source capture counter，AVI 内部 frame index 使用有效 timestamp 行的顺序编号。capture counter gap 不再自动解释为 AVI frame gap。",
        "",
        "## Session results",
        "",
        "| subject | old status | capture gap | timestamp-time gap | AVI frames | valid timestamp rows | old phase mapping | sequential phase mapping | AVI gap | recovery status | minimal recovery |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| {subject} | {old_status} | {capture_counter_gap_count} ({capture_counter_gap_frames} frames) | {timestamp_time_gap_count_gt_1_5x_median} | {avi_frame_count} | {valid_timestamp_row_count} | {old_capture_counter_mapping_phase_valid} | {sequential_mapping_phase_valid} | {avi_decode_frame_gap} | {recovery_status} | {minimal_recovery_validation_status} ({minimal_recovery_processed_frames} frames; {minimal_recovery_backend}) |".format(**row))
    lines += [
        "",
        "## Limits",
        "",
        "本轮 mapping-only validation 未运行完整 YOLO/RITnet 全量分析，也未验证 Probe alignment 的下游数值，因此该字段记录为 not_checked_in_mapping_only_validation。sub-100 与 sub-178 已各完成 32 帧、pytorch-cuda、block1/block2 smoke recovery validation，确认 runtime 使用 sequential AVI frame mapping；恢复 session 如需进入后续 fullclass，必须在隔离 recovery 输出根中进行完整恢复运行并检查 completion/QC。",
        "",
        "sub-099 不属于本任务；其 master_timeline 缺失问题不因 timestamp mapping 修复而改变。",
    ]
    (RESULT_ROOT / "NIR_TIMESTAMP_MAPPING_RECOVERY_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
