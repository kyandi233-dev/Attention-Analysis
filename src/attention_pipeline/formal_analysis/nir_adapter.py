from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .cohort import canonical_session_id

PIR_PATTERNS = (
    re.compile(r"(^|_)pir($|_)", re.IGNORECASE),
    re.compile(r"pupil_to_iris", re.IGNORECASE),
    re.compile(r"centered_pir", re.IGNORECASE),
)

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "pupil_equivalent_diameter": ("pupil_equivalent_diameter", "pupil_equiv_diameter"),
    "pupil_geom_mean_diameter": ("pupil_geom_mean_diameter",),
    "pupil_contour_area": ("pupil_contour_area",),
    "pupil_ellipse_area": ("pupil_ellipse_area",),
    "pupil_axis_a": ("pupil_axis_a",),
    "pupil_axis_b": ("pupil_axis_b",),
    "pupil_center_x": ("pupil_center_x",),
    "pupil_center_y": ("pupil_center_y",),
    "hard_pupil_fraction": ("hard_pupil_fraction",),
    "soft_pupil_fraction": ("soft_pupil_fraction",),
    "hard_iris_fraction": ("hard_iris_fraction",),
    "soft_iris_fraction": ("soft_iris_fraction",),
    "ocular_aperture_ratio_median": (
        "fullclass_ocular_aperture_ratio_median",
        "ocular_aperture_ratio_median",
    ),
    "ocular_aperture_ratio_p90": (
        "fullclass_ocular_aperture_ratio_p90",
        "ocular_aperture_ratio_p90",
    ),
}

PROVENANCE_COLUMNS = (
    "phase", "phase_segment", "frame_idx", "video_time_ms", "unix_ms", "phase_time_ms",
    "source_detection_source", "source_frame_status", "source_eye_status",
    "source_redetect_reason", "source_yolo_batch_size", "roi_width", "roi_height",
    "roi_valid_content_width", "roi_valid_content_height", "analysis_domain_version",
    "analysis_valid_pixel_count", "analysis_valid_pixel_fraction",
    "predicted_in_padding_fraction", "touches_roi_edge", "pupil_found", "pupil_fit_valid",
    "pupil_component_count", "pupil_largest_component_fraction", "qc_pupil_fragmented",
    "soft_max_probability", "soft_margin", "soft_entropy", "prev_frame_idx",
    "prev_video_time_ms", "frame_gap", "time_gap_ms", "temporal_reset", "diameter_delta",
    "center_jump", "temporal_anomaly",
)

_EYE_MAP = {
    "frame_left": "left", "left": "left", "eye_left": "left",
    "frame_right": "right", "right": "right", "eye_right": "right",
}


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    for name in aliases:
        if name in frame.columns:
            return name
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(False, index=series.index, dtype=bool)
    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].ne(0)
    text_mask = ~numeric_mask & series.notna()
    if text_mask.any():
        result.loc[text_mask] = series.loc[text_mask].astype(str).str.strip().str.lower().isin(
            {"true", "yes", "y", "valid", "success"}
        )
    return result


def _forbidden_pir_columns(columns: Iterable[str]) -> list[str]:
    found: list[str] = []
    for column in columns:
        if any(pattern.search(str(column)) for pattern in PIR_PATTERNS):
            found.append(str(column))
    return sorted(found)


def adapt_nir_frame_table(
    frame: pd.DataFrame,
    *,
    session_id: str | None = None,
    schema_version: str | int | None = None,
    source_path: str | Path | None = None,
    reject_pir: bool = True,
) -> pd.DataFrame:
    """Convert fullclass-final eye rows into a version-neutral pupil-only table."""
    if frame.empty:
        raise ValueError("NIR eye table is empty")

    forbidden = _forbidden_pir_columns(frame.columns)
    if reject_pir and forbidden:
        raise ValueError(
            "当前正式 pupil-only 适配器拒绝 PIR/iris-normalization 历史字段: "
            + ", ".join(forbidden)
        )

    required = {"phase", "phase_segment", "frame_idx", "eye", "unix_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "当前正式 NIR 接入必须使用真实时间键并保留帧/眼主键；缺少: "
            + ", ".join(sorted(missing))
        )
    if frame["unix_ms"].isna().all():
        raise ValueError("unix_ms 全为空；禁止使用 frame_idx/固定FPS 伪造跨模态时间")

    if session_id is None:
        if "subject" not in frame.columns or frame["subject"].dropna().empty:
            raise ValueError("缺少 session_id 参数且输入没有 subject 列")
        unique = frame["subject"].dropna().astype(str).unique()
        if len(unique) != 1:
            raise ValueError(f"单场适配器发现多个 subject: {unique.tolist()}")
        session_id = str(unique[0])
    session_id = canonical_session_id(session_id)

    out = pd.DataFrame(index=frame.index)
    out["session_id"] = session_id
    out["eye_raw"] = frame["eye"].astype("string")
    out["eye"] = frame["eye"].astype(str).str.strip().str.lower().map(_EYE_MAP).astype("string")
    if out["eye"].isna().any():
        unknown = sorted(frame.loc[out["eye"].isna(), "eye"].astype(str).unique().tolist())
        raise ValueError(f"未知 eye 标签，不能静默改写: {unknown}")

    for column in PROVENANCE_COLUMNS:
        if column in frame.columns:
            out[column] = frame[column]
    for canonical, aliases in COLUMN_ALIASES.items():
        source = _find_column(frame, aliases)
        if source is not None:
            out[canonical] = pd.to_numeric(frame[source], errors="coerce")

    if "source_eye_status" in frame.columns:
        out["qc_source_observed"] = frame["source_eye_status"].astype(str).str.strip().str.lower().eq("observed")
    else:
        out["qc_source_observed"] = pd.NA
    found = _bool_series(frame["pupil_found"]) if "pupil_found" in frame.columns else pd.Series(True, index=frame.index)
    fitted = _bool_series(frame["pupil_fit_valid"]) if "pupil_fit_valid" in frame.columns else pd.Series(False, index=frame.index)
    out["qc_pupil_geometry_valid"] = found & fitted
    out["qc_touches_roi_edge"] = _bool_series(frame["touches_roi_edge"]) if "touches_roi_edge" in frame.columns else pd.NA
    out["qc_temporal_anomaly"] = _bool_series(frame["temporal_anomaly"]) if "temporal_anomaly" in frame.columns else pd.NA
    if "analysis_valid_pixel_fraction" in frame.columns:
        out["qc_analysis_domain_fraction"] = pd.to_numeric(frame["analysis_valid_pixel_fraction"], errors="coerce")
    if "predicted_in_padding_fraction" in frame.columns:
        out["qc_predicted_in_padding_fraction"] = pd.to_numeric(frame["predicted_in_padding_fraction"], errors="coerce")

    has_oar = "ocular_aperture_ratio_median" in out.columns
    out["ocular_aperture_available"] = bool(has_oar)
    out["ocular_aperture_role"] = "nir_eye_opening_candidate_qc" if has_oar else "unavailable_not_reconstructed"
    out["nir_schema_version"] = pd.NA if schema_version is None else str(schema_version)
    out["nir_source_path"] = pd.NA if source_path is None else str(Path(source_path))
    out["nir_contract"] = "fullclass-final-pupil-only-v2"

    key = ["phase", "phase_segment", "frame_idx", "eye"]
    if out.duplicated(key).any():
        dup_n = int(out.duplicated(key, keep=False).sum())
        raise ValueError(f"NIR canonical主键重复: {dup_n} rows")
    return out.reset_index(drop=True)


def adapt_nir_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    session_id: str,
    schema_version: str | int | None = None,
    reject_pir: bool = True,
) -> dict[str, object]:
    source = Path(input_csv).resolve()
    target = Path(output_csv).resolve()
    if target.exists():
        raise FileExistsError(f"拒绝覆盖已有适配结果: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, encoding="utf-8-sig")
    adapted = adapt_nir_frame_table(
        frame, session_id=session_id, schema_version=schema_version,
        source_path=source, reject_pir=reject_pir,
    )
    adapted.to_csv(target, index=False, encoding="utf-8-sig")
    return {
        "session_id": canonical_session_id(session_id),
        "input_rows": int(len(frame)), "output_rows": int(len(adapted)),
        "schema_version": None if schema_version is None else str(schema_version),
        "ocular_aperture_available": bool(adapted["ocular_aperture_available"].iloc[0]),
        "output_csv": str(target),
    }
