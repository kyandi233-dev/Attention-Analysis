"""Read-only NIR/RGB pupil audit for Issue #22.

This module intentionally stops at an engineering/provenance audit.  It does
not repair NIR geometry, create a formal analysis table, or fit a scientific
model.  The paired sample is local and is written below an ignored
``artifacts/`` directory by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats


AUDIT_VERSION = "issue22-multimodal-pupil-audit-v2"
DEFAULT_NIR = Path(
    r"D:\_AttentionData\Beijing-NIR\amd-directml\sub-031_formal_v3.1.3_yolo_b16_fp32\sub-031_ritnet_fullclass.csv"
)
DEFAULT_RGB = Path(r"D:\_AttentionData\Beijing-RGB\sub-031\sub-031_face_raw.parquet")
DEFAULT_SART = Path(
    r"D:\AAAWORK\07-竞赛\厚璨杯\021-analysisplan\SART\问卷\问卷分析\subject_repeat_registry.csv"
)
DEFAULT_PARTICIPANT_INFO = Path(
    r"D:\AAAWORK\07-竞赛\厚璨杯\020-Experiment\北京被试信息表.xlsx"
)
DEFAULT_IDENTITY_SUMMARY = Path(
    r"D:\AAAWORK\07-竞赛\厚璨杯\021-analysisplan\被试信息.xlsx"
)
DEFAULT_NIR_ROOT = Path(r"D:\_AttentionData\Beijing-NIR\amd-directml")
DEFAULT_RGB_ROOT = Path(r"D:\_AttentionData\Beijing-RGB")

RIGHT_IRIS = (469, 470, 471, 472)
LEFT_IRIS = (474, 475, 476, 477)

NIR_REQUIRED = {
    "subject",
    "video",
    "phase",
    "phase_segment",
    "frame_idx",
    "video_time_ms",
    "unix_ms",
    "phase_time_ms",
    "eye",
    "frame_status",
    "status",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "roi_clipped",
    "fullclass_pupil_found",
    "fullclass_pupil_fit_valid",
    "fullclass_pupil_center_x",
    "fullclass_pupil_center_y",
    "fullclass_pupil_axis_a",
    "fullclass_pupil_axis_b",
    "fullclass_pupil_equiv_diameter",
    "fullclass_pupil_geom_mean_diameter",
    "fullclass_pupil_contour_area",
    "fullclass_pupil_touches_roi_edge",
    "fullclass_iris_outer_found",
    "fullclass_iris_outer_fit_valid",
    "fullclass_pupil_to_iris_diameter_ratio",
    "fullclass_normalization_valid",
}

RGB_REQUIRED = {
    "schema_version",
    "subject",
    "sample_index",
    "unix_ms",
    "phase",
    "block",
    "trial_num",
    "condition",
    "position_in_cycle",
    "is_no_go",
    "response",
    "correct",
    "is_probe",
    "detected",
    "face_rank",
    "FaceRectWidth",
    "FaceRectHeight",
    "FaceScore",
    "Pitch",
    "Roll",
    "Yaw",
    "X",
    "Y",
    "Z",
    "gaze_pitch",
    "gaze_yaw",
    "temporal_gap",
    "capture_gap_before",
}


def _json_value(value: Any) -> Any:
    """Convert numpy/pandas values into strict JSON-compatible values."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "t"})


def finite_fraction(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.notna().mean()) if len(numeric) else 0.0


def quantiles(values: Iterable[Any], probabilities: Sequence[float] = (0.01, 0.1, 0.5, 0.9, 0.95, 0.99)) -> dict[str, float | None]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return {f"q{int(p * 100):02d}": None for p in probabilities}
    result: dict[str, float | None] = {}
    for p in probabilities:
        result[f"q{int(p * 100):02d}"] = float(np.quantile(numeric, p))
    return result


def timeline_stats(timestamps: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(timestamps, errors="coerce").dropna().astype("int64").drop_duplicates().sort_values()
    if values.empty:
        return {"unique_timestamps": 0, "min_unix_ms": None, "max_unix_ms": None, "span_ms": None, "estimated_hz": None, "dt_ms": quantiles([]), "gaps_gt_200ms": 0, "gaps_gt_1000ms": 0}
    diffs = values.diff().dropna()
    span_ms = int(values.iloc[-1] - values.iloc[0])
    return {
        "unique_timestamps": int(values.size),
        "min_unix_ms": int(values.iloc[0]),
        "max_unix_ms": int(values.iloc[-1]),
        "span_ms": span_ms,
        "estimated_hz": float(values.size / (span_ms / 1000.0)) if span_ms else None,
        "dt_ms": quantiles(diffs),
        "gaps_gt_200ms": int((diffs > 200).sum()),
        "gaps_gt_1000ms": int((diffs > 1000).sum()),
    }


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> dict[str, Any]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna() & np.isfinite(x_num) & np.isfinite(y_num)
    n = int(mask.sum())
    if n < 3 or x_num[mask].nunique() < 2 or y_num[mask].nunique() < 2:
        return {"n": n, "r": None, "p": None, "method": method}
    if method == "spearman":
        result = stats.spearmanr(x_num[mask], y_num[mask])
    else:
        result = stats.pearsonr(x_num[mask], y_num[mask])
    return {"n": n, "r": float(result.statistic), "p": float(result.pvalue), "method": method}


def summarize_numeric(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return {"n": 0, "valid_fraction": 0.0, "mean": None, "sd": None, "cv": None, "quantiles": quantiles([])}
    mean = float(finite.mean())
    sd = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    return {
        "n": int(len(finite)),
        "valid_fraction": float(numeric.notna().mean()),
        "mean": mean,
        "sd": sd,
        "cv": float(sd / mean) if mean else None,
        "quantiles": quantiles(finite),
    }


def mesh_column(axis: str, index: int) -> str:
    return f"mesh_{axis}_{index}"


def numeric_column_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def pairwise_max_distance(frame: pd.DataFrame, indices: Sequence[int]) -> pd.Series:
    pairs: list[pd.Series] = []
    for left_position, left in enumerate(indices):
        for right in indices[left_position + 1 :]:
            dx = pd.to_numeric(frame[mesh_column("x", left)], errors="coerce") - pd.to_numeric(frame[mesh_column("x", right)], errors="coerce")
            dy = pd.to_numeric(frame[mesh_column("y", left)], errors="coerce") - pd.to_numeric(frame[mesh_column("y", right)], errors="coerce")
            pairs.append(np.hypot(dx, dy))
    return pd.concat(pairs, axis=1).max(axis=1)


def mean_point(frame: pd.DataFrame, indices: Sequence[int]) -> tuple[pd.Series, pd.Series]:
    xs = [pd.to_numeric(frame[mesh_column("x", index)], errors="coerce") for index in indices]
    ys = [pd.to_numeric(frame[mesh_column("y", index)], errors="coerce") for index in indices]
    return pd.concat(xs, axis=1).mean(axis=1), pd.concat(ys, axis=1).mean(axis=1)


def derive_rgb_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    width = numeric_column_or_nan(result, "FaceRectWidth")
    height = numeric_column_or_nan(result, "FaceRectHeight")
    result["rgb_face_bbox_area_px2"] = width * height
    result["rgb_face_bbox_scale_px"] = np.sqrt(result["rgb_face_bbox_area_px2"].clip(lower=0))
    result["rgb_eye_outer_corner_distance_px"] = np.hypot(
        pd.to_numeric(result[mesh_column("x", 33)], errors="coerce") - pd.to_numeric(result[mesh_column("x", 263)], errors="coerce"),
        pd.to_numeric(result[mesh_column("y", 33)], errors="coerce") - pd.to_numeric(result[mesh_column("y", 263)], errors="coerce"),
    )
    result["rgb_eye_inner_canthus_distance_px"] = np.hypot(
        pd.to_numeric(result[mesh_column("x", 133)], errors="coerce") - pd.to_numeric(result[mesh_column("x", 362)], errors="coerce"),
        pd.to_numeric(result[mesh_column("y", 133)], errors="coerce") - pd.to_numeric(result[mesh_column("y", 362)], errors="coerce"),
    )
    result["rgb_right_iris_diameter_px"] = pairwise_max_distance(result, RIGHT_IRIS)
    result["rgb_left_iris_diameter_px"] = pairwise_max_distance(result, LEFT_IRIS)
    result["rgb_iris_diameter_px"] = result[["rgb_right_iris_diameter_px", "rgb_left_iris_diameter_px"]].mean(axis=1)
    right_x, right_y = mean_point(result, RIGHT_IRIS)
    left_x, left_y = mean_point(result, LEFT_IRIS)
    result["rgb_iris_center_distance_px"] = np.hypot(right_x - left_x, right_y - left_y)
    pose_columns = ["Pitch", "Roll", "Yaw"]
    translation_columns = ["X", "Y", "Z"]
    result["rgb_head_rotation_magnitude"] = np.sqrt(sum(numeric_column_or_nan(result, column) ** 2 for column in pose_columns))
    result["rgb_head_translation_magnitude"] = np.sqrt(sum(numeric_column_or_nan(result, column) ** 2 for column in translation_columns))
    return result


def read_nir(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(NIR_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"NIR CSV missing required columns: {missing}")
    return frame


def read_rgb(path: Path) -> pd.DataFrame:
    available = set(pq.read_schema(path).names)
    mesh_columns = [mesh_column(axis, index) for axis in ("x", "y") for index in range(478)]
    columns = sorted((RGB_REQUIRED | set(mesh_columns)) & available)
    missing = sorted(RGB_REQUIRED - available)
    if missing:
        raise ValueError(f"RGB parquet missing required columns: {missing}")
    frame = pq.read_table(path, columns=columns).to_pandas()
    frame = frame.loc[bool_series(frame["face_rank"]) | pd.to_numeric(frame["face_rank"], errors="coerce").eq(0)] if frame["face_rank"].dtype != object else frame
    frame["_face_rank_numeric"] = pd.to_numeric(frame["face_rank"], errors="coerce")
    frame = frame.loc[frame["_face_rank_numeric"].eq(0)].copy()
    frame["unix_ms"] = pd.to_numeric(frame["unix_ms"], errors="coerce")
    frame = frame.dropna(subset=["unix_ms"]).copy()
    frame["unix_ms"] = frame["unix_ms"].astype("int64")
    frame = frame.sort_values("unix_ms").drop_duplicates("unix_ms", keep="first")
    return derive_rgb_features(frame.drop(columns=["_face_rank_numeric"]))


XLSX_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _xlsx_column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if not letters:
        raise ValueError(f"invalid Excel cell reference: {reference}")
    result = 0
    for char in letters.group(0):
        result = result * 26 + ord(char) - ord("A") + 1
    return result


def _xlsx_shared_string(element: ET.Element) -> str:
    return "".join(text.text or "" for text in element.iter("{%s}t" % XLSX_NS["m"]))


def _normalize_identity_value(value: Any, kind: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    if kind == "phone":
        return re.sub(r"\D", "", normalized)
    if kind == "id_or_student":
        return re.sub(r"[^0-9A-Z]", "", normalized)
    if kind == "name":
        return re.sub(r"\s+", "", normalized)
    return normalized


def _normalize_experiment_id(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    if re.fullmatch(r"\d+\.0+", normalized):
        normalized = normalized.split(".", 1)[0]
    normalized = re.sub(r"[^0-9A-Z]", "", normalized)
    return normalized.zfill(3) if normalized.isdigit() else normalized


def _nullable_text(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def read_repeat_registry_csv(path: Path) -> dict[str, Any]:
    """Read an external, non-PII repeat registry keyed by experiment ID.

    The registry is deliberately separate from the raw Beijing workbook.  It
    may carry provisional local IDs and an optional authoritative global ID,
    but must not carry names, phones, student IDs, or other raw identity
    fields.  Each experiment ID must map to exactly one local participant
    group so downstream grouping cannot silently split repeat visits.
    """

    if not path.exists():
        raise FileNotFoundError(f"repeat registry does not exist: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {"local_repeat_participant_id", "experiment_ids", "session_count"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"repeat registry missing required columns: {missing}")
    forbidden = {"name", "phone", "id_or_student", "身份证", "学号", "手机号", "电话"}
    forbidden_present = sorted(forbidden & set(frame.columns))
    if forbidden_present:
        raise ValueError(f"repeat registry must not contain raw identity columns: {forbidden_present}")

    by_experiment_id: dict[str, dict[str, Any]] = {}
    for row_index, row in frame.iterrows():
        local_id = str(row["local_repeat_participant_id"]).strip()
        if not local_id:
            raise ValueError(f"repeat registry row {row_index + 2} has an empty local_repeat_participant_id")
        experiment_ids = sorted(
            {
                normalized
                for value in str(row["experiment_ids"]).split("|")
                if (normalized := _normalize_experiment_id(value))
            }
        )
        if not experiment_ids:
            raise ValueError(f"repeat registry row {row_index + 2} has no experiment_ids")
        entry = {
            "local_repeat_participant_id": local_id,
            "global_repeat_participant_id": _nullable_text(row.get("global_repeat_participant_id")),
            "experiment_ids": experiment_ids,
            "session_count": int(float(row["session_count"])),
            "repeat_visits_beyond_first": int(float(row["repeat_visits_beyond_first"])) if str(row.get("repeat_visits_beyond_first", "")).strip() else None,
            "identity_match_basis": _nullable_text(row.get("identity_match_basis")),
            "row_number": row_index + 2,
        }
        for experiment_id in experiment_ids:
            previous = by_experiment_id.get(experiment_id)
            if previous and previous["local_repeat_participant_id"] != local_id:
                raise ValueError(
                    f"experiment_id {experiment_id} maps to multiple local repeat groups: "
                    f"{previous['local_repeat_participant_id']} and {local_id}"
                )
            by_experiment_id[experiment_id] = entry
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "local_repeat_participant_groups": int(frame["local_repeat_participant_id"].nunique()),
        "experiment_id_mappings": int(len(by_experiment_id)),
        "contains_raw_identity_values": False,
        "join_key": "experiment_id",
        "by_experiment_id": by_experiment_id,
    }


def read_participant_info_xlsx(path: Path) -> pd.DataFrame:
    """Read only identity columns from the local Excel workbook.

    The workbook contains PII.  This function returns normalized identity
    tokens for in-memory matching only; callers must never serialize them.
    It uses the XLSX XML parts directly because the bundled openpyxl version
    cannot read this workbook's legacy DataValidation metadata.
    """

    if not path.exists():
        return pd.DataFrame(columns=["sheet", "experiment_id", "name", "phone", "id_or_student"])
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [_xlsx_shared_string(element) for element in root.findall("m:si", XLSX_NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_map = {element.attrib["Id"]: element.attrib["Target"] for element in relationships}
        records: list[dict[str, str]] = []
        for sheet in workbook.find("m:sheets", XLSX_NS):
            name = sheet.attrib["name"]
            target = relationship_map[sheet.attrib["{%s}id" % XLSX_NS["r"]]]
            if not target.startswith("xl/"):
                target = "xl/" + target
            sheet_root = ET.fromstring(archive.read(target))
            for row in sheet_root.findall(".//m:sheetData/m:row", XLSX_NS):
                if int(row.attrib.get("r", "0")) == 1:
                    continue
                values: dict[int, str] = {}
                for cell in row.findall("m:c", XLSX_NS):
                    index = _xlsx_column_index(cell.attrib["r"])
                    if index > 9:
                        continue
                    value_element = cell.find("m:v", XLSX_NS)
                    if value_element is None:
                        value = ""
                    elif cell.attrib.get("t") == "s":
                        value = shared[int(value_element.text or "0")]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = _xlsx_shared_string(cell)
                    else:
                        value = value_element.text or ""
                    values[index] = value
                records.append(
                    {
                        "sheet": name,
                        "experiment_id": _normalize_experiment_id(values.get(4)),
                        "name": _normalize_identity_value(values.get(5), "name"),
                        "phone": _normalize_identity_value(values.get(6), "phone"),
                        "id_or_student": _normalize_identity_value(values.get(9), "id_or_student"),
                    }
                )
    return pd.DataFrame.from_records(records, columns=["sheet", "experiment_id", "name", "phone", "id_or_student"])


def read_identity_summary_xlsx(path: Path) -> pd.DataFrame:
    """Read the separate phone/student-ID/participation-count workbook."""

    if not path.exists():
        return pd.DataFrame(columns=["sheet", "phone", "id_or_student", "participation"])
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [_xlsx_shared_string(element) for element in root.findall("m:si", XLSX_NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_map = {element.attrib["Id"]: element.attrib["Target"] for element in relationships}
        records: list[dict[str, str]] = []
        for sheet in workbook.find("m:sheets", XLSX_NS):
            name = sheet.attrib["name"]
            target = relationship_map[sheet.attrib["{%s}id" % XLSX_NS["r"]]]
            if not target.startswith("xl/"):
                target = "xl/" + target
            sheet_root = ET.fromstring(archive.read(target))
            for row in sheet_root.findall(".//m:sheetData/m:row", XLSX_NS):
                if int(row.attrib.get("r", "0")) == 1:
                    continue
                values: dict[int, str] = {}
                for cell in row.findall("m:c", XLSX_NS):
                    index = _xlsx_column_index(cell.attrib["r"])
                    if index > 5:
                        continue
                    value_element = cell.find("m:v", XLSX_NS)
                    if value_element is None:
                        value = ""
                    elif cell.attrib.get("t") == "s":
                        value = shared[int(value_element.text or "0")]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = _xlsx_shared_string(cell)
                    else:
                        value = value_element.text or ""
                    values[index] = value
                record = {
                    "sheet": name,
                    "phone": _normalize_identity_value(values.get(2), "phone"),
                    "id_or_student": _normalize_identity_value(values.get(4), "id_or_student"),
                    "participation": values.get(5, ""),
                }
                if record["phone"] or record["id_or_student"] or record["participation"]:
                    records.append(record)
    return pd.DataFrame.from_records(records, columns=["sheet", "phone", "id_or_student", "participation"])


def identity_groups(records: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for field in ("phone", "id_or_student"):
            token = str(record.get(field, ""))
            if not token:
                continue
            key = (field, token)
            if key in first_seen:
                union(index, first_seen[key])
            else:
                first_seen[key] = index
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    return list(groups.values())


def participation_count(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def identity_roster_audit(participant_info_path: Path, summary_path: Path) -> dict[str, Any]:
    """Compare the experiment log with the separate participation summary."""

    experiment_sheets = read_participant_info_xlsx(participant_info_path)
    experiment_rows = experiment_sheets.loc[experiment_sheets["experiment_id"].ne("")].to_dict("records")
    experiment_identity_rows = [row for row in experiment_rows if row["phone"] or row["id_or_student"]]
    experiment_groups = identity_groups(experiment_identity_rows)
    direct_repeat_groups = []
    for group in experiment_groups:
        experiment_ids = sorted({str(row["experiment_id"]) for row in group if row["experiment_id"]})
        if len(experiment_ids) > 1:
            matched_fields = []
            if len({row["phone"] for row in group if row["phone"]}) < len(group):
                matched_fields.append("phone")
            if len({row["id_or_student"] for row in group if row["id_or_student"]}) < len(group):
                matched_fields.append("id_or_student")
            direct_repeat_groups.append(
                {
                    "experiment_ids": experiment_ids,
                    "session_count": len(experiment_ids),
                    "matched_fields": matched_fields,
                    "source_sheets": sorted({str(row["sheet"]) for row in group if row.get("sheet")}),
                }
            )

    summary_rows = read_identity_summary_xlsx(summary_path).to_dict("records")
    summary_groups = identity_groups(summary_rows)
    summary_group_items = []
    for group in summary_groups:
        values = [participation_count(row["participation"]) for row in group]
        values = [value for value in values if value is not None]
        summary_group_items.append({"group": group, "reported_participation_count": max(values) if values else None})

    mapped_repeat_groups = []
    unmapped_repeat_groups = []
    for item in summary_group_items:
        reported = item["reported_participation_count"]
        if reported is None or reported <= 1:
            continue
        group = item["group"]
        summary_tokens = {(field, str(row.get(field, ""))) for row in group for field in ("phone", "id_or_student") if row.get(field, "")}
        experiment_ids: set[str] = set()
        matched_fields: set[str] = set()
        for experiment_group in experiment_groups:
            experiment_tokens = {(field, str(row.get(field, ""))) for row in experiment_group for field in ("phone", "id_or_student") if row.get(field, "")}
            overlap = summary_tokens & experiment_tokens
            if overlap:
                experiment_ids.update(str(row["experiment_id"]) for row in experiment_group if row["experiment_id"])
                matched_fields.update(field for field, _ in overlap)
        item_out = {
            "experiment_ids": sorted(experiment_ids),
            "session_count_from_experiment_log": len(experiment_ids),
            "reported_participation_count": reported,
            "identity_rows_in_summary": len(group),
            "matched_fields": sorted(matched_fields),
            "mapped_to_experiment_log": bool(experiment_ids),
        }
        if experiment_ids:
            mapped_repeat_groups.append(item_out)
        else:
            unmapped_repeat_groups.append({"reported_participation_count": reported, "identity_rows_in_summary": len(group)})

    mismatch_groups = [
        item for item in mapped_repeat_groups
        if item["session_count_from_experiment_log"] != item["reported_participation_count"]
    ]
    single_summary_but_experiment_repeat = []
    for item in summary_group_items:
        if item["reported_participation_count"] != 1:
            continue
        group = item["group"]
        summary_tokens = {(field, str(row.get(field, ""))) for row in group for field in ("phone", "id_or_student") if row.get(field, "")}
        experiment_ids: set[str] = set()
        for experiment_group in experiment_groups:
            experiment_tokens = {(field, str(row.get(field, ""))) for row in experiment_group for field in ("phone", "id_or_student") if row.get(field, "")}
            if summary_tokens & experiment_tokens:
                experiment_ids.update(str(row["experiment_id"]) for row in experiment_group if row["experiment_id"])
        if len(experiment_ids) > 1:
            single_summary_but_experiment_repeat.append({"experiment_ids": sorted(experiment_ids), "reported_participation_count": 1, "session_count_from_experiment_log": len(experiment_ids)})

    participation_distribution: dict[str, int] = {}
    for item in summary_group_items:
        value = item["reported_participation_count"]
        if value is not None:
            participation_distribution[str(value)] = participation_distribution.get(str(value), 0) + 1
    repeat_distribution = {key: value for key, value in participation_distribution.items() if int(key) > 1}
    return {
        "experiment_log": {
            "workbook": str(participant_info_path),
            "experiment_sessions": len(experiment_rows),
            "sessions_with_phone_or_id": len(experiment_identity_rows),
            "identity_groups": len(experiment_groups),
            "direct_repeat_groups": len(direct_repeat_groups),
            "direct_repeat_groups_detail": sorted(direct_repeat_groups, key=lambda item: item["experiment_ids"]),
        },
        "participation_summary": {
            "workbook": str(summary_path),
            "identity_rows": len(summary_rows),
            "identity_groups": len(summary_groups),
            "participation_count_group_distribution": participation_distribution,
            "repeat_participant_count": sum(repeat_distribution.values()),
            "repeat_participant_count_distribution": repeat_distribution,
            "repeat_visits_beyond_first_declared": sum((int(key) - 1) * value for key, value in repeat_distribution.items()),
            "declared_participation_slots": sum(int(key) * value for key, value in participation_distribution.items()),
        },
        "cross_file": {
            "experiment_identity_groups_matched_to_summary": sum(
                1 for group in experiment_groups
                if any(
                    {(field, str(row.get(field, ""))) for row in group for field in ("phone", "id_or_student") if row.get(field, "")}
                    & {(field, str(row.get(field, ""))) for row in summary_rows for field in ("phone", "id_or_student") if row.get(field, "")}
                    for _ in [0]
                )
            ),
            "mapped_repeat_groups": len(mapped_repeat_groups),
            "unmapped_repeat_groups": len(unmapped_repeat_groups),
            "mapped_repeat_groups_detail": sorted(mapped_repeat_groups, key=lambda item: (item["reported_participation_count"], item["experiment_ids"])),
            "unmapped_repeat_groups_detail": unmapped_repeat_groups,
            "count_mismatch_groups": mismatch_groups,
            "summary_single_but_experiment_log_repeat_groups": single_summary_but_experiment_repeat,
            "all_identity_fields_consistent": not mismatch_groups and not single_summary_but_experiment_repeat and not unmapped_repeat_groups,
        },
    }


def write_beijing_repeat_csv(path: Path, identity_roster: Mapping[str, Any], participant_info_path: Path) -> None:
    """Write a non-PII repeat-participant table from the Beijing experiment log."""

    groups = identity_roster["experiment_log"]["direct_repeat_groups_detail"]
    fieldnames = [
        "site",
        "local_repeat_participant_id",
        "experiment_ids",
        "session_count",
        "repeat_visits_beyond_first",
        "identity_match_basis",
        "source_sheets",
        "identity_status",
        "local_participant_linkage_key",
        "global_repeat_participant_id",
        "source_workbook",
        "note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, group in enumerate(groups, start=1):
            experiment_ids = group["experiment_ids"]
            stable_id = f"beijing_xlsx_repeat_{index:03d}"
            writer.writerow(
                {
                    "site": "Beijing",
                    "local_repeat_participant_id": stable_id,
                    "experiment_ids": "|".join(experiment_ids),
                    "session_count": group["session_count"],
                    "repeat_visits_beyond_first": group["session_count"] - 1,
                    "identity_match_basis": "+".join(group["matched_fields"]),
                    "source_sheets": "|".join(group.get("source_sheets", [])),
                    "identity_status": "local_repeat_confirmed_provisional",
                    "local_participant_linkage_key": stable_id,
                    "global_repeat_participant_id": "",
                    "source_workbook": str(participant_info_path),
                    "note": "Derived from normalized phone/ID-student matches; raw identity values intentionally excluded.",
                }
            )


def _local_session_output_presence(experiment_ids: Sequence[str]) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = {}
    for experiment_id in experiment_ids:
        result[experiment_id] = {
            "nir_output_present": any(DEFAULT_NIR_ROOT.glob(f"sub-{experiment_id}_*")),
            "rgb_output_present": (DEFAULT_RGB_ROOT / f"sub-{experiment_id}").exists(),
        }
    return result


def identity_audit(
    path: Path,
    subject: str,
    participant_info_path: Path,
    repeat_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = {
        "site": "Beijing",
        "session_id": subject,
        "session_key": f"Beijing:{subject}",
        "subject": subject,
        "local_participant_linkage_key": None,
        "identity_status": "unresolved_local_bridge",
        "local_repeat_participant_id": None,
        "global_repeat_participant_id": None,
        "repeat_session_count": None,
        "repeat_visits_beyond_first": None,
        "identity_match_basis": None,
        "current_session_repeat_status": "undetermined",
        "central_authoritative_identity_map_present": False,
        "identity_evidence_source": str(participant_info_path) if participant_info_path.exists() else (str(path) if path.exists() else None),
        "identity_evidence_basis": "local workbook identity match on normalized phone and/or ID/student fields; raw identity values are never serialized",
    }
    participant_info = read_participant_info_xlsx(participant_info_path)
    experiment_match = re.search(r"(\d+)$", subject)
    experiment_id = experiment_match.group(1).zfill(3) if experiment_match else ""
    session_rows = participant_info.loc[participant_info["experiment_id"].eq(experiment_id)]
    matches: dict[str, set[str]] = {}
    if len(session_rows):
        for _, current in session_rows.iterrows():
            for _, other in participant_info.loc[participant_info["experiment_id"].ne(experiment_id)].iterrows():
                methods: set[str] = set()
                if current["phone"] and current["phone"] == other["phone"]:
                    methods.add("phone")
                if current["id_or_student"] and current["id_or_student"] == other["id_or_student"]:
                    methods.add("id_or_student")
                if methods and other["experiment_id"]:
                    matches.setdefault(other["experiment_id"], set()).update(methods)
    other_ids = sorted(matches)
    identity_tokens = [
        f"phone:{row['phone']}" for _, row in session_rows.iterrows() if row["phone"]
    ] + [
        f"id_or_student:{row['id_or_student']}" for _, row in session_rows.iterrows() if row["id_or_student"]
    ]
    if identity_tokens:
        linkage_digest = hashlib.sha256("|".join(sorted(identity_tokens)).encode("utf-8")).hexdigest()[:16]
        base["local_participant_linkage_key"] = f"local_xlsx_{linkage_digest}"
    if other_ids:
        base.update(
            {
                "identity_status": "local_repeat_confirmed_provisional",
                "local_repeat_participant_id": f"provisional_xlsx_{linkage_digest}",
                "current_session_repeat_status": "repeat_participant_confirmed_locally",
            }
        )
    elif len(session_rows):
        base["identity_status"] = "local_single_visit_or_unmatched_provisional"
        base["current_session_repeat_status"] = "no_repeat_match_in_workbook"
    registry_entry = None
    if repeat_registry is not None:
        registry_entry = repeat_registry.get("by_experiment_id", {}).get(experiment_id)
        base.update(
            {
                "repeat_registry_present": True,
                "repeat_registry_path": repeat_registry.get("path"),
                "repeat_registry_join_key": repeat_registry.get("join_key", "experiment_id"),
                "repeat_registry_match_found": registry_entry is not None,
                "repeat_registry_match_experiment_ids": registry_entry.get("experiment_ids", []) if registry_entry else [],
            }
        )
        if registry_entry is not None:
            base.update(
                {
                    "identity_evidence_source": repeat_registry.get("path"),
                    "identity_evidence_basis": "external non-PII repeat registry join on experiment_id; raw identity values are not serialized",
                    "local_participant_linkage_key": registry_entry["local_repeat_participant_id"],
                    "local_repeat_participant_id": registry_entry["local_repeat_participant_id"],
                    "global_repeat_participant_id": registry_entry.get("global_repeat_participant_id"),
                    "repeat_session_count": registry_entry.get("session_count"),
                    "repeat_visits_beyond_first": registry_entry.get("repeat_visits_beyond_first"),
                    "identity_match_basis": registry_entry.get("identity_match_basis"),
                    "identity_status": "local_repeat_confirmed_provisional",
                    "current_session_repeat_status": "repeat_participant_joined_from_external_registry",
                }
            )
    else:
        base.update(
            {
                "repeat_registry_present": False,
                "repeat_registry_path": None,
                "repeat_registry_join_key": None,
                "repeat_registry_match_found": False,
                "repeat_registry_match_experiment_ids": [],
            }
        )
    base.update(
        {
            "participant_info_workbook_present": participant_info_path.exists(),
            "participant_info_rows": int(len(participant_info)),
            "participant_info_experiment_id": experiment_id or None,
            "participant_info_session_rows": int(len(session_rows)),
            "participant_info_session_identity_presence": {
                "name": int(session_rows["name"].ne("").sum()),
                "phone": int(session_rows["phone"].ne("").sum()),
                "id_or_student": int(session_rows["id_or_student"].ne("").sum()),
            },
            "participant_info_other_experiment_ids": other_ids,
            "participant_info_match_methods": {experiment: sorted(methods) for experiment, methods in matches.items()},
            "participant_info_other_session_output_presence": _local_session_output_presence(other_ids),
        }
    )
    if path.exists():
        registry = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        required = {"participant_key", "experiment_id", "is_repeat_participant", "is_cross_stage_repeat", "identity_conflict_flag"}
        missing = sorted(required - set(registry.columns))
        if not missing:
            participant_counts = registry.groupby("participant_key", dropna=False).size()
            repeat_participants = participant_counts[participant_counts > 1]
            registry_experiment_ids = registry["experiment_id"].map(_normalize_experiment_id)
            registry_session = registry.loc[registry_experiment_ids.eq(experiment_id)]
            base.update(
                {
                    "registry_present": True,
                    "registry_rows": int(len(registry)),
                    "registry_unique_participant_keys": int(registry["participant_key"].nunique(dropna=True)),
                    "registry_repeat_participant_keys": int(len(repeat_participants)),
                    "registry_repeat_rows": int(bool_series(registry["is_repeat_participant"]).sum()),
                    "registry_cross_stage_repeat_rows": int(bool_series(registry["is_cross_stage_repeat"]).sum()),
                    "registry_identity_conflict_rows": int(bool_series(registry["identity_conflict_flag"]).sum()),
                    "registry_nonempty_experiment_ids": int(registry["experiment_id"].astype("string").str.strip().ne("").sum()),
                    "registry_unique_experiment_ids": int(registry["experiment_id"].nunique(dropna=True)),
                    "registry_session_rows": int(len(registry_session)),
                    "registry_session_repeat_flags": [{
                        "is_repeat_participant": row.get("is_repeat_participant"),
                        "is_cross_stage_repeat": row.get("is_cross_stage_repeat"),
                        "identity_conflict_flag": row.get("identity_conflict_flag"),
                    } for _, row in registry_session.iterrows()],
                    "registry_vs_workbook_identity_note": "workbook personal-field match is the local repeat evidence; derived registry flags are retained separately for reconciliation",
                }
            )
        else:
            base["registry_present"] = False
            base["registry_schema_missing"] = missing
    else:
        base["registry_present"] = False
    return base


def relation_table(paired: pd.DataFrame, x_columns: Sequence[str], y_column: str = "log_pupil_diameter", method: str = "spearman") -> dict[str, Any]:
    return {column: safe_corr(paired[column], paired[y_column], method=method) for column in x_columns if column in paired}


def phase_subset(frame: pd.DataFrame, phases: set[str]) -> pd.DataFrame:
    return frame.loc[frame["nir_phase"].astype("string").isin(phases)].copy()


def relation_by_phase(paired: pd.DataFrame, x_columns: Sequence[str], y_column: str = "log_pupil_diameter") -> dict[str, Any]:
    return {
        "all": relation_table(paired, x_columns, y_column=y_column),
        "baseline": relation_table(phase_subset(paired, {"baseline"}), x_columns, y_column=y_column),
        "task_blocks": relation_table(phase_subset(paired, {"block1", "block2"}), x_columns, y_column=y_column),
    }


def rgb_iris_stability(rgb: pd.DataFrame) -> dict[str, Any]:
    valid = rgb.loc[pd.to_numeric(rgb["rgb_iris_diameter_px"], errors="coerce").notna()].sort_values("unix_ms").copy()
    sample_n = min(500, len(valid))
    if sample_n:
        sample = valid.iloc[np.linspace(0, len(valid) - 1, sample_n, dtype=int)]
    else:
        sample = valid
    summary = {"valid_rows": int(len(valid)), "sample_n": int(len(sample)), "all": summarize_numeric(valid["rgb_iris_diameter_px"]), "sample": summarize_numeric(sample["rgb_iris_diameter_px"])}
    if len(valid) > 1:
        diff = pd.to_numeric(valid["rgb_iris_diameter_px"], errors="coerce").diff().abs().dropna()
        summary["all"]["successive_abs_difference_px"] = {"median": float(diff.median()), "p95": float(diff.quantile(0.95)), "n": int(len(diff))}
    baseline = valid.loc[valid["phase"].astype("string").eq("baseline")]
    summary["baseline"] = summarize_numeric(baseline["rgb_iris_diameter_px"])
    for pose in ("Yaw", "Pitch"):
        pose_abs = pd.to_numeric(valid[pose], errors="coerce").abs()
        low_cut, high_cut = pose_abs.quantile([0.2, 0.8]) if pose_abs.notna().any() else (np.nan, np.nan)
        low = valid.loc[pose_abs <= low_cut, "rgb_iris_diameter_px"]
        high = valid.loc[pose_abs >= high_cut, "rgb_iris_diameter_px"]
        summary[f"abs_{pose}_low_high"] = {"low_cut": float(low_cut), "high_cut": float(high_cut), "low": summarize_numeric(low), "high": summarize_numeric(high), "high_minus_low_median": float(high.median() - low.median()) if len(low) and len(high) else None, "high_vs_low_ratio_median": float(high.median() / low.median()) if len(low) and len(high) and low.median() else None, "correlation_with_abs_pose": safe_corr(pose_abs, valid["rgb_iris_diameter_px"], method="spearman")}
    return summary


def select_sample_times(paired: pd.DataFrame, sample_size: int) -> np.ndarray:
    valid = paired.loc[paired["pupil_valid"]].dropna(subset=["unix_ms_nir", "unix_ms_rgb"])
    times = np.sort(valid["unix_ms_nir"].astype("int64").unique())
    if len(times) <= sample_size:
        return times
    return times[np.linspace(0, len(times) - 1, sample_size, dtype=int)]


def build_paired(nir: pd.DataFrame, rgb: pd.DataFrame, tolerance_ms: int) -> pd.DataFrame:
    left = nir.copy()
    left["unix_ms"] = pd.to_numeric(left["unix_ms"], errors="coerce")
    left = left.rename(columns={"unix_ms": "unix_ms_nir", "phase": "nir_phase", "phase_segment": "nir_phase_segment"}).sort_values("unix_ms_nir")
    right = rgb.copy()
    right["unix_ms"] = pd.to_numeric(right["unix_ms"], errors="coerce")
    right = right.rename(columns={"unix_ms": "unix_ms_rgb", "phase": "rgb_phase"}).sort_values("unix_ms_rgb")
    paired = pd.merge_asof(left, right, left_on="unix_ms_nir", right_on="unix_ms_rgb", direction="nearest", tolerance=tolerance_ms, suffixes=("_nir", "_rgb"))
    paired["delta_ms"] = paired["unix_ms_rgb"] - paired["unix_ms_nir"]
    paired["abs_delta_ms"] = paired["delta_ms"].abs()
    found = bool_series(paired["fullclass_pupil_found"])
    fit = bool_series(paired["fullclass_pupil_fit_valid"])
    diameter = pd.to_numeric(paired["fullclass_pupil_geom_mean_diameter"], errors="coerce")
    paired["pupil_valid"] = found & fit & diameter.gt(0) & diameter.notna()
    paired["pupil_diameter_px"] = diameter
    paired["log_pupil_diameter"] = np.log(diameter.where(diameter.gt(0)))
    axis_a = pd.to_numeric(paired["fullclass_pupil_axis_a"], errors="coerce")
    axis_b = pd.to_numeric(paired["fullclass_pupil_axis_b"], errors="coerce")
    paired["pupil_ellipse_ratio_a_over_b"] = axis_a / axis_b.where(axis_b.ne(0))
    return paired


def make_sample(paired: pd.DataFrame, sample_times: np.ndarray, identity: Mapping[str, Any], nir_manifest: Mapping[str, Any], rgb_manifest: Mapping[str, Any]) -> pd.DataFrame:
    sample = paired.loc[paired["unix_ms_nir"].isin(sample_times)].copy()
    columns: dict[str, Any] = {
        "site": identity["site"],
        "session_id": identity["session_id"],
        "session_key": identity["session_key"],
        "subject": identity["subject"],
        "local_participant_linkage_key": identity["local_participant_linkage_key"],
        "identity_status": identity["identity_status"],
        "local_repeat_participant_id": identity["local_repeat_participant_id"],
        "global_repeat_participant_id": identity["global_repeat_participant_id"],
        "identity_evidence_source": identity["identity_evidence_source"],
        "nir_schema_version": nir_manifest.get("extension_version", "unrecorded"),
        "rgb_schema_version": rgb_manifest.get("schema_version", "unrecorded"),
        "nir_git_commit": "unrecorded_in_manifest",
        "rgb_git_commit": "unrecorded_in_manifest",
        "unix_ms_nir": sample["unix_ms_nir"],
        "unix_ms_rgb": sample["unix_ms_rgb"],
        "delta_ms": sample["delta_ms"],
        "abs_delta_ms": sample["abs_delta_ms"],
        "eye": sample["eye"],
        "nir_phase": sample["nir_phase"],
        "rgb_phase": sample["rgb_phase"],
        "nir_frame_idx": sample["frame_idx"],
        "nir_video_time_ms": sample["video_time_ms"],
        "nir_phase_time_ms": sample["phase_time_ms"],
        "pupil_valid": sample["pupil_valid"],
        "pupil_diameter_px": sample["pupil_diameter_px"],
        "fullclass_pupil_geom_mean_diameter": sample["fullclass_pupil_geom_mean_diameter"],
        "fullclass_pupil_equiv_diameter": sample["fullclass_pupil_equiv_diameter"],
        "fullclass_pupil_axis_a": sample["fullclass_pupil_axis_a"],
        "fullclass_pupil_axis_b": sample["fullclass_pupil_axis_b"],
        "fullclass_pupil_ellipse_ratio_a_over_b": sample["pupil_ellipse_ratio_a_over_b"],
        "fullclass_pupil_contour_area": sample["fullclass_pupil_contour_area"],
        "fullclass_pupil_center_x": sample["fullclass_pupil_center_x"],
        "fullclass_pupil_center_y": sample["fullclass_pupil_center_y"],
        "fullclass_pupil_found": sample["fullclass_pupil_found"],
        "fullclass_pupil_fit_valid": sample["fullclass_pupil_fit_valid"],
        "fullclass_pupil_touches_roi_edge": sample["fullclass_pupil_touches_roi_edge"],
        "fullclass_pupil_to_iris_diameter_ratio": sample["fullclass_pupil_to_iris_diameter_ratio"],
        "fullclass_normalization_valid": sample["fullclass_normalization_valid"],
        "roi_clipped": sample["roi_clipped"],
        "bbox_x1": sample["bbox_x1"],
        "bbox_y1": sample["bbox_y1"],
        "bbox_x2": sample["bbox_x2"],
        "bbox_y2": sample["bbox_y2"],
        "roi_x1": sample["roi_x1"],
        "roi_y1": sample["roi_y1"],
        "roi_x2": sample["roi_x2"],
        "roi_y2": sample["roi_y2"],
        "rgb_face_bbox_width_px": sample["FaceRectWidth"],
        "rgb_face_bbox_height_px": sample["FaceRectHeight"],
        "rgb_face_bbox_area_px2": sample["rgb_face_bbox_area_px2"],
        "rgb_face_bbox_scale_px": sample["rgb_face_bbox_scale_px"],
        "rgb_eye_outer_corner_distance_px": sample["rgb_eye_outer_corner_distance_px"],
        "rgb_eye_inner_canthus_distance_px": sample["rgb_eye_inner_canthus_distance_px"],
        "rgb_right_iris_diameter_px": sample["rgb_right_iris_diameter_px"],
        "rgb_left_iris_diameter_px": sample["rgb_left_iris_diameter_px"],
        "rgb_iris_diameter_px": sample["rgb_iris_diameter_px"],
        "rgb_iris_center_distance_px": sample["rgb_iris_center_distance_px"],
        "rgb_face_score": sample["FaceScore"],
        "rgb_detected": sample["detected"],
        "rgb_face_rank": sample["face_rank"],
        "rgb_pitch": sample["Pitch"],
        "rgb_roll": sample["Roll"],
        "rgb_yaw": sample["Yaw"],
        "rgb_x": sample["X"],
        "rgb_y": sample["Y"],
        "rgb_z": sample["Z"],
        "rgb_gaze_pitch": sample["gaze_pitch"],
        "rgb_gaze_yaw": sample["gaze_yaw"],
        "rgb_temporal_gap": sample["temporal_gap"],
        "rgb_capture_gap_before": sample["capture_gap_before"],
    }
    return pd.DataFrame(columns)


def pupil_field_catalog(nir: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        ("fullclass_pupil_geom_mean_diameter", "pupil-only geometric mean diameter candidate", False),
        ("fullclass_pupil_equiv_diameter", "pupil-only equivalent diameter candidate", False),
        ("fullclass_pupil_axis_a", "pupil-only ellipse axis candidate", False),
        ("fullclass_pupil_axis_b", "pupil-only ellipse axis candidate", False),
        ("fullclass_pupil_contour_area", "pupil-only contour area candidate", False),
        ("fullclass_pupil_center_x", "pupil-only center x in ROI coordinates", False),
        ("fullclass_pupil_center_y", "pupil-only center y in ROI coordinates", False),
        ("fullclass_pupil_to_iris_diameter_ratio", "iris-dependent ratio; not used as a primary pupil field", True),
        ("fullclass_pupil_to_iris_ellipse_area_ratio", "iris-dependent ratio; not used as a primary pupil field", True),
        ("fullclass_normalization_valid", "iris-dependent normalization/QC flag", True),
        ("fullclass_iris_outer_*", "iris outer geometry; iris-dependent", True),
    ]
    result = []
    for field, meaning, iris_dependent in fields:
        present = field in nir.columns or field.endswith("_*")
        valid = finite_fraction(nir[field]) if field in nir.columns else None
        result.append({"field": field, "present": present, "iris_dependent": iris_dependent, "finite_fraction": valid, "meaning": meaning})
    return result


def rgb_field_catalog(rgb: pd.DataFrame) -> list[dict[str, Any]]:
    entries = [
        ("rgb_face_bbox_scale_px", "sqrt(FaceRectWidth*FaceRectHeight)", "FaceRectWidth/Height from primary face output"),
        ("rgb_eye_outer_corner_distance_px", "distance between mesh 33 and 263", "478-point original-frame mesh"),
        ("rgb_eye_inner_canthus_distance_px", "distance between mesh 133 and 362", "478-point original-frame mesh"),
        ("rgb_iris_diameter_px", "max pairwise distance within each 4-point iris, then bilateral mean", "right 469:472; left 474:477"),
        ("Pitch/Roll/Yaw", "canonical rotation fields", "script mapping from raw pose[0:3]; units not established"),
        ("X/Y/Z", "canonical translation-like fields", "script mapping from raw pose[3:6]; units/coordinate semantics unresolved"),
        ("gaze_pitch/gaze_yaw", "gaze-like proxy fields", "canonical gaze output; not pupil measurement"),
    ]
    return [{"field": field, "meaning": meaning, "source": source, "finite_fraction": finite_fraction(rgb[field]) if field in rgb else None} for field, meaning, source in entries]


def run_audit(
    nir_path: Path,
    rgb_path: Path,
    sart_path: Path,
    participant_info_path: Path,
    identity_summary_path: Path,
    output_dir: Path,
    sample_size: int = 1000,
    tolerance_ms: int = 1000,
    repeat_registry_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nir_manifest_path = nir_path.with_name("run_manifest.json")
    extension_manifest_path = nir_path.with_name("sub-031_ritnet_fullclass_manifest.json")
    rgb_manifest_path = rgb_path.with_name("sub-031_face_raw_manifest.json")
    nir_manifest = read_json_if_exists(nir_manifest_path)
    extension_manifest = read_json_if_exists(extension_manifest_path)
    rgb_manifest = read_json_if_exists(rgb_manifest_path)
    nir = read_nir(nir_path)
    rgb = read_rgb(rgb_path)
    subject = str(nir["subject"].dropna().iloc[0])
    repeat_registry = read_repeat_registry_csv(repeat_registry_path) if repeat_registry_path else None
    identity = identity_audit(sart_path, subject, participant_info_path, repeat_registry=repeat_registry)
    identity_roster = identity_roster_audit(participant_info_path, identity_summary_path)
    repeat_csv_path = output_dir / "beijing_repeat_participants.csv"
    write_beijing_repeat_csv(repeat_csv_path, identity_roster, participant_info_path)
    identity_roster["experiment_log"]["repeat_csv"] = {
        "path": str(repeat_csv_path),
        "rows": len(identity_roster["experiment_log"]["direct_repeat_groups_detail"]),
        "contains_raw_identity_values": False,
    }
    paired = build_paired(nir, rgb, tolerance_ms=tolerance_ms)
    matched = paired.dropna(subset=["unix_ms_rgb"]).copy()
    valid = matched.loc[matched["pupil_valid"]].copy()
    sample_times = select_sample_times(paired, sample_size)
    sample = make_sample(paired, sample_times, identity, extension_manifest, rgb_manifest)
    sample_path = output_dir / f"{subject}_paired_sample.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    signed_delta = pd.to_numeric(matched["delta_ms"], errors="coerce")
    alignment = {
        "nir_rows": int(len(nir)),
        "rgb_primary_unique_timestamp_rows": int(len(rgb)),
        "paired_rows": int(len(paired)),
        "matched_rows": int(len(matched)),
        "match_fraction_rows": float(len(matched) / len(paired)) if len(paired) else 0.0,
        "valid_pupil_matched_rows": int(len(valid)),
        "valid_pupil_match_fraction": float(len(valid) / len(paired)) if len(paired) else 0.0,
        "signed_delta_ms": quantiles(signed_delta),
        "absolute_delta_ms": quantiles(signed_delta.abs()),
        "match_fraction_by_window": {str(window): float((signed_delta.abs() <= window).mean()) if len(signed_delta) else 0.0 for window in (20, 40, 50, 100, 200, 1000)},
        "large_delta_gt_200ms": int((signed_delta.abs() > 200).sum()),
        "large_delta_gt_1000ms": int((signed_delta.abs() > 1000).sum()),
        "reasonable_window_recommendation_ms": 50,
        "clock_jump_or_gap_note": "Timeline gap counts are reported separately; nearest matching uses actual unix_ms fields, not target_unix_ms.",
    }

    scale_columns = ["rgb_face_bbox_scale_px", "rgb_eye_outer_corner_distance_px", "rgb_eye_inner_canthus_distance_px", "rgb_iris_diameter_px", "rgb_iris_center_distance_px"]
    pose_columns = ["Pitch", "Roll", "Yaw", "X", "Y", "Z", "gaze_pitch", "gaze_yaw"]
    center_columns = ["fullclass_pupil_center_x", "fullclass_pupil_center_y"]
    paired_rel = valid.copy()
    relations = {
        "log_pupil_vs_rgb_scale": relation_by_phase(paired_rel, scale_columns),
        "log_pupil_vs_pose_and_gaze": relation_by_phase(paired_rel, pose_columns),
        "pupil_ellipse_ratio_vs_pose_and_gaze": {"all": relation_table(paired_rel, pose_columns, y_column="pupil_ellipse_ratio_a_over_b"), "baseline": relation_table(phase_subset(paired_rel, {"baseline"}), pose_columns, y_column="pupil_ellipse_ratio_a_over_b"), "task_blocks": relation_table(phase_subset(paired_rel, {"block1", "block2"}), pose_columns, y_column="pupil_ellipse_ratio_a_over_b")},
        "pupil_center_vs_pose_and_gaze": {axis: relation_by_phase(paired_rel, pose_columns, y_column=axis) for axis in center_columns},
    }
    rotation = pd.to_numeric(paired_rel["rgb_head_rotation_magnitude"], errors="coerce")
    valid_motion = paired_rel.loc[rotation.notna()].copy()
    rotation = pd.to_numeric(valid_motion["rgb_head_rotation_magnitude"], errors="coerce")
    low_cut, high_cut = rotation.quantile([0.2, 0.8]) if len(rotation) else (np.nan, np.nan)
    low = valid_motion.loc[rotation <= low_cut, "pupil_diameter_px"]
    high = valid_motion.loc[rotation >= high_cut, "pupil_diameter_px"]
    relations["large_small_head_motion"] = {
        "rotation_magnitude_q20": float(low_cut) if np.isfinite(low_cut) else None,
        "rotation_magnitude_q80": float(high_cut) if np.isfinite(high_cut) else None,
        "low_motion": summarize_numeric(low),
        "high_motion": summarize_numeric(high),
        "high_minus_low_median": float(high.median() - low.median()) if len(low) and len(high) else None,
        "high_vs_low_ratio_median": float(high.median() / low.median()) if len(low) and len(high) and low.median() else None,
        "scale_correlation_low_motion": safe_corr(valid_motion.loc[rotation <= low_cut, "rgb_face_bbox_scale_px"], low, method="spearman"),
        "scale_correlation_high_motion": safe_corr(valid_motion.loc[rotation >= high_cut, "rgb_face_bbox_scale_px"], high, method="spearman"),
    }

    summary = {
        "audit_version": AUDIT_VERSION,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "input": {
            "nir_csv": str(nir_path),
            "nir_csv_sha256": sha256_file(nir_path),
            "rgb_parquet": str(rgb_path),
            "rgb_parquet_sha256": sha256_file(rgb_path),
            "nir_run_manifest": str(nir_manifest_path),
            "nir_extension_manifest": str(extension_manifest_path),
            "rgb_manifest": str(rgb_manifest_path),
            "sart_registry": str(sart_path),
            "participant_info_workbook": str(participant_info_path),
            "identity_summary_workbook": str(identity_summary_path),
            "repeat_registry": repeat_registry_path and str(repeat_registry_path),
        },
        "provenance": {
            "nir_schema_version": extension_manifest.get("extension_version", "unrecorded"),
            "rgb_schema_version": rgb_manifest.get("schema_version", "unrecorded"),
            "nir_git_commit": "unrecorded_in_manifest",
            "rgb_git_commit": "unrecorded_in_manifest",
            "nir_package_version": nir_manifest.get("package", {}).get("version"),
            "nir_model_sha256": nir_manifest.get("ritnet_sha256") or extension_manifest.get("ritnet_model_sha256"),
            "rgb_config_digest": rgb_manifest.get("config_digest"),
            "rgb_model_sha256": {key: value.get("sha256") for key, value in rgb_manifest.get("models", {}).items() if isinstance(value, Mapping)},
            "nir_analysis_size": extension_manifest.get("analysis_size") or nir_manifest.get("effective_parameters", {}).get("ritnet_analysis_size"),
            "nir_input_size": extension_manifest.get("input_size") or nir_manifest.get("effective_parameters", {}).get("ritnet_input_size"),
            "nir_class_mapping": extension_manifest.get("class_mapping"),
        },
        "identity": identity,
        "identity_roster": identity_roster,
        "nir": {
            "rows": int(len(nir)),
            "eyes": {str(key): int(value) for key, value in nir["eye"].value_counts(dropna=False).items()},
            "phases": {str(key): int(value) for key, value in nir["phase"].value_counts(dropna=False).items()},
            "status": {str(key): int(value) for key, value in nir["status"].value_counts(dropna=False).items()},
            "frame_status": {str(key): int(value) for key, value in nir["frame_status"].value_counts(dropna=False).items()},
            "timeline": timeline_stats(nir["unix_ms"]),
            "pupil_found_rows": int(bool_series(nir["fullclass_pupil_found"]).sum()),
            "pupil_fit_valid_rows": int(bool_series(nir["fullclass_pupil_fit_valid"]).sum()),
            "normalization_valid_rows": int(bool_series(nir["fullclass_normalization_valid"]).sum()),
            "pupil_touches_roi_edge_rows": int(bool_series(nir["fullclass_pupil_touches_roi_edge"]).sum()),
            "fields": pupil_field_catalog(nir),
        },
        "rgb": {
            "rows_after_primary_face_and_timestamp_dedup": int(len(rgb)),
            "raw_output_rows_from_manifest": rgb_manifest.get("output_rows"),
            "timeline": timeline_stats(rgb["unix_ms"]),
            "face_rank_raw_counts": {str(key): int(value) for key, value in pd.read_parquet(rgb_path, columns=["face_rank"])["face_rank"].value_counts(dropna=False).items()},
            "fields": rgb_field_catalog(rgb),
            "iris_stability": rgb_iris_stability(rgb),
            "pose_semantics": "Pitch/Roll/Yaw are canonical numeric rotations mapped from raw pose[0:3]; X/Y/Z are mapped from raw pose[3:6], but local code/docs do not establish units or coordinate convention.",
        },
        "alignment": alignment,
        "relations": relations,
        "paired_sample": {
            "path": str(sample_path),
            "sha256": sha256_file(sample_path),
            "requested_unique_nir_timestamps": int(sample_size),
            "actual_unique_nir_timestamps": int(len(sample_times)),
            "rows": int(len(sample)),
        "identity_fields_present": ["site", "session_id", "session_key", "subject", "local_participant_linkage_key", "identity_status", "local_repeat_participant_id", "global_repeat_participant_id", "repeat_session_count", "repeat_visits_beyond_first", "identity_match_basis"],
        },
        "blockers": [
            "NIR and RGB manifests do not record the producing Git commit; exact code provenance remains unresolved.",
            "No authoritative central participant map is available; local repeat IDs remain provisional and must not be used as global IDs.",
            "Pose X/Y/Z units and coordinate semantics are not established by the local output contract; use them only as raw numeric diagnostic fields pending source confirmation.",
            "Current NIR/PIR values are known invalid for formal scientific interpretation; this audit is validation/provenance only and does not change formal tables or statistics.",
            "NIR pupil pixels and RGB iris/face pixels are cross-camera nuisance proxies, not a calibrated physical pupil scale.",
        ],
        "interpretation_boundary": "Engineering audit only. Do not report relation directions, p-values, or candidate variables as scientific findings before NIR geometry validation, identity reconciliation, and a frozen multimodal analysis contract.",
    }
    schema = {"audit_version": AUDIT_VERSION, "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in sample.dtypes.items()]}
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "sample_schema.json", schema)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nir-csv", type=Path, default=DEFAULT_NIR)
    parser.add_argument("--rgb-parquet", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--sart-registry", type=Path, default=DEFAULT_SART)
    parser.add_argument("--participant-info-workbook", type=Path, default=DEFAULT_PARTICIPANT_INFO)
    parser.add_argument("--identity-summary-workbook", type=Path, default=DEFAULT_IDENTITY_SUMMARY)
    parser.add_argument("--repeat-registry", type=Path, default=None, help="External non-PII repeat registry keyed by experiment_ids.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multimodal_pupil_audit"))
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--tolerance-ms", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_audit(args.nir_csv, args.rgb_parquet, args.sart_registry, args.participant_info_workbook, args.identity_summary_workbook, args.output_dir, args.sample_size, args.tolerance_ms, args.repeat_registry)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "paired_sample": summary["paired_sample"], "alignment": summary["alignment"], "identity": summary["identity"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
