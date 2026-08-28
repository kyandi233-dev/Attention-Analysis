"""Compare pupil ellipse post-processors on frozen RITnet hard-label evidence.

This is a validation tool, not a production runner. It never changes the RITnet
model, YOLO boxes, source videos, or final cohort schema. The benchmark compares:

1. historical largest pupil contour -> OpenCV fitEllipse;
2. current project primary-iris-topology pupil component -> OpenCV fitEllipse;
3. EllSeg PartSeg semantic boundary -> ElliFit -> deterministic RANSAC.

The preferred input is ``qc/qc_pixel_evidence.npz`` produced by the final NIR
QC pipeline because it contains exact 400x640 hard labels plus the source-valid
mask for a bounded set of eyes. A plain ``.npy`` label map is also supported for
manually exported bad cases.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from ritnet_fullclass_metric_adapter import _primary_pupil_component
from ritnet_native_metrics import _component_metrics, _ellipse_geometry, validate_native_labels
from ritnet_pupil_geometry import (
    PUPIL_GEOMETRY_VERSION,
    _canonicalize_opencv_geometry,
    fit_ellseg_partseg_pupil_geometry,
)


BENCHMARK_VERSION = "pupil-geometry-benchmark-v1"
METHOD_LEGACY = "legacy-largest-contour-opencv"
METHOD_TOPOLOGY = "primary-iris-topology-opencv"
METHOD_ELLSEG = "ellseg-partseg-ellifit"
METHODS = (METHOD_LEGACY, METHOD_TOPOLOGY, METHOD_ELLSEG)

CSV_FIELDS = (
    "benchmark_version",
    "pupil_geometry_version",
    "source",
    "record_index",
    "subject",
    "phase",
    "phase_segment",
    "frame_idx",
    "eye",
    "reasons",
    "pupil_component_count",
    "pupil_largest_component_fraction",
    "method",
    "geometry_method",
    "fit_valid",
    "geometry_failure_reason",
    "center_x",
    "center_y",
    "short_axis",
    "long_axis",
    "angle_deg",
    "contour_area",
    "ellipse_area",
    "equiv_diameter",
    "geom_mean_diameter",
    "whole_mask_touches_edge",
    "largest_contour_touches_edge",
    "valid_boundary_point_count",
    "ransac_used",
    "ransac_inlier_count",
    "ransac_inlier_fraction",
    "ellipse_fit_error",
    "axis_ratio",
    "contour_to_ellipse_area_ratio",
)


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return str(value)


def _meta_value(array: np.ndarray | None, index: int, default: Any = "") -> Any:
    if array is None:
        return default
    values = np.asarray(array)
    if values.ndim == 0:
        return values.item()
    if index >= len(values):
        return default
    value = values[index]
    return value.item() if isinstance(value, np.generic) else value


def _full_valid_mask() -> np.ndarray:
    return np.ones((400, 640), dtype=bool)


def load_records(
    *,
    evidence: Path | None,
    labels_npy: Path | None,
    valid_mask_npy: Path | None,
) -> list[dict[str, Any]]:
    if (evidence is None) == (labels_npy is None):
        raise ValueError("provide exactly one of --evidence or --labels-npy")

    if evidence is not None:
        with np.load(evidence, allow_pickle=False) as payload:
            if "labels" not in payload:
                raise ValueError("evidence NPZ is missing labels")
            labels = np.asarray(payload["labels"])
            if labels.ndim != 3 or labels.shape[1:] != (400, 640):
                raise ValueError(f"evidence labels must be [N,400,640], got {labels.shape}")
            valid_masks = (
                np.asarray(payload["valid_source_mask"], dtype=bool)
                if "valid_source_mask" in payload
                else np.ones(labels.shape, dtype=bool)
            )
            if valid_masks.shape != labels.shape:
                raise ValueError(
                    f"valid_source_mask shape {valid_masks.shape} != labels shape {labels.shape}"
                )
            subject_value = _scalar_text(payload["subject"]) if "subject" in payload else ""
            records: list[dict[str, Any]] = []
            for index in range(labels.shape[0]):
                label_map = validate_native_labels(np.ascontiguousarray(labels[index], dtype=np.uint8))
                valid = np.ascontiguousarray(valid_masks[index], dtype=bool)
                if not valid.any():
                    raise ValueError(f"record {index} has no source-valid pixels")
                records.append(
                    {
                        "labels": label_map,
                        "valid": valid,
                        "source": str(evidence),
                        "record_index": index,
                        "subject": subject_value,
                        "phase": _meta_value(payload["phase"] if "phase" in payload else None, index),
                        "phase_segment": _meta_value(
                            payload["phase_segment"] if "phase_segment" in payload else None,
                            index,
                        ),
                        "frame_idx": _meta_value(
                            payload["frame_idx"] if "frame_idx" in payload else None,
                            index,
                        ),
                        "eye": _meta_value(payload["eye"] if "eye" in payload else None, index),
                        "reasons": _meta_value(
                            payload["reasons"] if "reasons" in payload else None,
                            index,
                        ),
                    }
                )
            return records

    assert labels_npy is not None
    labels = np.load(labels_npy, allow_pickle=False)
    if labels.ndim == 2:
        labels = labels[None, ...]
    if labels.ndim != 3 or labels.shape[1:] != (400, 640):
        raise ValueError(f"labels NPY must be [400,640] or [N,400,640], got {labels.shape}")

    if valid_mask_npy is None:
        valid_masks = np.ones(labels.shape, dtype=bool)
    else:
        valid_masks = np.load(valid_mask_npy, allow_pickle=False)
        if valid_masks.ndim == 2:
            valid_masks = valid_masks[None, ...]
        valid_masks = np.asarray(valid_masks, dtype=bool)
        if valid_masks.shape != labels.shape:
            raise ValueError(
                f"valid mask shape {valid_masks.shape} != labels shape {labels.shape}"
            )

    records = []
    for index in range(labels.shape[0]):
        records.append(
            {
                "labels": validate_native_labels(np.ascontiguousarray(labels[index], dtype=np.uint8)),
                "valid": np.ascontiguousarray(valid_masks[index], dtype=bool),
                "source": str(labels_npy),
                "record_index": index,
                "subject": "",
                "phase": "",
                "phase_segment": "",
                "frame_idx": "",
                "eye": "",
                "reasons": "",
            }
        )
    return records


def _legacy_geometry(labels: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    pupil = (labels == 3) & valid
    geometry, _ = _ellipse_geometry(pupil)
    result = dict(geometry)
    result["geometry_method"] = "opencv-fitellipse-largest-pupil-contour-legacy"
    result["geometry_failure_reason"] = None if result.get("fit_valid") else "opencv_fit_invalid"
    result["valid_boundary_point_count"] = None
    result["ransac_used"] = False
    result["ransac_inlier_count"] = None
    result["ransac_inlier_fraction"] = None
    result["ellipse_fit_error"] = None
    if result.get("fit_valid"):
        long_axis = float(result["long_axis"])
        short_axis = float(result["short_axis"])
        result["axis_ratio"] = float(short_axis / long_axis) if long_axis > 0 else None
        ellipse_area = float(result["ellipse_area"])
        contour_area = float(result["contour_area"])
        result["contour_to_ellipse_area_ratio"] = (
            float(contour_area / ellipse_area) if ellipse_area > 0 else None
        )
    else:
        result["axis_ratio"] = None
        result["contour_to_ellipse_area_ratio"] = None
    return result


def _topology_geometry(labels: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    selected = _primary_pupil_component(labels, valid)
    geometry, _ = _ellipse_geometry(selected)
    result = _canonicalize_opencv_geometry(geometry)
    result["geometry_method"] = "primary-iris-topology-opencv-fitellipse"
    return result


def compare_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels = np.asarray(record["labels"], dtype=np.uint8)
    valid = np.asarray(record["valid"], dtype=bool)
    pupil = (labels == 3) & valid
    component_count, largest_fraction = _component_metrics(pupil)

    geometries = {
        METHOD_LEGACY: _legacy_geometry(labels, valid),
        METHOD_TOPOLOGY: _topology_geometry(labels, valid),
        METHOD_ELLSEG: fit_ellseg_partseg_pupil_geometry(labels, valid),
    }

    rows: list[dict[str, Any]] = []
    base = {
        "benchmark_version": BENCHMARK_VERSION,
        "pupil_geometry_version": PUPIL_GEOMETRY_VERSION,
        "source": record.get("source", ""),
        "record_index": record.get("record_index", ""),
        "subject": record.get("subject", ""),
        "phase": record.get("phase", ""),
        "phase_segment": record.get("phase_segment", ""),
        "frame_idx": record.get("frame_idx", ""),
        "eye": record.get("eye", ""),
        "reasons": record.get("reasons", ""),
        "pupil_component_count": component_count,
        "pupil_largest_component_fraction": largest_fraction,
    }
    geometry_fields = CSV_FIELDS[13:]
    for method in METHODS:
        geometry = geometries[method]
        row = dict(base)
        row["method"] = method
        for field in geometry_fields:
            if field == "method":
                continue
            row[field] = geometry.get(field)
        rows.append(row)
    return rows


def _normalized_geometry_for_drawing(method: str, geometry: Mapping[str, Any]) -> dict[str, Any]:
    if not geometry.get("fit_valid"):
        return dict(geometry)
    if method == METHOD_LEGACY:
        return _canonicalize_opencv_geometry(dict(geometry))
    return dict(geometry)


def _segmentation_canvas(labels: np.ndarray) -> np.ndarray:
    canvas = np.zeros((400, 640, 3), dtype=np.uint8)
    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    }
    for class_id, color in palette.items():
        canvas[labels == class_id] = color
    return canvas


def _draw_ellipse(canvas: np.ndarray, geometry: Mapping[str, Any], color: tuple[int, int, int]) -> None:
    if not geometry.get("fit_valid"):
        return
    values = [
        geometry.get("center_x"),
        geometry.get("center_y"),
        geometry.get("long_axis"),
        geometry.get("short_axis"),
        geometry.get("angle_deg"),
    ]
    if any(value is None for value in values):
        return
    cx, cy, long_axis, short_axis, angle = map(float, values)
    if long_axis <= 0 or short_axis <= 0:
        return
    cv2.ellipse(
        canvas,
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(long_axis / 2))), max(1, int(round(short_axis / 2)))),
        angle,
        0,
        360,
        color,
        2,
        cv2.LINE_AA,
    )


def render_comparison(record: Mapping[str, Any]) -> np.ndarray:
    labels = np.asarray(record["labels"], dtype=np.uint8)
    valid = np.asarray(record["valid"], dtype=bool)
    geometries = {
        METHOD_LEGACY: _legacy_geometry(labels, valid),
        METHOD_TOPOLOGY: _topology_geometry(labels, valid),
        METHOD_ELLSEG: fit_ellseg_partseg_pupil_geometry(labels, valid),
    }
    colors = {
        METHOD_LEGACY: (255, 255, 255),
        METHOD_TOPOLOGY: (0, 255, 255),
        METHOD_ELLSEG: (255, 0, 255),
    }
    panels: list[np.ndarray] = []
    for method in METHODS:
        panel = _segmentation_canvas(labels)
        if not valid.all():
            panel[~valid] = (48, 48, 48)
        normalized = _normalized_geometry_for_drawing(method, geometries[method])
        _draw_ellipse(panel, normalized, colors[method])
        status = "valid" if geometries[method].get("fit_valid") else "invalid"
        cv2.putText(
            panel,
            f"{method} | {status}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence", type=Path, help="qc_pixel_evidence.npz")
    source.add_argument("--labels-npy", type=Path, help="one or more [400,640] hard-label maps")
    parser.add_argument("--valid-mask-npy", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_records(
        evidence=args.evidence,
        labels_npy=args.labels_npy,
        valid_mask_npy=args.valid_mask_npy,
    )
    rows = [row for record in records for row in compare_record(record)]
    write_rows(args.output_csv, rows)

    if args.overlay_dir is not None:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            image = render_comparison(record)
            frame = str(record.get("frame_idx", "")) or f"record{record['record_index']:03d}"
            eye = str(record.get("eye", "")) or "eye"
            name = f"{record['record_index']:03d}_{frame}_{eye}.png".replace("/", "-").replace("\\", "-")
            ok = cv2.imwrite(str(args.overlay_dir / name), image)
            if not ok:
                raise RuntimeError(f"failed to write overlay: {args.overlay_dir / name}")

    print(
        f"records={len(records)} methods={len(METHODS)} rows={len(rows)} "
        f"csv={args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
